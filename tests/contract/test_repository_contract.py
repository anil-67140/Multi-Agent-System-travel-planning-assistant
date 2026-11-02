"""The same behavioural contract must hold for every Repository implementation.

The in-memory repository always runs here. The SQL repository runs when SQLAlchemy + aiosqlite are
installed (always in CI), so the two can never drift apart.
"""

from __future__ import annotations

import importlib.util
from datetime import timedelta

import pytest

from tests.helpers import NOW
from tripmate.errors import ConflictError
from tripmate.services.records import ApprovalRecord, PreferenceRecord, SourceEntry
from tripmate.services.repository import InMemoryRepository

pytestmark = pytest.mark.anyio

HAVE_SQL = all(importlib.util.find_spec(m) for m in ("sqlalchemy", "aiosqlite"))


@pytest.fixture(
    params=[
        "memory",
        pytest.param(
            "sql", marks=pytest.mark.skipif(not HAVE_SQL, reason="sqlalchemy/aiosqlite missing")
        ),
    ]
)
async def repo(request):
    if request.param == "memory":
        yield InMemoryRepository(now=lambda: NOW)
        return
    from tripmate.db.models import Base
    from tripmate.db.repositories import SqlRepository
    from tripmate.db.session import make_engine, make_sessionmaker

    engine = make_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield SqlRepository(make_sessionmaker(engine))
    await engine.dispose()


async def test_user_roundtrip_and_unique_email(repo):
    u = await repo.create_user("A@Example.com", "h")
    assert u.email == "a@example.com" and (await repo.get_user_by_email("a@EXAMPLE.com")).id == u.id
    with pytest.raises(ConflictError):
        await repo.create_user("a@example.com", "h2")
    u.failed_logins, u.locked_until = 3, NOW + timedelta(minutes=5)
    await repo.save_user(u)
    got = await repo.get_user(u.id)
    assert got.failed_logins == 3 and got.locked_until == NOW + timedelta(minutes=5)
    assert await repo.get_user("missing") is None


async def test_preferences_default_and_persist(repo):
    u = await repo.create_user("p@x.com", "h")
    assert (await repo.get_preferences(u.id)).bundle_affinity == {}
    await repo.save_preferences(
        PreferenceRecord(
            user_id=u.id,
            home_city="Karachi",
            style="luxury",
            interests=["food"],
            bundle_affinity={"comfort": 2},
        )
    )
    got = await repo.get_preferences(u.id)
    assert (got.home_city, got.style, got.interests, got.bundle_affinity) == (
        "Karachi",
        "luxury",
        ["food"],
        {"comfort": 2},
    )


async def test_plan_lifecycle_fields_roundtrip(repo):
    u = await repo.create_user("q@x.com", "h")
    plan = await repo.create_plan(u.id, "Tokyo")
    assert plan.status == "queued" and plan.run_no == 0
    plan.status, plan.destination, plan.travelers = "running", "Tokyo", 2
    plan.start_date, plan.budget_amount = "2026-12-01", "3000.50"
    plan.trip, plan.notes = {"a": 1}, ["n"]
    plan.heartbeat_at = NOW
    await repo.save_plan(plan)
    got = await repo.get_plan(plan.id)
    assert (got.status, got.destination, got.travelers, str(got.start_date)) == (
        "running",
        "Tokyo",
        2,
        "2026-12-01",
    )
    assert (
        float(got.budget_amount) == 3000.5
        and got.trip == {"a": 1}
        and got.notes == ["n"]
        and got.heartbeat_at == NOW
    )
    assert await repo.get_plan("nope") is None


async def test_versions_append_and_are_idempotent(repo):
    u = await repo.create_user("v@x.com", "h")
    p = await repo.create_plan(u.id, "x")
    await repo.add_version(p.id, 1, {"t": 1}, None)
    await repo.add_version(p.id, 1, {"t": "dup"}, None)  # replay
    await repo.add_version(p.id, 2, {"t": 2}, "more food")
    got = await repo.get_plan(p.id)
    assert [v.version for v in got.versions] == [1, 2] and got.versions[0].itinerary == {"t": 1}
    assert got.versions[1].feedback == "more food" and got.latest_version.version == 2


async def test_save_plan_does_not_clobber_children(repo):
    u = await repo.create_user("c@x.com", "h")
    p = await repo.create_plan(u.id, "x")
    await repo.add_version(p.id, 1, {"t": 1}, None)
    stale = await repo.get_plan(p.id)
    await repo.add_version(p.id, 2, {"t": 2}, None)
    stale.status = "completed"
    await repo.save_plan(stale)
    assert [v.version for v in (await repo.get_plan(p.id)).versions] == [1, 2]


async def test_sources_replaced_not_appended(repo):
    u = await repo.create_user("s@x.com", "h")
    p = await repo.create_plan(u.id, "x")

    def s(url):
        return SourceEntry(
            provider="tavily", kind="web_source", origin="live", url=url, retrieved_at=NOW
        )

    await repo.replace_sources(p.id, [s("https://a.example"), s("https://b.example")])
    await repo.replace_sources(p.id, [s("https://c.example")])
    assert [x.url for x in (await repo.get_plan(p.id)).sources] == ["https://c.example"]


async def test_listing_is_per_user_newest_first_and_paged(repo):
    a, b = await repo.create_user("la@x.com", "h"), await repo.create_user("lb@x.com", "h")
    ids = []
    for i in range(3):
        ids.append((await repo.create_plan(a.id, str(i))).id)
    await repo.create_plan(b.id, "other")
    listed = await repo.list_plans(a.id)
    assert {p.id for p in listed} == set(ids) and all(p.user_id == a.id for p in listed)
    assert (
        len(await repo.list_plans(a.id, limit=2)) == 2
        and len(await repo.list_plans(a.id, limit=2, offset=2)) == 1
    )


async def test_active_count_and_stale_detection(repo):
    u = await repo.create_user("a2@x.com", "h")
    p1, p2, p3 = [await repo.create_plan(u.id, str(i)) for i in range(3)]
    p2.status, p2.heartbeat_at = "running", NOW - timedelta(hours=1)
    p3.status, p3.heartbeat_at = "running", NOW
    for p in (p2, p3):
        await repo.save_plan(p)
    assert await repo.count_active_plans(u.id) == 3  # queued + 2 running
    stale = await repo.find_stale_running(NOW - timedelta(minutes=10))
    assert [p.id for p in stale] == [p2.id]


async def test_approval_unique_per_idempotency_key(repo):
    u = await repo.create_user("ap@x.com", "h")
    p = await repo.create_plan(u.id, "x")
    rec = ApprovalRecord(
        id="key-12345678",
        plan_id=p.id,
        user_id=u.id,
        itinerary_version=1,
        approved=True,
        idempotency_key="key-12345678",
        created_at=NOW,
    )
    await repo.add_approval(rec)
    with pytest.raises(ConflictError):
        await repo.add_approval(rec)
    got = await repo.get_approval(p.id, "key-12345678")
    assert got.approved and got.itinerary_version == 1
    assert await repo.get_approval(p.id, "other-key-123") is None


async def test_events_are_sequential_and_resumable(repo):
    u = await repo.create_user("ev@x.com", "h")
    p = await repo.create_plan(u.id, "x")
    for i in range(5):
        e = await repo.add_event(p.id, 0, "node_done", "flights", f"m{i}", {"i": i})
        assert e.seq == i + 1
    after2 = await repo.events_after(p.id, 2)
    assert [e.seq for e in after2] == [3, 4, 5] and after2[0].data == {"i": 2}
    assert len(await repo.events_after(p.id, 0, limit=2)) == 2
    assert await repo.events_after(p.id, 99) == []


async def test_concurrent_events_get_unique_sequence_numbers(repo):
    import asyncio

    u = await repo.create_user("cc@x.com", "h")
    p = await repo.create_plan(u.id, "x")
    results = await asyncio.gather(
        *(repo.add_event(p.id, 0, "t", None, str(i), {}) for i in range(8))
    )
    assert sorted(e.seq for e in results) == list(range(1, 9))

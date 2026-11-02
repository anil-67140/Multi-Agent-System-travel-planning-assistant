from __future__ import annotations

import pytest

from tripmate.observability.redaction import REDACTED, redact, redact_text


@pytest.mark.parametrize(
    "secret",
    [
        "gsk_" + "a" * 30,
        "AIza" + "b" * 30,
        "tvly-" + "c" * 20,
        "sk-" + "d" * 20,
        "Bearer abcdefghijkl",
        "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4",
    ],
)
def test_known_secret_shapes_are_removed(secret):
    assert secret not in redact_text(f"calling api with {secret} now")


def test_db_url_password_is_masked_but_host_kept():
    out = redact_text("postgresql+psycopg://app:hunter2@db.internal:5432/trip")
    assert "hunter2" not in out and "db.internal" in out and "app:" in out


def test_email_is_partially_masked():
    assert redact_text("mail nova@example.com now") == "mail n***@example.com now"


def test_sensitive_keys_are_redacted_at_any_depth():
    out = redact(
        {
            "user": "a",
            "Authorization": "x",
            "nested": {"api_key": "y", "ok": [{"password": "z", "n": 1}]},
        }
    )
    assert out["Authorization"] == REDACTED and out["nested"]["api_key"] == REDACTED
    assert (
        out["nested"]["ok"][0]["password"] == REDACTED
        and out["nested"]["ok"][0]["n"] == 1
        and out["user"] == "a"
    )


def test_deep_structures_do_not_recurse_forever():
    deep: dict = {}
    cur = deep
    for _ in range(50):
        cur["a"] = {}
        cur = cur["a"]
    redact(deep)


def test_non_strings_pass_through():
    assert redact({"n": 5, "f": 1.5, "b": None}) == {"n": 5, "f": 1.5, "b": None}

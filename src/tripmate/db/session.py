"""Async engine/session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def make_engine(url: str, *, echo: bool = False) -> AsyncEngine:
    kwargs: dict = {"echo": echo, "pool_pre_ping": True}
    if not url.startswith("sqlite"):
        kwargs.update(pool_size=10, max_overflow=10, pool_recycle=1800)
    return create_async_engine(url, **kwargs)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all_for_dev(engine: AsyncEngine) -> None:
    """Dev/test convenience. Production uses Alembic migrations."""
    from tripmate.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

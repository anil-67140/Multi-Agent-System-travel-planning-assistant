"""LangGraph checkpoint storage.

PostgreSQL (``AsyncPostgresSaver``) in production so a paused plan survives restarts and any
replica can resume it; ``MemorySaver`` for tests/dev (explicitly not durable).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

log = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def open_checkpointer(postgres_url: str | None) -> AsyncIterator[Any]:
    if not postgres_url:
        from langgraph.checkpoint.memory import MemorySaver

        log.warning("using in-memory checkpoints: workflows will not survive a restart")
        yield MemorySaver()
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    # autocommit + dict_row are required by the saver; a pool lets many plans checkpoint concurrently.
    pool = AsyncConnectionPool(
        conninfo=postgres_url,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()  # idempotent: creates/migrates checkpoint tables
        yield saver
    finally:
        await pool.close()

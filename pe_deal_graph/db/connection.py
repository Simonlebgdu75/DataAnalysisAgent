from __future__ import annotations

import asyncpg

from pe_deal_graph.config import (
    LINKEDIN_DB,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_USER,
)

_linkedin_pool: asyncpg.Pool | None = None


async def get_linkedin_pool() -> asyncpg.Pool:
    global _linkedin_pool
    if _linkedin_pool is None or _linkedin_pool._closed:
        _linkedin_pool = await asyncpg.create_pool(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=LINKEDIN_DB,
            min_size=2,
            max_size=5,
            command_timeout=60,
            max_inactive_connection_lifetime=120,
        )
    return _linkedin_pool


async def get_pool() -> asyncpg.Pool:
    return await get_linkedin_pool()


async def close_pools() -> None:
    global _linkedin_pool
    if _linkedin_pool and not _linkedin_pool._closed:
        await _linkedin_pool.close()
        _linkedin_pool = None

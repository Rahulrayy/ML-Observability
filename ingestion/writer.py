import asyncio
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base, InferenceLog


class BatchWriter:
    def __init__(self, database_url: str, batch_size: int = 100, flush_interval: float = 2.0):
        async_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        self._engine = create_async_engine(async_url, pool_size=5)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._pending: list[dict] = []
        self._lock = asyncio.Lock()

    async def init_db(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def enqueue(self, records: list[dict[str, Any]]) -> None:
        async with self._lock:
            self._pending.extend(records)
            if len(self._pending) >= self._batch_size:
                await self._flush()

    async def run_flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            async with self._lock:
                await self._flush()

    async def _flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending[: self._batch_size], self._pending[self._batch_size :]
        async with self._session_factory() as session:
            async with session.begin():
                session.add_all([
                    InferenceLog(
                        inference_id=r["inference_id"],
                        model_id=r["model_id"],
                        timestamp=datetime.fromtimestamp(r["timestamp"], tz=timezone.utc),
                        features=r["features"],
                        prediction=r["prediction"],
                        confidence=r.get("confidence"),
                        ground_truth=r.get("ground_truth"),
                        latency_ms=r.get("latency_ms"),
                        metadata_=r.get("metadata", {}),
                    )
                    for r in batch
                ])

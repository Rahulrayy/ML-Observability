import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .models import Base, InferenceLog
from .writer import BatchWriter

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
API_KEY = os.environ.get("API_KEY", "dev")

writer: BatchWriter
redis_client: aioredis.Redis
_async_engine = None
_async_sessions = None
_drift_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drift")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

DRIFT_INTERVAL = int(os.environ.get("DRIFT_INTERVAL_SECONDS", 900))  # 15 min default


def _run_drift_sync(database_url: str) -> None:
    from sqlalchemy import create_engine, text
    from drift.engine import DriftEngine
    from alerts.manager import AlertManager
    from graph.dag import PipelineDAG
    from graph.propagator import RiskPropagator

    sync_engine = create_engine(database_url)
    with sync_engine.connect() as conn:
        model_ids = [
            row[0]
            for row in conn.execute(text("SELECT DISTINCT model_id FROM inference_logs")).fetchall()
        ]
    sync_engine.dispose()

    dag = PipelineDAG.from_yaml("configs/dag_definition.yaml")
    propagator = RiskPropagator(dag, model_path="models/pipeline_gnn.pt")
    alert_manager = AlertManager(database_url)
    drift = DriftEngine(database_url)

    for mid in model_ids:
        results = drift.run(mid)
        if not results:
            continue

        psi_scores = {r.feature_name: r.statistic for r in results if r.test_type == "psi"}
        kl_score = next((r.statistic for r in results if r.test_type == "kl"), 0.0)
        max_psi = max(psi_scores.values(), default=0.0)
        feature_risk = min(max_psi / 0.25, 1.0)
        pred_risk = min(kl_score / 0.3, 1.0)
        node_features = {
            "raw_data": [feature_risk] * 8,
            "churn_model_v2": [pred_risk] * 8,
        }
        dag_risk_scores = propagator.propagate(node_features)
        alert_manager.process(mid, results, dag_risk_scores)


async def _drift_loop(database_url: str) -> None:
    while True:
        await asyncio.sleep(DRIFT_INTERVAL)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(_drift_executor, _run_drift_sync, database_url)
        except Exception as exc:
            print(f"[drift] error: {exc}")


def require_api_key(key: str = Security(api_key_header)) -> str:
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


@asynccontextmanager
async def lifespan(app: FastAPI):
    global writer, redis_client, _async_engine, _async_sessions
    writer = BatchWriter(DATABASE_URL)
    await writer.init_db()
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    async_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    _async_engine = create_async_engine(async_url)
    _async_sessions = async_sessionmaker(_async_engine, expire_on_commit=False)
    asyncio.create_task(writer.run_flush_loop())
    drift_task = asyncio.create_task(_drift_loop(DATABASE_URL))
    yield
    drift_task.cancel()
    await redis_client.aclose()
    await _async_engine.dispose()


app = FastAPI(title="ML Observability Ingestion API", lifespan=lifespan)


class InferenceBatch(BaseModel):
    records: list[dict[str, Any]]


class GroundTruthPayload(BaseModel):
    inference_id: str
    model_id: str
    ground_truth: float


class PipelineEvent(BaseModel):
    node_id: str
    event_type: str
    model_id: str
    details: dict[str, Any] = {}


@app.post("/inferences", dependencies=[Security(require_api_key)])
async def ingest_inferences(payload: InferenceBatch):
    await writer.enqueue(payload.records)
    minute_key = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    for r in payload.records:
        model_id = r.get("model_id", "unknown")
        await redis_client.incr(f"throughput:{model_id}:total")
        await redis_client.incr(f"throughput:{model_id}:minute:{minute_key}")
        await redis_client.expire(f"throughput:{model_id}:minute:{minute_key}", 3600)
    return {"accepted": len(payload.records)}


@app.post("/ground-truth", dependencies=[Security(require_api_key)])
async def ingest_ground_truth(payload: GroundTruthPayload):
    async with _async_sessions() as session:
        await session.execute(
            update(InferenceLog)
            .where(InferenceLog.inference_id == payload.inference_id)
            .values(ground_truth=payload.ground_truth)
        )
        await session.commit()
    return {"updated": payload.inference_id}


@app.post("/pipeline-event", dependencies=[Security(require_api_key)])
async def ingest_pipeline_event(payload: PipelineEvent):
    key = f"pipeline_events:{payload.model_id}"
    value = f"{payload.node_id}:{payload.event_type}:{datetime.now(timezone.utc).isoformat()}"
    await redis_client.lpush(key, value)
    await redis_client.ltrim(key, 0, 999)
    return {"received": True}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", 8000)),
    )

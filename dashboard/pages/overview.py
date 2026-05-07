import os

import pandas as pd
import redis
import streamlit as st
from sqlalchemy import create_engine, text


def render():
    st.title("Live Overview")

    db_url = os.environ.get("DATABASE_URL", "")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    if not db_url:
        st.warning("DATABASE_URL not configured.")
        return

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            model_ids = [row[0] for row in conn.execute(text("SELECT DISTINCT model_id FROM inference_logs")).fetchall()]
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return

    if not model_ids:
        st.info("No models instrumented yet. Run `python demo/setup_demo.py` to populate data.")
        engine.dispose()
        return

    model_id = st.selectbox("Model", model_ids)

    col1, col2, col3 = st.columns(3)

    try:
        r = redis.from_url(redis_url, decode_responses=True)
        redis_total = int(r.get(f"throughput:{model_id}:total") or 0)
    except Exception:
        redis_total = 0

    with engine.connect() as conn:
        db_total = conn.execute(
            text("SELECT COUNT(*) FROM inference_logs WHERE model_id = :mid"),
            {"mid": model_id},
        ).scalar() or 0

    total = redis_total if redis_total > 0 else int(db_total)
    col1.metric("Total Inferences", f"{total:,}")

    with engine.connect() as conn:
        active_alerts = conn.execute(
            text("SELECT COUNT(*) FROM alerts WHERE model_id = :mid AND resolved = false"),
            {"mid": model_id},
        ).scalar() or 0

        avg_pred = conn.execute(
            text("SELECT AVG(prediction) FROM inference_logs WHERE model_id = :mid AND timestamp > NOW() - INTERVAL '1 hour'"),
            {"mid": model_id},
        ).scalar()

    col2.metric("Active Alerts", int(active_alerts))
    col3.metric("Avg Prediction (1h)", f"{float(avg_pred or 0):.3f}")

    st.subheader("Recent Inferences")
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT timestamp, prediction, ground_truth, latency_ms FROM inference_logs WHERE model_id = :mid ORDER BY timestamp DESC LIMIT 100"),
            {"mid": model_id},
        ).fetchall()

    engine.dispose()
    df = pd.DataFrame(rows, columns=["timestamp", "prediction", "ground_truth", "latency_ms"])
    st.dataframe(df, use_container_width=True)

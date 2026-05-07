import os

import plotly.express as px
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text


def render():
    st.title("Model Performance")

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        st.warning("DATABASE_URL not configured.")
        return

    engine = create_engine(db_url)
    with engine.connect() as conn:
        model_ids = [row[0] for row in conn.execute(text("SELECT DISTINCT model_id FROM inference_logs")).fetchall()]

    if not model_ids:
        st.info("No models instrumented yet.")
        engine.dispose()
        return

    model_id = st.selectbox("Model", model_ids)

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT
                    date_trunc('hour', timestamp) AS hour,
                    COUNT(*) AS count,
                    AVG(prediction) AS avg_pred,
                    AVG(CASE WHEN ground_truth IS NOT NULL THEN ABS(prediction - ground_truth) END) AS mae
                FROM inference_logs
                WHERE model_id = :mid
                GROUP BY 1 ORDER BY 1
            """),
            {"mid": model_id},
        ).fetchall()
    engine.dispose()

    df = pd.DataFrame(rows, columns=["hour", "count", "avg_pred", "mae"])

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(df, x="hour", y="count", title="Inference Volume"), use_container_width=True)
    with col2:
        st.plotly_chart(px.line(df, x="hour", y="mae", title="Mean Absolute Error (when GT available)"), use_container_width=True)

    st.plotly_chart(
        px.line(df, x="hour", y="avg_pred", title="Average Prediction Score Over Time"),
        use_container_width=True,
    )

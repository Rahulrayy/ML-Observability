import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

from dashboard.components.drift_gauge import drift_gauge


def render():
    st.title("Feature Drift")

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        st.warning("DATABASE_URL not configured.")
        return

    engine = create_engine(db_url)
    with engine.connect() as conn:
        model_ids = [row[0] for row in conn.execute(text("SELECT DISTINCT model_id FROM drift_results")).fetchall()]

    if not model_ids:
        st.info("No drift results yet. The drift engine runs every 15 minutes.")
        engine.dispose()
        return

    model_id = st.selectbox("Model", model_ids)

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT feature_name, test_type, statistic, p_value, drift_detected, severity, computed_at
                FROM drift_results WHERE model_id = :mid
                ORDER BY computed_at DESC LIMIT 1000
            """),
            {"mid": model_id},
        ).fetchall()
    engine.dispose()

    df = pd.DataFrame(rows, columns=["feature", "test_type", "statistic", "p_value", "drift_detected", "severity", "computed_at"])

    st.subheader("Current Drift (PSI)")
    latest_psi = df[df["test_type"] == "psi"].groupby("feature").first().reset_index()
    if not latest_psi.empty:
        cols = st.columns(min(len(latest_psi), 4))
        for i, (_, row) in enumerate(latest_psi.iterrows()):
            with cols[i % len(cols)]:
                st.plotly_chart(drift_gauge(float(row["statistic"]), row["feature"]), use_container_width=True)

    st.subheader("PSI Over Time")
    psi_df = df[df["test_type"] == "psi"]
    if not psi_df.empty:
        fig = px.line(psi_df, x="computed_at", y="statistic", color="feature",
                      labels={"statistic": "PSI", "computed_at": "Time"})
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Warning")
        fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Critical")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("All Drift Results"):
        st.dataframe(df, use_container_width=True)

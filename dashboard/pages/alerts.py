import os

import streamlit as st
from sqlalchemy import create_engine, text


def render():
    st.title("Alerts")

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        st.warning("DATABASE_URL not configured.")
        return

    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, model_id, severity, alert_type, title, message, explanation, resolved, created_at
                FROM alerts ORDER BY created_at DESC LIMIT 100
            """)
        ).fetchall()
    engine.dispose()

    if not rows:
        st.success("No alerts - all systems healthy.")
        return

    severity_filter = st.multiselect(
        "Severity", ["CRITICAL", "WARNING", "INFO"],
        default=["CRITICAL", "WARNING", "INFO"],
    )

    for row in rows:
        _, model_id, severity, _, title, message, explanation, resolved, created_at = row
        if severity not in severity_filter:
            continue

        status = "Resolved" if resolved else "Active"
        label = f"[{severity}] {title} - {created_at.strftime('%Y-%m-%d %H:%M')} - {status}"

        with st.expander(label):
            st.write(f"**Model:** {model_id}")
            st.write(f"**Message:** {message}")
            if explanation:
                st.subheader("LLM Root Cause Analysis")
                st.markdown(explanation)

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ML Observability",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

_PAGES = {
    "Overview": "overview",
    "Feature Drift": "drift",
    "Pipeline DAG": "pipeline",
    "Alerts": "alerts",
    "Model Performance": "models",
}

with st.sidebar:
    st.title("ML Observability")
    st.caption("Real-time model monitoring")
    st.divider()
    selected = st.radio("Navigate", list(_PAGES.keys()), label_visibility="collapsed")

page = _PAGES[selected]

if page == "overview":
    from dashboard.pages.overview import render
elif page == "drift":
    from dashboard.pages.drift import render
elif page == "pipeline":
    from dashboard.pages.pipeline import render
elif page == "alerts":
    from dashboard.pages.alerts import render
elif page == "models":
    from dashboard.pages.models import render

render()

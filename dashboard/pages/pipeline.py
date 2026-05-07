import os

import streamlit as st
from sqlalchemy import create_engine, text

from graph.dag import PipelineDAG
from graph.propagator import RiskPropagator
from dashboard.components.dag_visualiser import render_dag


def _get_node_features(db_url: str, model_id: str) -> dict[str, list[float]]:
    """Build node feature vectors from latest drift results."""
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT feature_name, test_type, statistic
                FROM drift_results
                WHERE model_id = :mid
                AND computed_at = (SELECT MAX(computed_at) FROM drift_results WHERE model_id = :mid)
            """), {"mid": model_id}).fetchall()
    finally:
        engine.dispose()

    if not rows:
        return {}

    psi_scores = {r[0]: float(r[2]) for r in rows if r[1] == "psi"}
    kl_score = next((float(r[2]) for r in rows if r[1] == "kl"), 0.0)

    max_psi = max(psi_scores.values(), default=0.0)
    # Normalise PSI by critical threshold (0.25)
    feature_risk = min(max_psi / 0.25, 1.0)
    pred_risk = min(kl_score / 0.3, 1.0)

    return {
        "raw_data": [feature_risk] * 8,
        "churn_model_v2": [pred_risk] * 8,
    }


def render():
    st.title("Pipeline DAG")

    db_url = os.environ.get("DATABASE_URL", "")
    dag_path = "configs/dag_definition.yaml"

    try:
        dag = PipelineDAG.from_yaml(dag_path)
    except FileNotFoundError:
        st.error(f"DAG config not found at {dag_path}")
        return

    propagator = RiskPropagator(dag, model_path="models/pipeline_gnn.pt")

    if db_url:
        try:
            node_features = _get_node_features(db_url, "churn_model_v2")
            if node_features:
                propagator.propagate(node_features)
        except Exception:
            pass

    risk_scores = {n.id: n.risk_score for n in dag.nodes()}

    st.caption("Node colour: green = low risk, orange = medium, red = high. Scores propagated from latest drift results.")
    render_dag(dag.nodes(), dag.edges(), risk_scores)

    st.subheader("Node Risk Scores")
    for node in sorted(dag.nodes(), key=lambda n: -risk_scores.get(n.id, 0.0)):
        score = risk_scores.get(node.id, 0.0)
        risk_label = "LOW" if score < 0.3 else "MED" if score < 0.7 else "HIGH"
        st.write(f"[{risk_label}] **{node.id}** ({node.node_type}) - risk: `{score:.2f}` - {node.description}")

from streamlit_agraph import Config, Edge, Node, agraph


def _risk_color(score: float) -> str:
    if score < 0.3:
        return "#4caf50"
    elif score < 0.7:
        return "#ff9800"
    return "#f44336"


def render_dag(nodes: list, edges: list[tuple[str, str]], risk_scores: dict[str, float]) -> None:
    ag_nodes = [
        Node(
            id=n.id,
            label=n.id,
            title=f"{n.description}\nType: {n.node_type}\nRisk: {risk_scores.get(n.id, 0.0):.2f}",
            color=_risk_color(risk_scores.get(n.id, 0.0)),
            size=20,
        )
        for n in nodes
    ]
    ag_edges = [Edge(source=src, target=dst) for src, dst in edges]
    config = Config(
        width="100%",
        height=400,
        directed=True,
        physics=True,
        hierarchical=True,
    )
    agraph(nodes=ag_nodes, edges=ag_edges, config=config)

import plotly.graph_objects as go


def drift_gauge(value: float, title: str, max_value: float = 0.5) -> go.Figure:
    normalized = min(value / max_value, 1.0) if max_value > 0 else 0.0
    color = "green" if normalized < 0.4 else "orange" if normalized < 0.8 else "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, max_value * 0.4], "color": "#e8f5e9"},
                {"range": [max_value * 0.4, max_value * 0.8], "color": "#fff3e0"},
                {"range": [max_value * 0.8, max_value], "color": "#ffebee"},
            ],
        },
        number={"font": {"size": 18}, "valueformat": ".3f"},
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=0, l=20, r=20))
    return fig

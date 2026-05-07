"""Export a static HTML snapshot of the dashboard for GitHub Pages deployment.

Run while PostgreSQL is live:
    .venv\Scripts\python.exe dashboard/export_static.py
    # output: docs/index.html  (ready for GitHub Pages)
"""
import os
import sys
from datetime import timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

_CSS = """
body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; margin: 0; padding: 0; }
.header { background: #1a1d27; padding: 1.5rem 2rem; border-bottom: 1px solid #2d3046; }
.header h1 { margin: 0; font-size: 1.6rem; color: #fff; }
.header p  { margin: 0.25rem 0 0; color: #8892b0; font-size: 0.9rem; }
.content { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
.metrics { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }
.metric { background: #1a1d27; border: 1px solid #2d3046; border-radius: 8px;
           padding: 1rem 1.5rem; flex: 1; min-width: 150px; }
.metric .label { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: .05em; }
.metric .value { font-size: 2rem; font-weight: 700; color: #fff; margin-top: 0.25rem; }
.card { background: #1a1d27; border: 1px solid #2d3046; border-radius: 8px;
        padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; }
.card h2 { margin: 0 0 1rem; font-size: 1.1rem; color: #cdd6f4; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { text-align: left; padding: 0.5rem 0.75rem; color: #8892b0;
     border-bottom: 1px solid #2d3046; font-weight: 600; }
td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #1e2130; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 0.75rem; font-weight: 700; }
.CRITICAL { background: #3d0f0f; color: #f87171; }
.WARNING  { background: #3d2a0f; color: #fbbf24; }
.INFO     { background: #0f2d3d; color: #60a5fa; }
.footer { text-align: center; color: #4b5563; font-size: 0.8rem; padding: 2rem; }
"""


def _fig_html(fig: go.Figure) -> str:
    fig.update_layout(
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#1a1d27",
        font_color="#e0e0e0",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def export(output_path: str = "docs/index.html") -> None:
    db_url = os.environ.get("DATABASE_URL", "postgresql://mlobs:mlobs@localhost:5432/mlobs")
    engine = create_engine(db_url)

    with engine.connect() as conn:
        model_ids = [
            r[0] for r in conn.execute(text("SELECT DISTINCT model_id FROM inference_logs")).fetchall()
        ]
        if not model_ids:
            print("No inference data found. Run demo/setup_demo.py first.")
            sys.exit(1)

        model_id = model_ids[0]

        total_inferences = conn.execute(
            text("SELECT COUNT(*) FROM inference_logs WHERE model_id = :mid"), {"mid": model_id}
        ).scalar() or 0

        active_alerts = conn.execute(
            text("SELECT COUNT(*) FROM alerts WHERE model_id = :mid AND resolved = false"), {"mid": model_id}
        ).scalar() or 0

        avg_pred = conn.execute(
            text("SELECT AVG(prediction) FROM inference_logs WHERE model_id = :mid"), {"mid": model_id}
        ).scalar() or 0.0

        drift_rows = conn.execute(text("""
            SELECT feature_name, test_type, statistic, p_value, drift_detected, severity, computed_at
            FROM drift_results WHERE model_id = :mid ORDER BY computed_at DESC LIMIT 2000
        """), {"mid": model_id}).fetchall()

        alert_rows = conn.execute(text("""
            SELECT severity, alert_type, title, message, created_at
            FROM alerts WHERE model_id = :mid ORDER BY created_at DESC LIMIT 50
        """), {"mid": model_id}).fetchall()

    engine.dispose()

    drift_df = pd.DataFrame(
        drift_rows,
        columns=["feature", "test_type", "statistic", "p_value", "drift_detected", "severity", "computed_at"],
    )

    # PSI trend chart
    psi_df = drift_df[drift_df["test_type"] == "psi"].copy()
    if not psi_df.empty:
        psi_fig = px.line(
            psi_df, x="computed_at", y="statistic", color="feature",
            title="PSI Over Time", labels={"statistic": "PSI", "computed_at": "Time"},
        )
        psi_fig.add_hline(y=0.1, line_dash="dash", line_color="#fbbf24", annotation_text="Warning (0.10)")
        psi_fig.add_hline(y=0.25, line_dash="dash", line_color="#f87171", annotation_text="Critical (0.25)")
        psi_html = _fig_html(psi_fig)
    else:
        psi_html = "<p style='color:#8892b0'>No PSI data yet — run the drift engine first.</p>"

    # Latest drift summary bar chart
    if not drift_df.empty:
        latest = drift_df[drift_df["test_type"] == "ks"].groupby("feature").first().reset_index()
        bar_fig = px.bar(
            latest, x="feature", y="statistic",
            color="severity",
            color_discrete_map={"CRITICAL": "#f87171", "WARNING": "#fbbf24", "INFO": "#60a5fa"},
            title="Latest KS Statistic per Feature",
        )
        bar_html = _fig_html(bar_fig)
    else:
        bar_html = "<p style='color:#8892b0'>No drift results yet.</p>"

    # Alerts table
    def alert_rows_html(rows):
        if not rows:
            return "<p style='color:#4ade80'>No alerts — all systems healthy.</p>"
        cells = ""
        for r in rows:
            sev, atype, title, message, created_at = r
            ts = created_at.strftime("%Y-%m-%d %H:%M") if hasattr(created_at, "strftime") else str(created_at)
            cells += (
                f"<tr><td><span class='badge {sev}'>{sev}</span></td>"
                f"<td>{ts}</td><td>{title}</td><td style='color:#8892b0'>{message[:120]}...</td></tr>"
            )
        return f"<table><thead><tr><th>Severity</th><th>Time</th><th>Title</th><th>Message</th></tr></thead><tbody>{cells}</tbody></table>"

    from datetime import datetime
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Observability — {model_id}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
  <h1>ML Observability Platform</h1>
  <p>Model: <strong>{model_id}</strong> &nbsp;·&nbsp; Snapshot: {generated_at}</p>
</div>
<div class="content">

  <div class="metrics">
    <div class="metric"><div class="label">Total Inferences</div><div class="value">{int(total_inferences):,}</div></div>
    <div class="metric"><div class="label">Active Alerts</div><div class="value">{int(active_alerts)}</div></div>
    <div class="metric"><div class="label">Avg Prediction</div><div class="value">{float(avg_pred):.3f}</div></div>
    <div class="metric"><div class="label">Features Tracked</div><div class="value">{drift_df['feature'].nunique() if not drift_df.empty else 0}</div></div>
  </div>

  <div class="card">
    <h2>PSI Drift Over Time</h2>
    {psi_html}
  </div>

  <div class="card">
    <h2>Latest KS Statistics by Feature</h2>
    {bar_html}
  </div>

  <div class="card">
    <h2>Recent Alerts</h2>
    {alert_rows_html(alert_rows)}
  </div>

</div>
<div class="footer">
  Static snapshot generated {generated_at} &nbsp;·&nbsp;
  <a href="https://github.com/rahulraypm2002" style="color:#60a5fa">GitHub</a>
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Static dashboard written to {output_path}")
    print("Deploy: push docs/ to GitHub and enable Pages from the docs/ folder.")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/index.html"
    export(out)

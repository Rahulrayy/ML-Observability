import json
import os
from typing import Optional

import diskcache
from groq import Groq

_cache = diskcache.Cache(".mlobs_explainer_cache")

_PROMPT = """You are an ML observability expert. Analyze the following monitoring data and provide a root cause analysis.

ACTIVE ALERTS:
{alerts_summary}

DRIFT STATISTICS:
{drift_stats}

PIPELINE RISK SCORES:
{dag_risks}

Respond in exactly this format:

ROOT CAUSE ANALYSIS
[2-3 sentences explaining what is happening and why]

PIPELINE IMPACT
[Bullet points for each at-risk component with risk level]

RECOMMENDED ACTIONS
[Numbered list of concrete remediation steps]
"""


def explain_alert(
    alerts: list,
    drift_results: list,
    dag_risk_scores: dict[str, float],
    groq_api_key: Optional[str] = None,
) -> str:
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "LLM explanation unavailable: GROQ_API_KEY not set."

    cache_key = _build_cache_key(alerts, drift_results, dag_risk_scores)
    cached = _cache.get(cache_key)
    if cached:
        return cached

    alerts_summary = "\n".join(
        f"- [{a.severity}] {a.title}: {a.message}" for a in alerts
    ) or "None"

    drift_stats = "\n".join(
        f"- {r.feature_name} ({r.test_type}): stat={r.statistic:.4f}"
        + (f", p={r.p_value:.4f}" if r.p_value is not None else "")
        + f", severity={r.severity}"
        for r in drift_results
        if getattr(r, "drift_detected", False)
    ) or "None"

    dag_risks = "\n".join(
        f"- {node}: {score:.2f}"
        for node, score in sorted(dag_risk_scores.items(), key=lambda x: -x[1])
    ) or "None"

    prompt = _PROMPT.format(alerts_summary=alerts_summary, drift_stats=drift_stats, dag_risks=dag_risks)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )
        explanation = response.choices[0].message.content
        _cache.set(cache_key, explanation, expire=3600)
        return explanation
    except Exception as e:
        return f"LLM explanation failed: {e}"


def _build_cache_key(alerts, drift_results, dag_risks) -> str:
    data = {
        "alerts": [(getattr(a, "alert_type", ""), getattr(a, "model_id", "")) for a in alerts],
        "drifts": [
            (r.feature_name, r.test_type, round(r.statistic, 2))
            for r in drift_results
            if getattr(r, "drift_detected", False)
        ],
        "risks": {k: round(v, 1) for k, v in dag_risks.items()},
    }
    return json.dumps(data, sort_keys=True)

from dataclasses import dataclass, field


@dataclass
class SeverityScore:
    level: str  # INFO, WARNING, CRITICAL
    score: float  # 0.0 - 1.0
    reasons: list[str] = field(default_factory=list)


def compute_severity(
    drift_results: list,
    dag_risk_scores: dict[str, float],
) -> SeverityScore:
    reasons: list[str] = []
    score = 0.0

    critical_drifts = [r for r in drift_results if getattr(r, "severity", "") == "CRITICAL" and getattr(r, "drift_detected", False)]
    warning_drifts = [r for r in drift_results if getattr(r, "severity", "") == "WARNING" and getattr(r, "drift_detected", False)]

    if critical_drifts:
        score = max(score, 0.9)
        reasons.append(f"{len(critical_drifts)} critical drift(s) detected")
    if warning_drifts:
        score = max(score, 0.5)
        reasons.append(f"{len(warning_drifts)} warning drift(s) detected")

    if len(critical_drifts) + len(warning_drifts) > 3:
        score = min(score + 0.1, 1.0)
        reasons.append("Multiple simultaneous drift signals")

    high_risk = {k: v for k, v in dag_risk_scores.items() if v > 0.7}
    if high_risk:
        score = max(score, 0.7)
        reasons.append(f"High-risk pipeline nodes: {', '.join(high_risk)}")

    if score >= 0.8 or critical_drifts:
        level = "CRITICAL"
    elif score >= 0.4 or warning_drifts:
        level = "WARNING"
    else:
        level = "INFO"

    return SeverityScore(level=level, score=score, reasons=reasons)

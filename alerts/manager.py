from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.models import Alert, DriftResult
from .explainer import explain_alert
from .router import AlertRouter
from .severity import compute_severity

load_dotenv()


class AlertManager:
    def __init__(
        self,
        database_url: str,
        router: Optional[AlertRouter] = None,
        dedup_window_hours: int = 1,
    ):
        self._engine = create_engine(database_url)
        self._router = router
        self._dedup_window = timedelta(hours=dedup_window_hours)

    def process(
        self,
        model_id: str,
        drift_results: list[DriftResult],
        dag_risk_scores: dict[str, float],
    ) -> list[Alert]:
        if not any(getattr(r, "drift_detected", False) for r in drift_results):
            return []

        if not self._should_fire(model_id, "drift"):
            return []

        severity = compute_severity(drift_results, dag_risk_scores)
        title = f"Drift detected in {model_id}"
        message = f"Severity: {severity.level}. " + " | ".join(severity.reasons)

        explanation = explain_alert(
            alerts=[],
            drift_results=drift_results,
            dag_risk_scores=dag_risk_scores,
        )

        alert = Alert(
            model_id=model_id,
            severity=severity.level,
            alert_type="drift",
            title=title,
            message=message,
            details={
                "drift_count": sum(1 for r in drift_results if getattr(r, "drift_detected", False)),
                "dag_risk_scores": dag_risk_scores,
                "severity_score": severity.score,
                "reasons": severity.reasons,
            },
            explanation=explanation,
        )

        with Session(self._engine) as session:
            session.add(alert)
            session.commit()
            session.refresh(alert)

        if self._router:
            self._router.dispatch(alert)

        return [alert]

    def _should_fire(self, model_id: str, alert_type: str) -> bool:
        cutoff = datetime.now(timezone.utc) - self._dedup_window
        with Session(self._engine) as session:
            recent = (
                session.query(Alert)
                .filter(
                    Alert.model_id == model_id,
                    Alert.alert_type == alert_type,
                    Alert.created_at >= cutoff,
                    Alert.resolved == False,
                )
                .first()
            )
        return recent is None

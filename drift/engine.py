import os
from datetime import datetime, timedelta, timezone

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from .categorical import detect_categorical_drift
from .numeric import detect_numeric_drift
from .prediction import detect_prediction_drift
from .reference import ReferenceWindow
from ingestion.models import DriftResult

load_dotenv()


class DriftEngine:
    def __init__(self, database_url: str, window_minutes: int = 15, reference_days: int = 14):
        self._db_url = database_url
        self._window_minutes = window_minutes
        self._reference_days = reference_days
        self._engine = create_engine(database_url)

    def run(self, model_id: str) -> list[DriftResult]:
        reference = ReferenceWindow.from_database(model_id, self._db_url, days=self._reference_days)
        if not reference.features:
            return []

        current_rows = self._fetch_current_window(model_id)
        if not current_rows:
            return []

        current_features: dict[str, list] = {}
        current_predictions: list[float] = []
        for row in current_rows:
            current_predictions.append(row["prediction"])
            for k, v in row["features"].items():
                current_features.setdefault(k, []).append(v)

        results: list[DriftResult] = []

        for feature_name, ref_values in reference.features.items():
            cur_values = current_features.get(feature_name, [])
            if not cur_values:
                continue

            if all(isinstance(v, (int, float)) for v in cur_values):
                r = detect_numeric_drift(feature_name, ref_values, np.array(cur_values, dtype=float))
                results.append(DriftResult(
                    model_id=model_id, feature_name=feature_name, test_type="ks",
                    statistic=r.ks_statistic, p_value=r.ks_p_value,
                    drift_detected=r.drift_detected, severity=r.severity,
                ))
                results.append(DriftResult(
                    model_id=model_id, feature_name=feature_name, test_type="psi",
                    statistic=r.psi, p_value=None,
                    drift_detected=r.drift_detected, severity=r.severity,
                ))
            else:
                r = detect_categorical_drift(feature_name, list(ref_values), cur_values)
                results.append(DriftResult(
                    model_id=model_id, feature_name=feature_name, test_type="chi2",
                    statistic=r.chi2_statistic, p_value=r.chi2_p_value,
                    drift_detected=r.drift_detected, severity=r.severity,
                ))

        pred_result = detect_prediction_drift(reference.predictions, np.array(current_predictions, dtype=float))
        results.append(DriftResult(
            model_id=model_id, feature_name="__prediction__", test_type="kl",
            statistic=pred_result.kl_divergence, p_value=None,
            drift_detected=pred_result.drift_detected, severity=pred_result.severity,
        ))

        self._save_results(results)
        return results

    def _fetch_current_window(self, model_id: str) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self._window_minutes)
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT features, prediction FROM inference_logs WHERE model_id = :mid AND timestamp >= :cutoff"),
                {"mid": model_id, "cutoff": cutoff},
            ).fetchall()
        return [{"features": r[0], "prediction": r[1]} for r in rows]

    def _save_results(self, results: list[DriftResult]) -> None:
        with Session(self._engine, expire_on_commit=False) as session:
            session.add_all(results)
            session.commit()

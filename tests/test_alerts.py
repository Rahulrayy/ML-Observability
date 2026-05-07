from unittest.mock import MagicMock

from alerts.severity import compute_severity


def _drift(severity: str, detected: bool = True):
    r = MagicMock()
    r.severity = severity
    r.drift_detected = detected
    return r


def test_critical_on_critical_drift():
    result = compute_severity([_drift("CRITICAL"), _drift("WARNING")], {})
    assert result.level == "CRITICAL"


def test_warning_on_warning_only():
    result = compute_severity([_drift("WARNING")], {})
    assert result.level == "WARNING"


def test_info_when_no_drift():
    result = compute_severity([_drift("INFO", detected=False)], {})
    assert result.level == "INFO"


def test_escalates_on_high_dag_risk():
    result = compute_severity([], {"feature_store": 0.9, "model": 0.85})
    assert result.level in ("WARNING", "CRITICAL")
    assert any("High-risk" in r for r in result.reasons)


def test_multiple_signals_noted_in_reasons():
    result = compute_severity([_drift("WARNING")] * 5, {})
    assert any("Multiple simultaneous" in r for r in result.reasons)

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class NumericDriftResult:
    feature: str
    ks_statistic: float
    ks_p_value: float
    psi: float
    drift_detected: bool
    severity: str


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    stat, p_value = stats.ks_2samp(reference, current)
    return float(stat), float(p_value)


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref_min, ref_max = reference.min(), reference.max()
    if ref_min == ref_max:
        return 0.0

    edges = np.linspace(ref_min, ref_max, bins + 1)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing to avoid log(0)
    ref_pct = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * bins)
    cur_pct = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * bins)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_numeric_drift(
    feature: str,
    reference: np.ndarray,
    current: np.ndarray,
    ks_threshold: float = 0.05,
    psi_warning: float = 0.1,
    psi_critical: float = 0.25,
) -> NumericDriftResult:
    if len(reference) < 30 or len(current) < 30:
        return NumericDriftResult(feature, 0.0, 1.0, 0.0, False, "INFO")

    ks_stat, ks_p = ks_test(reference, current)
    psi = population_stability_index(reference, current)

    if psi >= psi_critical or (ks_p < 0.001 and psi >= psi_warning):
        severity = "CRITICAL"
        drift_detected = True
    elif psi >= psi_warning or ks_p < ks_threshold:
        severity = "WARNING"
        drift_detected = True
    else:
        severity = "INFO"
        drift_detected = False

    return NumericDriftResult(feature, ks_stat, ks_p, psi, drift_detected, severity)

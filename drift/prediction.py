from dataclasses import dataclass

import numpy as np
from scipy.special import kl_div


@dataclass
class PredictionDriftResult:
    kl_divergence: float
    mean_shift: float
    std_shift: float
    drift_detected: bool
    severity: str


def compute_kl_divergence(reference: np.ndarray, current: np.ndarray, bins: int = 20) -> float:
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    if min_val == max_val:
        return 0.0

    edges = np.linspace(min_val, max_val, bins + 1)
    eps = 1e-10
    ref_hist, _ = np.histogram(reference, bins=edges, density=True)
    cur_hist, _ = np.histogram(current, bins=edges, density=True)

    ref_hist = ref_hist + eps
    cur_hist = cur_hist + eps
    ref_hist /= ref_hist.sum()
    cur_hist /= cur_hist.sum()

    return float(np.sum(kl_div(ref_hist, cur_hist)))


def detect_prediction_drift(
    reference: np.ndarray,
    current: np.ndarray,
    kl_warning: float = 0.1,
    kl_critical: float = 0.3,
) -> PredictionDriftResult:
    if len(reference) < 30 or len(current) < 30:
        return PredictionDriftResult(0.0, 0.0, 0.0, False, "INFO")

    kl = compute_kl_divergence(reference, current)
    mean_shift = abs(float(current.mean()) - float(reference.mean()))
    std_shift = abs(float(current.std()) - float(reference.std()))

    if kl >= kl_critical:
        severity = "CRITICAL"
        drift_detected = True
    elif kl >= kl_warning:
        severity = "WARNING"
        drift_detected = True
    else:
        severity = "INFO"
        drift_detected = False

    return PredictionDriftResult(kl, mean_shift, std_shift, drift_detected, severity)

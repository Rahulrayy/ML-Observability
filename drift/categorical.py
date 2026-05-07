from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class CategoricalDriftResult:
    feature: str
    chi2_statistic: float
    chi2_p_value: float
    new_categories: list[str]
    drift_detected: bool
    severity: str


def detect_categorical_drift(
    feature: str,
    reference: list,
    current: list,
    p_threshold: float = 0.05,
) -> CategoricalDriftResult:
    ref_set = set(reference)
    cur_set = set(current)
    new_cats = [str(c) for c in cur_set - ref_set]

    all_cats = sorted(ref_set | cur_set, key=str)
    ref_counts = np.array([reference.count(c) for c in all_cats], dtype=float)
    cur_counts = np.array([current.count(c) for c in all_cats], dtype=float)

    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        detected = bool(new_cats)
        return CategoricalDriftResult(feature, 0.0, 1.0, new_cats, detected, "WARNING" if detected else "INFO")

    ref_expected = ref_counts / ref_counts.sum() * cur_counts.sum()
    ref_expected = np.where(ref_expected == 0, 0.5, ref_expected)
    ref_expected = ref_expected / ref_expected.sum() * cur_counts.sum()

    chi2_stat, p_value = stats.chisquare(cur_counts, f_exp=ref_expected)

    if new_cats or p_value < 0.001:
        severity = "CRITICAL"
        drift_detected = True
    elif p_value < p_threshold:
        severity = "WARNING"
        drift_detected = True
    else:
        severity = "INFO"
        drift_detected = False

    return CategoricalDriftResult(feature, float(chi2_stat), float(p_value), new_cats, drift_detected, severity)

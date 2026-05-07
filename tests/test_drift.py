import numpy as np
import pytest

from drift.categorical import detect_categorical_drift
from drift.numeric import detect_numeric_drift, population_stability_index
from drift.prediction import detect_prediction_drift


def test_ks_no_drift_same_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(0, 1, 500)
    assert not detect_numeric_drift("f", ref, cur).drift_detected


def test_ks_detects_large_shift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(5, 1, 500)
    result = detect_numeric_drift("f", ref, cur)
    assert result.drift_detected
    assert result.severity in ("WARNING", "CRITICAL")


def test_psi_stable():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(0, 1, 1000), rng.normal(0, 1, 1000)) < 0.1


def test_psi_high_drift():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(0, 1, 1000), rng.normal(5, 1, 1000)) > 0.25


def test_categorical_new_category_flagged():
    ref = ["A", "B", "C"] * 100
    cur = ["A", "B", "D"] * 100
    result = detect_categorical_drift("cat", ref, cur)
    assert "D" in result.new_categories
    assert result.drift_detected


def test_categorical_stable():
    ref = ["A", "B", "C"] * 200
    cur = ["A", "B", "C"] * 200
    assert not detect_categorical_drift("cat", ref, cur).drift_detected


def test_prediction_drift_detected():
    rng = np.random.default_rng(7)
    assert detect_prediction_drift(rng.uniform(0, 0.4, 500), rng.uniform(0.6, 1.0, 500)).drift_detected


def test_prediction_no_drift():
    rng = np.random.default_rng(7)
    assert not detect_prediction_drift(rng.uniform(0.3, 0.7, 500), rng.uniform(0.3, 0.7, 500)).drift_detected

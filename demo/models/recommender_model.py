"""Recommendation CTR model. Drift scenario: new low-engagement user cohort onboards — engagement collapses suddenly."""
import math
import random

from sdk import Monitor


def create_monitor(api_endpoint: str = "http://localhost:8000", api_key: str = "dev") -> Monitor:
    return Monitor(model_id="recommender_v3", api_endpoint=api_endpoint, api_key=api_key)


def predict(features: dict, monitor: Monitor) -> float:
    session = min(features["session_duration_s"] / 1800, 1.0)
    pages = min(features["pages_per_session"] / 20, 1.0)
    ctr = features["historical_ctr"]
    age = min(features["user_age_days"] / 365, 1.0)
    cart = min(features["items_in_cart"] / 10, 1.0)

    logit = -1.5 + 2.0 * ctr + 1.0 * session + 0.8 * pages + 0.5 * age + 1.2 * cart
    logit += random.gauss(0, 0.25)
    prediction = 1 / (1 + math.exp(-logit))

    monitor.log(features=features, prediction=prediction, confidence=prediction)
    return prediction


def sample_features(drift: bool = False) -> dict:
    if drift:
        # New cohort: brand new users, very low engagement, no cart activity
        return {
            "user_age_days":       max(0, random.gauss(3, 3)),
            "session_duration_s":  max(5, random.gauss(28, 20)),
            "pages_per_session":   max(1, random.gauss(1.4, 0.8)),
            "historical_ctr":      max(0, min(1, random.gauss(0.01, 0.01))),
            "items_in_cart":       max(0, int(random.gauss(0.05, 0.2))),
            "days_since_last_visit": max(0, random.gauss(0.2, 0.5)),
        }
    return {
        "user_age_days":       max(0, random.gauss(180, 120)),
        "session_duration_s":  max(5, random.gauss(420, 180)),
        "pages_per_session":   max(1, random.gauss(8, 4)),
        "historical_ctr":      max(0, min(1, random.gauss(0.12, 0.05))),
        "items_in_cart":       max(0, int(random.gauss(2.5, 2))),
        "days_since_last_visit": max(0, random.gauss(4, 3)),
    }

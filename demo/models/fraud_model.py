"""Fraud detection model. Drift scenario: class imbalance shift — fraud rate jumps from ~1% to ~15%."""
import math
import random

from sdk import Monitor


def create_monitor(api_endpoint: str = "http://localhost:8000", api_key: str = "dev") -> Monitor:
    return Monitor(model_id="fraud_detection_v2", api_endpoint=api_endpoint, api_key=api_key)


def predict(features: dict, monitor: Monitor) -> float:
    amount_norm = min(features["transaction_amount"] / 5000, 1.0)
    foreign = features["is_foreign_transaction"]
    distance = min(features["distance_from_home_km"] / 1000, 1.0)
    hour = features["hour_of_day"]
    night = 1.0 if (hour < 5 or hour > 22) else 0.0

    logit = -4 + 2.0 * foreign + 1.5 * distance + 1.0 * night + 0.8 * amount_norm
    logit += random.gauss(0, 0.2)
    prediction = 1 / (1 + math.exp(-logit))

    monitor.log(features=features, prediction=prediction, confidence=prediction)
    return prediction


def sample_features(drift: bool = False) -> dict:
    if drift:
        # New fraud pattern: higher amounts, more foreign, abnormal hours
        return {
            "transaction_amount":      max(1, random.gauss(1800, 900)),
            "merchant_category":       random.choice(["online", "atm", "foreign", "luxury", "crypto"]),
            "hour_of_day":             random.choice([1, 2, 3, 23, 0] + list(range(9, 17))),
            "distance_from_home_km":   max(0, random.gauss(400, 200)),
            "is_foreign_transaction":  1 if random.random() < 0.45 else 0,
        }
    return {
        "transaction_amount":      max(1, random.gauss(85, 120)),
        "merchant_category":       random.choice(["grocery", "gas", "restaurant", "retail", "online"]),
        "hour_of_day":             int(random.gauss(14, 4)) % 24,
        "distance_from_home_km":   max(0, random.gauss(12, 15)),
        "is_foreign_transaction":  1 if random.random() < 0.02 else 0,
    }

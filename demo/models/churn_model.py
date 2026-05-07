"""Example churn prediction model instrumented with the ML Observability SDK."""
import math
import random

from sdk import Monitor


def create_monitor(api_endpoint: str = "http://localhost:8000", api_key: str = "dev") -> Monitor:
    return Monitor(model_id="churn_model_v2", api_endpoint=api_endpoint, api_key=api_key)


def predict(features: dict, monitor: Monitor) -> float:
    tenure = features.get("customer_tenure_days", 365) / 365
    freq = features.get("transaction_frequency", 5) / 10
    value = features.get("avg_transaction_value", 100) / 500
    support = features.get("support_tickets", 0)

    logit = -2 + 1.5 * (1 - tenure) - 0.8 * freq + 0.3 * support - 0.5 * value
    logit += random.gauss(0, 0.2)
    prediction = 1 / (1 + math.exp(-logit))

    monitor.log(features=features, prediction=prediction)
    return prediction


def sample_features(drift: bool = False) -> dict:
    if drift:
        tenure = max(0, random.gauss(700, 200))  # heavy right-tail shift
        freq = max(0, random.gauss(2, 1))
    else:
        tenure = max(0, random.gauss(365, 150))
        freq = max(0, random.gauss(5, 2))
    return {
        "customer_tenure_days": tenure,
        "transaction_frequency": freq,
        "avg_transaction_value": max(0, random.gauss(100, 40)),
        "support_tickets": max(0, int(random.gauss(1, 1))),
    }

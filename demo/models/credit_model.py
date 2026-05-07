"""Credit risk scoring model. Drift scenario: economic downturn — income drops, credit scores worsen, DTI rises."""
import math
import random

from sdk import Monitor


def create_monitor(api_endpoint: str = "http://localhost:8000", api_key: str = "dev") -> Monitor:
    return Monitor(model_id="credit_model_v1", api_endpoint=api_endpoint, api_key=api_key)


def predict(features: dict, monitor: Monitor) -> float:
    score_norm = (features["credit_score"] - 300) / 550
    income_norm = min(features["annual_income"] / 100000, 1.0)
    dti = features["debt_to_income_ratio"]
    emp = min(features["employment_years"] / 20, 1.0)

    logit = -3 + 2.5 * score_norm + 1.5 * income_norm - 2.0 * dti + 0.5 * emp
    logit += random.gauss(0, 0.3)
    prediction = 1 / (1 + math.exp(-logit))

    monitor.log(features=features, prediction=prediction)
    return prediction


def sample_features(drift: bool = False) -> dict:
    if drift:
        # Economic downturn: income collapses, credit scores worsen, DTI rises
        return {
            "annual_income":        max(0, random.gauss(28000, 8000)),
            "credit_score":         max(300, min(850, random.gauss(570, 65))),
            "debt_to_income_ratio": max(0, min(1, random.gauss(0.58, 0.15))),
            "loan_amount":          max(1000, random.gauss(38000, 12000)),
            "employment_years":     max(0, random.gauss(1.5, 1.5)),
        }
    return {
        "annual_income":        max(0, random.gauss(65000, 18000)),
        "credit_score":         max(300, min(850, random.gauss(700, 55))),
        "debt_to_income_ratio": max(0, min(1, random.gauss(0.28, 0.10))),
        "loan_amount":          max(1000, random.gauss(22000, 9000)),
        "employment_years":     max(0, random.gauss(9, 5)),
    }

"""Demand forecasting model (regression). Drift scenario: price sensitivity changes — competitor undercuts market."""
import random

from sdk import Monitor


def create_monitor(api_endpoint: str = "http://localhost:8000", api_key: str = "dev") -> Monitor:
    return Monitor(model_id="demand_forecast_v1", api_endpoint=api_endpoint, api_key=api_key)


def predict(features: dict, monitor: Monitor) -> float:
    price_ratio = features["our_price"] / max(features["competitor_price"], 1)
    stock = min(features["stock_level"] / 1000, 1.0)
    holiday = features["is_holiday"]
    day_factor = features["day_of_week"] / 6.0

    # Demand score 0-1 (normalised expected units sold / 1000)
    demand = max(0, min(1, 0.5 - 0.4 * (price_ratio - 1) + 0.2 * holiday + 0.1 * day_factor + random.gauss(0, 0.05)))

    monitor.log(features=features, prediction=demand)
    return demand


def sample_features(drift: bool = False) -> dict:
    if drift:
        # Competitor slashes prices — our price ratio worsens, demand drops
        return {
            "our_price":          max(1, random.gauss(52, 8)),
            "competitor_price":   max(1, random.gauss(28, 5)),   # competitor now 45% cheaper
            "stock_level":        max(0, random.gauss(300, 100)),
            "is_holiday":         1 if random.random() < 0.08 else 0,
            "day_of_week":        random.randint(0, 6),
            "marketing_spend":    max(0, random.gauss(500, 200)),
        }
    return {
        "our_price":          max(1, random.gauss(50, 8)),
        "competitor_price":   max(1, random.gauss(51, 6)),   # roughly equal
        "stock_level":        max(0, random.gauss(700, 200)),
        "is_holiday":         1 if random.random() < 0.08 else 0,
        "day_of_week":        random.randint(0, 6),
        "marketing_spend":    max(0, random.gauss(2000, 500)),
    }

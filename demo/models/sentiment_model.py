"""Sentiment classification model. Drift scenario: data source changes — new platform with aggressive, short-form text."""
import math
import random

from sdk import Monitor


def create_monitor(api_endpoint: str = "http://localhost:8000", api_key: str = "dev") -> Monitor:
    return Monitor(model_id="sentiment_v1", api_endpoint=api_endpoint, api_key=api_key)


def predict(features: dict, monitor: Monitor) -> float:
    # Positive sentiment score 0-1
    length_norm = min(features["text_length"] / 500, 1.0)
    caps = features["caps_ratio"]
    excl = min(features["exclamation_count"] / 10, 1.0)
    questions = min(features["question_count"] / 5, 1.0)
    word_len = features["avg_word_length"]

    logit = 1.0 + 0.3 * length_norm - 1.5 * caps + 0.5 * excl - 0.4 * questions + 0.2 * (word_len - 5)
    logit += random.gauss(0, 0.3)
    prediction = 1 / (1 + math.exp(-logit))

    monitor.log(features=features, prediction=prediction, confidence=max(prediction, 1 - prediction))
    return prediction


def sample_features(drift: bool = False) -> dict:
    if drift:
        # New platform: short aggressive text, lots of caps and exclamations
        return {
            "text_length":        max(10, random.gauss(45, 20)),
            "avg_word_length":    max(2, random.gauss(3.2, 0.8)),
            "caps_ratio":         max(0, min(1, random.gauss(0.38, 0.15))),
            "exclamation_count":  max(0, int(random.gauss(5, 3))),
            "question_count":     max(0, int(random.gauss(0.3, 0.5))),
            "unique_word_ratio":  max(0, min(1, random.gauss(0.45, 0.12))),
        }
    return {
        "text_length":        max(10, random.gauss(180, 80)),
        "avg_word_length":    max(2, random.gauss(5.2, 1.0)),
        "caps_ratio":         max(0, min(1, random.gauss(0.06, 0.04))),
        "exclamation_count":  max(0, int(random.gauss(0.8, 1.0))),
        "question_count":     max(0, int(random.gauss(1.2, 1.2))),
        "unique_word_ratio":  max(0, min(1, random.gauss(0.72, 0.10))),
    }

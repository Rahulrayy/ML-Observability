"""Injects ~200 drifted inferences per model with current timestamps directly to PostgreSQL.
Bypasses the API BatchWriter to avoid duplicate-key conflicts.
"""
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from ingestion.models import Base, InferenceLog
from demo.models.churn_model import sample_features as churn_sample
from demo.models.credit_model import sample_features as credit_sample
from demo.models.fraud_model import sample_features as fraud_sample
from demo.models.demand_model import sample_features as demand_sample
from demo.models.sentiment_model import sample_features as sentiment_sample
from demo.models.recommender_model import sample_features as recommender_sample

import math


def _logit_to_prob(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


MODELS = [
    {
        "model_id": "churn_model_v2",
        "sampler": churn_sample,
        "prefix": "live-churn",
        "predictor": lambda feats: _logit_to_prob(
            -2 + 1.5 * (1 - feats["customer_tenure_days"] / 365)
            - 0.8 * feats["transaction_frequency"] / 10
            + random.gauss(0, 0.2)
        ),
    },
    {
        "model_id": "credit_model_v1",
        "sampler": credit_sample,
        "prefix": "live-credit",
        "predictor": lambda feats: _logit_to_prob(
            -3 + 2.5 * (feats["credit_score"] - 300) / 550
            + 1.5 * min(feats["annual_income"] / 100000, 1.0)
            - 2.0 * feats["debt_to_income_ratio"]
            + random.gauss(0, 0.3)
        ),
    },
    {
        "model_id": "fraud_detection_v2",
        "sampler": fraud_sample,
        "prefix": "live-fraud",
        "predictor": lambda feats: _logit_to_prob(
            -4 + 2.0 * feats["is_foreign_transaction"]
            + 1.5 * min(feats["distance_from_home_km"] / 1000, 1.0)
            + 1.0 * (1.0 if feats["hour_of_day"] < 5 or feats["hour_of_day"] > 22 else 0.0)
            + 0.8 * min(feats["transaction_amount"] / 5000, 1.0)
            + random.gauss(0, 0.2)
        ),
    },
    {
        "model_id": "demand_forecast_v1",
        "sampler": demand_sample,
        "prefix": "live-demand",
        "predictor": lambda feats: max(0, min(1,
            0.5 - 0.4 * (feats["our_price"] / max(feats["competitor_price"], 1) - 1)
            + 0.2 * feats["is_holiday"]
            + random.gauss(0, 0.05)
        )),
    },
    {
        "model_id": "sentiment_v1",
        "sampler": sentiment_sample,
        "prefix": "live-sentiment",
        "predictor": lambda feats: _logit_to_prob(
            1.0 - 1.5 * feats["caps_ratio"]
            + 0.5 * min(feats["exclamation_count"] / 10, 1.0)
            + 0.2 * (feats["avg_word_length"] - 5)
            + random.gauss(0, 0.3)
        ),
    },
    {
        "model_id": "recommender_v3",
        "sampler": recommender_sample,
        "prefix": "live-recommender",
        "predictor": lambda feats: _logit_to_prob(
            -1.5 + 2.0 * feats["historical_ctr"]
            + 1.0 * min(feats["session_duration_s"] / 1800, 1.0)
            + 0.8 * min(feats["pages_per_session"] / 20, 1.0)
            + random.gauss(0, 0.25)
        ),
    },
]

N = 250  # inferences per model


def inject(database_url: str) -> None:
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(engine)

    now = datetime.now(timezone.utc)

    for cfg in MODELS:
        model_id = cfg["model_id"]
        batch: list[InferenceLog] = []

        for i in range(N):
            # Spread within the last 10 minutes so they fall in the current drift window
            ts = now - timedelta(seconds=random.randint(0, 600))
            feats = cfg["sampler"](drift=True)
            pred = float(cfg["predictor"](feats))
            gt = 1.0 if random.random() < (0.15 if "fraud" in model_id else 0.5) else 0.0

            batch.append(InferenceLog(
                inference_id=f"{cfg['prefix']}-{uuid.uuid4().hex[:12]}",
                model_id=model_id,
                timestamp=ts,
                features=feats,
                prediction=pred,
                ground_truth=gt,
            ))

        with SessionFactory() as session:
            session.add_all(batch)
            session.commit()

        print(f"  [{model_id}] injected {N} drifted inferences")

    engine.dispose()
    print("Done.")


if __name__ == "__main__":
    db_url = os.environ.get("DATABASE_URL", "postgresql://mlobs:mlobs@localhost:5432/mlobs")
    inject(db_url)

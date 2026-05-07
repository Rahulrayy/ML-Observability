"""Seeds the database with 30 days of synthetic inference history for all demo models."""
import math
import os
import random
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

load_dotenv()

from ingestion.models import Base, InferenceLog
from demo.models.churn_model import sample_features as churn_sample
from demo.models.credit_model import sample_features as credit_sample
from demo.models.fraud_model import sample_features as fraud_sample
from demo.models.demand_model import sample_features as demand_sample
from demo.models.sentiment_model import sample_features as sentiment_sample
from demo.models.recommender_model import sample_features as recommender_sample


def _logit_to_prob(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


def _seed_model(
    session_factory,
    model_id: str,
    sampler,
    predictor,
    days: int,
    per_day: int,
    drift_start_day: int,
    prefix: str,
) -> int:
    start = datetime.now(timezone.utc) - timedelta(days=days)
    batch: list[InferenceLog] = []
    total = 0

    for d in range(days):
        day_start = start + timedelta(days=d)
        has_drift = d >= drift_start_day

        for i in range(per_day):
            ts = day_start + timedelta(seconds=random.randint(0, 86399))
            feats = sampler(drift=has_drift)
            pred, gt = predictor(feats, has_drift)

            batch.append(InferenceLog(
                inference_id=f"{prefix}-{d}-{i}",
                model_id=model_id,
                timestamp=ts,
                features=feats,
                prediction=pred,
                ground_truth=gt,
            ))

            if len(batch) >= 500:
                with session_factory() as session:
                    session.add_all(batch)
                    session.commit()
                total += len(batch)
                batch.clear()

        print(f"  [{model_id}] Day {d + 1}/{days}: {total + len(batch):,} inferences")

    if batch:
        with session_factory() as session:
            session.add_all(batch)
            session.commit()
        total += len(batch)

    return total


def seed(database_url: str, days: int = 30, per_day: int = 300) -> None:
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)

    from sqlalchemy.orm import sessionmaker
    SessionFactory = sessionmaker(engine)

    drift_day = days - 2  # last 2 days have drift

    # --- churn_model_v2 ---
    def churn_pred(feats, drifted):
        tenure = feats["customer_tenure_days"] / 365
        freq = feats["transaction_frequency"] / 10
        logit = -2 + 1.5 * (1 - tenure) - 0.8 * freq
        pred = _logit_to_prob(logit + random.gauss(0, 0.2))
        gt = 1.0 if random.random() < pred * 0.9 else 0.0
        return pred, gt

    n = _seed_model(SessionFactory, "churn_model_v2", churn_sample, churn_pred, days, per_day, drift_day, "churn")
    print(f"  churn_model_v2: {n:,} total\n")

    # --- credit_model_v1 ---
    def credit_pred(feats, drifted):
        score_norm = (feats["credit_score"] - 300) / 550
        income_norm = min(feats["annual_income"] / 100000, 1.0)
        dti = feats["debt_to_income_ratio"]
        logit = -3 + 2.5 * score_norm + 1.5 * income_norm - 2.0 * dti
        pred = _logit_to_prob(logit + random.gauss(0, 0.3))
        gt = 1.0 if random.random() < pred * 0.85 else 0.0
        return pred, gt

    n = _seed_model(SessionFactory, "credit_model_v1", credit_sample, credit_pred, days, per_day, drift_day, "credit")
    print(f"  credit_model_v1: {n:,} total\n")

    # --- fraud_detection_v2 (imbalanced: ~1% fraud baseline, ~15% drifted) ---
    def fraud_pred(feats, drifted):
        foreign = feats["is_foreign_transaction"]
        distance = min(feats["distance_from_home_km"] / 1000, 1.0)
        amount_norm = min(feats["transaction_amount"] / 5000, 1.0)
        hour = feats["hour_of_day"]
        night = 1.0 if (hour < 5 or hour > 22) else 0.0
        logit = -4 + 2.0 * foreign + 1.5 * distance + 1.0 * night + 0.8 * amount_norm
        pred = _logit_to_prob(logit + random.gauss(0, 0.2))
        fraud_rate = 0.15 if drifted else 0.01
        gt = 1.0 if random.random() < fraud_rate else 0.0
        return pred, gt

    n = _seed_model(SessionFactory, "fraud_detection_v2", fraud_sample, fraud_pred, days, per_day, drift_day, "fraud")
    print(f"  fraud_detection_v2: {n:,} total\n")

    # --- demand_forecast_v1 (regression: continuous output) ---
    def demand_pred(feats, drifted):
        price_ratio = feats["our_price"] / max(feats["competitor_price"], 1)
        holiday = feats["is_holiday"]
        demand = max(0, min(1, 0.5 - 0.4 * (price_ratio - 1) + 0.2 * holiday + random.gauss(0, 0.05)))
        gt = max(0, min(1, demand + random.gauss(0, 0.08)))
        return demand, gt

    n = _seed_model(SessionFactory, "demand_forecast_v1", demand_sample, demand_pred, days, per_day, drift_day, "demand")
    print(f"  demand_forecast_v1: {n:,} total\n")

    # --- sentiment_v1 ---
    def sentiment_pred(feats, drifted):
        caps = feats["caps_ratio"]
        excl = min(feats["exclamation_count"] / 10, 1.0)
        word_len = feats["avg_word_length"]
        logit = 1.0 - 1.5 * caps + 0.5 * excl + 0.2 * (word_len - 5)
        pred = _logit_to_prob(logit + random.gauss(0, 0.3))
        gt = 1.0 if pred > 0.5 + random.gauss(0, 0.1) else 0.0
        return pred, gt

    n = _seed_model(SessionFactory, "sentiment_v1", sentiment_sample, sentiment_pred, days, per_day, drift_day, "sentiment")
    print(f"  sentiment_v1: {n:,} total\n")

    # --- recommender_v3 ---
    def recommender_pred(feats, drifted):
        ctr = feats["historical_ctr"]
        session = min(feats["session_duration_s"] / 1800, 1.0)
        pages = min(feats["pages_per_session"] / 20, 1.0)
        logit = -1.5 + 2.0 * ctr + 1.0 * session + 0.8 * pages
        pred = _logit_to_prob(logit + random.gauss(0, 0.25))
        gt = 1.0 if random.random() < pred else 0.0
        return pred, gt

    n = _seed_model(SessionFactory, "recommender_v3", recommender_sample, recommender_pred, days, per_day, drift_day, "recommender")
    print(f"  recommender_v3: {n:,} total\n")

    engine.dispose()
    print("Done. All models seeded. Last 2 days contain drifted data.")
    print("Open http://localhost:8501 to see the dashboard.")


if __name__ == "__main__":
    db_url = os.environ.get("DATABASE_URL", "postgresql://mlobs:mlobs@localhost:5432/mlobs")
    seed(db_url)

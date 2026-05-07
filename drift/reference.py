from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from sqlalchemy import create_engine, text


@dataclass
class ReferenceWindow:
    model_id: str
    features: dict[str, np.ndarray] = field(default_factory=dict)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    computed_at: Optional[datetime] = None

    @classmethod
    def from_database(cls, model_id: str, database_url: str, days: int = 14) -> "ReferenceWindow":
        engine = create_engine(database_url)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = datetime.now(timezone.utc) - timedelta(days=1)

        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT features, prediction FROM inference_logs
                    WHERE model_id = :mid AND timestamp < :recent AND timestamp >= :cutoff
                    ORDER BY timestamp DESC LIMIT 10000
                """),
                {"mid": model_id, "cutoff": cutoff, "recent": recent},
            ).fetchall()
        engine.dispose()

        if not rows:
            return cls(model_id=model_id, computed_at=datetime.now(timezone.utc))

        features: dict[str, list] = {}
        predictions = []
        for row in rows:
            feat_dict = row[0]
            predictions.append(row[1])
            for k, v in feat_dict.items():
                features.setdefault(k, []).append(v)

        return cls(
            model_id=model_id,
            features={k: np.array(v) for k, v in features.items()},
            predictions=np.array(predictions),
            computed_at=datetime.now(timezone.utc),
        )

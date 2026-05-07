import time
import uuid
import threading
from typing import Any, Optional

import requests

from .buffer import LocalBuffer


class Monitor:
    def __init__(
        self,
        model_id: str,
        api_endpoint: str,
        api_key: str,
        flush_interval: float = 5.0,
        batch_size: int = 50,
    ):
        self.model_id = model_id
        self._endpoint = api_endpoint.rstrip("/")
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self._buffer = LocalBuffer()
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._start_flush_worker()

    def log(
        self,
        features: dict[str, Any],
        prediction: float,
        ground_truth: Optional[float] = None,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        inference_id = str(uuid.uuid4())
        self._buffer.push({
            "inference_id": inference_id,
            "model_id": self.model_id,
            "timestamp": time.time(),
            "features": features,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "metadata": metadata or {},
        })
        return inference_id

    def log_ground_truth(self, inference_id: str, ground_truth: float) -> None:
        try:
            requests.post(
                f"{self._endpoint}/ground-truth",
                json={
                    "inference_id": inference_id,
                    "model_id": self.model_id,
                    "ground_truth": ground_truth,
                },
                headers=self._headers,
                timeout=5,
            )
        except requests.RequestException:
            pass

    def _flush(self) -> None:
        batch = self._buffer.drain(self._batch_size)
        if not batch:
            return
        try:
            requests.post(
                f"{self._endpoint}/inferences",
                json={"records": batch},
                headers=self._headers,
                timeout=10,
            )
        except requests.RequestException:
            for record in batch:
                self._buffer.push(record)

    def _start_flush_worker(self) -> None:
        def worker():
            while True:
                time.sleep(self._flush_interval)
                self._flush()

        threading.Thread(target=worker, daemon=True).start()

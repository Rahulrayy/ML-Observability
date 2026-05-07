import json
import threading
from collections import deque
from pathlib import Path
from typing import Any


class LocalBuffer:
    """Thread-safe in-memory queue with disk fallback for offline resilience."""

    def __init__(self, path: str = ".mlobs_buffer.jsonl", maxsize: int = 10_000):
        self._path = Path(path)
        self._maxsize = maxsize
        self._queue: deque = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._load_from_disk()

    def push(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._queue.append(record)

    def drain(self, n: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            batch = []
            for _ in range(min(n, len(self._queue))):
                batch.append(self._queue.popleft())
            return batch

    def __len__(self) -> int:
        return len(self._queue)

    def flush_to_disk(self) -> None:
        with self._lock:
            with open(self._path, "w") as f:
                for record in self._queue:
                    f.write(json.dumps(record) + "\n")

    def _load_from_disk(self) -> None:
        if not self._path.exists():
            return
        with open(self._path) as f:
            for line in f:
                try:
                    self._queue.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass

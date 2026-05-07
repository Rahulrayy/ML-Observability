import time
from unittest.mock import patch

import pytest

from sdk.buffer import LocalBuffer
from sdk.monitor import Monitor


def test_buffer_push_and_drain():
    buf = LocalBuffer(path=".test_buffer.jsonl")
    buf.push({"a": 1})
    buf.push({"b": 2})
    assert len(buf) == 2
    batch = buf.drain(1)
    assert batch == [{"a": 1}]
    assert len(buf) == 1


def test_buffer_drain_empty():
    buf = LocalBuffer(path=".test_buffer_empty.jsonl")
    assert buf.drain(10) == []


def test_monitor_log_returns_uuid():
    with patch("requests.post"):
        monitor = Monitor("test_model", "http://localhost:8000", "key")
        inf_id = monitor.log({"x": 1.0}, prediction=0.7)
    assert len(inf_id) == 36


def test_monitor_buffers_on_api_failure():
    monitor = Monitor.__new__(Monitor)
    monitor.model_id = "test"
    monitor._endpoint = "http://localhost:9999"
    monitor._headers = {}
    monitor._batch_size = 50
    monitor._buffer = LocalBuffer(path=".test_fail.jsonl")

    monitor._buffer.push({
        "inference_id": "x", "model_id": "test",
        "timestamp": time.time(), "features": {}, "prediction": 0.5, "metadata": {},
    })
    monitor._flush()
    assert len(monitor._buffer) == 1  # re-queued after failure

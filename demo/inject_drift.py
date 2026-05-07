"""Injects drift into a live model for demo purposes."""
import argparse
import os
import time

import requests

from demo.models.churn_model import sample_features

API = os.environ.get("API_ENDPOINT", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ.get("API_KEY", "dev")}


def inject_feature_drift(node: str = "raw_data", n: int = 200) -> None:
    print(f"Injecting feature distribution drift at node '{node}'...")
    records = [
        {
            "inference_id": f"drift-inject-{int(time.time())}-{i}",
            "model_id": "churn_model_v2",
            "timestamp": time.time(),
            "features": sample_features(drift=True),
            "prediction": 0.5,
            "metadata": {"injected_drift": True, "node": node},
        }
        for i in range(n)
    ]
    resp = requests.post(f"{API}/inferences", json={"records": records}, headers=HEADERS, timeout=30)
    print(f"  Sent {n} drifted inferences -> HTTP {resp.status_code}")
    print("  Wait ~15 minutes for the drift engine to detect the shift.")
    print("  Dashboard: upstream node will turn yellow/red; GNN propagates risk forward.")


def inject_label_drift(severity: str = "high", n: int = 200) -> None:
    print(f"Injecting label drift (severity={severity})...")
    records = [
        {
            "inference_id": f"label-drift-{int(time.time())}-{i}",
            "model_id": "churn_model_v2",
            "timestamp": time.time(),
            "features": sample_features(drift=True),
            "prediction": 0.15 + 0.05 * (i % 4),  # model predicts low churn
            "ground_truth": 1.0,                    # reality: everyone is churning
            "metadata": {"injected_label_drift": True, "severity": severity},
        }
        for i in range(n)
    ]
    resp = requests.post(f"{API}/inferences", json={"records": records}, headers=HEADERS, timeout=30)
    print(f"  Sent {n} label-drifted inferences -> HTTP {resp.status_code}")
    print("  Accuracy will degrade visibly in the Model Performance page.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject drift for demo")
    parser.add_argument("--type", choices=["feature_distribution", "label_drift"], required=True)
    parser.add_argument("--node", default="raw_data", help="Pipeline node for feature drift")
    parser.add_argument("--severity", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--n", type=int, default=200, help="Number of inferences to inject")
    args = parser.parse_args()

    if args.type == "feature_distribution":
        inject_feature_drift(node=args.node, n=args.n)
    else:
        inject_label_drift(severity=args.severity, n=args.n)


if __name__ == "__main__":
    main()

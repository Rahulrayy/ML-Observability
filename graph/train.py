from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .dag import PipelineDAG
from .gnn import PipelineGNN


def build_graph_data(dag: PipelineDAG, node_features: dict[str, list[float]]) -> Data:
    nodes = dag.nodes()
    node_ids = [n.id for n in nodes]
    idx = {nid: i for i, nid in enumerate(node_ids)}

    feature_dim = max((len(v) for v in node_features.values()), default=8)
    x = torch.zeros(len(node_ids), feature_dim)
    for nid, feats in node_features.items():
        if nid in idx:
            x[idx[nid], : len(feats)] = torch.tensor(feats, dtype=torch.float)

    edges = [(e[0], e[1]) for e in dag.edges() if e[0] in idx and e[1] in idx]
    if edges:
        src, dst = zip(*edges)
        edge_index = torch.tensor([[idx[s] for s in src], [idx[d] for d in dst]], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def train(
    dag: PipelineDAG,
    historical_snapshots: list[dict[str, list[float]]],
    save_path: str = "models/pipeline_gnn.pt",
    epochs: int = 200,
    lr: float = 1e-3,
) -> PipelineGNN:
    if not historical_snapshots:
        raise ValueError("Need at least one historical snapshot to train")

    sample_data = build_graph_data(dag, historical_snapshots[0])
    in_channels = sample_data.x.shape[1]

    model = PipelineGNN(in_channels=in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for snapshot in historical_snapshots:
            data = build_graph_data(dag, snapshot)
            optimizer.zero_grad()
            _, reconstructed = model(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0:
            avg = total_loss / len(historical_snapshots)
            print(f"Epoch {epoch:3d}: loss={avg:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model


if __name__ == "__main__":
    import os
    import sys

    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text

    load_dotenv()

    db_url = os.environ.get("DATABASE_URL", "postgresql://mlobs:mlobs@localhost:5432/mlobs")
    model_id = sys.argv[1] if len(sys.argv) > 1 else "churn_model_v2"

    engine = create_engine(db_url)
    print(f"Fetching daily snapshots for '{model_id}'...")
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                DATE(timestamp)                                               AS day,
                AVG(prediction)                                               AS mean_pred,
                COALESCE(STDDEV(prediction), 0)                               AS std_pred,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY prediction)      AS p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY prediction)      AS p75,
                AVG(CASE WHEN prediction > 0.5 THEN 1.0 ELSE 0.0 END)        AS pos_rate,
                COUNT(*)                                                       AS n,
                COALESCE(AVG(confidence), AVG(prediction))                     AS mean_conf,
                COALESCE(STDDEV(confidence), STDDEV(prediction) * 0.5, 0)     AS std_conf
            FROM inference_logs
            WHERE model_id = :mid
            GROUP BY DATE(timestamp)
            ORDER BY day
        """), {"mid": model_id}).fetchall()
    engine.dispose()

    if not rows:
        print(f"No data for model_id='{model_id}'. Run demo/setup_demo.py first.")
        sys.exit(1)

    dag = PipelineDAG.from_yaml("configs/dag_definition.yaml")
    max_n = max(float(r.n) for r in rows)

    snapshots: list[dict[str, list[float]]] = []
    for r in rows:
        feats = [
            float(r.mean_pred),
            float(r.std_pred),
            float(r.p25),
            float(r.p75),
            float(r.pos_rate),
            float(r.n) / max_n,
            float(r.mean_conf),
            float(r.std_conf),
        ]
        snapshots.append({node.id: feats for node in dag.nodes()})

    print(f"Training GNN on {len(snapshots)} daily snapshots...")
    train(dag, snapshots)

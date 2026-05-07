from pathlib import Path
from typing import Optional

import torch

from .dag import PipelineDAG
from .gnn import PipelineGNN
from .train import build_graph_data


class RiskPropagator:
    def __init__(self, dag: PipelineDAG, model_path: Optional[str] = None, in_channels: int = 8):
        self._dag = dag
        self._model: Optional[PipelineGNN] = None
        if model_path and Path(model_path).exists():
            self._model = PipelineGNN(in_channels=in_channels)
            self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self._model.eval()

    def propagate(self, node_features: dict[str, list[float]]) -> dict[str, float]:
        """Returns risk score [0, 1] per node. Falls back to simple propagation if GNN not trained."""
        if self._model is None:
            return self._simple_propagate(node_features)

        data = build_graph_data(self._dag, node_features)
        raw_scores = self._model.anomaly_score(data.x, data.edge_index)

        nodes = self._dag.nodes()
        result = {}
        for i, node in enumerate(nodes):
            score = min(float(raw_scores[i].item()), 1.0)
            result[node.id] = score
            self._dag.update_risk_score(node.id, score)
        return result

    def _simple_propagate(self, node_features: dict[str, list[float]]) -> dict[str, float]:
        """Threshold-based fallback used before the GNN accumulates training history."""
        scores: dict[str, float] = {n.id: 0.0 for n in self._dag.nodes()}
        for node_id, feats in node_features.items():
            if feats:
                scores[node_id] = min(sum(abs(f) for f in feats) / len(feats), 1.0)
            for d_id in self._dag.downstream(node_id):
                scores[d_id] = max(scores.get(d_id, 0.0), scores[node_id] * 0.7)
        for node in self._dag.nodes():
            self._dag.update_risk_score(node.id, scores[node.id])
        return scores

import pytest

try:
    import torch
    from graph.dag import PipelineDAG, PipelineNode
    from graph.gnn import PipelineGNN
    from graph.train import build_graph_data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

pytestmark = pytest.mark.skipif(not HAS_TORCH_GEOMETRIC, reason="torch-geometric not installed")


def _make_dag() -> "PipelineDAG":
    dag = PipelineDAG()
    dag.add_node(PipelineNode("raw_data", "data_source", "Raw data"))
    dag.add_node(PipelineNode("transform", "transform", "Feature engineering"))
    dag.add_node(PipelineNode("model", "model", "Prediction model"))
    dag.add_edge("raw_data", "transform")
    dag.add_edge("transform", "model")
    return dag


def test_gnn_forward_output_shape():
    model = PipelineGNN(in_channels=8, hidden_channels=16, out_channels=8)
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    z, reconstructed = model(x, edge_index)
    assert z.shape == (3, 8)
    assert reconstructed.shape == (3, 8)


def test_anomaly_score_is_non_negative_per_node():
    model = PipelineGNN(in_channels=8, hidden_channels=16, out_channels=8)
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    scores = model.anomaly_score(x, edge_index)
    assert scores.shape == (3,)
    assert (scores >= 0).all()


def test_build_graph_data_correct_shape():
    dag = _make_dag()
    feats = {
        "raw_data": [1.0] * 8,
        "transform": [0.5] * 8,
        "model": [0.1] * 8,
    }
    data = build_graph_data(dag, feats)
    assert data.x.shape == (3, 8)
    assert data.edge_index.shape[0] == 2

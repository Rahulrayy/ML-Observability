import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv


class PipelineGNN(torch.nn.Module):
    """GraphSAGE model that learns normal pipeline behaviour via reconstruction."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32, num_layers: int = 3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.score_head = torch.nn.Linear(out_channels, in_channels)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        return self.convs[-1](x, edge_index)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x, edge_index)
        reconstructed = self.score_head(z)
        return z, reconstructed

    def anomaly_score(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Per-node reconstruction error — higher means more anomalous."""
        self.eval()
        with torch.no_grad():
            _, reconstructed = self(x, edge_index)
        return F.mse_loss(reconstructed, x, reduction="none").mean(dim=1)

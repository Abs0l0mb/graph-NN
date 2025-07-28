import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2

class A3TGCN(nn.Module):
    def __init__(
        self,
        node_features: int,
        input_length: int = 12,
        periods: int      = 12,
        hidden_channels: int = 32,
        batch_size: int      = 64,
    ):
        super().__init__()
        # in_channels=1 because each node has a single feature (e.g. speed)
        # periods=input_length (this drives the attention over T_in steps)
        self.tgnn = A3TGCN2(
            in_channels   = 1,
            out_channels  = hidden_channels,
            periods       = periods,
            batch_size    = batch_size,
        )
        # project the hidden vector at each node to your H‑step forecast
        self.linear = nn.Linear(hidden_channels, periods)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          ──>  (B, input_length, num_nodes)
        edge_index ──>  (2, E)
        returns    ──>  (B, horizon, num_nodes)
        """
        B, T, N = x.shape

        # 1) add feature‑dim: (B, T, N, 1)
        X = x.unsqueeze(-1)

        # 2) permute to (B, N, F_in=1, T_in)
        X = X.permute(0, 2, 3, 1)

        # 3) run the A3TGCN2 cell → (B, N, hidden_channels)
        H = self.tgnn(X, edge_index) 

        # 4) nonlinearity
        H = F.relu(H)

        # 5) project hidden → horizon per node: (B, N, horizon)
        Y = self.linear(H)

        # 6) permute back to (B, horizon, N) to match your y tensors
        return Y.permute(0, 2, 1)
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

'''
Goal of a STGCN block

The idea is to jointly learn:

- Temporal dependencies (how traffic changes over time)
- Spatial dependencies (how traffic at one sensor affects its neighbors)

Each STGCN block does this in three main parts:

- Temporal convolution (across time)
- Graph convolution (across space)
- Another temporal convolution (to refine temporal features)
'''

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        '''
        A 2D convolution with a (3, 1) kernel:
        3 on time axis (looks at 3 time steps: past, current, next)
        1 on the node axis (acts independently per node)
        Result: You get 32 feature maps per time-node pair
        This extracts local time patterns independently for each node.

        This is not usable, as the runs take too long ; we would have to vectorize a lot of data to aprallelize compute
        '''

        self.temporal1 = nn.Conv2d(in_channels, 32, (3, 1), padding=(1, 0))
        self.gcn = GCNConv(32, 32)
        self.temporal2 = nn.Conv2d(32, out_channels, (3, 1), padding=(1, 0))

    def forward(self, x, edge_index):
        # x: (Batch, Channels_in=1, Time, Nodes)

        # Step 1: Temporal convolution (learn time patterns independently for each node)
        x = self.temporal1(x)  # → shape: (B, C_mid, T, N)

        # Step 2: Prepare for graph convolution
        x = x.permute(0, 2, 3, 1)  # → (B, T, N, C_mid)
        B, T, N, C = x.shape

        # Step 3: Apply GCN for each time step independently
        x = x.reshape(B * T, N, C)         # → (B*T, N, C)
        x_out = []

        for t in range(B * T):
            x_t = x[t]  # shape: (N, C)
            x_gcn = self.gcn(x_t, edge_index)  # shape: (N, C)
            x_out.append(x_gcn)

        # Step 4: Stack and reshape back to 4D
        x = torch.stack(x_out, dim=0)          # → (B*T, N, C)
        x = x.view(B, T, N, C).permute(0, 3, 1, 2)  # → (B, C, T, N)

        # Step 5: Second temporal convolution
        x = self.temporal2(x)  # → (B, channel_out, T, N)

        return x
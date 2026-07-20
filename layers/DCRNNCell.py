import math

import torch
from torch import nn

from .GRUCell import GraphGRUCellBase


class DiffConv(nn.Module):
    """Diffusion graph convolution used by the final ALFGraph DCRNN cell."""

    def __init__(self, in_channels, out_channels, k_hop):
        super(DiffConv, self).__init__()
        self.orders = k_hop
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(
            torch.FloatTensor(in_channels * self.num_matrices, out_channels)
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def _concat(x, x_next):
        return torch.cat([x, x_next.unsqueeze(0)], dim=0)

    def forward(self, x, A_q, A_h):
        batch_size, num_nodes, input_size = x.shape
        x0 = x.permute(1, 2, 0).reshape(num_nodes, input_size * batch_size)
        features = torch.unsqueeze(x0, 0)
        for support in (A_q, A_h):
            x1 = torch.mm(support, x0)
            features = self._concat(features, x1)
            for _ in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                features = self._concat(features, x2)
                x1, x0 = x2, x1

        features = features.reshape(
            self.num_matrices,
            num_nodes,
            input_size,
            batch_size,
        )
        features = features.permute(3, 1, 2, 0).reshape(
            batch_size,
            num_nodes,
            input_size * self.num_matrices,
        )
        return torch.matmul(features, self.Theta1) + self.bias


class DCRNNCell(GraphGRUCellBase):
    """Diffusion convolutional recurrent cell (Li et al., ICLR 2018)."""

    def __init__(self, input_size: int, hidden_size: int, k: int = 2):
        forget_gate = DiffConv(input_size + hidden_size, hidden_size, k)
        update_gate = DiffConv(input_size + hidden_size, hidden_size, k)
        candidate_gate = DiffConv(input_size + hidden_size, hidden_size, k)
        super(DCRNNCell, self).__init__(
            hidden_size=hidden_size,
            forget_gate=forget_gate,
            update_gate=update_gate,
            candidate_gate=candidate_gate,
        )

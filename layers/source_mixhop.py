import torch
from torch import nn
import torch.nn.functional as F


class SourceDenseNGCNLayer(nn.Module):
    # Copied from references/MixHop/src/layers.py::DenseNGCNLayer.
    # Local changes: accepts batched dense node features [B, N, C] and a dense
    # normalized adjacency [N, N] instead of torch_sparse.spmm dictionaries.
    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(SourceDenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        self.weight_matrix = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        base_features = torch.matmul(features, self.weight_matrix)
        base_features = F.dropout(base_features, p=self.dropout_rate, training=self.training)
        for _ in range(self.iterations - 1):
            base_features = torch.einsum("nm,bmc->bnc", normalized_adjacency_matrix, base_features).contiguous()
        base_features = base_features + self.bias
        return base_features


class SourceMixHopDiffConv(nn.Module):
    # Adapted from references/MixHop/src/trainer_and_networks.py::MixHopNetwork:
    # multiple NGCN layers with different adjacency powers are concatenated and
    # then fused for the recurrent gate output dimension.
    def __init__(self, in_channels, out_channels, k_hop, dropout_rate=0.0):
        super(SourceMixHopDiffConv, self).__init__()
        self.orders = max(1, int(k_hop))
        self.self_layer = nn.Linear(in_channels, out_channels)
        self.forward_layers = nn.ModuleList(
            [SourceDenseNGCNLayer(in_channels, out_channels, i, dropout_rate) for i in range(1, self.orders + 1)]
        )
        self.backward_layers = nn.ModuleList(
            [SourceDenseNGCNLayer(in_channels, out_channels, i, dropout_rate) for i in range(1, self.orders + 1)]
        )
        self.fuse = nn.Linear((1 + 2 * self.orders) * out_channels, out_channels)

    def forward(self, X, A_q, A_h):
        abstract_features = [self.self_layer(X)]
        abstract_features.extend(layer(A_q, X) for layer in self.forward_layers)
        abstract_features.extend(layer(A_h, X) for layer in self.backward_layers)
        return self.fuse(torch.cat(abstract_features, dim=-1))

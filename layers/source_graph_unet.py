import numpy as np
import torch
from torch import nn


class GCN(nn.Module):
    """Copied from references/Graph-U-Nets/src/utils/ops.py."""

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h


class Pool(nn.Module):
    """Copied from references/Graph-U-Nets/src/utils/ops.py."""

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):
    """Copied from references/Graph-U-Nets/src/utils/ops.py."""

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    """Copied from references/Graph-U-Nets/src/utils/ops.py."""

    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    """Copied from references/Graph-U-Nets/src/utils/ops.py."""

    degrees = torch.sum(g, 1)
    return g / degrees.clamp_min(1e-6).unsqueeze(-1)


def _degree_scores(adj):
    adj_sym = (adj + adj.T) / 2.0
    degree = adj_sym.sum(axis=1)
    tie_break = np.linspace(0.0, 1e-6, adj.shape[0], dtype=np.float64)
    return degree + tie_break


def _split_with_graph_unet_pool(adj, nodes):
    if len(nodes) <= 1:
        return nodes, []
    local_adj = adj[np.ix_(nodes, nodes)]
    scores_np = _degree_scores(local_adj)
    scores = torch.as_tensor(scores_np, dtype=torch.float32)
    g = torch.as_tensor((local_adj > 0).astype(np.float32), dtype=torch.float32)
    h = scores.unsqueeze(-1)
    keep = max(1, len(nodes) // 2)
    if len(nodes) <= 3:
        local_top = torch.argsort(scores, descending=True)[:keep]
    else:
        _, _, idx = top_k_graph(scores, g, h, keep / float(len(nodes)))
        local_top = idx[:keep]
    top_set = {int(i) for i in local_top.cpu().tolist()}
    left = [nodes[i] for i in range(len(nodes)) if i in top_set]
    right = [nodes[i] for i in range(len(nodes)) if i not in top_set]
    return left, right


def graph_unet_parts(adj, recur_times):
    """Build static PatchSTG leaves by recursively applying Graph-U-Net top-k pooling scores."""

    leaves = [list(range(adj.shape[0]))]
    for _ in range(int(recur_times)):
        next_leaves = []
        for nodes in leaves:
            left, right = _split_with_graph_unet_pool(adj, nodes)
            if left:
                next_leaves.append(left)
            if right:
                next_leaves.append(right)
        leaves = next_leaves
    return [np.sort(np.asarray(nodes, dtype=int)) for nodes in leaves if len(nodes) > 0]

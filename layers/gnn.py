import numpy as np
import scipy.sparse as sp
import torch


def calculate_random_walk_matrix(adj):
    """Return the row-normalized random-walk adjacency matrix."""
    device = adj.device
    sparse_adj = sp.coo_matrix(adj.cpu().numpy())
    degree = np.asarray(sparse_adj.sum(1))
    degree = np.where(degree == 0, 1, degree)
    inverse_degree = np.power(degree, -1).flatten()
    inverse_degree[np.isinf(inverse_degree)] = 0.0
    random_walk = sp.diags(inverse_degree).dot(sparse_adj).tocoo()
    return torch.from_numpy(random_walk.toarray()).float().to(device)

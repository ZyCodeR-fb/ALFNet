"""Inductive node embeddings: shared mapping over frozen structure.

The problem this replaces
------------------------
Every PatchSTG-derived backbone (and SLPF, STID, STBP) carried a
``nn.Parameter(node_num, node_dims)`` indexed by node id. Under this project's
protocol the node set is split into training-known nodes and held-out nodes that
the model is supposed to have never seen, and a per-node parameter breaks that:
the held-out rows still take gradient, because the spatial attention mixes every
node before the known ones are gathered back out.

Measured on ``fg1`` after five optimiser steps: all 50 held-out rows of
``node_emb`` received non-zero gradient -- with a *larger* mean magnitude than
the known rows -- and perturbing only those 50 rows moved the forecast for the
known nodes by 1.056. A model with node-specific capacity for nodes it claims
not to have seen is not inductive, whatever it is trained on.

What replaces it
----------------
One small MLP, shared by every node, applied to features that are functions of
frozen structure only:

* normalised coordinates, when the dataset ships them;
* weighted in-degree, out-degree and total degree, each normalised;
* a log-scaled total degree, because degree distributions are heavy-tailed;
* an isolated flag.

No node owns a parameter. A node the model never trained on gets an embedding
from the same mapping as every other node, which is what makes the setting
genuinely inductive: the mapping generalises, the identity does not.

Two properties follow by construction and are pinned by tests:

* held-out nodes have no dedicated trainable parameter, so nothing of theirs can
  receive gradient;
* the encoder is permutation equivariant -- permuting the nodes, the adjacency
  and the coordinates together permutes the output the same way.

Lifecycle features are deliberately absent. They are ALFNet's method inputs and
must not leak into a baseline's node representation.
"""

import hashlib

import torch
from torch import nn


# coordinates (2) + in/out/total degree (3) + log total degree (1) + isolated (1)
STRUCTURAL_FEATURE_NAMES = (
    "pos_x", "pos_y", "in_degree", "out_degree", "total_degree",
    "log_total_degree", "isolated",
)
STRUCTURAL_FEATURE_DIM = len(STRUCTURAL_FEATURE_NAMES)
DEFAULT_HIDDEN_DIM = 64


def _fingerprint(features):
    """A stable scalar digest of the structural features.

    Stored as a buffer so it travels in the checkpoint and can be compared on
    load. It is a digest of the *derived* features rather than of the raw
    adjacency, because those are exactly what the encoder consumes.
    """
    payload = features.detach().cpu().double().contiguous().numpy().tobytes()
    return float(int(hashlib.sha256(payload).hexdigest()[:12], 16))


def structural_fingerprints(model):
    """{module path: fingerprint} for every structural encoder in a model."""
    return {
        name: float(module.structure_fingerprint)
        for name, module in model.named_modules()
        if isinstance(module, StructuralNodeEncoder)
    }


def _normalise(values):
    """Zero mean, unit scale, robust to a constant column."""
    centred = values - values.mean()
    scale = centred.abs().max().clamp_min(1e-6)
    return centred / scale


def build_structural_features(adjacency, positions=None):
    """[N, STRUCTURAL_FEATURE_DIM] from frozen topology and geometry.

    `adjacency` is used as given -- weighted and possibly asymmetric, so in- and
    out-degree are genuinely different features. `positions` may be None, in
    which case the coordinate columns are zero and the encoder degrades to a
    purely topological one, as required for datasets that ship no geometry.
    """
    weights = torch.nan_to_num(adjacency.float(), nan=0.0)
    if weights.dim() != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"adjacency must be square, got {tuple(weights.shape)}")
    num_nodes = weights.shape[0]
    self_loop = torch.eye(num_nodes, device=weights.device, dtype=weights.dtype)
    off_diagonal = weights * (1.0 - self_loop)

    in_degree = off_diagonal.sum(dim=0)
    out_degree = off_diagonal.sum(dim=1)
    total_degree = in_degree + out_degree
    isolated = (total_degree <= 0).to(weights.dtype)

    if positions is None:
        coordinates = torch.zeros(num_nodes, 2, device=weights.device, dtype=weights.dtype)
    else:
        coordinates = torch.as_tensor(positions, device=weights.device, dtype=weights.dtype)
        if coordinates.shape[0] != num_nodes:
            raise ValueError(
                f"positions cover {coordinates.shape[0]} nodes, adjacency has {num_nodes}"
            )
        coordinates = coordinates[:, :2] if coordinates.shape[1] >= 2 else torch.cat(
            [coordinates, torch.zeros(num_nodes, 2 - coordinates.shape[1],
                                      device=weights.device, dtype=weights.dtype)],
            dim=1,
        )
        coordinates = torch.stack(
            [_normalise(coordinates[:, 0]), _normalise(coordinates[:, 1])], dim=1
        )

    return torch.stack(
        [
            coordinates[:, 0],
            coordinates[:, 1],
            _normalise(in_degree),
            _normalise(out_degree),
            _normalise(total_degree),
            _normalise(torch.log1p(total_degree.clamp_min(0.0))),
            isolated,
        ],
        dim=1,
    )


class StructuralNodeEncoder(nn.Module):
    """Node embedding = shared MLP over frozen structural features.

    Drop-in replacement for a ``[node_num, node_dims]`` parameter: call it with
    no arguments and it returns exactly that shape, so call sites change by one
    line. The features are a buffer, not a parameter, so they travel with the
    checkpoint and with `.to(device)` while never receiving gradient.
    """

    def __init__(self, adjacency, positions, out_dim, hidden_dim=None):
        super().__init__()
        features = build_structural_features(adjacency, positions)
        # A buffer, deliberately: it is data about the graph, not something the
        # optimiser may touch. Persistent so a checkpoint records the geometry
        # it was trained on.
        self.register_buffer("features", features)
        # Provenance. Two datasets with the same node count produce different
        # structure, and a checkpoint trained on one must not load against the
        # other -- the shapes match, so nothing else would notice.
        self.register_buffer(
            "structure_fingerprint",
            torch.tensor(_fingerprint(features), dtype=torch.float64),
            persistent=True,
        )
        self.num_nodes = int(features.shape[0])
        self.out_dim = int(out_dim)
        # Fixed, not a multiple of out_dim: the input is seven numbers, so the
        # bottleneck is the input side and widening the hidden layer only buys
        # parameters. Scaling with out_dim was measurably wrong -- at STBP's
        # out_dim=576 it produced a 673k-parameter encoder replacing a 119k
        # bank, i.e. the fix cost five times what it removed.
        hidden_dim = int(hidden_dim or DEFAULT_HIDDEN_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(STRUCTURAL_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, node_ids=None):
        """[N, out_dim], or [len(node_ids), out_dim] for a subset."""
        embedding = self.mlp(self.features)
        if node_ids is None:
            return embedding
        index = torch.as_tensor(node_ids, device=embedding.device, dtype=torch.long)
        return embedding.index_select(0, index)

    def extra_repr(self):
        return f"num_nodes={self.num_nodes}, out_dim={self.out_dim}, shared=True"

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

# Provenance schema. v1 hashed the raw float bytes of the *derived* features and
# was therefore not portable between machines; v2 splits exact raw identity from
# a quantised record of the derived tensor.
STRUCTURE_PROVENANCE_VERSION = 2
# Grid for the derived features. An order of magnitude above the largest drift
# measured across machines (1.31e-07) and far below any real structural change.
FEATURE_QUANTUM = 1e-6
# Tolerance when a stored feature buffer is compared directly. rtol is zero on
# purpose: these are normalised values that legitimately sit near zero, and a
# relative tolerance there is meaningless.
FEATURE_ATOL = 1e-6


def _canonical_bytes(tensor):
    """Shape and dtype alongside the values, so neither can change silently."""
    if tensor is None:
        return b"none"
    values = torch.as_tensor(tensor).detach().cpu().double().contiguous()
    header = f"{tuple(values.shape)}|{values.dtype}".encode("utf-8")
    return header + b"|" + values.numpy().tobytes()


def structure_digest(adjacency, positions, features):
    """Version 1: hex digest over the raw graph *and* what was derived from it.

    Hashing only the derived features is not enough: the seven features are
    coordinates plus degree summaries, so two graphs with identical coordinates
    and identical degree sequences but different edges produce identical
    features and would have collided. The adjacency itself is included, which is
    what actually distinguishes a rewiring.

    Superseded by `structure_provenance` because it folds exact float bytes of
    *derived* features into a single digest, which makes it machine dependent --
    see that function. Kept because checkpoints written before the change store
    it, and because decomposing a v1 digest is how those are still verified.
    """
    digest = hashlib.sha256()
    for part in (adjacency, positions, features):
        digest.update(_canonical_bytes(part))
        digest.update(b"::")
    return digest.hexdigest()


def raw_structure_digest(adjacency, positions):
    """Exact digest over the graph as loaded, with no derived quantity in it.

    `load_project_adjacency` and `load_project_positions` are file reads with a
    dtype cast and no arithmetic, so these bytes are identical on every machine.
    This is the part of the identity that may be compared for exact equality.
    """
    digest = hashlib.sha256()
    for part in (adjacency, positions):
        digest.update(_canonical_bytes(part))
        digest.update(b"::")
    return digest.hexdigest()


def quantised_feature_digest(features, quantum=FEATURE_QUANTUM):
    """A recorded fingerprint of the derived features. Not a portability check.

    `build_structural_features` sums rows of the adjacency, and a reduction's
    order depends on the BLAS build and thread count. Measured between this
    machine and the training server: identical graph, features differing by
    1.31e-07 -- one float32 ulp -- which changed every bit of the schema-1
    raw-byte digest and made the server's checkpoints unloadable here.

    Quantising reduces that churn but does **not** remove it, and it is worth
    being exact about why rather than assuming it does. Rounding to a grid still
    has boundaries, and a value sitting near one crosses under an arbitrarily
    small perturbation. Measured over 30 random graphs at this quantum, 9.65% of
    individual values changed bucket under a one-ulp nudge, and because a digest
    covers the whole tensor, **all 30 digests differed**. So this value must not
    be used as the deciding comparison.

    What decides identity is `raw_structure_digest`, which covers only tensors
    that are read from a file with no arithmetic and is therefore bit-stable,
    together with the algorithm version -- the features are a deterministic
    function of those two. A numerical divergence is caught by comparing the
    stored feature buffer within `FEATURE_ATOL`. This digest is kept because it
    records shape, dtype, quantum and algorithm alongside the values, and
    because equality is a cheap fast path when a checkpoint has not moved.
    """
    values = torch.as_tensor(features).detach().cpu()
    header = (
        f"v{STRUCTURE_PROVENANCE_VERSION}|{tuple(values.shape)}|{values.dtype}|"
        f"{quantum:.17g}"
    ).encode("utf-8")
    # Rounding to a grid, not truncating: truncation moves a value by up to a
    # whole quantum, so two machines could straddle a boundary far more often.
    grid = torch.round(values.double() / float(quantum)).to(torch.int64)
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"::")
    digest.update(grid.contiguous().numpy().tobytes())
    return digest.hexdigest()


def structure_provenance(adjacency, positions, features, quantum=FEATURE_QUANTUM):
    """Version 2 identity: exact on what was loaded, tolerant on what was derived.

    `raw` is the part that decides whether this is the same graph and is compared
    for exact equality. `features` is a consistency record over the derived
    tensor; it is quantised, and a mismatch there is checked against the stored
    buffer with a tolerance rather than treated as a different graph.
    """
    return {
        "version": STRUCTURE_PROVENANCE_VERSION,
        "raw": raw_structure_digest(adjacency, positions),
        "features": quantised_feature_digest(features, quantum),
        "shape": list(torch.as_tensor(features).shape),
        "dtype": str(torch.as_tensor(features).dtype),
        "quantum": float(quantum),
        "algorithm": "build_structural_features",
    }


def _fingerprint_scalar(hex_digest):
    """A float view of the digest, so it can live in a tensor buffer.

    float64 holds 53 bits exactly; 12 hex characters is 48 bits, so the value
    round-trips through the checkpoint without loss.
    """
    return float(int(hex_digest[:12], 16))


def structural_fingerprints(model):
    """{module path: v2 provenance dict} for every structural encoder in a model.

    A dict rather than the float buffer: the buffer exists so a value survives a
    state_dict round trip, but provenance comparison needs the whole record --
    the exact raw digest and the quantised feature digest are checked
    differently, and collapsing them to one string is what made v1 unportable.
    """
    return {
        name: dict(module.structure_provenance)
        for name, module in model.named_modules()
        if isinstance(module, StructuralNodeEncoder)
    }


def legacy_structural_fingerprints(model):
    """{module path: v1 hex digest}, for checking checkpoints written before v2."""
    return {
        name: module.structure_hash
        for name, module in model.named_modules()
        if isinstance(module, StructuralNodeEncoder)
    }


def structural_encoders(model):
    """{module path: module} for every structural encoder in a model."""
    return {
        name: module
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
        # Non-persistent: they must not enter the state_dict, or every existing
        # checkpoint gains unexpected keys and stops loading. Retained so a v1
        # digest can be recomposed against a stored feature buffer, which is how
        # a pre-v2 checkpoint is verified without trusting a machine-dependent
        # hash -- see `legacy_digest_for`.
        self.register_buffer(
            "_raw_adjacency",
            torch.as_tensor(adjacency).detach().cpu().clone(),
            persistent=False,
        )
        self.register_buffer(
            "_raw_positions",
            None if positions is None
            else torch.as_tensor(positions).detach().cpu().clone(),
            persistent=False,
        )
        # Provenance over the raw graph as well as the derived features. Two
        # datasets with the same node count produce shape-compatible state
        # dicts, and two graphs with the same coordinates and degree sequence
        # produce identical features, so neither alone identifies the graph.
        self.structure_provenance = structure_provenance(adjacency, positions, features)
        # v1, kept so a checkpoint written before the schema change still has
        # something to compare against.
        self.structure_hash = structure_digest(adjacency, positions, features)
        self.register_buffer(
            "structure_fingerprint",
            torch.tensor(_fingerprint_scalar(self.structure_hash), dtype=torch.float64),
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

    def legacy_digest_for(self, features):
        """The v1 digest this encoder's graph would produce for `features`.

        Passing a checkpoint's stored feature buffer decomposes a v1 digest: if
        the result equals what the checkpoint recorded, then this machine's
        adjacency and positions are byte-identical to the training machine's,
        and any remaining difference is confined to the derived features. That
        distinction is unavailable from the v1 digest alone, which is why a
        1-ulp drift used to be indistinguishable from a different graph.
        """
        return structure_digest(self._raw_adjacency, self._raw_positions, features)

    def forward(self, node_ids=None):
        """[N, out_dim], or [len(node_ids), out_dim] for a subset."""
        embedding = self.mlp(self.features)
        if node_ids is None:
            return embedding
        index = torch.as_tensor(node_ids, device=embedding.device, dtype=torch.long)
        return embedding.index_select(0, index)

    def extra_repr(self):
        return f"num_nodes={self.num_nodes}, out_dim={self.out_dim}, shared=True"

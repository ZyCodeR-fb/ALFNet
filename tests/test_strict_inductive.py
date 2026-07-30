"""Held-out nodes must not own trainable parameters.

The setting claims to be inductive: a node in the held-out set is one the model
never trained on. A `nn.Parameter(node_num, node_dims)` indexed by node id
breaks that claim outright, and not in a subtle way -- measured on the previous
`fg1`, after five optimiser steps all 50 held-out rows carried non-zero
gradient, with a *larger* mean magnitude than the known rows, and perturbing
only those rows moved the known-node forecast by 1.056.

`StructuralNodeEncoder` replaces it with one MLP shared by every node, over
features derived from frozen adjacency and coordinates. These tests pin the
three properties that make that substitution meaningful:

  * no parameter is indexed by node id, so a held-out node has nothing of its
    own to receive gradient;
  * the encoder is permutation equivariant, which is what "the mapping
    generalises, the identity does not" means operationally;
  * the features carry structure, not identity -- two graphs that differ only by
    a relabelling produce the same embeddings up to that relabelling.
"""

import pytest
import torch
import pytorch_lightning as pl

from args import load_args_from_config
from layers.structural_node_encoder import (
    STRUCTURAL_FEATURE_DIM,
    StructuralNodeEncoder,
    build_structural_features,
    structural_fingerprints,
)
from lib.utils import resolve_known_unknown_nodes
from model.regist_model import load_model_class


# Every model in the comparison table, not only the ones that happened to be
# listed first: GWNET's `nodevec1/nodevec2` were missed on the first pass
# precisely because the audit was scoped to a hand-written list.
ALL_MODELS = [
    "ALFNet", "MoGERNN", "PatchSTG", "FCLSTM", "GWNET", "HA", "MCPST_FSL",
    "SLPF", "STBP", "STFiT", "STGCN", "STGODE", "STID",
]

NODE_ID_FREE_CONFIGS = ["configs/ALFNet/fg1.yaml"]

# A shape whose axis equals the node count is a *candidate*, not a verdict --
# SLPF's `rank_emb` is sized by node count but indexed by a data-derived rank
# (`_build_rank_feature` uses `torch.argsort(local_values)`, recomputed per
# sample and timestep), so no node owns a row. Anything not on this list must be
# justified before it is added, which is the point of keeping the list explicit
# rather than loosening the check.
JUSTIFIED_NODE_SHAPED_PARAMETERS = {
    ("SLPF", "backbone.rank_emb"),
}


def _config_for(model):
    if model == "ALFNet":
        return "configs/ALFNet/fg1.yaml"
    return f"configs/{model}/METRLA_lcbt_v3.yaml"


def _build(model):
    pl.seed_everything(32, workers=True)
    args = load_args_from_config(_config_for(model), [])
    args.device = "cpu"
    return args, load_model_class(args.model)(args)


def _node_counts(args):
    known, unknown = resolve_known_unknown_nodes({
        "data_path": args.data_path,
        "unknown_nodes_path": args.unknown_nodes_path,
        "unknown_nodes_list": args.unknown_nodes_list,
        "unknown_nodes_num": args.unknown_nodes_num,
        "seed": args.seed,
    })
    return known, unknown


# --- every model in the table, not a hand-picked subset ----------------------


@pytest.mark.parametrize("model", ALL_MODELS)
def test_model_has_no_unjustified_node_indexed_parameter(model):
    args, net = _build(model)
    known, unknown = _node_counts(args)
    full = len(known) + len(unknown)

    candidates = {
        name for name, param in net.named_parameters()
        if param.requires_grad and full in tuple(param.shape)
    }
    unjustified = {
        name for name in candidates
        if (model, name) not in JUSTIFIED_NODE_SHAPED_PARAMETERS
    }
    assert not unjustified, (
        f"{model}: these trainable parameters have an axis per graph node, so "
        f"held-out nodes own capacity: {sorted(unjustified)}. Either replace "
        "them with a shared structural mapping or add them to "
        "JUSTIFIED_NODE_SHAPED_PARAMETERS with the reason."
    )


@pytest.mark.parametrize("model", ALL_MODELS)
def test_model_builds_and_forwards_finite(model):
    """A compliance check that never runs the model would prove nothing."""
    args, net = _build(model)
    known, _ = _node_counts(args)
    nodes = len(getattr(net, "known_nodes", known))
    generator = torch.Generator().manual_seed(1)
    adjacency = torch.rand(nodes, nodes, generator=generator)
    x = torch.randn(2, args.input_len, nodes, args.enc_in, generator=generator)
    marks = torch.rand(2, args.input_len, 3, generator=generator) * 5
    horizon = torch.rand(2, args.pred_len, 3, generator=generator) * 5

    net.eval()
    with torch.no_grad():
        out = net(adjacency, x, marks, None, None, horizon, 0)
    prediction = out["pred"] if isinstance(out, dict) else out
    assert torch.isfinite(prediction).all(), f"{model} produced a non-finite forecast"
    assert prediction.shape[0] == 2 and prediction.shape[2] == nodes


@pytest.mark.parametrize("model", ["ALFNet", "PatchSTG", "STID", "SLPF", "STBP", "GWNET"])
def test_adapted_model_uses_a_shared_structural_encoder(model):
    """The models that carried a node-id bank must now carry an encoder."""
    _, net = _build(model)
    encoders = [
        module for module in net.modules()
        if isinstance(module, StructuralNodeEncoder)
    ]
    assert encoders, f"{model} lost its structural encoder"
    for encoder in encoders:
        for name, param in encoder.named_parameters():
            assert encoder.num_nodes not in tuple(param.shape), (
                f"{model}: {name} inside the encoder is still node-indexed"
            )


@pytest.mark.parametrize("model", ["GWNET"])
def test_gwnet_adaptive_adjacency_stays_directed(model):
    """Two independent encoders, so `softmax(relu(L R^T))` is not symmetric.

    One encoder reused on both sides would give `relu(E E^T)`, a symmetric
    operator, which is a different model from Graph WaveNet's.
    """
    _, net = _build(model)
    backbone = net.backbone
    assert hasattr(backbone, "nodevec_left") and hasattr(backbone, "nodevec_right")
    assert backbone.nodevec_left is not backbone.nodevec_right
    left, right = backbone.nodevec_left(), backbone.nodevec_right()
    product = torch.mm(left, right.t())
    assert not torch.allclose(product, product.t(), atol=1e-4), (
        "the adaptive adjacency became symmetric; the two factors are tied"
    )


def _graph(num_nodes=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    adjacency = (torch.rand(num_nodes, num_nodes, generator=generator) < 0.3).float()
    adjacency = adjacency * torch.rand(num_nodes, num_nodes, generator=generator)
    adjacency.fill_diagonal_(0.0)
    positions = torch.rand(num_nodes, 2, generator=generator)
    return adjacency, positions


# --- no parameter may be indexed by node id ----------------------------------


@pytest.mark.parametrize("config_path", NODE_ID_FREE_CONFIGS)
def test_no_parameter_is_sized_by_the_node_count(config_path):
    pl.seed_everything(32, workers=True)
    args = load_args_from_config(config_path, [])
    args.device = "cpu"
    model = load_model_class(args.model)(args)
    known, unknown = resolve_known_unknown_nodes({
        "data_path": args.data_path,
        "unknown_nodes_path": args.unknown_nodes_path,
        "unknown_nodes_list": args.unknown_nodes_list,
        "unknown_nodes_num": args.unknown_nodes_num,
        "seed": args.seed,
    })
    full = len(known) + len(unknown)

    offenders = [
        (name, tuple(param.shape))
        for name, param in model.named_parameters()
        if param.requires_grad and full in tuple(param.shape)
    ]
    assert not offenders, (
        "these trainable parameters have one axis per graph node, so held-out "
        f"nodes own capacity: {offenders}"
    )
    assert not any("node_emb" in name for name, _ in model.named_parameters()), (
        "a node-id embedding came back"
    )


@pytest.mark.parametrize("config_path", NODE_ID_FREE_CONFIGS)
def test_structural_features_are_a_buffer_not_a_parameter(config_path):
    """They describe the graph; the optimiser must not be able to move them."""
    pl.seed_everything(32, workers=True)
    args = load_args_from_config(config_path, [])
    args.device = "cpu"
    model = load_model_class(args.model)(args)
    buffers = {name for name, _ in model.named_buffers()}
    assert any(name.endswith("node_encoder.features") for name in buffers), buffers
    for name, param in model.named_parameters():
        assert not name.endswith("node_encoder.features")


def test_training_leaves_no_gradient_on_anything_node_indexed():
    """The end-to-end version of the check, after the FiLM head has left zero-init.

    film_head's last layer is zero-initialised, so at step 0 the whole patch
    branch legitimately receives no gradient. Probing there measures the
    initialisation rather than the training regime, which is why this takes a
    few steps first.
    """
    pl.seed_everything(32, workers=True)
    args = load_args_from_config("configs/ALFNet/fg1.yaml", [])
    args.device = "cpu"
    model = load_model_class(args.model)(args)
    known, unknown = resolve_known_unknown_nodes({
        "data_path": args.data_path,
        "unknown_nodes_path": args.unknown_nodes_path,
        "unknown_nodes_list": args.unknown_nodes_list,
        "unknown_nodes_num": args.unknown_nodes_num,
        "seed": args.seed,
    })
    nodes = len(known)
    generator = torch.Generator().manual_seed(5)
    adjacency = torch.rand(nodes, nodes, generator=generator)
    x = torch.randn(2, args.input_len, nodes, args.enc_in, generator=generator)
    tm = torch.rand(2, args.input_len, 3, generator=generator) * 5
    dec = torch.randn(2, args.pred_len, nodes, args.c_out, generator=generator)
    dm = torch.rand(2, args.pred_len, 3, generator=generator) * 5

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(5):
        optimiser.zero_grad(set_to_none=True)
        out = model(adjacency, x, tm, None, dec, dm, 0)
        (out["pred"] - dec).pow(2).mean().backward()
        optimiser.step()

    full = len(known) + len(unknown)
    node_indexed = [
        name for name, param in model.named_parameters()
        if param.requires_grad and full in tuple(param.shape)
    ]
    assert not node_indexed, node_indexed

    # and the structural features never moved
    features = dict(model.named_buffers())["patch_expert.backbone.node_encoder.features"]
    assert features.grad is None
    assert not features.requires_grad


# --- permutation equivariance ------------------------------------------------


def test_features_are_permutation_equivariant():
    adjacency, positions = _graph()
    base = build_structural_features(adjacency, positions)
    perm = torch.randperm(adjacency.shape[0], generator=torch.Generator().manual_seed(1))
    permuted = build_structural_features(
        adjacency[perm][:, perm], positions[perm]
    )
    assert torch.allclose(permuted, base[perm], atol=1e-5), (
        "relabelling the nodes changed the structure they describe"
    )


def test_encoder_is_permutation_equivariant():
    """Jointly permuting adjacency and coordinates permutes the embeddings."""
    adjacency, positions = _graph()
    torch.manual_seed(0)
    encoder = StructuralNodeEncoder(adjacency, positions, out_dim=8)
    base = encoder()

    perm = torch.randperm(adjacency.shape[0], generator=torch.Generator().manual_seed(2))
    torch.manual_seed(0)
    permuted_encoder = StructuralNodeEncoder(
        adjacency[perm][:, perm], positions[perm], out_dim=8
    )
    permuted = permuted_encoder()
    assert torch.allclose(permuted, base[perm], atol=1e-5), (
        "the encoder is not permutation equivariant, so it is reading identity"
    )


def test_two_nodes_with_identical_structure_get_identical_embeddings():
    """Identity is not an input: structurally interchangeable nodes coincide."""
    # a 4-cycle with no coordinates -- every node has the same degree profile
    adjacency = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    torch.manual_seed(0)
    encoder = StructuralNodeEncoder(adjacency, None, out_dim=6)
    embedding = encoder()
    assert torch.allclose(embedding[0], embedding[1], atol=1e-6)
    assert torch.allclose(embedding[0], embedding[3], atol=1e-6)


# --- the features themselves -------------------------------------------------


def test_features_degrade_to_topology_without_coordinates():
    adjacency, _ = _graph()
    features = build_structural_features(adjacency, None)
    assert features.shape == (adjacency.shape[0], STRUCTURAL_FEATURE_DIM)
    assert torch.all(features[:, 0] == 0) and torch.all(features[:, 1] == 0), (
        "coordinate columns must be zero when the dataset ships no geometry"
    )
    assert torch.isfinite(features).all()


def test_isolated_nodes_are_flagged_and_finite():
    adjacency = torch.zeros(5, 5)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    features = build_structural_features(adjacency, None)
    isolated = features[:, -1]
    assert isolated.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert torch.isfinite(features).all(), "a zero-degree node produced a non-finite feature"


# --- the graph enters provenance ---------------------------------------------


def test_structure_fingerprint_separates_graphs_with_the_same_node_count():
    """The failure this guards: same shapes, different graph, silent load."""
    adjacency_a, positions_a = _graph(seed=0)
    adjacency_b, positions_b = _graph(seed=99)
    assert adjacency_a.shape == adjacency_b.shape

    torch.manual_seed(0)
    encoder_a = StructuralNodeEncoder(adjacency_a, positions_a, out_dim=8)
    torch.manual_seed(0)
    encoder_b = StructuralNodeEncoder(adjacency_b, positions_b, out_dim=8)

    assert float(encoder_a.structure_fingerprint) != float(encoder_b.structure_fingerprint)
    # and the state dicts are shape-compatible, which is why the fingerprint has
    # to exist at all
    assert {k: v.shape for k, v in encoder_a.state_dict().items()} == \
           {k: v.shape for k, v in encoder_b.state_dict().items()}


def test_structure_fingerprint_is_stable_for_the_same_graph():
    adjacency, positions = _graph(seed=3)
    torch.manual_seed(0)
    first = StructuralNodeEncoder(adjacency, positions, out_dim=8)
    torch.manual_seed(7)                       # different weights, same graph
    second = StructuralNodeEncoder(adjacency, positions, out_dim=8)
    assert float(first.structure_fingerprint) == float(second.structure_fingerprint)


def test_structural_fingerprints_are_reported_for_a_real_model():
    _, net = _build("ALFNet")
    fingerprints = structural_fingerprints(net)
    assert fingerprints, "a model with a structural encoder reported none"
    assert all(value != 0.0 for value in fingerprints.values())


def test_asymmetric_adjacency_gives_different_in_and_out_degree():
    """Direction is real information; collapsing it would waste a feature."""
    adjacency = torch.zeros(3, 3)
    adjacency[0, 1] = 1.0            # 0 -> 1 only
    adjacency[0, 2] = 1.0
    features = build_structural_features(adjacency, None)
    in_degree, out_degree = features[:, 2], features[:, 3]
    assert not torch.allclose(in_degree, out_degree)

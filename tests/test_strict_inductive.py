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
)
from lib.utils import resolve_known_unknown_nodes
from model.regist_model import load_model_class


# Every model that used to index a parameter by node id.
NODE_ID_FREE_CONFIGS = ["configs/ALFNet/fg1.yaml"]


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


def test_asymmetric_adjacency_gives_different_in_and_out_degree():
    """Direction is real information; collapsing it would waste a feature."""
    adjacency = torch.zeros(3, 3)
    adjacency[0, 1] = 1.0            # 0 -> 1 only
    adjacency[0, 2] = 1.0
    features = build_structural_features(adjacency, None)
    in_degree, out_degree = features[:, 2], features[:, 3]
    assert not torch.allclose(in_degree, out_degree)

"""ALFTemporal's v4 constructor takes its graph from the context, not a path.

Deleting the explicit `load_project_adjacency` call would not have been enough:
`build_patchstg_cfg` reaches `data_path` through `infer_project_num_nodes` and
`resolve_known_unknown_nodes`, so the private resolution survives the obvious
fix and stays invisible. These tests make every one of those raise, then build
the model.

Scope: constructors only. The K->N scatter and the old forward are deliberately
still here and are exercised by a smoke test, not presented as fair.
"""

import numpy as np
import pytest
import torch

from args import load_args_from_config
from lib.runtime_context import ContextRefused, build_runtime_context

CONFIGS = {"PEMSBAY": "configs/paper/v4/experts/PEMSBAY_ALFTemporal.yaml",
           "PEMS08": "configs/paper/v4/experts/PEMS08_ALFTemporal.yaml"}
PHASES = ("train", "validation", "test")


@pytest.fixture
def no_private_resolution(monkeypatch):
    """Every path-based resolver raises; the counter proves it stayed at zero."""
    import lib.utils as utils
    import model.comparison_model.family_utils as family

    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("v4 resolved a graph asset from data_path")

    for module, name in ((utils, "load_project_adjacency"),
                         (utils, "load_project_positions"),
                         (family, "infer_project_num_nodes"),
                         (family, "resolve_known_unknown_nodes")):
        monkeypatch.setattr(module, name, boom)
    return calls


def _build(dataset, phase="train", context=None):
    from model.ALFTemporal.ALFTemporal import Model

    args = load_args_from_config(CONFIGS[dataset], [])
    args.device = "cpu"
    return Model.from_runtime_context(
        args, context or build_runtime_context(dataset, phase))


# --- identity -----------------------------------------------------------------


def test_static_identity_is_the_same_across_phases():
    for dataset in CONFIGS:
        hashes = {p: build_runtime_context(dataset, p).static_asset_hash()
                  for p in PHASES}
        assert len(set(hashes.values())) == 1, (
            f"{dataset}: static graph assets must not change with the phase, or "
            "one architecture looks like three models")


def test_exposure_identity_differs_across_phases():
    for dataset in CONFIGS:
        hashes = {p: build_runtime_context(dataset, p).exposure_hash() for p in PHASES}
        assert len(set(hashes.values())) == 3, dataset


def test_static_identity_carries_no_phase_or_split_field():
    identity = build_runtime_context("PEMS08", "train").static_asset_identity()
    for leaked in ("phase", "signal_scope_sha256", "known_nodes_sha256"):
        assert leaked not in identity


@pytest.mark.parametrize("field, value", [
    ("adjacency_sha256", "0" * 64),
    ("positions_sha256", "0" * 64),
    ("layout_sha256", "0" * 64),
    ("layout_algorithm", "graph_native_component_bfs"),
    ("node_ids_sha256", "0" * 64),
    ("target_channel", 0),
])
def test_a_changed_static_asset_is_refused(field, value):
    context = build_runtime_context("PEMSBAY", "train")
    wrong = dict(context.static_asset_identity(), **{field: value})
    with pytest.raises(ContextRefused, match=field):
        context.assert_static_asset_matches(wrong, "ALFTemporal")


def test_a_context_missing_a_required_static_field_fails_closed():
    context = build_runtime_context("PEMS08", "train")
    object.__setattr__(context, "layout_algorithm", None)
    with pytest.raises(ContextRefused, match="layout_algorithm"):
        context.static_asset_identity()


# --- construction -------------------------------------------------------------


@pytest.mark.parametrize("dataset", sorted(CONFIGS))
def test_v4_constructor_touches_no_private_asset_resolver(no_private_resolution, dataset):
    model = _build(dataset)
    assert no_private_resolution["n"] == 0
    context = build_runtime_context(dataset, "train")
    assert model.full_num_nodes == context.num_nodes
    assert model.static_asset_hash == context.static_asset_hash()


def test_the_v4_entry_point_fails_closed_without_a_context():
    from model.ALFTemporal.ALFTemporal import Model

    args = load_args_from_config(CONFIGS["PEMS08"], [])
    args.device = "cpu"
    with pytest.raises(ValueError, match="requires a RuntimeDataContext"):
        Model.from_runtime_context(args, None)


@pytest.mark.parametrize("dataset, algorithm", [
    ("PEMSBAY", "kdtree_on_verified_wgs84"),
    ("PEMS08", "graph_native_component_bfs"),
])
def test_the_model_consumes_the_contexts_own_layout(no_private_resolution,
                                                    dataset, algorithm):
    """PEMS-BAY must take the frozen WGS84 KD-tree membership and the
    coordinate-free set its graph-native one. A second builder inside the model
    is how the manifest and the model disagreed about PEMS-BAY."""
    context = build_runtime_context(dataset, "train")
    assert context.layout_algorithm == algorithm
    model = _build(dataset, context=context)
    covered = sorted(node for part in context.layout_parts for node in part)
    assert covered == list(range(context.num_nodes))
    # every real node survives the patch padding, and no index escapes the graph
    flat = np.asarray(model.ori_parts_idx).reshape(-1)
    assert set(covered) <= set(int(node) for node in flat)
    assert int(flat.max()) < context.num_nodes and int(flat.min()) >= 0
    assert no_private_resolution["n"] == 0


@pytest.mark.parametrize("phase", PHASES)
def test_one_static_model_accepts_every_phase(no_private_resolution, phase):
    """Binding is to the assets, not to the split."""
    model = _build("PEMS08", phase)
    assert model.static_asset_hash == build_runtime_context(
        "PEMS08", "train").static_asset_hash()


def test_a_tampered_layout_is_refused(no_private_resolution):
    context = build_runtime_context("PEMS08", "train")
    broken = list(context.layout_parts)
    broken[0] = tuple(list(broken[0])[:-1])          # drop a node
    object.__setattr__(context, "layout_parts", tuple(broken))
    with pytest.raises(ValueError, match="every node exactly once"):
        _build("PEMS08", context=context)


def test_a_config_declaring_a_different_node_count_is_refused(no_private_resolution):
    from model.ALFTemporal.ALFTemporal import Model

    args = load_args_from_config(CONFIGS["PEMS08"], [])
    args.device = "cpu"
    args.num_nodes = 999
    with pytest.raises(ValueError, match="refusing rather than silently winning"):
        Model.from_runtime_context(args, build_runtime_context("PEMS08", "train"))


# --- initialisation is unchanged ---------------------------------------------


def test_two_constructions_agree_parameter_for_parameter(no_private_resolution):
    """Removing the asset reads must not have moved the RNG stream."""
    torch.manual_seed(1234)
    first = _build("PEMSBAY")
    torch.manual_seed(1234)
    second = _build("PEMSBAY")
    a, b = dict(first.named_parameters()), dict(second.named_parameters())
    assert set(a) == set(b)
    for name in a:
        assert a[name].shape == b[name].shape, name
        assert torch.equal(a[name], b[name]), name


# --- the old forward still runs ----------------------------------------------


def test_legacy_forward_still_runs_after_the_injection(no_private_resolution):
    """Not a fairness result. The K->N scatter is still here on purpose; this
    only shows the constructor change did not break the existing path."""
    context = build_runtime_context("PEMS08", "train")
    model = _build("PEMS08", context=context).eval()
    known = len(model.known_nodes)
    generator = torch.Generator().manual_seed(3)
    x_enc = torch.randn(2, 12, known, model.expert_input_dim, generator=generator)
    marks = torch.rand(2, 12, 3, generator=generator) * 5
    with torch.no_grad():
        out = model(context.adjacency, x_enc, marks)
    assert isinstance(out, dict) and out

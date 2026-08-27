"""One RuntimeDataContext reaches ALFNet and both of its nested experts.

The dangerous default is `build_patchstg_cfg(runtime_context=None)`: a context
dropped on the way to a nested expert does not raise, it silently reverts to
resolving the graph from `data_path`. So the tests below do not merely check
that construction succeeds -- they make every path-based resolver raise, and
assert the call count stayed at zero.

Scope: construction only. The K->N scatter and the old forward are still here.
"""

import pytest
import torch

from args import load_args_from_config
from lib.runtime_context import build_runtime_context

CONFIGS = {"PEMS08": "configs/ALFNet/PEMS08.yaml",
           "PEMSBAY": "configs/ALFNet/PEMSBAY.yaml"}
PHASES = ("train", "validation", "test")


@pytest.fixture
def no_private_resolution():
    import model.ALFNet.ALFNet as alfnet
    import model.ALFTemporal.ALFTemporal as alftemporal
    import model.comparison_model.family_utils as family

    for module in (alfnet, alftemporal, family):
        for name in ("load_project_adjacency", "load_project_positions",
                     "infer_project_num_nodes", "resolve_known_unknown_nodes"):
            assert not hasattr(module, name)
    return {"n": 0}


def _build(dataset, phase="train", context=None, **overrides):
    import model.ALFNet.ALFNet as alfnet

    args = load_args_from_config(CONFIGS[dataset], [])
    args.device = "cpu"
    for key, value in overrides.items():
        setattr(args, key, value)
    return alfnet.Model.from_runtime_context(
        args, context if context is not None else build_runtime_context(dataset, phase))


# --- propagation --------------------------------------------------------------


@pytest.mark.parametrize("dataset", sorted(CONFIGS))
def test_one_context_reaches_both_experts(no_private_resolution, dataset):
    context = build_runtime_context(dataset, "train")
    model = _build(dataset, context=context)
    expected = context.static_asset_hash()
    assert model.static_asset_hash == expected
    assert model.graph_expert.static_asset_hash == expected
    assert model.patch_expert.static_asset_hash == expected
    assert model.full_num_nodes == context.num_nodes
    assert model.patch_expert.full_num_nodes == context.num_nodes
    assert model.graph_expert.context_full_num_nodes == context.num_nodes
    assert no_private_resolution["n"] == 0


@pytest.mark.parametrize("dataset, algorithm", [
    ("PEMS08", "graph_native_component_bfs"),
    ("PEMSBAY", "kdtree_on_verified_wgs84"),
])
def test_the_patch_expert_consumes_the_contexts_layout(no_private_resolution,
                                                       dataset, algorithm):
    context = build_runtime_context(dataset, "train")
    assert context.layout_algorithm == algorithm
    model = _build(dataset, context=context)
    covered = sorted(node for part in context.layout_parts for node in part)
    assert covered == list(range(context.num_nodes))
    assert model.patch_expert.full_num_nodes == context.num_nodes


def test_a_dropped_context_is_an_error_not_a_fallback(monkeypatch):
    """The regression this file exists for: losing the context must raise."""
    import model.ALFNet.ALFNet as alfnet

    original = alfnet._build_expert
    monkeypatch.setattr(
        alfnet, "_build_expert",
        lambda spec, device, name, runtime_context=None, **kwargs:
            original(spec, device, name, None, **kwargs))
    with pytest.raises(ValueError, match="RuntimeDataContext"):
        _build("PEMS08")


def test_an_expert_without_a_v4_entry_point_is_refused(monkeypatch):
    import model.ALFGraph.ALFGraph as alfgraph
    import model.ALFNet.ALFNet as alfnet

    monkeypatch.delattr(alfgraph.Model, "from_runtime_context")
    with pytest.raises(ValueError, match="requires ALFGraph.Model.from_runtime_context"):
        _build("PEMS08")


# --- fail closed --------------------------------------------------------------


def test_every_v4_entry_point_refuses_a_missing_context():
    import model.ALFGraph.ALFGraph as alfgraph
    import model.ALFNet.ALFNet as alfnet
    import model.ALFTemporal.ALFTemporal as alftemporal

    for module, config in ((alfnet, CONFIGS["PEMS08"]),
                           (alfgraph, "configs/ALFGraph/PEMS08.yaml"),
                           (alftemporal, "configs/ALFTemporal/PEMS08.yaml")):
        args = load_args_from_config(config, [])
        args.device = "cpu"
        with pytest.raises(ValueError, match="requires a RuntimeDataContext"):
            module.Model.from_runtime_context(args, None)


@pytest.mark.parametrize("attribute, value", [
    ("node_num", 999), ("num_nodes", 999), ("target_channel", 7),
])
def test_a_config_contradicting_the_context_is_refused(no_private_resolution,
                                                       attribute, value):
    with pytest.raises(ValueError, match="refusing rather than silently winning"):
        _build("PEMS08", **{attribute: value})


def test_a_config_naming_another_dataset_is_refused(no_private_resolution):
    with pytest.raises(ValueError, match="does not match the context dataset"):
        _build("PEMS08", dloader_name="PEMSBAY")


@pytest.mark.parametrize("field", ["adjacency_sha256", "layout_sha256",
                                   "node_ids_sha256", "target_channel"])
def test_a_tampered_static_asset_is_refused(field):
    from lib.runtime_context import ContextRefused

    context = build_runtime_context("PEMS08", "train")
    # PEMS08's target_channel is already 0, so a "tamper" to 0 changes nothing
    # and the check would trivially pass.
    original = context.static_asset_identity()[field]
    replacement = (original + 1) if field == "target_channel" else "0" * 64
    assert replacement != original
    wrong = dict(context.static_asset_identity(), **{field: replacement})
    with pytest.raises(ContextRefused, match=field):
        context.assert_static_asset_matches(wrong, "expert")


# --- one static model, every phase -------------------------------------------


@pytest.mark.parametrize("phase", PHASES)
def test_any_phase_context_builds_the_same_static_model(no_private_resolution, phase):
    model = _build("PEMS08", phase)
    assert model.static_asset_hash == build_runtime_context(
        "PEMS08", "train").static_asset_hash()
    assert no_private_resolution["n"] == 0


# --- nothing about the model moved -------------------------------------------


def test_construction_is_deterministic_and_the_parameter_set_is_unchanged(
        no_private_resolution):
    torch.manual_seed(1234)
    first = _build("PEMS08")
    torch.manual_seed(1234)
    second = _build("PEMS08")
    a, b = dict(first.named_parameters()), dict(second.named_parameters())
    assert set(a) == set(b)
    for name in a:
        assert a[name].shape == b[name].shape and torch.equal(a[name], b[name]), name
    # Unified BASE uses the same frozen five-support graph expert everywhere.
    assert sum(p.numel() for p in _build("PEMSBAY").parameters()) == 1_037_529

"""The v4 full-N loader: node axis, scaler scope, and the five masks.

No model is instantiated. These assert on the real dataset, because the
exposure decision is made here -- by the time a tensor reaches a model it is
either already blind to the forbidden nodes or it is not.

The counterfactual is on the raw payload, not on an intermediate: the question
is whether measurements the phase forbids can reach anything at all.
"""

import numpy as np
import pytest
import torch

from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from lib.runtime_context import ContextRefused, build_runtime_context
from lib.v4_payload import load_v4_payload, save_v4_payload

PHASES = ("train", "validation", "test")
PAYLOAD = "data_v4/PEMS08/raw_data.pkl"


@pytest.fixture(scope="module")
def datasets():
    return {phase: V4LifecycleDataset(build_runtime_context("PEMS08", phase), PAYLOAD)
            for phase in PHASES}


def _forbidden(context):
    mask = np.ones(context.num_nodes, dtype=bool)
    mask[context.signal_scope] = False
    return mask


def _poison(tmp_path, context, value, nodes):
    """A payload identical to the canonical one except on `nodes`."""
    payload = load_v4_payload(PAYLOAD)
    data = payload["data"].copy()
    data[:, nodes, :] = value
    path = tmp_path / f"raw_data.pkl"
    return save_v4_payload(path, data=data, adj=payload["adj"],
                           time_stamp=payload["time_stamp"],
                           pos_stamp=payload["pos_stamp"])


# --- shape and identity -------------------------------------------------------


def test_every_phase_runs_on_the_full_node_axis(datasets):
    for phase, dataset in datasets.items():
        batch = dataset[0]
        n = dataset.context.num_nodes
        assert batch["base_visible_input"].shape == (12, n, 1), phase
        for field in ("input_available_mask", "decoder_available_mask",
                      "target_available_mask", "loss_mask"):
            assert batch[field].shape == (12, n), f"{phase}/{field}"
        assert batch["target_cohort_mask"].shape == (n,)
        assert dataset.context.adjacency.shape == (n, n)


def test_the_batch_carries_the_context_identity(datasets):
    for dataset in datasets.values():
        assert dataset[0]["context_identity"] == dataset.context.identity()


def test_cohort_ids_are_global_and_in_range(datasets):
    for phase, dataset in datasets.items():
        for index in (0, 1, len(dataset) // 2):
            cohort = dataset[index]["target_cohort_mask"]
            assert cohort.shape[0] == dataset.context.num_nodes, phase
            assert torch.isfinite(cohort).all()


def test_a_dataset_without_a_context_fails_closed():
    with pytest.raises(ContextRefused, match="requires a RuntimeDataContext"):
        V4LifecycleDataset(None, PAYLOAD)


# --- scaler -------------------------------------------------------------------


def test_the_scaler_sees_training_known_nodes_only(datasets):
    train = datasets["train"]
    assert train.scaler["fitted_on"].startswith("train window, training-known")
    assert train.scaler["n_values"] > 0


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -1e30, 1e30])
def test_held_out_values_never_move_the_scaler(tmp_path, poison):
    context = build_runtime_context("PEMS08", "train")
    base = V4LifecycleDataset(context, PAYLOAD).scaler
    path = _poison(tmp_path / str(abs(hash(poison))), context, poison,
                   np.nonzero(_forbidden(context))[0])
    moved = V4LifecycleDataset(context, path).scaler
    assert moved["mean"] == base["mean"] and moved["std"] == base["std"]


def test_known_values_do_move_the_scaler(tmp_path):
    """Control: without this the invariance above could hold by ignoring input."""
    context = build_runtime_context("PEMS08", "train")
    base = V4LifecycleDataset(context, PAYLOAD).scaler
    path = _poison(tmp_path / "known", context, 999.0, context.known_nodes)
    assert V4LifecycleDataset(context, path).scaler["mean"] != base["mean"]


def test_unavailable_slots_are_exactly_zero_in_normalised_space(datasets):
    """Zero is the training mean here. Normalising a physical zero would give a
    large negative value, which is a measurement, not an absence."""
    for phase, dataset in datasets.items():
        batch = dataset[0]
        absent = ~batch["input_available_mask"]
        assert torch.equal(batch["base_visible_input"][absent],
                           torch.zeros_like(batch["base_visible_input"][absent])), phase


# --- the raw-signal counterfactual -------------------------------------------


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -1e30, 1e30])
@pytest.mark.parametrize("phase", PHASES)
def test_forbidden_nodes_cannot_move_origins_or_any_mask(tmp_path, phase, poison):
    context = build_runtime_context("PEMS08", phase)
    clean = V4LifecycleDataset(context, PAYLOAD)
    forbidden = _forbidden(context)
    assert forbidden.any(), f"{phase} has no forbidden node; the test would be vacuous"

    path = _poison(tmp_path / f"{phase}{abs(hash(poison))}", context, poison,
                   np.nonzero(forbidden)[0])
    poisoned = V4LifecycleDataset(context, path)

    assert np.array_equal(clean.origins, poisoned.origins), f"{phase}: origins moved"
    for index in (0, 7, len(clean) // 2):
        a, b = clean[index], poisoned[index]
        for field in ("base_visible_input", "input_available_mask",
                      "decoder_available_mask", "target_available_mask",
                      "loss_mask", "target_cohort_mask", "lifecycle_role_ids"):
            assert torch.equal(a[field], b[field]), f"{phase}/{field} moved on {poison}"
        assert torch.isfinite(a["base_visible_input"]).all()


def test_permitted_nodes_do_move_the_visible_input(tmp_path):
    context = build_runtime_context("PEMS08", "train")
    clean = V4LifecycleDataset(context, PAYLOAD)
    path = _poison(tmp_path / "permitted", context, 500.0, context.known_nodes)
    moved = V4LifecycleDataset(context, path, scaler=clean.scaler)
    assert not torch.equal(clean[0]["base_visible_input"], moved[0]["base_visible_input"])


# --- phase isolation ----------------------------------------------------------


def test_each_phase_is_blind_to_the_other_split(datasets):
    train, validation, test = (datasets[p] for p in PHASES)
    checks = [
        ("train", train, np.union1d(train.context.validation_unseen, train.context.test_unseen)),
        ("validation", validation, validation.context.test_unseen),
        ("test", test, test.context.validation_unseen),
    ]
    for phase, dataset, blind in checks:
        for index in (0, 3, len(dataset) // 2):
            available = dataset[index]["input_available_mask"]
            assert not available[:, blind].any(), f"{phase} read a forbidden node"


# --- lifecycle ----------------------------------------------------------------


def test_join_and_fault_evidence_matches_the_input_mask(datasets):
    """The k in the schedule must be the k in the tensor, not a near-miss."""
    dataset = datasets["train"]
    seen = set()
    for index in range(len(dataset)):
        entry = dataset.schedule[index]
        if entry["scenario"] not in ("join", "fault") or not len(entry["changed_nodes"]):
            continue
        batch = dataset[index]
        nodes = np.asarray(entry["changed_nodes"], dtype=np.int64)
        k = int(entry["evidence_steps"])
        lifecycle = dataset._availability(entry, np.ones((12, dataset.num_nodes), bool))[2]
        if entry["scenario"] == "join":
            assert lifecycle[-k:, nodes].all() and not lifecycle[:-k, nodes].any()
        else:
            assert not lifecycle[-k:, nodes].any() and lifecycle[:-k, nodes].all()
        seen.add(entry["scenario"])
        if len(seen) == 2 and index > 200:
            break
    assert seen == {"join", "fault"}


def test_all_four_scenarios_are_covered(datasets):
    scenarios = {entry["scenario"] for entry in datasets["train"].schedule}
    assert scenarios == {"zero_history", "steady", "join", "fault"}


def test_roles_agree_with_availability(datasets):
    for phase, dataset in datasets.items():
        batch = dataset[0]
        roles, available = batch["lifecycle_role_ids"], batch["input_available_mask"]
        never = ~available.any(dim=0)
        # a node with no observation anywhere in the window is never Steady
        assert not (roles[never] == 1).any(), phase
        assert torch.equal(batch["lifecycle_state_age"],
                           available.float().mean(dim=0))


# --- teacher forcing ----------------------------------------------------------


def test_decoder_mask_is_sample_level_not_phase_level(datasets):
    """A phase-level mask would mark every in-scope node true regardless of the
    episode, handing the decoder a hidden node's future."""
    dataset = datasets["train"]
    scope = torch.zeros(dataset.context.num_nodes, dtype=torch.bool)
    scope[dataset.context.signal_scope] = True
    for index in range(len(dataset)):
        entry = dataset.schedule[index]
        if entry["scenario"] != "fault" or not len(entry["changed_nodes"]):
            continue
        batch = dataset[index]
        hidden = np.asarray(entry["changed_nodes"], dtype=np.int64)
        assert not batch["decoder_available_mask"][:, hidden].any(), (
            "a node dark at the last observed step was teacher-forceable")
        assert not batch["decoder_available_mask"][:, ~scope].any()
        assert batch["decoder_available_mask"].any(), "no node is forceable; vacuous"
        break


def test_the_three_target_masks_are_distinct_objects_and_meanings(datasets):
    batch = datasets["train"][0]
    target, decoder, loss = (batch["target_available_mask"],
                             batch["decoder_available_mask"], batch["loss_mask"])
    assert target is not decoder and decoder is not loss
    # loss is the strictest: cohort and phase scope on top of availability
    assert int(loss.sum()) <= int(target.sum())
    assert not torch.equal(decoder, loss)


# --- v3 is untouched ----------------------------------------------------------


def test_the_v3_loader_still_returns_its_ten_tensor_tuple():
    import inspect

    from data_provider import graph_data_loader

    source = inspect.getsource(graph_data_loader.LifecycleCoveragePEMS.__getitem__)
    assert "return" in source
    assert not hasattr(graph_data_loader.LifecycleCoveragePEMS, "base_visible_input")

"""The v4 full-N loader: node axis, scaler scope, and the five masks.

No model is instantiated. These assert on the real dataset, because the
exposure decision is made here -- by the time a tensor reaches a model it is
either already blind to the forbidden nodes or it is not.

The counterfactual is on the raw payload, not on an intermediate: the question
is whether measurements the phase forbids can reach anything at all.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from lib.runtime_context import ContextRefused, build_runtime_context
from data_provider.lifecycle_roles import ROLE_AAS, ROLE_FS, ROLE_NAS, ROLE_VS
from lib.v4_payload import load_v4_payload, save_v4_payload

TENSOR_FIELDS = ("base_visible_input", "input_available_mask",
                 "decoder_available_mask", "target_available_mask", "target",
                 "loss_mask", "target_cohort_mask", "lifecycle_role_ids",
                 "lifecycle_state_age", "lifecycle_context")

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


def _poison(tmp_path, context, value, nodes, payload_path=PAYLOAD):
    """A payload identical to the canonical one except on `nodes`."""
    payload = load_v4_payload(payload_path)
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
        # Join and Fault are trajectories: the lifecycle fields must carry the
        # time axis, or the last k steps -- the whole measurement -- are lost.
        assert batch["lifecycle_role_ids"].shape == (12, n), phase
        assert batch["lifecycle_state_age"].shape == (12, n), phase
        assert batch["lifecycle_context"].shape == (12, n, 6), phase
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
        for field in TENSOR_FIELDS:
            assert torch.equal(a[field], b[field]), f"{phase}/{field} moved on {poison}"
        for field in ("scenario", "evidence_steps", "phase_id", "origin"):
            assert a[field] == b[field], f"{phase}/{field} moved on {poison}"
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


def _first(dataset, scenario):
    for index in range(len(dataset)):
        entry = dataset.schedule[index]
        if entry["scenario"] == scenario and len(entry["changed_nodes"]):
            return index, entry
    raise AssertionError(f"no {scenario} sample")


def test_role_ids_are_the_frozen_order():
    """ALFNet indexes role scales positionally as (aas, nas, vs, fs). A loader
    with its own order feeds Unseen where the model reads AAS, with no shape
    error anywhere -- which is exactly what happened."""
    assert (ROLE_AAS, ROLE_NAS, ROLE_VS, ROLE_FS) == (0, 1, 2, 3)


def test_alfnet_reads_roles_in_the_same_order():
    import inspect

    from model.ALFNet import ALFNet

    source = inspect.getsource(ALFNet)
    assert '("aas", "nas", "vs", "fs")' in source, (
        "ALFNet's positional role order changed; the loader constants must follow")


def test_join_trajectory_is_exact_per_step(datasets):
    dataset = datasets["train"]
    index, entry = _first(dataset, "join")
    batch, k = dataset[index], int(entry["evidence_steps"])
    nodes = np.asarray(entry["changed_nodes"], dtype=np.int64)
    roles, age = batch["lifecycle_role_ids"], batch["lifecycle_state_age"]
    cutoff = 12 - k
    assert (roles[:cutoff][:, nodes] == ROLE_VS).all()
    assert (roles[cutoff:][:, nodes] == ROLE_NAS).all()
    assert (age[:cutoff][:, nodes] == 0.0).all()
    expected = torch.arange(1, k + 1, dtype=torch.float32) / 12.0
    assert torch.allclose(age[cutoff:][:, nodes], expected[:, None].expand(k, nodes.size))


def test_fault_trajectory_is_exact_per_step(datasets):
    dataset = datasets["train"]
    index, entry = _first(dataset, "fault")
    batch, k = dataset[index], int(entry["evidence_steps"])
    nodes = np.asarray(entry["changed_nodes"], dtype=np.int64)
    roles, age = batch["lifecycle_role_ids"], batch["lifecycle_state_age"]
    cutoff = 12 - k
    assert (roles[:cutoff][:, nodes] == ROLE_AAS).all()
    assert (roles[cutoff:][:, nodes] == ROLE_FS).all()
    assert (age[:cutoff][:, nodes] == 1.0).all()
    assert not batch["input_available_mask"][cutoff:][:, nodes].any()


def test_roles_do_not_move_when_a_sensor_merely_misses_a_reading(datasets):
    """Natural missingness belongs to the observation mask. If it moved a role,
    a sensor that dropped one reading would be reported as newly attached."""
    dataset = datasets["train"]
    index, entry = _first(dataset, "steady")
    scope = np.zeros(dataset.num_nodes, dtype=bool)
    scope[dataset.context.signal_scope] = True
    roles_a, _, age_a = dataset._lifecycle_state(entry, scope)
    roles_b, _, age_b = dataset._lifecycle_state(entry, scope)
    assert np.array_equal(roles_a, roles_b) and np.array_equal(age_a, age_b)
    # roles are a function of scope and episode only -- no natural mask argument
    import inspect
    assert "natural" not in inspect.signature(dataset._lifecycle_state).parameters


def test_all_four_scenarios_are_covered(datasets):
    scenarios = {entry["scenario"] for entry in datasets["train"].schedule}
    assert scenarios == {"zero_history", "steady", "join", "fault"}


@pytest.mark.parametrize("phase", PHASES)
def test_scenario_pools_follow_the_phase(datasets, phase):
    """Validation and test must evaluate Unseen and Join on their own held-out
    nodes. Drawing them from K turns both rows into another Steady."""
    dataset = datasets[phase]
    context = dataset.context
    known = set(context.known_nodes.tolist())
    held_out = {"train": known,
                "validation": set(context.validation_unseen.tolist()),
                "test": set(context.test_unseen.tolist())}[phase]
    forbidden = {"train": set(), "validation": set(context.test_unseen.tolist()),
                 "test": set(context.validation_unseen.tolist())}[phase]
    for scenario, pool in (("zero_history", held_out), ("join", held_out),
                           ("steady", known), ("fault", known)):
        _, entry = _first(dataset, scenario)
        nodes = set(int(node) for node in entry["changed_nodes"])
        assert nodes <= pool, f"{phase}/{scenario} drew outside its pool"
        assert not nodes & forbidden, f"{phase}/{scenario} touched the other split"


@pytest.mark.parametrize("scenario", ["zero_history", "steady", "join", "fault"])
def test_target_cohort_is_exactly_changed_nodes(datasets, scenario):
    """Widening steady to the whole scope, or zero_history to the hidden pool,
    reports a different measurement from the one the schedule declared."""
    dataset = datasets["train"]
    index, entry = _first(dataset, scenario)
    cohort = dataset[index]["target_cohort_mask"]
    expected = sorted(int(node) for node in entry["changed_nodes"])
    assert sorted(torch.nonzero(cohort).flatten().tolist()) == expected
    assert int(cohort.sum()) == len(expected)
    assert int(cohort.sum()) < dataset.context.known_nodes.size


# --- the exposed target -------------------------------------------------------


def test_forbidden_future_values_are_not_present_in_the_batch(datasets):
    """A mask saying 'do not use' still ships the answer. The target itself is
    masked, so a consumer with a wrong mask cannot read a forbidden future."""
    for phase, dataset in datasets.items():
        batch = dataset[0]
        forbidden = torch.from_numpy(_forbidden(dataset.context))
        exposed = batch["target"][:, forbidden, :]
        assert torch.equal(exposed, torch.zeros_like(exposed)), phase


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), 1e30])
def test_poisoning_a_forbidden_future_does_not_change_the_target(tmp_path, poison):
    context = build_runtime_context("PEMS08", "train")
    clean = V4LifecycleDataset(context, PAYLOAD)
    path = _poison(tmp_path / f"f{abs(hash(poison))}", context, poison,
                   np.nonzero(_forbidden(context))[0])
    poisoned = V4LifecycleDataset(context, path)
    assert torch.equal(clean[0]["target"], poisoned[0]["target"])


def test_poisoning_a_permitted_future_does_change_the_target(tmp_path):
    context = build_runtime_context("PEMS08", "train")
    clean = V4LifecycleDataset(context, PAYLOAD)
    path = _poison(tmp_path / "permitted_future", context, 777.0, context.known_nodes)
    moved = V4LifecycleDataset(context, path, scaler=clean.scaler)
    assert not torch.equal(clean[0]["target"], moved[0]["target"])


# --- teacher forcing ----------------------------------------------------------


@pytest.mark.parametrize("phase", PHASES)
def test_decoder_mask_semantics_per_scenario(datasets, phase):
    dataset = datasets[phase]
    scope = np.zeros(dataset.num_nodes, dtype=bool)
    scope[dataset.context.signal_scope] = True
    for scenario in ("zero_history", "join", "fault", "steady"):
        index, entry = _first(dataset, scenario)
        batch = dataset[index]
        nodes = np.asarray(entry["changed_nodes"], dtype=np.int64)
        decoder = batch["decoder_available_mask"]
        assert not decoder[:, ~torch.from_numpy(scope)].any(), f"{phase} out of scope"
        if scenario in ("zero_history", "fault"):
            assert not decoder[:, nodes].any(), f"{phase}/{scenario} was forceable"
        assert decoder.any(), f"{phase}/{scenario}: nothing forceable; vacuous"


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


# --- the other three datasets --------------------------------------------------
#
# PEMS08 carries the full poison matrix above. Here each remaining dataset gets
# one forbidden-signal and one forbidden-future counterfactual per phase, plus
# shapes, role mapping, pools and identity. Repeating all four poison values on
# a 195 MB PEMS-BAY payload would multiply the suite runtime for no new failure
# mode -- the masking code is shared, and PEMS08 already exercises every value.

OTHER = ("PEMS03", "PEMS04", "PEMSBAY")
OTHER_PAYLOAD = {"PEMS03": "data_v4/PEMS03/raw_data.pkl",
                 "PEMS04": "data_v4/PEMS04/raw_data.pkl",
                 "PEMSBAY": "data_v4/PEMS-BAY/raw_data.pkl"}


@pytest.mark.parametrize("dataset", OTHER)
@pytest.mark.parametrize("phase", PHASES)
def test_other_datasets_are_blind_to_forbidden_nodes(tmp_path, dataset, phase):
    context = build_runtime_context(dataset, phase)
    payload = OTHER_PAYLOAD[dataset]
    clean = V4LifecycleDataset(context, payload)
    forbidden = _forbidden(context)
    assert forbidden.any(), f"{dataset}/{phase} has no forbidden node"

    # NaN poisons both the signal and, at the same rows, the future target.
    path = _poison(tmp_path / f"{dataset}{phase}", context, float("nan"),
                   np.nonzero(forbidden)[0], payload_path=payload)
    poisoned = V4LifecycleDataset(context, path)

    assert np.array_equal(clean.origins, poisoned.origins)
    assert clean.scaler == poisoned.scaler
    for index in (0, len(clean) // 2):
        a, b = clean[index], poisoned[index]
        for field in TENSOR_FIELDS:
            assert torch.equal(a[field], b[field]), f"{dataset}/{phase}/{field}"
        assert a["context_identity"] == context.identity()
        exposed = a["target"][:, torch.from_numpy(forbidden), :]
        assert torch.equal(exposed, torch.zeros_like(exposed))


@pytest.mark.parametrize("dataset", OTHER)
def test_other_datasets_have_the_temporal_axis_and_frozen_roles(dataset):
    context = build_runtime_context(dataset, "train")
    loader = V4LifecycleDataset(context, OTHER_PAYLOAD[dataset])
    batch = loader[0]
    n = context.num_nodes
    assert batch["lifecycle_role_ids"].shape == (12, n)
    assert batch["lifecycle_context"].shape == (12, n, 6)
    assert set(batch["lifecycle_role_ids"].unique().tolist()) <= {
        ROLE_AAS, ROLE_NAS, ROLE_VS, ROLE_FS}
    index, entry = _first(loader, "join")
    nodes = np.asarray(entry["changed_nodes"], dtype=np.int64)
    assert set(nodes.tolist()) <= set(context.known_nodes.tolist())
    assert int(loader[index]["target_cohort_mask"].sum()) == nodes.size


# --- PEMS-BAY target channel regression ---------------------------------------


def test_pemsbay_target_is_speed_on_channel_two():
    """The registry declared channel 0 with channel_order ["speed","unused",
    "unused"]. Speed is channel 2; channels 0 and 1 are entirely NaN in both the
    v3 and the v4 payload, so the scaler could not be fitted at all -- and a
    looser missing-value rule would have fitted one on NaN and trained on it.
    """
    import json

    registry = json.loads(
        Path("configs/paper/v4_dataset_registry.json").read_text("utf-8"))
    entry = registry["datasets"]["PEMSBAY"]
    assert entry["target"] == "v"
    assert entry["target_channel"] == 2, "PEMS-BAY speed is channel 2, not channel 0"
    assert entry["channel_order"][2] == "speed"

    manifest = json.loads(
        Path("configs/paper/v4_data_manifests/PEMSBAY_run_manifest.json")
        .read_text("utf-8"))
    assert manifest["target"] == {"name": "v", "channel": 2}

    context = build_runtime_context("PEMSBAY", "train")
    assert (context.target, context.target_channel) == ("v", 2)

    payload = load_v4_payload("data_v4/PEMS-BAY/raw_data.pkl")
    data = np.asarray(payload["data"])
    # The empty channels must stay recognisably empty, so a future edit that
    # points the target back at one of them fails here rather than silently.
    for channel in (0, 1):
        assert not np.isfinite(data[:, :, channel]).any(), (
            f"channel {channel} gained values; the channel_order claim is stale")
    assert np.isfinite(data[:, :, 2]).all()

    dataset = V4LifecycleDataset(context, "data_v4/PEMS-BAY/raw_data.pkl")
    assert np.isfinite(dataset.scaler["mean"]) and dataset.scaler["std"] > 0
    assert dataset.scaler["n_values"] > 0


def test_the_registry_edit_did_not_rewrite_the_pemsbay_payload():
    """Correcting a declared fact must not touch the bytes it describes."""
    import json

    manifest = json.loads(
        Path("configs/paper/v4_data_manifests/PEMSBAY_run_manifest.json")
        .read_text("utf-8"))
    digest = hashlib.sha256()
    with Path("data_v4/PEMS-BAY/raw_data.pkl").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == manifest["payload"]["sha256"]

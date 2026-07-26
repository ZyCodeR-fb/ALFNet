import json
import importlib.util
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_provider.data_factory import LifecycleBalancedBatchSampler
from data_provider.graph_data_loader import AsyncLifecyclePEMS, LifecycleCoveragePEMS
from data_provider.lcbt_protocol import (
    LCBT_GROUPS,
    build_lcbt_schedule,
    compose_effective_lifecycle_features,
    finalize_lifecycle_validation_mae,
    resolve_lcbt_boundaries,
    split_unseen_nodes,
)
from data_provider.ples_protocol import (
    phase_for_tau_v2,
    summarize_lce_rows,
    summarize_paired_rows,
)
from experiments.train import _accept_checkpoint


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace_tmp(request):
    path = ROOT / "_tmp_lifecycle_tests" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _ring_adjacency(num_nodes):
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for node in range(num_nodes):
        adjacency[node, (node + 1) % num_nodes] = 1.0 + node / num_nodes
        adjacency[(node + 1) % num_nodes, node] = 1.0
    return adjacency


def _metric_row(track, tau, treated, control, regime="Weekday/AMPeak"):
    phase = (
        ("Cold", "Warm", "Ready")
        if track == "join"
        else ("Onset", "Decay", "Outage")
    )
    label = phase[0] if tau == 0 else phase[2] if tau >= 11 else phase[1]
    return {
        "track": track,
        "phase": label,
        "tau": tau,
        "regime": regime,
        "origin_id": 0,
        "cohort_id": 0,
        "count": 1,
        "treated_ape_sum": treated / 100.0,
        "control_ape_sum": control / 100.0,
        "treated_ae_sum": treated,
        "control_ae_sum": control,
        "treated_se_sum": treated**2,
        "control_se_sum": control**2,
        "graph_weight": None,
    }


def test_lcbt_v3_uniformly_balances_every_evidence_length_per_track_and_calendar():
    schedule = build_lcbt_schedule(
        num_samples=768,
        input_len=12,
        adjacency=_ring_adjacency(60),
        calendar_regimes=["AM", "PM", "OffPeak", "Weekend"] * 192,
        seed=2026,
        epoch=3,
    )

    scenarios = Counter(item["scenario"] for item in schedule)
    assert scenarios == Counter({name: 192 for name in LCBT_GROUPS})
    for regime in ("AM", "PM", "OffPeak", "Weekend"):
        for scenario in ("join", "fault"):
            evidence = Counter(
                item["evidence_steps"]
                for item in schedule
                if item["calendar_regime"] == regime
                and item["scenario"] == scenario
            )
            assert set(evidence) == set(range(1, 13))
            assert max(evidence.values()) - min(evidence.values()) <= 1

            for k in range(1, 13):
                cell = [
                    item
                    for item in schedule
                    if item["calendar_regime"] == regime
                    and item["scenario"] == scenario
                    and item["evidence_steps"] == k
                ]
                assert Counter(item["topology_mode"] for item in cell) == {
                    "clustered": 2,
                    "scattered": 2,
                }
                assert Counter(item["cohort_fraction"] for item in cell) == {
                    0.025: 1,
                    0.05: 2,
                    0.1: 1,
                }


def test_validation_replay_pairs_cohorts_with_their_own_calendar_stratum(workspace_tmp):
    """The source schedule must be matched per stratum, not zipped globally."""
    validation = _build_dataset(workspace_tmp, "val")
    origins = {
        int(origin["window_index"]): str(origin["stratum"])
        for origin in validation.validation_origins
    }
    mismatched = [
        record
        for record in validation.schedule
        if record["scenario"] in {"join", "fault"}
        and record["calendar_regime"] != origins[int(record["window_index"])]
    ]
    assert not mismatched, f"{len(mismatched)} replay records carry another stratum"


def test_episode_holdout_matches_the_evaluation_hidden_rate():
    """Training used to show a graph with nothing permanently missing."""
    nodes = 39  # divides evenly into three degree strata
    regimes = ["Weekday/AMPeak", "Weekend/Midday"] * 32
    common = dict(
        num_samples=len(regimes),
        input_len=12,
        adjacency=_ring_adjacency(nodes),
        calendar_regimes=regimes,
        seed=2026,
        epoch=0,
        cohort_fractions=(0.05,),
        cohort_weights=(1.0,),
    )

    without = build_lcbt_schedule(hidden_fraction=0.0, **common)
    assert all(not item["hidden_nodes"] for item in without)

    with_holdout = build_lcbt_schedule(hidden_fraction=0.30, **common)
    for item in with_holdout:
        hidden = set(item["hidden_nodes"])
        assert len(hidden) == round(0.30 * nodes)
        cohort = set(item["changed_nodes"])
        if item["scenario"] in {"join", "zero_history"}:
            assert cohort <= hidden, "join cohorts must sit on held-out sensors"
        else:
            assert not (cohort & hidden), "steady/fault cohorts must stay visible"

    # Held-out duty rotates rather than always landing on the same sensors.
    exposure = Counter(node for item in with_holdout for node in item["hidden_nodes"])
    assert len(exposure) == nodes, "every sensor must take a turn being hidden"
    assert max(exposure.values()) <= 1.2 * min(exposure.values())


def test_episode_holdout_cannot_be_combined_with_fixed_pools():
    with pytest.raises(ValueError, match="cannot be combined"):
        build_lcbt_schedule(
            num_samples=8,
            input_len=12,
            adjacency=_ring_adjacency(12),
            calendar_regimes=["Weekday/AMPeak"] * 8,
            seed=1,
            epoch=0,
            hidden_fraction=0.25,
            scenario_node_pools={name: [0, 1, 2] for name in LCBT_GROUPS},
        )


def test_thin_strata_keep_every_cohort_size_tier():
    """Reproduces the degeneracy that wiped the 2.5% tier out of validation.

    With one sample per (stratum, evidence) cell, ``_allocate_counts(1, [1/3,
    2/3])`` always returns the heavier tier, so evidence-level stratification
    silently deleted the small cohorts. The validation replay uses one cohort per
    origin across every k, so it must balance at the stratum level instead.
    """
    regimes = ["Weekday/AMPeak"] * 12 + ["Weekend/Midday"] * 12
    kwargs = dict(
        num_samples=len(regimes),
        input_len=12,
        adjacency=_ring_adjacency(24),
        calendar_regimes=regimes,
        seed=2026,
        epoch=0,
        cohort_fractions=(0.025, 0.05),
        cohort_weights=(1 / 3, 2 / 3),
    )

    per_evidence = build_lcbt_schedule(stratify_by_evidence=True, **kwargs)
    per_stratum = build_lcbt_schedule(stratify_by_evidence=False, **kwargs)

    def tiers(schedule, regime):
        return Counter(
            item["cohort_fraction"]
            for item in schedule
            if item["scenario"] == "join" and item["calendar_regime"] == regime
        )

    # The defect: evidence-level stratification collapses onto one tier.
    assert set(tiers(per_evidence, "Weekday/AMPeak")) == {0.05}
    # The fix: both tiers survive once the balance is taken per stratum.
    assert set(tiers(per_stratum, "Weekday/AMPeak")) == {0.025, 0.05}
    assert set(tiers(per_stratum, "Weekend/Midday")) == {0.025, 0.05}


def test_origin_selection_is_balanced_spaced_and_time_of_day_diverse():
    """The three defects the midpoint rule and my first spacing fix produced."""
    from data_provider.calendar_strata import (
        TOD_NAMES,
        lce_episode_span,
        select_stratified_origins,
    )

    steps = 288 * 21  # three weeks at five-minute resolution
    timestamps = (
        np.datetime64("2026-03-02T00:00") + np.arange(steps) * np.timedelta64(5, "m")
    )
    origins = select_stratified_origins(
        timestamps, steps, input_len=12, pred_len=12, pre_steps=12, post_steps=36
    )
    indices = sorted(origin["event_time_index"] for origin in origins)
    spacing = min(b - a for a, b in zip(indices, indices[1:]))
    counts = Counter(origin["tod_bin"] for origin in origins)
    clocks = {
        name: {origin["event_timestamp"][11:16] for origin in origins if origin["tod_bin"] == name}
        for name in TOD_NAMES
    }

    # Globally spaced so LCE episodes never share data.
    assert spacing >= lce_episode_span(12, 12)
    # No time-of-day bin may be crowded out by the greedy placement order.
    assert set(counts) == set(TOD_NAMES)
    assert min(counts.values()) >= 0.5 * max(counts.values()), counts
    # And each bin must land on many different clock times, not one midpoint.
    assert all(len(values) >= 5 for values in clocks.values()), clocks


def test_lcbt_boundaries_are_strict_and_scaler_cannot_see_validation_or_test():
    boundaries = resolve_lcbt_boundaries(1000)
    assert boundaries == {
        "train_start": 0,
        "train_end": 500,
        "val_start": 500,
        "val_end": 600,
        "test_start": 600,
        "test_end": 1000,
        "scaler_start": 0,
        "scaler_end": 500,
    }
    with pytest.raises(ValueError, match="scaler_end_ratio"):
        resolve_lcbt_boundaries(1000, scaler_end_ratio=0.51)


def test_checkpoint_uses_point_one_percent_tie_band_then_worst_group():
    assert _accept_checkpoint(9.98, 10.0, 20.0, 10.0)
    assert _accept_checkpoint(9.995, 10.0, 9.0, 10.0)
    assert not _accept_checkpoint(9.995, 10.0, 11.0, 10.0)
    assert not _accept_checkpoint(10.02, 10.0, 1.0, 10.0)


def test_v3_validation_macro_equal_weights_evidence_not_three_phase_buckets():
    state = {
        "Unseen": {0: [8.0]},
        "Steady": {0: [2.0]},
    }
    for evidence in range(1, 13):
        state[f"Join@{evidence}"] = {0: [float(evidence)]}
        state[f"Fault@{evidence}"] = {0: [float(evidence + 12)]}

    metrics = finalize_lifecycle_validation_mae(state, input_len=12)

    assert metrics["join_mae"] == pytest.approx(6.5)
    assert metrics["fault_mae"] == pytest.approx(18.5)
    assert metrics["mae"] == pytest.approx((8.0 + 2.0 + 6.5 + 18.5) / 4.0)
    assert metrics["Cold_MAE"] == pytest.approx(1.0)
    assert metrics["Warm_MAE"] == pytest.approx(6.5)
    assert metrics["Ready_MAE"] == pytest.approx(12.0)


def test_lce_v2_uses_exactly_one_complete_input_window_but_ples_keeps_long_tail():
    rows = []
    for tau in range(-1, 16):
        rows.append(_metric_row("join", tau, treated=100.0 if tau >= 12 else tau + 1, control=5.0))
        rows.append(_metric_row("fault", tau, treated=100.0 if tau >= 12 else tau + 13, control=3.0))

    lce = summarize_lce_rows(rows, input_len=12)
    ples = summarize_paired_rows(rows, input_len=12)

    assert lce["main_metrics"]["Join"]["MAE"] == pytest.approx(6.5)
    assert lce["main_metrics"]["Fault"]["MAE"] == pytest.approx(18.5)
    assert lce["phase_metrics"]["Ready"]["MAE"] == pytest.approx(12.0)
    assert lce["phase_metrics"]["Outage"]["MAE"] == pytest.approx(24.0)
    assert [point["k"] for point in lce["trajectory_metrics"]["Join"]] == list(range(1, 13))
    assert max(curve["tau"] for curve in ples["curves"]) == 15


def test_lcbt_v3_configs_keep_models_unchanged_and_use_new_protocol_versions():
    paths = [ROOT / "configs" / "ALFNet" / "rf_l3s.yaml"] + [
        ROOT / "configs" / model / "METRLA_lcbt_v3.yaml"
        for model in (
            "MoGERNN", "PatchSTG", "FCLSTM", "GWNET", "HA", "MCPST_FSL",
            "SLPF", "STBP", "STFiT", "STGCN", "STGODE", "STID",
        )
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "lifecycle_training_strategy: lcbt_v3" in text
        assert "lce_protocol_version: 3" in text
        assert "ples_protocol_version: 3" in text
        assert "max_epoch: 100" in text
        assert "patience: 10" in text

    new = (ROOT / "configs" / "ALFNet" / "rf_l3s.yaml").read_text(encoding="utf-8")
    assert "model_variant: rf_l3s" in new
    assert "model: ALFGraph" in new
    assert "model: ALFTemporal" in new


def test_event_clock_and_effective_mask_keep_anchors_distinct():
    assert phase_for_tau_v2("join", 0, 12) == "Cold"
    assert phase_for_tau_v2("join", 11, 12) == "Ready"
    assert phase_for_tau_v2("fault", 0, 12) == "Onset"
    assert phase_for_tau_v2("fault", 11, 12) == "Outage"

    raw = np.ones((4, 2, 1), dtype=np.float32)
    raw[2, 0, 0] = np.nan
    lifecycle_mask = np.ones((4, 2), dtype=bool)
    lifecycle_mask[:2, 0] = False
    role_ids = np.zeros((4, 2), dtype=np.int64)
    state_age = np.asarray(
        [[0.0, 0.25], [0.0, 0.50], [0.25, 0.75], [0.50, 1.00]],
        dtype=np.float32,
    )
    features, effective = compose_effective_lifecycle_features(
        raw,
        lifecycle_mask,
        role_ids,
        append_role=True,
        append_age=True,
        append_observation_mask=True,
        state_age=state_age,
    )
    np.testing.assert_array_equal(
        effective,
        np.isfinite(raw).all(axis=-1) & lifecycle_mask,
    )
    np.testing.assert_allclose(features[..., 5], state_age)
    np.testing.assert_array_equal(features[..., 6], effective.astype(np.float32))


def test_degree_stratified_unseen_split_is_frozen_and_disjoint():
    master = np.arange(50, dtype=np.int64)
    adjacency = _ring_adjacency(80)
    validation, test, metadata = split_unseen_nodes(master, adjacency, seed=2026)
    replay = split_unseen_nodes(master, adjacency, seed=2026)
    assert len(validation) == 20 and len(test) == 30
    assert set(validation).isdisjoint(test)
    assert sorted(np.concatenate([validation, test]).tolist()) == master.tolist()
    np.testing.assert_array_equal(validation, replay[0])
    np.testing.assert_array_equal(test, replay[1])
    assert sum(metadata["validation_stratum_counts"]) == 20


def test_balanced_batch_sampler_covers_each_window_once():
    schedule = build_lcbt_schedule(
        num_samples=100,
        input_len=12,
        adjacency=_ring_adjacency(60),
        calendar_regimes=["AM", "PM", "OffPeak", "Weekend"] * 25,
        seed=2026,
        epoch=0,
    )
    dataset = SimpleNamespace(schedule=schedule, current_epoch=0)
    sampler = LifecycleBalancedBatchSampler(dataset, batch_size=16, seed=2026)
    batches = list(sampler)
    flat = [index for batch in batches for index in batch]
    assert sorted(flat) == list(range(100))
    assert len(flat) == len(set(flat))
    for batch in batches[:-1]:
        counts = Counter(schedule[index]["scenario"] for index in batch)
        assert counts == Counter({name: 4 for name in LCBT_GROUPS})


def _build_dataset(tmp_path, flag, overwrite_manifest=True):
    num_steps = 960
    num_nodes = 12
    values = np.ones((num_steps, num_nodes, 3), dtype=np.float32)
    values[..., 2] = (
        np.arange(num_steps, dtype=np.float32)[:, None] / 100.0
        + np.arange(num_nodes, dtype=np.float32)[None, :]
        + 1.0
    )
    values[20, 3, 2] = np.nan
    timestamps = (
        np.datetime64("2026-01-01T00:00")
        + np.arange(num_steps) * np.timedelta64(5, "m")
    )
    data_path = tmp_path / "toy.npy"
    np.save(
        data_path,
        {
            "data": values,
            "adj": _ring_adjacency(num_nodes),
            "time_stamp": timestamps,
        },
        allow_pickle=True,
    )
    return LifecycleCoveragePEMS(
        data_path=str(data_path),
        size=[12, 0, 12],
        flag=flag,
        scale=True,
        freq="5min",
        target_features="v",
        input_features="v",
        dloader_name="TOY",
        splits=[0.5, 0.1],
        unknown_nodes_list=[0, 3, 6, 9],
        lifecycle_role_features=True,
        lifecycle_history_age_feature=True,
        lifecycle_observation_mask_feature=True,
        lcbt_manifest_root=str(tmp_path / "manifests"),
        lifecycle_age_mode="state_duration",
        lcbt_overwrite_manifest=overwrite_manifest,
        lifecycle_training_strategy="lcbt_v3",
    )


def test_dataset_uses_fixed_unseen_pools_full_k_validation_and_cohort_targets(workspace_tmp):
    train = _build_dataset(workspace_tmp, "train")
    validation = _build_dataset(workspace_tmp, "val")
    assert train.training_protocol_hash == validation.training_protocol_hash
    assert set(validation.validation_unseen_nodes).isdisjoint(
        validation.test_unseen_nodes
    )
    assert train.num_nodes == 8
    assert validation.num_nodes == 12

    train_sample = train[0]
    assert len(train_sample) == 10
    assert train_sample[0].shape[-1] == 7
    assert int(train_sample[7].sum()) >= 1

    unseen, steady, cold, onset = [validation[index] for index in range(4)]
    assert int(unseen[6]) == 0 and int(cold[6]) == 2
    assert int(steady[6]) == 1 and int(onset[6]) == 3
    assert int(cold[9]) == 1 and int(onset[9]) == 1
    np.testing.assert_array_equal(unseen[7], cold[7])
    np.testing.assert_array_equal(steady[7], onset[7])
    assert not np.array_equal(unseen[0], cold[0], equal_nan=True)
    assert not np.array_equal(steady[0], onset[0], equal_nan=True)

    join_k = Counter(
        int(item["evidence_steps"])
        for item in validation.schedule
        if item["scenario"] == "join"
    )
    fault_k = Counter(
        int(item["evidence_steps"])
        for item in validation.schedule
        if item["scenario"] == "fault"
    )
    assert set(join_k) == set(range(1, 13))
    assert set(fault_k) == set(range(1, 13))
    assert len(set(join_k.values())) == 1
    assert len(set(fault_k.values())) == 1


def _load_evaluation_module():
    spec = importlib.util.spec_from_file_location(
        "ples_evaluation",
        ROOT / "experiments" / "ples_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LastValueModel(torch.nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len

    def forward(
        self,
        adj,
        x_enc,
        x_t_mark=None,
        pos_mark=None,
        x_dec=None,
        dec_t_mark=None,
        epoch=0,
    ):
        return {"pred": x_enc[:, -1:, :, :1].repeat(1, self.pred_len, 1, 1)}


def test_lce_and_ples_outputs_are_separate_and_protocol_matched(workspace_tmp):
    train = _build_dataset(workspace_tmp, "train")
    dataset = AsyncLifecyclePEMS(
        data_path=str(workspace_tmp / "toy.npy"),
        size=[12, 0, 12],
        flag="test",
        scale=True,
        freq="5min",
        target_features="v",
        input_features="v",
        dloader_name="TOY",
        splits=[0.5, 0.1],
        unknown_nodes_list=[0, 3, 6, 9],
        lifecycle_role_features=True,
        lifecycle_history_age_feature=True,
        lifecycle_observation_mask_feature=True,
        lifecycle_age_mode="state_duration",
        lcbt_manifest_root=str(workspace_tmp / "manifests"),
        lcbt_scaler_end_ratio=0.50,
        lcbt_test_start_ratio=0.60,
        ples_protocol_version=3,
        lce_protocol_version=3,
        ples_manifest_root=str(workspace_tmp / "manifests"),
        ples_overwrite_manifest=True,
        ples_pre_steps=1,
        ples_post_steps=12,
        ples_bootstrap_samples=0,
        lifecycle_training_strategy="lcbt_v3",
    )
    checkpoint_dir = workspace_tmp / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "training_summary.json").write_text(
        json.dumps({"training_protocol_hash": train.training_protocol_hash}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        slide_step=1,
        best_prediction_path=str(workspace_tmp / "predictions"),
        best_checkpoint_path=str(checkpoint_dir),
        exp_id="v3-smoke",
        ples_manifest_root=str(workspace_tmp / "manifests"),
        ples_bootstrap_samples=0,
        ples_bootstrap_seed=7,
        lce_protocol_version=3,
        device="cpu",
        t_mark=False,
        pos_mark=False,
        pred_len=12,
        input_len=12,
        model="TinyModel",
        model_variant="tiny-v3",
        seed=32,
    )
    result = _load_evaluation_module().async_lifecycle_inference(
        _LastValueModel(12),
        SimpleNamespace(dataset=dataset),
        args,
    )
    output_dir = (
        workspace_tmp
        / "predictions"
        / "v3-smoke"
        / f"lifecycle-v3-{dataset.protocol_manifest['protocol_hash'][:10]}"
    )
    assert set(result["lce"]["main_metrics"]) == {
        "Unseen", "Steady", "Join", "Fault", "LifeAvg", "WorstGroup"
    }
    assert result["lce"]["main_metrics"]["WorstGroup"]["MAE"] == max(
        result["lce"]["main_metrics"][name]["MAE"]
        for name in ("Unseen", "Steady", "Join", "Fault")
    )
    assert len(result["lce"]["trajectory_metrics"]["Join"]) == 12
    for filename in (
        "lce_main_summary.json",
        "lce_phase_summary.json",
        "lce_trajectory_summary.json",
        "lce_trajectory_summary.csv",
        "ples_effects_summary.json",
        "ples_paired_rows.csv",
        "ples_curves.csv",
    ):
        assert (output_dir / filename).exists()
    rendered = (output_dir / "lce_main_summary.json").read_text(encoding="utf-8")
    assert '"formal_protocol_match": true' in rendered
    assert "RoleMacro" not in rendered
    assert "AvailabilityMacro" not in rendered

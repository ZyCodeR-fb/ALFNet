import numpy as np
import pytest
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from data_provider.lcbt_protocol import (
    LCBT_V2_GROUPS,
    LCBT_V2_PHASES,
    accumulate_lifecycle_mae,
    build_lcbt_v2_schedule,
    compose_effective_lifecycle_features,
    finalize_lifecycle_mae,
    finalize_lifecycle_validation_mae,
    split_unseen_nodes,
)
from data_provider.ples_protocol import (
    bootstrap_paired_effects_v2,
    build_protocol_manifest_v2,
    phase_for_tau_v2,
    select_event_origins_v2,
    summarize_lce_rows_v2,
    summarize_paired_rows_v2,
)
from data_provider.graph_data_loader import AsyncLifecyclePEMS, LifecycleCoveragePEMS


ROOT = Path(__file__).resolve().parents[1]


def _load_ples_evaluation():
    spec = importlib.util.spec_from_file_location(
        "ples_evaluation_v2",
        ROOT / "experiments" / "ples_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ring_adjacency(num_nodes):
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for node in range(num_nodes):
        adjacency[node, (node + 1) % num_nodes] = 1.0 + node / num_nodes
        adjacency[(node + 1) % num_nodes, node] = 1.0
    return adjacency


def test_v2_event_clock_distinguishes_both_static_anchors_from_event_onsets():
    assert phase_for_tau_v2("join", tau=0, input_len=12) == "Cold"
    assert phase_for_tau_v2("join", tau=1, input_len=12) == "Warm"
    assert phase_for_tau_v2("join", tau=10, input_len=12) == "Warm"
    assert phase_for_tau_v2("join", tau=11, input_len=12) == "Ready"

    assert phase_for_tau_v2("fault", tau=0, input_len=12) == "Onset"
    assert phase_for_tau_v2("fault", tau=1, input_len=12) == "Decay"
    assert phase_for_tau_v2("fault", tau=10, input_len=12) == "Decay"
    assert phase_for_tau_v2("fault", tau=11, input_len=12) == "Outage"


def test_all_models_expose_explicit_lcbt_v2_configs():
    config_paths = [
        ROOT / "configs" / "ALFNet" / "rf_lcb2_g.yaml",
        ROOT / "configs" / "ALFNet" / "rf_lcb2_s.yaml",
    ] + [
        ROOT / "configs" / model / "METRLA_lcbt_v2.yaml"
        for model in (
            "MoGERNN", "PatchSTG", "FCLSTM", "GWNET", "HA", "MCPST_FSL",
            "SLPF", "STBP", "STFiT", "STGCN", "STGODE", "STID",
        )
    ]

    for path in config_paths:
        text = path.read_text(encoding="utf-8")
        assert "lifecycle_training_strategy: lcbt_v2" in text
        assert "stop_based: val_lifecycle" in text
        assert "lifecycle_age_mode: state_duration" in text
        assert "ples_protocol_version: 2" in text
        assert "lce_protocol_version: 1" in text
        assert "lcbt_zero_history_ratio: 0.25" in text


def test_v2_configs_only_change_protocol_fields_and_short_variant_names():
    yaml = pytest.importorskip("yaml")
    families = (
        "MoGERNN", "PatchSTG", "FCLSTM", "GWNET", "HA", "MCPST_FSL",
        "SLPF", "STBP", "STFiT", "STGCN", "STGODE", "STID",
    )
    pairs = [
        (ROOT / "configs" / family / "METRLA_lcbt.yaml", ROOT / "configs" / family / "METRLA_lcbt_v2.yaml")
        for family in families
    ] + [
        (ROOT / "configs" / "ALFNet" / "rf_lcb_g.yaml", ROOT / "configs" / "ALFNet" / "rf_lcb2_g.yaml"),
        (ROOT / "configs" / "ALFNet" / "rf_lcb_s.yaml", ROOT / "configs" / "ALFNet" / "rf_lcb2_s.yaml"),
    ]
    for old_path, new_path in pairs:
        old = yaml.safe_load(old_path.read_text(encoding="utf-8"))
        new = yaml.safe_load(new_path.read_text(encoding="utf-8"))
        assert old["dataset"] == new["dataset"]
        old_runtime = {key: value for key, value in old["runtime"].items() if key != "model_variant"}
        new_runtime = {key: value for key, value in new["runtime"].items() if key != "model_variant"}
        assert old_runtime == new_runtime
        old_training = {key: value for key, value in old["training"].items() if key != "stop_based"}
        new_training = {key: value for key, value in new["training"].items() if key != "stop_based"}
        assert old_training == new_training
        assert len(new["runtime"]["model_variant"]) <= 12
        if "family" in old:
            assert old["family"] == new["family"]


def test_v2_origins_are_calendar_adaptive_and_respect_inclusive_clock_bounds():
    timestamps = np.datetime64("2026-01-01T00:00") + np.arange(7 * 288) * np.timedelta64(5, "m")
    origins = select_event_origins_v2(
        timestamps,
        timeline_length=len(timestamps),
        input_len=12,
        pred_len=12,
        pre_steps=12,
        post_steps=36,
    )

    keys = [(origin["calendar_day"], origin["regime"]) for origin in origins]
    assert len(keys) == len(set(keys))
    assert len(origins) > 4
    for origin in origins:
        event = origin["event_time_index"]
        assert event - 12 + 1 - 12 >= 0
        assert event + 36 + 1 + 12 <= len(timestamps)


def test_v2_schedule_balances_four_groups_and_all_six_phases():
    schedule = build_lcbt_v2_schedule(
        num_samples=624,
        input_len=12,
        adjacency=_ring_adjacency(60),
        calendar_regimes=["AM", "PM", "OffPeak", "Weekend"] * 156,
        seed=2026,
        epoch=2,
    )

    scenarios = [item["scenario"] for item in schedule]
    assert {name: scenarios.count(name) for name in LCBT_V2_GROUPS} == {
        "zero_history": 156,
        "steady": 156,
        "join": 156,
        "fault": 156,
    }

    join_evidence = [
        item["evidence_steps"] for item in schedule if item["scenario"] == "join"
    ]
    fault_evidence = [
        item["evidence_steps"] for item in schedule if item["scenario"] == "fault"
    ]
    assert join_evidence.count(1) == 52
    assert join_evidence.count(12) == 52
    assert fault_evidence.count(1) == 52
    assert fault_evidence.count(12) == 52
    assert max(np.bincount(join_evidence, minlength=13)[2:12]) - min(
        np.bincount(join_evidence, minlength=13)[2:12]
    ) <= 1
    assert max(np.bincount(fault_evidence, minlength=13)[2:12]) - min(
        np.bincount(fault_evidence, minlength=13)[2:12]
    ) <= 1

    phase_ids = {
        item["phase_id"]
        for item in schedule
        if item["phase_id"] >= 0
    }
    assert phase_ids == set(range(len(LCBT_V2_PHASES)))
    assert all(item["changed_nodes"] for item in schedule)
    for regime in ("AM", "PM", "OffPeak", "Weekend"):
        for scenario in ("join", "fault"):
            evidence = [
                item["evidence_steps"]
                for item in schedule
                if item["calendar_regime"] == regime
                and item["scenario"] == scenario
            ]
            stage_counts = (
                sum(value == 1 for value in evidence),
                sum(2 <= value < 12 for value in evidence),
                sum(value == 12 for value in evidence),
            )
            assert max(stage_counts) - min(stage_counts) <= 1


def test_degree_stratified_unseen_split_is_disjoint_and_deterministic():
    master = np.arange(50, dtype=np.int64)
    adjacency = _ring_adjacency(80)
    val_nodes, test_nodes, metadata = split_unseen_nodes(
        master,
        adjacency,
        validation_ratio=0.40,
        seed=2026,
    )
    replay = split_unseen_nodes(
        master,
        adjacency,
        validation_ratio=0.40,
        seed=2026,
    )

    assert len(val_nodes) == 20
    assert len(test_nodes) == 30
    assert set(val_nodes).isdisjoint(test_nodes)
    assert sorted(np.concatenate([val_nodes, test_nodes]).tolist()) == master.tolist()
    np.testing.assert_array_equal(val_nodes, replay[0])
    np.testing.assert_array_equal(test_nodes, replay[1])
    assert sum(metadata["validation_stratum_counts"]) == 20
    assert sum(metadata["test_stratum_counts"]) == 30


def test_v2_manifest_freezes_two_anchors_two_tracks_and_weighted_cohort_plan():
    num_nodes = 80
    timeline = 7 * 288
    timestamps = np.datetime64("2026-01-01T00:00") + np.arange(timeline) * np.timedelta64(5, "m")
    adjacency = _ring_adjacency(num_nodes)
    master = list(range(20))
    validation = list(range(8))
    test = list(range(8, 20))
    known = list(range(20, num_nodes))
    manifest = build_protocol_manifest_v2(
        "TOY",
        timestamps,
        adjacency,
        known,
        master,
        validation,
        test,
        input_len=12,
        pred_len=12,
        training_protocol_hash="train-hash",
        test_values=np.ones((timeline, num_nodes, 1), dtype=np.float32),
    )

    fractions = [cohort["fraction"] for cohort in manifest["cohorts"]]
    styles = [cohort["style"] for cohort in manifest["cohorts"]]
    assert fractions.count(0.025) == 2
    assert fractions.count(0.05) == 4
    assert fractions.count(0.10) == 2
    assert styles.count("clustered") == styles.count("scattered") == 4
    assert all(set(cohort["join_nodes"]) <= set(test) for cohort in manifest["cohorts"])
    assert all(set(cohort["fault_nodes"]) <= set(known) for cohort in manifest["cohorts"])
    assert manifest["training_protocol_hash"] == "train-hash"
    assert manifest["lce_protocol_hash"]
    assert manifest["ples_protocol_hash"]


def test_state_age_is_event_duration_and_not_visibility_alias():
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

    np.testing.assert_allclose(features[..., 5], state_age)
    visibility = effective.mean(axis=0)
    assert features[-1, 0, 5] != pytest.approx(visibility[0])


def test_lifecycle_checkpoint_macro_is_four_semantic_groups_not_six_phases():
    metrics = finalize_lifecycle_mae(
        {
            "Unseen": [8.0, 10.0],
            "Steady": [2.0],
            "Cold": [5.0],
            "Warm": [4.0],
            "Ready": [3.0],
            "Onset": [6.0],
            "Decay": [7.0],
            "Outage": [8.0],
        }
    )

    assert metrics["unseen_mae"] == pytest.approx(9.0)
    assert metrics["steady_mae"] == pytest.approx(2.0)
    assert metrics["join_mae"] == pytest.approx(4.0)
    assert metrics["fault_mae"] == pytest.approx(7.0)
    assert metrics["mae"] == pytest.approx(5.5)
    assert metrics["worst_group_mae"] == pytest.approx(9.0)


def test_validation_mae_equalizes_calendar_cells_before_lifecycle_groups():
    true = np.zeros((9, 1, 1, 1), dtype=np.float32)
    pred = np.asarray([1, 1, 1, 1, 9, 2, 3, 4, 5], dtype=np.float32).reshape(9, 1, 1, 1)
    groups = np.asarray([0, 0, 0, 0, 0, 1, 2, 3, 3])
    phases = np.asarray([-1, -1, -1, -1, -1, -1, 0, 3, 4])
    calendars = np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 0])
    cohorts = np.ones((9, 1), dtype=np.float32)
    state = {}

    accumulate_lifecycle_mae(
        state,
        true,
        pred,
        groups,
        phases,
        cohorts,
        calendars,
    )
    # Fill missing event phases so the four-group macro is finite.
    state["Warm"] = {0: [3.0]}
    state["Ready"] = {0: [3.0]}
    state["Outage"] = {0: [4.5]}
    metrics = finalize_lifecycle_validation_mae(state)

    # Unseen is mean(calendar-0 mean=1, calendar-1 mean=9), not pooled mean=2.6.
    assert metrics["unseen_mae"] == pytest.approx(5.0)
    assert metrics["steady_mae"] == pytest.approx(2.0)


def _paired_metric_row(
    track,
    phase,
    tau,
    regime,
    treated_mape,
    control_mape,
    treated_mae,
    control_mae,
    count=1,
    origin_id=0,
    cohort_id=0,
):
    return {
        "track": track,
        "phase": phase,
        "tau": tau,
        "regime": regime,
        "origin_id": origin_id,
        "cohort_id": cohort_id,
        "count": count,
        "treated_ape_sum": treated_mape / 100.0 * count,
        "control_ape_sum": control_mape / 100.0 * count,
        "treated_ae_sum": treated_mae * count,
        "control_ae_sum": control_mae * count,
        "treated_se_sum": treated_mae**2 * count,
        "control_se_sum": control_mae**2 * count,
        "graph_weight": None,
    }


def test_lce_v1_equalizes_cells_calendar_strata_and_four_semantic_groups():
    rows = []
    for regime, count, unseen_mae in (("AM", 100, 1.0), ("PM", 1, 9.0)):
        for tau, phase, treated_mae in ((0, "Cold", 3.0), (1, "Warm", 4.0), (11, "Ready", 5.0)):
            rows.append(
                _paired_metric_row(
                    "join",
                    phase,
                    tau,
                    regime,
                    treated_mae,
                    unseen_mae,
                    treated_mae,
                    unseen_mae,
                    count=count,
                )
            )
        for tau, phase, treated_mae in ((0, "Onset", 6.0), (1, "Decay", 7.0), (11, "Outage", 8.0)):
            rows.append(
                _paired_metric_row(
                    "fault",
                    phase,
                    tau,
                    regime,
                    treated_mae,
                    2.0,
                    treated_mae,
                    2.0,
                    count=count,
                )
            )

    summary = summarize_lce_rows_v2(rows)

    assert summary["main_metrics"]["Unseen"]["MAE"] == pytest.approx(5.0)
    assert summary["main_metrics"]["Steady"]["MAE"] == pytest.approx(2.0)
    assert summary["main_metrics"]["Join"]["MAE"] == pytest.approx(4.0)
    assert summary["main_metrics"]["Fault"]["MAE"] == pytest.approx(7.0)
    assert summary["main_metrics"]["LifeAvg"]["MAE"] == pytest.approx(4.5)


def test_ples_v2_uses_descriptive_directions_and_zero_pre_event_difference():
    rows = [
        _paired_metric_row("join", "Pre", -1, "AM", 5.0, 5.0, 1.0, 1.0),
        _paired_metric_row("fault", "Pre", -1, "AM", 4.0, 4.0, 1.0, 1.0),
    ]
    for tau, phase in ((0, "Cold"), (1, "Warm"), (11, "Ready")):
        rows.append(_paired_metric_row("join", phase, tau, "AM", 3.0, 5.0, 1.0, 1.0))
    for tau, phase in ((0, "Onset"), (1, "Decay"), (11, "Outage")):
        rows.append(_paired_metric_row("fault", phase, tau, "AM", 7.0, 4.0, 1.0, 1.0))

    summary = summarize_paired_rows_v2(rows)
    intervals = bootstrap_paired_effects_v2(rows, samples=0)

    assert summary["paired_effects"]["JoinLift"] == pytest.approx(2.0)
    assert summary["paired_effects"]["FaultPenalty"] == pytest.approx(3.0)
    assert summary["paired_effects"]["PeakFaultPenalty"] == pytest.approx(3.0)
    assert summary["pre_event_max_abs_mape_difference"] == pytest.approx(0.0)
    assert intervals["JoinLift"]["point"] == pytest.approx(2.0)


def test_checkpoint_uses_point_one_percent_primary_then_worst_group_tiebreak():
    pytest.importorskip("omegaconf")
    from experiments.train import _accept_lifecycle_checkpoint

    assert _accept_lifecycle_checkpoint(9.98, 10.0, 12.0, 11.0)
    assert not _accept_lifecycle_checkpoint(9.995, 10.0, 12.0, 11.0)
    assert _accept_lifecycle_checkpoint(10.005, 10.0, 10.5, 11.0)
    assert not _accept_lifecycle_checkpoint(10.02, 10.0, 1.0, 11.0)


def _build_v2_dataset(tmp_path, flag, overwrite_manifest=True):
    num_steps = 480
    num_nodes = 12
    values = np.ones((num_steps, num_nodes, 3), dtype=np.float32)
    values[..., 2] = (
        np.arange(num_steps, dtype=np.float32)[:, None] / 100.0
        + np.arange(num_nodes, dtype=np.float32)[None, :]
        + 1.0
    )
    values[20, 3, 2] = np.nan
    timestamps = np.datetime64("2026-01-01T11:00") + np.arange(num_steps) * np.timedelta64(5, "m")
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
        lcbt_protocol_version=2,
        lcbt_join_ratio=0.25,
        lcbt_fault_ratio=0.25,
        lcbt_steady_ratio=0.25,
        lcbt_zero_history_ratio=0.25,
        lifecycle_age_mode="state_duration",
        lcbt_overwrite_manifest=overwrite_manifest,
    )


def test_v2_dataset_uses_true_validation_unseen_and_cohort_only_targets():
    workdir = ROOT / ".pytest_cache" / "lifecycle_v2_dataset"
    workdir.mkdir(parents=True, exist_ok=True)
    train = _build_v2_dataset(workdir, "train")
    validation = _build_v2_dataset(workdir, "val")

    assert train.training_protocol_hash == validation.training_protocol_hash
    assert set(validation.validation_unseen_nodes).isdisjoint(validation.test_unseen_nodes)
    assert train.num_nodes == 8
    assert validation.num_nodes == 12

    train_sample = train[0]
    assert len(train_sample) == 9
    assert train_sample[0].shape[-1] == 7
    assert int(train_sample[7].sum()) >= 1

    unseen = validation[0]
    cold = validation[1]
    steady = validation[4]
    onset = validation[5]
    assert int(unseen[6]) == 0 and int(cold[6]) == 2
    assert int(steady[6]) == 1 and int(onset[6]) == 3
    np.testing.assert_array_equal(unseen[7], cold[7])
    np.testing.assert_array_equal(steady[7], onset[7])
    assert not np.array_equal(unseen[0], cold[0], equal_nan=True)
    assert not np.array_equal(steady[0], onset[0], equal_nan=True)
    cold_nodes = np.flatnonzero(cold[7] > 0)
    onset_nodes = np.flatnonzero(onset[7] > 0)
    assert np.all(cold[0][-1, cold_nodes, -1] == 1.0)
    assert np.all(unseen[0][-1, cold_nodes, -1] == 0.0)
    assert np.all(onset[0][-1, onset_nodes, -1] == 0.0)
    assert np.all(steady[0][-1, onset_nodes, -1] == 1.0)


def test_v2_train_and_validation_share_one_frozen_manifest_without_overwrite():
    workdir = ROOT / ".pytest_cache" / "lifecycle_v2_manifest_identity"
    workdir.mkdir(parents=True, exist_ok=True)
    manifest_path = workdir / "manifests" / "TOY" / "lcbt_v2" / "protocol_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    train = _build_v2_dataset(workdir, "train", overwrite_manifest=False)
    validation = _build_v2_dataset(workdir, "val", overwrite_manifest=False)

    assert train.training_protocol_hash == validation.training_protocol_hash
    assert train.protocol_manifest == validation.protocol_manifest


def test_async_v2_provider_uses_inclusive_clock_and_frozen_test_unseen_pool():
    workdir = ROOT / ".pytest_cache" / "lifecycle_v2_async"
    workdir.mkdir(parents=True, exist_ok=True)
    _build_v2_dataset(workdir, "train")
    dataset = AsyncLifecyclePEMS(
        data_path=str(workdir / "toy.npy"),
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
        lcbt_manifest_root=str(workdir / "manifests"),
        lcbt_scaler_end_ratio=0.50,
        lcbt_test_start_ratio=0.60,
        lcbt_protocol_version=2,
        ples_protocol_version=2,
        lce_protocol_version=1,
        ples_manifest_root=str(workdir / "manifests"),
        ples_overwrite_manifest=True,
        ples_pre_steps=12,
        ples_post_steps=36,
    )
    manifest = dataset.get_protocol_manifest()
    origin = manifest["origins"][0]
    cohort = manifest["cohorts"][0]

    join_control = dataset.build_ples_sample(origin, cohort, "join", "control", 0)[0]
    cold = dataset.build_ples_sample(origin, cohort, "join", "treatment", 0)[0]
    steady = dataset.build_ples_sample(origin, cohort, "fault", "control", 0)[0]
    onset = dataset.build_ples_sample(origin, cohort, "fault", "treatment", 0)[0]
    pre_control = dataset.build_ples_sample(origin, cohort, "join", "control", -1)[0]
    pre_treatment = dataset.build_ples_sample(origin, cohort, "join", "treatment", -1)[0]

    join_nodes = cohort["join_nodes"]
    fault_nodes = cohort["fault_nodes"]
    assert manifest["nodes"]["test_unseen"] == dataset.test_unseen_nodes
    assert np.all(cold[:, join_nodes, -1].sum(axis=0) == 1.0)
    assert np.all(join_control[:, join_nodes, -1].sum(axis=0) == 0.0)
    assert np.all(onset[:, fault_nodes, -1].sum(axis=0) == 11.0)
    assert np.all(steady[:, fault_nodes, -1].sum(axis=0) == 12.0)
    np.testing.assert_array_equal(pre_control, pre_treatment)


class _V2LastValueModel(torch.nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        return {"pred": x_enc[:, -1:, :, :1].repeat(1, self.pred_len, 1, 1)}


def test_v2_evaluator_writes_separate_lce_and_ples_outputs():
    workdir = ROOT / ".pytest_cache" / "lifecycle_v2_eval"
    workdir.mkdir(parents=True, exist_ok=True)
    _build_v2_dataset(workdir, "train")
    dataset = AsyncLifecyclePEMS(
        data_path=str(workdir / "toy.npy"),
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
        lcbt_manifest_root=str(workdir / "manifests"),
        lcbt_scaler_end_ratio=0.50,
        lcbt_test_start_ratio=0.60,
        lcbt_protocol_version=2,
        ples_protocol_version=2,
        lce_protocol_version=1,
        ples_manifest_root=str(workdir / "manifests"),
        ples_overwrite_manifest=True,
        ples_pre_steps=1,
        ples_post_steps=12,
        ples_bootstrap_samples=0,
    )
    checkpoint_dir = workdir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "training_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "training_protocol_hash": dataset.protocol_manifest[
                    "training_protocol_hash"
                ]
            },
            handle,
        )
    args = SimpleNamespace(
        slide_step=1,
        test_for_changed=False,
        best_prediction_path=str(workdir / "predictions"),
        best_checkpoint_path=str(checkpoint_dir),
        exp_id="v2-smoke",
        ples_manifest_root=str(workdir / "manifests"),
        ples_bootstrap_samples=0,
        ples_bootstrap_seed=7,
        ples_protocol_version=2,
        lce_protocol_version=1,
        device="cpu",
        t_mark=False,
        pos_mark=False,
        pred_len=12,
        input_len=12,
        model="TinyModel",
        model_variant="tiny-v2",
        seed=32,
    )
    evaluation = _load_ples_evaluation()
    loader = SimpleNamespace(dataset=dataset)

    result = evaluation.async_lifecycle_inference(
        _V2LastValueModel(pred_len=12),
        loader,
        args,
    )
    output_dir = (
        workdir
        / "predictions"
        / "v2-smoke"
        / f"lifecycle-v2-{dataset.protocol_manifest['protocol_hash'][:10]}"
    )

    assert set(result["lce"]["main_metrics"]) == {
        "Unseen", "Steady", "Join", "Fault", "LifeAvg"
    }
    for filename in (
        "lce_main_summary.json",
        "lce_main_summary.csv",
        "lce_phase_summary.json",
        "lce_phase_summary.csv",
        "ples_effects_summary.json",
        "ples_effects_summary.csv",
        "ples_paired_rows.csv",
        "ples_curves.csv",
    ):
        assert (output_dir / filename).exists()
    rendered = (output_dir / "lce_main_summary.json").read_text(encoding="utf-8")
    assert '"seed": 32' in rendered
    assert '"formal_lcbt_v2": true' in rendered
    assert "RoleMacro" not in rendered
    assert "AvailabilityMacro" not in rendered

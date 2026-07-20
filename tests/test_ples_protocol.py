from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_provider.graph_data_loader import AsyncLifecyclePEMS, ROLE_AAS, ROLE_FS, ROLE_NAS, ROLE_VS
from data_provider.ples_protocol import (
    PHASE_ORDER,
    build_balanced_cohorts,
    build_protocol_manifest,
    phase_for_tau,
    resolve_cohort_plan,
    resolve_time_blocks,
    select_event_origins,
    summarize_phase_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_ples_evaluation():
    spec = importlib.util.spec_from_file_location(
        "ples_evaluation",
        ROOT / "experiments" / "ples_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_ples_comparison():
    spec = importlib.util.spec_from_file_location(
        "plot_counterfactual_shockwave",
        ROOT / "scripts" / "plot_counterfactual_shockwave.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timestamps(days, start=datetime(2024, 1, 1), step_minutes=5):
    count = days * 24 * 60 // step_minutes
    return np.array([start + timedelta(minutes=step_minutes * index) for index in range(count)])


def test_cohort_plan_is_dataset_adaptive():
    metrla = resolve_cohort_plan(
        num_nodes=207,
        unknown_count=50,
        known_count=157,
        requested_cohorts=5,
        cohort_fraction=0.05,
    )
    pemsbay = resolve_cohort_plan(
        num_nodes=325,
        unknown_count=65,
        known_count=260,
        requested_cohorts=5,
        cohort_fraction=0.05,
    )
    tiny = resolve_cohort_plan(
        num_nodes=12,
        unknown_count=2,
        known_count=10,
        requested_cohorts=5,
        cohort_fraction=0.05,
    )

    assert metrla == {"cohort_count": 5, "cohort_size": 10}
    assert pemsbay == {"cohort_count": 5, "cohort_size": 13}
    assert tiny == {"cohort_count": 2, "cohort_size": 1}


def test_balanced_cohorts_are_disjoint_and_cover_degree_range():
    nodes = list(range(20))
    degree = np.arange(20, dtype=np.float64)

    cohorts = build_balanced_cohorts(nodes, degree, cohort_count=4, cohort_size=3)

    assert len(cohorts) == 4
    assert all(len(cohort) == 3 for cohort in cohorts)
    flattened = [node for cohort in cohorts for node in cohort]
    assert len(flattened) == len(set(flattened))
    assert min(flattened) <= 2
    assert max(flattened) >= 17
    assert all(max(degree[cohort]) - min(degree[cohort]) >= 8 for cohort in cohorts)


def test_seeded_balanced_cohorts_are_reproducible():
    nodes = list(range(100))
    degree = np.arange(100, dtype=np.float64)

    first = build_balanced_cohorts(nodes, degree, cohort_count=5, cohort_size=4, seed=2026)
    repeat = build_balanced_cohorts(nodes, degree, cohort_count=5, cohort_size=4, seed=2026)
    changed = build_balanced_cohorts(nodes, degree, cohort_count=5, cohort_size=4, seed=2027)

    assert first == repeat
    assert first != changed
    assert len({node for cohort in first for node in cohort}) == 20


def test_phase_boundaries_follow_actual_input_length():
    assert phase_for_tau("join", -1, input_len=12) == "Pre"
    assert phase_for_tau("join", 0, input_len=12) == "Cold"
    assert phase_for_tau("join", 1, input_len=12) == "Warm"
    assert phase_for_tau("join", 11, input_len=12) == "Warm"
    assert phase_for_tau("join", 12, input_len=12) == "Ready"

    assert phase_for_tau("fault", -1, input_len=6) == "Pre"
    assert phase_for_tau("fault", 0, input_len=6) == "Onset"
    assert phase_for_tau("fault", 1, input_len=6) == "Decay"
    assert phase_for_tau("fault", 5, input_len=6) == "Decay"
    assert phase_for_tau("fault", 6, input_len=6) == "Outage"


def test_origin_selection_adapts_to_timeline_and_keeps_minimum_gap():
    long_timestamps = _timestamps(days=56)
    short_timestamps = _timestamps(days=2)

    long_origins = select_event_origins(
        long_timestamps,
        min_event_index=24,
        max_event_index=len(long_timestamps) - 48,
        temporal_blocks=4,
        min_gap=72,
    )
    short_origins = select_event_origins(
        short_timestamps,
        min_event_index=24,
        max_event_index=len(short_timestamps) - 48,
        temporal_blocks=4,
        min_gap=72,
    )

    assert len(long_origins) == 16
    assert 0 < len(short_origins) < 16
    assert len({(item["block_id"], item["regime"]) for item in long_origins}) == 16
    selected = sorted(item["event_time_index"] for item in long_origins)
    assert min(np.diff(selected)) >= 72
    assert all(24 <= item["event_time_index"] <= len(long_timestamps) - 48 for item in long_origins)


def test_time_blocks_record_dataset_specific_boundaries():
    short_timestamps = _timestamps(days=2)
    long_timestamps = _timestamps(days=56)

    short_blocks = resolve_time_blocks(short_timestamps, 24, len(short_timestamps) - 48, 4)
    long_blocks = resolve_time_blocks(long_timestamps, 24, len(long_timestamps) - 48, 4)

    assert len(short_blocks) == len(long_blocks) == 4
    assert sum(block["time_steps"] for block in short_blocks) == len(short_timestamps) - 71
    assert sum(block["time_steps"] for block in long_blocks) == len(long_timestamps) - 71
    assert short_blocks[0]["end_time_index"] < long_blocks[0]["end_time_index"]
    assert short_blocks[0]["start_timestamp"] == "2024-01-01 02:00:00"


def test_manifest_records_resolved_dataset_specific_protocol():
    timestamps = _timestamps(days=28)
    adj_207 = np.eye(207, dtype=np.float32)
    adj_325 = np.eye(325, dtype=np.float32)

    metrla = build_protocol_manifest(
        dataset_name="METRLA",
        timestamps=timestamps,
        adjacency=adj_207,
        known_nodes=list(range(50, 207)),
        unknown_nodes=list(range(50)),
        input_len=12,
        pred_len=12,
        pre_steps=12,
        post_steps=36,
        temporal_blocks=4,
        requested_cohorts=5,
        cohort_fraction=0.05,
        min_event_gap=72,
        seed=2026,
    )
    pemsbay = build_protocol_manifest(
        dataset_name="PEMSBAY",
        timestamps=np.concatenate([timestamps, timestamps + timedelta(days=28)]),
        adjacency=adj_325,
        known_nodes=list(range(65, 325)),
        unknown_nodes=list(range(65)),
        input_len=12,
        pred_len=12,
        pre_steps=12,
        post_steps=36,
        temporal_blocks=4,
        requested_cohorts=5,
        cohort_fraction=0.05,
        min_event_gap=72,
        seed=2026,
    )

    assert metrla["dataset"]["num_nodes"] == 207
    assert pemsbay["dataset"]["num_nodes"] == 325
    assert metrla["resolved"]["cohort_size"] == 10
    assert pemsbay["resolved"]["cohort_size"] == 13
    assert metrla["dataset"]["fingerprint"] != pemsbay["dataset"]["fingerprint"]
    assert metrla["metric_names"] == ["Join", "Fault", "LifeMacro"]
    assert metrla["phase_names"] == list(PHASE_ORDER)
    assert metrla["evaluation"]["main_metric_rule"]["LifeMacro"] == "equal mean of all six phases"
    assert metrla["evaluation"]["bootstrap"]["samples"] == 2000
    assert len(metrla["origins"]) <= 16
    assert metrla["resolved"]["temporal_block_count"] == 4
    assert sum(block["origin_count"] for block in metrla["resolved"]["time_blocks"]) == len(metrla["origins"])


def test_main_metrics_are_equal_phase_macros():
    phase_metrics = {
        phase: {"MAPE": float(index + 1), "MAE": float(index + 2), "RMSE": float(index + 3)}
        for index, phase in enumerate(PHASE_ORDER)
    }

    summary = summarize_phase_metrics(phase_metrics)

    assert summary["Join"]["MAPE"] == 2.0
    assert summary["Fault"]["MAPE"] == 5.0
    assert summary["LifeMacro"]["MAPE"] == 3.5
    assert set(summary) == {"Join", "Fault", "LifeMacro"}


def _tiny_async_dataset():
    dataset = AsyncLifecyclePEMS.__new__(AsyncLifecyclePEMS)
    dataset.input_len = 4
    dataset.look_back = 0
    dataset.pred_len = 2
    dataset.raw_enc_in = 1
    dataset.lifecycle_role_features = True
    dataset.lifecycle_history_age_feature = True
    dataset.initial_aas_nodes = [2, 3]
    dataset.initial_vs_nodes = [0, 1]
    dataset.num_nodes = 4
    dataset.data = np.arange(48, dtype=np.float32).reshape(12, 4, 1) + 1.0
    dataset.data_y = dataset.data.copy()
    dataset.time_mark = np.zeros((12, 3), dtype=np.float32)
    return dataset


def test_paired_samples_reset_each_episode_and_share_targets():
    dataset = _tiny_async_dataset()
    origin = {"event_time_index": 5}
    cohort = {"join_nodes": [0], "fault_nodes": [2]}

    join_control = dataset.build_ples_sample(origin, cohort, "join", "control", tau=2)
    join_treated = dataset.build_ples_sample(origin, cohort, "join", "treatment", tau=2)
    fault_control = dataset.build_ples_sample(origin, cohort, "fault", "control", tau=2)
    fault_treated = dataset.build_ples_sample(origin, cohort, "fault", "treatment", tau=2)

    join_x = join_treated[0]
    fault_x = fault_treated[0]
    assert np.isnan(join_control[0][:, 0, 0]).all()
    assert np.isnan(join_x[:2, 0, 0]).all()
    assert np.isfinite(join_x[2:, 0, 0]).all()
    assert np.isfinite(fault_control[0][:, 2, 0]).all()
    assert np.isfinite(fault_x[:2, 2, 0]).all()
    assert np.isnan(fault_x[2:, 2, 0]).all()
    assert np.array_equal(join_control[1], join_treated[1])
    assert np.array_equal(fault_control[1], fault_treated[1])

    role_start = 1
    assert np.all(join_x[:2, 0, role_start + ROLE_VS] == 1.0)
    assert np.all(join_x[2:, 0, role_start + ROLE_NAS] == 1.0)
    assert np.all(fault_x[:2, 2, role_start + ROLE_AAS] == 1.0)
    assert np.all(fault_x[2:, 2, role_start + ROLE_FS] == 1.0)
    assert np.allclose(join_x[:, 0, -1], [0.0, 0.0, 0.25, 0.5])


def _paired_rows_for_summary():
    rows = []
    phase_by_track = {
        "join": ((0, "Cold"), (1, "Warm"), (12, "Ready")),
        "fault": ((0, "Onset"), (1, "Decay"), (12, "Outage")),
    }
    for origin_id in (1, 2):
        for cohort_id in (1, 2):
            for track, cells in phase_by_track.items():
                for tau, phase in cells:
                    control_error = 1.0
                    treated_error = 0.5 if track == "join" else 1.5
                    rows.append(
                        {
                            "origin_id": origin_id,
                            "cohort_id": cohort_id,
                            "track": track,
                            "tau": tau,
                            "phase": phase,
                            "count": 10,
                            "treated_ape_sum": treated_error,
                            "control_ape_sum": control_error,
                            "treated_ae_sum": treated_error * 10,
                            "control_ae_sum": control_error * 10,
                            "treated_se_sum": treated_error * 10,
                            "control_se_sum": control_error * 10,
                        }
                    )
    return rows


def test_paired_summary_uses_new_names_and_effect_directions():
    rows = _paired_rows_for_summary()
    evaluation = _load_ples_evaluation()

    summary = evaluation.summarize_paired_rows(rows)

    assert set(summary["main_metrics"]) == {"Join", "Fault", "LifeMacro"}
    assert set(summary["phase_metrics"]) == set(PHASE_ORDER)
    assert summary["paired_effects"]["JoinBenefitAUC"] > 0
    assert summary["paired_effects"]["FaultShockAUC"] > 0
    assert summary["paired_effects"]["PeakFaultShock"] > 0


def test_metric_sums_and_hierarchical_bootstrap_are_deterministic():
    evaluation = _load_ples_evaluation()
    sums = {"count": 4, "treated_ape_sum": 1.0, "treated_ae_sum": 8.0, "treated_se_sum": 20.0}
    metrics = evaluation.metric_from_sums(sums, "treated")
    intervals = evaluation.bootstrap_paired_effects(_paired_rows_for_summary(), samples=50, seed=7)

    assert metrics == {"MAPE": 25.0, "MAE": 2.0, "RMSE": np.sqrt(5.0)}
    assert intervals["JoinBenefitAUC"]["low"] == intervals["JoinBenefitAUC"]["high"]
    assert intervals["FaultShockAUC"]["low"] == intervals["FaultShockAUC"]["high"]


def test_paired_model_comparison_reports_positive_attenuation_gain():
    comparison = _load_ples_comparison()
    ours = _paired_rows_for_summary()
    baseline = []
    for row in ours:
        treated_delta = 0.2 if row["track"] == "join" else 0.3
        baseline.append(
            {
                **row,
                "treated_ape_sum": row["treated_ape_sum"] + treated_delta,
                "treated_ae_sum": row["treated_ae_sum"] + treated_delta * 10,
                "treated_se_sum": row["treated_se_sum"] + treated_delta * 10,
            }
        )

    curves = comparison.compare_rows(ours, baseline)
    intervals = comparison.bootstrap_gains(ours, baseline, samples=20, seed=9)

    assert all(row["attenuation_gain_mape"] > 0 for row in curves)
    assert intervals["Join"]["point"] > 0
    assert intervals["Fault"]["point"] > 0
    assert intervals["LifeMacro"]["low"] == intervals["LifeMacro"]["high"]


class _LastValueModel(torch.nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        pred = x_enc[:, -1:, :, :1].repeat(1, self.pred_len, 1, 1)
        return {"pred": pred}


def test_full_ples_evaluator_writes_only_new_protocol_outputs():
    evaluation = _load_ples_evaluation()
    tmp_path = ROOT / ".pytest_cache" / "ples_outputs"
    dataset = _tiny_async_dataset()
    dataset.adj = np.eye(4, dtype=np.float32)
    dataset.pos_mark = None
    dataset.tar_feas = [0]
    dataset.mean_ = np.zeros((1, 1, 1), dtype=np.float32)
    dataset.std_ = np.ones((1, 1, 1), dtype=np.float32)
    dataset.protocol_manifest = {
        "schema_version": 1,
        "protocol": "Paired Lifecycle Event Study",
        "protocol_hash": "a" * 64,
        "dataset": {"name": "TINY", "time_steps": 12, "fingerprint": "tiny"},
        "requested": {"pre_steps": 1, "post_steps": 4},
        "resolved": {"origin_count": 1, "cohort_count": 2, "cohort_size": 1},
        "phase_names": list(PHASE_ORDER),
        "metric_names": ["Join", "Fault", "LifeMacro"],
        "origins": [
            {
                "origin_id": 1,
                "block_id": 1,
                "regime": "AM",
                "event_time_index": 5,
                "event_timestamp": "2024-01-01 08:00:00",
            }
        ],
        "cohorts": [
            {"cohort_id": 1, "join_nodes": [0], "fault_nodes": [2]},
            {"cohort_id": 2, "join_nodes": [1], "fault_nodes": [3]},
        ],
    }
    loader = SimpleNamespace(dataset=dataset)
    args = SimpleNamespace(
        slide_step=1,
        test_for_changed=False,
        best_prediction_path=str(tmp_path / "predictions"),
        best_checkpoint_path="",
        exp_id="tiny-exp",
        ples_manifest_root=str(tmp_path / "protocols"),
        ples_overwrite_manifest=False,
        ples_bootstrap_samples=0,
        ples_bootstrap_seed=7,
        device="cpu",
        t_mark=False,
        pos_mark=False,
        pred_len=2,
        input_len=4,
        model="TinyModel",
    )

    result = evaluation.async_lifecycle_inference(_LastValueModel(pred_len=2), loader, args)
    output_dir = tmp_path / "predictions" / "tiny-exp" / "ples-v1-aaaaaaaaaa"

    assert set(result["main_metrics"]) == {"Join", "Fault", "LifeMacro"}
    assert (tmp_path / "protocols" / "tiny" / "protocol_manifest.json").exists()
    for filename in (
        "protocol_manifest.json",
        "paired_event_rows.csv",
        "paired_event_curves.csv",
        "paired_event_summary.json",
        "paired_event_results.txt",
        "paired_event_trajectory.png",
    ):
        assert (output_dir / filename).exists()

"""Regressions for the seven protocol issues raised in the 2026-07 audit.

Each test names the issue it pins down so a future refactor that reintroduces
the old behaviour fails here rather than silently in a reported number.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from data_provider.calendar_strata import STRATA, stratum_weights
from data_provider.ples_protocol import (
    PLES_PROTOCOL_VERSION,
    bootstrap_units,
    overlap_blocks,
)
from experiments.inductive_epoch import _element_loss, _stratum_balanced_loss

try:  # pytest puts tests/ on sys.path; a package-style run resolves the other way
    from test_lifecycle_protocol import _build_dataset, workspace_tmp  # noqa: F401
except ImportError:  # pragma: no cover
    from tests.test_lifecycle_protocol import _build_dataset, workspace_tmp  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def _build_async_dataset(tmp_path):
    """LCE/PLES test dataset over the toy series `_build_dataset` already wrote."""
    from data_provider.graph_data_loader import AsyncLifecyclePEMS

    return AsyncLifecyclePEMS(
        data_path=str(tmp_path / "toy.npy"),
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
        lcbt_manifest_root=str(tmp_path / "manifests"),
        lcbt_scaler_end_ratio=0.50,
        lcbt_test_start_ratio=0.60,
        ples_protocol_version=3,
        lce_protocol_version=3,
        ples_manifest_root=str(tmp_path / "manifests"),
        ples_overwrite_manifest=True,
        ples_pre_steps=1,
        ples_post_steps=12,
        ples_bootstrap_samples=0,
        lifecycle_training_strategy="lcbt_v3",
    )


def _row(origin_id, event_time_index, cohort_id=0, regime="Weekday/Midday", day="2026-01-01"):
    return {
        "origin_id": origin_id,
        "event_time_index": event_time_index,
        "cohort_id": cohort_id,
        "regime": regime,
        "calendar_day": day,
    }


# --- issue 5: origins whose trajectories overlap are not independent ---------


def test_overlap_blocks_chain_transitively_and_split_only_beyond_the_span():
    # 0-40-80 is a chain: 0 and 40 overlap, 40 and 80 overlap, so all three
    # share data and belong to one block even though 0 and 80 do not touch.
    assignment = overlap_blocks({1: 0, 2: 40, 3: 80, 4: 200}, span=72)
    assert assignment[1] == assignment[2] == assignment[3]
    assert assignment[4] != assignment[1]
    # Exactly `span` apart is the first independent gap.
    assert overlap_blocks({1: 0, 2: 72}, span=72) == {1: 0, 2: 1}
    assert overlap_blocks({1: 0, 2: 71}, span=72) == {1: 0, 2: 0}


def test_bootstrap_units_block_across_midnight_when_a_span_is_given():
    # Two origins 43 steps apart that a calendar day would have separated.
    rows = [
        _row(1, 280, day="2026-01-01"),
        _row(2, 323, day="2026-01-02"),
        _row(3, 900, day="2026-01-04"),
    ]
    by_day = bootstrap_units(rows)
    assert len(by_day["blocks"]) == 3

    overlapping = bootstrap_units(rows, span=72)
    assert len(overlapping["blocks"]) == 2
    blocks = {name: sorted(r["origin_id"] for r in items)
              for name, items in overlapping["blocks"].items()}
    assert sorted(blocks.values()) == [[1, 2], [3]]


def test_bootstrap_units_keep_cohorts_crossed_with_blocks():
    rows = [_row(1, 0, cohort_id=c) for c in (1, 2)] + [_row(2, 500, cohort_id=c) for c in (1, 2)]
    units = bootstrap_units(rows, span=72)
    assert units["cohort_ids"] == [1, 2]
    assert sum(len(items) for items in units["blocks"].values()) == len(rows)


# --- issue 7: the manifest has to describe the bootstrap that actually runs --


def test_lcbt_manifest_describes_the_overlap_block_bootstrap(workspace_tmp):  # noqa: F811
    dataset = _build_dataset(workspace_tmp, "train")
    manifest = dataset.get_protocol_manifest()
    assert "overlap-block" in manifest["ples"]["bootstrap"]
    assert "stratified_origin_then_cohort" not in json.dumps(manifest)


def test_evaluation_manifest_describes_the_overlap_block_bootstrap(workspace_tmp):  # noqa: F811
    _build_dataset(workspace_tmp, "train")
    dataset = _build_async_dataset(workspace_tmp)
    bootstrap = dataset.get_protocol_manifest()["ples"]["bootstrap"]
    assert "overlap-connected" in bootstrap["outer_unit"]
    assert bootstrap["design"] == "origins and cohorts are crossed, not nested"
    assert int(bootstrap["trajectory_span"]) > 0


# --- issue 6: training and the main table must optimise the same cohort tiers -


def test_training_cohort_plan_matches_the_main_table_by_default(workspace_tmp):  # noqa: F811
    dataset = _build_dataset(workspace_tmp, "train")
    assert dataset.include_stress_cohort is False
    train_plan = dataset._train_plan
    assert train_plan["stress_cohort_included"] is False
    assert tuple(train_plan["fractions"]) == tuple(dataset.validation_cohort_plan["fractions"])
    assert tuple(train_plan["weights"]) == tuple(dataset.validation_cohort_plan["weights"])

    manifest = dataset.get_protocol_manifest()
    recorded = manifest["training_cohort_plan"]
    assert recorded["stress_cohort_included"] is False
    assert recorded["fractions"] == list(manifest["validation_cohort_plan"]["fractions"])


def test_training_schedule_only_uses_main_table_cohort_sizes(workspace_tmp):  # noqa: F811
    dataset = _build_dataset(workspace_tmp, "train")
    allowed = set(dataset._train_plan["fractions"])
    scheduled = {float(item["cohort_fraction"]) for item in dataset.schedule
                 if "cohort_fraction" in item}
    assert scheduled <= allowed


# --- issue 4: the optional stratum-balanced training loss --------------------


def test_stratum_balanced_loss_is_off_unless_requested():
    import argparse

    from args import _add_arguments

    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    assert float(parser.get_default("lcbt_stratum_balanced_loss")) == 0.0


def test_stratum_balanced_loss_reweights_towards_the_evaluation_weights():
    torch.manual_seed(0)
    # Two samples in different strata, deliberately unbalanced in size so the
    # plain mean and the reweighted mean cannot coincide by accident.
    y_hat = torch.zeros(2, 3, 4, 1)
    y = torch.zeros(2, 3, 4, 1)
    y[0] = 1.0  # stratum A error 1
    y[1] = 3.0  # stratum B error 3
    valid = torch.ones_like(y, dtype=torch.bool)
    valid[1, :, 2:] = False  # stratum B contributes fewer elements

    calendar_ids = torch.tensor([0, 1])
    plain = _element_loss(y_hat, y, "mae")[valid].mean()
    balanced = _stratum_balanced_loss(y_hat, y, valid, calendar_ids, "mae")

    weights = stratum_weights()
    expected = (
        weights[STRATA[0]] * 1.0 + weights[STRATA[1]] * 3.0
    ) / (weights[STRATA[0]] + weights[STRATA[1]])
    assert balanced == pytest.approx(expected, rel=1e-6)
    assert balanced != pytest.approx(float(plain), rel=1e-6)


def test_stratum_balanced_loss_ignores_strata_absent_from_the_batch():
    y_hat = torch.zeros(1, 2, 2, 1)
    y = torch.full_like(y_hat, 2.0)
    valid = torch.ones_like(y, dtype=torch.bool)
    balanced = _stratum_balanced_loss(y_hat, y, valid, torch.tensor([3]), "mae")
    # A single stratum renormalises to itself, so it equals the plain mean.
    assert balanced == pytest.approx(2.0, rel=1e-6)


# --- issues 2 and 3: stale predictions must not be replotted as if current ---


def _write_bundle(directory, schema_version, protocol_hash="abc123"):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "ples_effects_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"schema_version": schema_version, "protocol_hash": protocol_hash}, handle)
    header = (
        "dataset,model,origin_id,calendar_day,regime,event_time_index,cohort_id,"
        "track,tau,phase,count,treated_ape_sum,control_ape_sum,treated_ae_sum,"
        "control_ae_sum,treated_se_sum,control_se_sum\n"
    )
    body = "toy,m,1,2026-01-01,Weekday/Midday,100,1,join,0,Cold,4,1,1,1,1,1,1\n"
    (directory / "ples_paired_rows.csv").write_text(header + body, encoding="utf-8")


def _run_plot(ours, baseline, output_dir):
    return subprocess.run(
        [sys.executable, "scripts/plot_counterfactual_shockwave.py",
         "--ours", str(ours), "--baseline", str(baseline),
         "--output-dir", str(output_dir), "--bootstrap-samples", "0"],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_plot_script_reads_the_v3_row_file(workspace_tmp):  # noqa: F811
    _write_bundle(workspace_tmp / "ours", PLES_PROTOCOL_VERSION)
    _write_bundle(workspace_tmp / "base", PLES_PROTOCOL_VERSION)
    result = _run_plot(workspace_tmp / "ours", workspace_tmp / "base", workspace_tmp / "out")
    # It must not die looking for the retired v1 filename.
    assert "paired_event_rows.csv" not in result.stderr
    assert "No such file" not in result.stderr


def test_plot_script_refuses_predictions_from_a_retired_schema(workspace_tmp):  # noqa: F811
    _write_bundle(workspace_tmp / "ours", 2)
    _write_bundle(workspace_tmp / "base", 2)
    result = _run_plot(workspace_tmp / "ours", workspace_tmp / "base", workspace_tmp / "out")
    assert result.returncode != 0
    assert "predate the maintained protocol" in result.stderr


def test_evaluator_requires_the_maintained_lce_version():
    from experiments.ples_evaluation import _async_lifecycle_inference
    from types import SimpleNamespace

    class _Dataset:
        def get_protocol_manifest(self):
            return {"protocol_hash": "0" * 40}

    with pytest.raises(ValueError, match="LCE v3"):
        _async_lifecycle_inference(None, _Dataset(), SimpleNamespace(lce_protocol_version=2))

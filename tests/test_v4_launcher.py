from pathlib import Path

import pytest

from scripts import build_v4_five_dataset_plan as builder
from scripts.v4_launcher import (
    LauncherRefused, assign_slots, parse_devices, plan_runs, resolve_plan_and_root,
)

ROOT = Path(__file__).resolve().parents[1]


def test_device_parsing_and_explicit_sharing():
    assert parse_devices("0,1") == ["cuda:0", "cuda:1"]
    assert assign_slots(2, ["cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1"]
    with pytest.raises(LauncherRefused, match="ALLOW_GPU_SHARING"):
        assign_slots(2, ["cuda:0"])
    assert assign_slots(2, ["cuda:0"], allow_sharing=True) == ["cuda:0"] * 2


def test_plan_enumerates_exactly_30_new_runs(tmp_path):
    plan = builder.load_plan()
    runs = plan_runs(tmp_path, plan)
    assert len(runs) == 30
    assert {run.execution_action for run in runs} == {"train"}
    assert {(run.dataset, run.model, run.seed) for run in runs} == {
        (dataset, model, seed) for dataset in plan["datasets"]
        for model in plan["models"] for seed in plan["seeds"]}


def test_launcher_uses_plan_output_root(monkeypatch):
    plan = builder.load_plan()
    monkeypatch.setattr("scripts.build_v4_readiness.assert_ready", lambda value: True)
    resolved, root = resolve_plan_and_root(builder.PLAN_PATH)
    assert resolved["plan_sha256"] == plan["plan_sha256"]
    assert root == ROOT / plan["output_root"] / plan["plan_sha256"]

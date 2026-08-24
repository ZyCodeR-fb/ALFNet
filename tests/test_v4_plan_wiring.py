from pathlib import Path

import pytest

from scripts import build_v4_five_dataset_plan as builder
from scripts.train_v4 import RunnerRefused, load_plan_authority
from scripts.v4_launcher import _make_run, build_command

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/paper/v4_five_dataset_plan.json"


def test_launcher_passes_the_exact_active_plan(tmp_path):
    run = _make_run(tmp_path, "PEMS08", "ALFNet", 32, 0)
    run.device = "cuda:0"
    command = build_command(run, tmp_path, PLAN)
    assert command[command.index("--plan") + 1] == str(PLAN)
    assert "--development_plan" not in command


def test_plan_is_clean_30_run_reproduction():
    plan = builder.assert_plan_still_holds(builder.load_plan(PLAN))
    assert plan["trial_budget"] == 30
    assert all(run["execution_action"] == "train" for run in plan["runs"])
    assert plan["execution"]["reuse_forbidden"] is True
    assert plan["training"]["slide_step"] == 1
    assert plan["endpoints"]["validation"]["cells"] == 33


def test_an_archive_plan_is_not_an_active_training_authority():
    archive = ROOT / "configs/paper/archive/v4_result_946d/plan.json"
    with pytest.raises((RunnerRefused, ValueError)):
        load_plan_authority(archive)

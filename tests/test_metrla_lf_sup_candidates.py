"""Contracts for the bounded METR-LA validation-only candidate study."""

from pathlib import Path

import pytest

from args import load_args_from_config
from experiments.v4_runner import RunnerRefused, build_v4_session, resolve_context
from layers.lifecycle_moge import (
    DISTINCT_MULTISCALE_SUPPORT_SET,
    EXPERIMENTAL_LEGACY_SUPPORT_SET,
    LEGACY_SUPPORT_NAMES,
)
from scripts import build_v4_metrla_lf_sup_plan as candidate_plan
from scripts.train_v4 import load_plan_authority
from scripts.v4_launcher import plan_runs

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "configs/paper/v4_candidates/METRLA"


def test_plan_is_exactly_five_validation_only_runs():
    # Completed evidence: its own content hash still holds, but current source
    # code is intentionally allowed to move on to the next bounded study.
    plan = candidate_plan.load_plan()
    assert plan["candidates"] == ["BASE", "LF", "SUP", "LF_SUP", "CONTROL"]
    assert plan["seed"] == 32 and plan["trial_budget"] == 5
    assert plan["selection"]["split"] == "validation"
    assert plan["selection"]["test_forbidden"] is True
    assert {run["candidate"] for run in plan["runs"]} == set(plan["candidates"])


@pytest.mark.parametrize("candidate,weight,support", [
    ("BASE", 0.0, EXPERIMENTAL_LEGACY_SUPPORT_SET),
    ("LF", 0.05, EXPERIMENTAL_LEGACY_SUPPORT_SET),
    ("SUP", 0.0, DISTINCT_MULTISCALE_SUPPORT_SET),
    ("LF_SUP", 0.05, DISTINCT_MULTISCALE_SUPPORT_SET),
])
def test_alfnet_candidate_factorial_is_exact(candidate, weight, support):
    args = load_args_from_config(CANDIDATE_DIR / f"{candidate}_ALFNet.yaml")
    recipe = args.model_family_cfg["recipe"]
    assert args.dloader_name == "METRLA" and args.seed == 32
    assert recipe["forecast_safe_relative_weight"] == weight
    assert recipe["candidate_validation_only"] is True
    assert recipe["experimental_speed_safe_relative"] is (weight > 0)
    graph = load_args_from_config(
        args.model_family_cfg["experts"]["graph"]["config_path"])
    assert graph.model_family_cfg["recipe"]["moge_support_set"] == support
    assert graph.model_family_cfg["recipe"]["candidate_validation_only"] is True


def test_final_speed_config_remains_sup_without_lf_or_candidate_flags():
    args = load_args_from_config(ROOT / "configs/paper/v4/METRLA_ALFNet.yaml")
    recipe = args.model_family_cfg["recipe"]
    assert recipe["forecast_safe_relative_weight"] == 0.0
    assert not recipe.get("candidate_validation_only", False)
    assert not recipe.get("experimental_speed_safe_relative", False)


def test_candidate_models_resolve_the_declared_support_inventory():
    legacy = build_v4_session(CANDIDATE_DIR / "BASE_ALFNet.yaml",
                              device="cpu", run_seed=32, batch_size=1,
                              num_workers=0)
    sup = build_v4_session(CANDIDATE_DIR / "SUP_ALFNet.yaml",
                           device="cpu", run_seed=32, batch_size=1,
                           num_workers=0)
    assert legacy.model.graph_expert.moge_support_set == EXPERIMENTAL_LEGACY_SUPPORT_SET
    assert legacy.model.graph_expert.resolved_support_names == LEGACY_SUPPORT_NAMES
    assert sup.model.graph_expert.moge_support_set == DISTINCT_MULTISCALE_SUPPORT_SET
    assert sup.model.graph_expert.resolved_support_names != LEGACY_SUPPORT_NAMES


def test_completed_candidate_plan_is_no_longer_a_trainer_authority():
    with pytest.raises(RunnerRefused, match="neither the active final plan"):
        load_plan_authority(
            ROOT / "configs/paper/v4_metrla_lf_sup_candidate_plan.json")


def test_launcher_enumerates_five_unique_candidate_configs(tmp_path):
    plan = candidate_plan.load_plan()
    runs = plan_runs(tmp_path, plan)
    assert len(runs) == 5
    assert len({run.name for run in runs}) == 5
    assert len({run.config for run in runs}) == 5
    assert {run.candidate for run in runs} == set(plan["candidates"])


def test_training_code_still_refuses_test_context():
    with pytest.raises(RunnerRefused, match="will not build"):
        resolve_context("METRLA", "test", 32)

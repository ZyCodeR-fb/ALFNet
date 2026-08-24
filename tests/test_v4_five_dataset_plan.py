from pathlib import Path

from scripts import build_v4_five_dataset_plan as builder

ROOT = Path(__file__).resolve().parents[1]


def test_generated_plan_matches_disk():
    disk = builder.load_plan()
    rebuilt = builder.build_plan(include_commit=False)
    rebuilt["git_commit"] = disk.get("git_commit")
    rebuilt["plan_sha256"] = builder.plan_hash(rebuilt)
    assert rebuilt == disk


def test_plan_has_five_datasets_two_models_three_seeds_and_no_reuse():
    plan = builder.load_plan()
    assert plan["datasets"] == ["PEMS03", "PEMS04", "PEMS08", "METRLA", "PEMSBAY"]
    assert plan["models"] == ["ALFNet", "MoGERNN"]
    assert plan["seeds"] == [32, 42, 52]
    assert len(plan["runs"]) == 30
    assert {run["execution_action"] for run in plan["runs"]} == {"train"}
    assert plan["status"] == "final_lf_sup_reproduction"
    assert plan["output_root"] == "outputs_v4/final_lf_sup"


def test_plan_freezes_the_selected_flow_and_speed_policy():
    policy = builder.load_plan()["training"]["selected_model_policy"]
    assert policy["selection_split"] == "validation"
    assert policy["test_used_for_selection"] is False
    assert policy["ALFNet_flow"]["name"] == "LF_SUP"
    assert policy["ALFNet_flow"]["forecast_safe_relative_weight"] == 0.05
    assert policy["ALFNet_speed"]["name"] == "SUP"
    assert policy["ALFNet_speed"]["forecast_safe_relative_weight"] == 0.0


def test_problem_definition_is_topology_known_signal_inductive():
    definition = builder.load_plan()["problem_definition"]
    assert definition["static_topology_known"] is True
    assert definition["full_node_axis_known"] is True
    assert definition["strict_node_unseen_claim"] is False


def test_completed_result_plan_is_preserved_as_an_archive():
    path = ROOT / "configs/paper/archive/v4_result_946d/plan.json"
    assert path.exists()
    assert builder.load_plan(path)["plan_sha256"].startswith("946d0b450bb10e3e")

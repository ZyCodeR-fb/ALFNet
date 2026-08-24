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


def test_completed_result_plan_is_preserved_as_an_archive():
    path = ROOT / "configs/paper/archive/v4_result_946d/plan.json"
    assert path.exists()
    assert builder.load_plan(path)["plan_sha256"].startswith("946d0b450bb10e3e")

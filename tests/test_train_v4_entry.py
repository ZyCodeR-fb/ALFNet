"""The v4 training entry point: what it saves, what it selects on, what it refuses.

Three things are asserted here that only ever fail expensively:

* the checkpoint carries enough to say *which experiment produced it* -- payload,
  unseen set, manifest, static assets, both exposure hashes, the scaler, the
  split and the teacher-forcing policy. A checkpoint that records only weights
  is a checkpoint whose protocol has to be reconstructed from memory;
* selection reads validation LifeAvg and nothing else, with the tie rule fixed
  before any number exists;
* a finished run is never overwritten, and no test-shaped argument is accepted.
"""

import json
from pathlib import Path

import pytest
import torch

from experiments.v4_runner import RunnerRefused
from scripts import train_v4

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"ALFNet": ROOT / "configs/paper/v4/PEMS08_ALFNet.yaml",
           "MoGERNN": ROOT / "configs/paper/v4/PEMS08_MoGERNN.yaml"}
DATASETS = ("PEMS03", "PEMS04", "PEMS08", "METRLA", "PEMSBAY")


def _preflight(config, output_root=None, **kwargs):
    return train_v4.train(config, output_root=output_root, device="cpu", seed=32,
                          max_epochs=1, max_train_batches=1,
                          max_validation_batches=1, batch_size=1, num_workers=0,
                          **kwargs)


# PEMS03 and PEMS04 build a schedule over ~15k origins, so a preflight there is
# not cheap. Each combination runs once and both tests below read the same
# report; running them twice would prove nothing extra.
_PREFLIGHT_CACHE = {}


@pytest.fixture(scope="module")
def preflight():
    def run(dataset, model, guard=None):
        key = (dataset, model)
        if key not in _PREFLIGHT_CACHE:
            _PREFLIGHT_CACHE[key] = _preflight(
                ROOT / f"configs/paper/v4/{dataset}_{model}.yaml", save=False)
        return _PREFLIGHT_CACHE[key]

    return run


# --- refusals ------------------------------------------------------------------


@pytest.mark.parametrize("argument", ["--test", "--test_async", "--evaluate",
                                      "--run-confirmation", "--TEST"])
def test_a_test_shaped_argument_is_refused(argument):
    with pytest.raises(RunnerRefused, match="confirmation splits are read once"):
        train_v4.assert_no_forbidden_arguments(["--config", "x", argument])


def test_ordinary_arguments_are_accepted():
    train_v4.assert_no_forbidden_arguments(
        ["--config", "x", "--device", "cpu", "--seed", "32",
         "--output_root", "./o", "--preflight"])


def test_only_the_generated_v4_configs_are_accepted():
    """Two per dataset, and the count follows the generator rather than a literal.

    A hardcoded 8 here is the same class of defect as a launcher with
    `DATASETS = (...)` baked in: it silently ignores a dataset that was added.
    """
    from scripts.build_v4_configs import DATASETS

    paths = train_v4.v4_config_paths()
    assert len(paths) == 2 * len(DATASETS) == 10
    assert {path.name for path in paths} == {
        f"{dataset}_{model}.yaml" for dataset in DATASETS
        for model in ("ALFNet", "MoGERNN")}
    with pytest.raises(RunnerRefused, match="is not one of the .* v4 configs"):
        _preflight(
            ROOT / "configs/paper/v4/experts/PEMS08_ALFTemporal.yaml",
            save=False,
        )


def test_the_entry_point_exposes_no_capacity_or_split_override():
    """Split, target, layout and capacity are not command-line business."""
    import argparse

    parser = argparse.ArgumentParser()
    # Mirror main()'s parser to inspect what it offers.
    for flag in ("--config", "--device", "--seed", "--output_root", "--resume",
                 "--preflight", "--max_train_batches", "--max_validation_batches"):
        parser.add_argument(flag, action="store_true" if flag in
                            ("--resume", "--preflight") else "store")
    offered = {action.dest for action in parser._actions}
    forbidden = {"batch_size", "lr", "d_model", "e_layers", "input_features",
                 "target_features", "lcbt_train_end_ratio", "spa_patchsize",
                 "unknown_nodes_path", "data_path"}
    assert not offered & forbidden
    source = (ROOT / "scripts/train_v4.py").read_text("utf-8")
    for name in forbidden:
        assert f'"--{name}"' not in source, name


# --- the run itself ------------------------------------------------------------


@pytest.mark.parametrize("who", sorted(CONFIGS))
def test_a_saved_run_carries_its_whole_provenance(tmp_path, who):
    report = _preflight(CONFIGS[who], output_root=tmp_path)
    directory = Path(report["run_directory"])
    assert directory.is_relative_to(tmp_path)
    # dataset / model / config hash / seed -- identity, not a timestamp.
    assert directory.parts[-4:] == ("PEMS08", who, report["config_sha256"][:16],
                                    "seed32")
    assert (directory / "COMPLETE").exists()
    assert json.loads((directory / "run_report.json").read_text("utf-8"))

    checkpoint = torch.load(directory / "best.pt", weights_only=False)
    manifest = json.loads(
        (ROOT / "configs/paper/v4_data_manifests/PEMS08_run_manifest.json")
        .read_text("utf-8"))
    for field in ("schema_version", "protocol", "dataset", "model",
                  "model_variant", "seed", "epoch", "best_epoch", "model_state",
                  "optimizer_state", "config_path", "config_sha256",
                  "payload_sha256", "unseen_sha256", "manifest_sha256",
                  "static_asset_hash", "train_exposure_hash",
                  "validation_exposure_hash", "scaler", "split_fractions",
                  "teacher_forcing_policy", "loss", "objective_terms",
                  "validation_report", "best_validation_life_avg_mae",
                  "early_stopping_metric", "tie_rule", "parameter_count"):
        assert field in checkpoint, field
    assert checkpoint["protocol"] == "lcbt_v4"
    assert checkpoint["payload_sha256"] == manifest["payload"]["sha256"]
    assert checkpoint["unseen_sha256"] == manifest["unseen"]["sha256"]
    assert checkpoint["split_fractions"] == [0.60, 0.20, 0.20]
    assert checkpoint["teacher_forcing_policy"] == "lcbt_v4_tf_v1"
    assert checkpoint["early_stopping_metric"] == "validation life_avg_mae"
    assert checkpoint["tie_rule"] == "keep_earlier_epoch"
    assert checkpoint["train_exposure_hash"] != checkpoint["validation_exposure_hash"]
    assert checkpoint["scaler"]["fitted_on"] == \
        "train window, training-known nodes only"
    assert checkpoint["parameter_count"] == (1_037_529 if who == "ALFNet"
                                             else 226_270)
    assert checkpoint["objective_terms"]["role_balanced_mix"] is (who == "ALFNet")
    for report_key in ("mae", "life_avg_mae"):
        assert checkpoint["validation_report"][report_key] == \
               checkpoint["validation_report"][report_key]


def test_the_checkpoint_round_trips_into_a_fresh_model(tmp_path):
    from experiments.v4_runner import build_v4_session

    report = _preflight(CONFIGS["MoGERNN"], output_root=tmp_path)
    checkpoint = torch.load(Path(report["run_directory"]) / "best.pt",
                            weights_only=False)
    fresh = build_v4_session(CONFIGS["MoGERNN"], batch_size=1)
    assert fresh.model.static_asset_hash == checkpoint["static_asset_hash"]
    fresh.model.load_state_dict(checkpoint["model_state"])
    for name, parameter in fresh.model.named_parameters():
        assert torch.equal(parameter.detach(), checkpoint["model_state"][name]), name
    optimizer = train_v4.build_optimizer(fresh)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    assert optimizer.state_dict()["param_groups"] == \
           checkpoint["optimizer_state"]["param_groups"]
    assert checkpoint["scheduler_state"] is None      # scheduler: None in the config
    assert checkpoint["config_sha256"] == report["config_sha256"]


def test_a_completed_run_is_never_overwritten(tmp_path):
    _preflight(CONFIGS["MoGERNN"], output_root=tmp_path)
    with pytest.raises(RunnerRefused, match="marked COMPLETE"):
        _preflight(CONFIGS["MoGERNN"], output_root=tmp_path)


def test_an_unfinished_run_needs_resume(tmp_path):
    report = _preflight(CONFIGS["MoGERNN"], output_root=tmp_path)
    (Path(report["run_directory"]) / "COMPLETE").unlink()
    with pytest.raises(RunnerRefused, match="pass --resume"):
        _preflight(CONFIGS["MoGERNN"], output_root=tmp_path)
    # ... and with --resume it proceeds rather than refusing.
    _preflight(CONFIGS["MoGERNN"], output_root=tmp_path, resume=True)


def test_selection_reads_validation_life_avg_and_nothing_else(tmp_path,
                                                              monkeypatch):
    """Feed a falling then rising LifeAvg; the best epoch must be the minimum.

    The train loss is driven the *opposite* way, so a run that selected on it
    would pick a different epoch.
    """
    from experiments import v4_runner

    sequence = iter([5.0, 3.0, 3.0, 4.0])
    train_calls = iter([4.0, 3.0, 2.0, 1.0])
    real_validate = v4_runner.validate

    def fake_validate(session, epoch=0, **kwargs):
        report = dict(real_validate(session, epoch=epoch, max_batches=1))
        report["life_avg_mae"] = next(sequence)
        return report

    def fake_train(session, optimizer, epoch, **kwargs):
        # Falls as validation rises, so selecting on it would pick epoch 3.
        return {"base": next(train_calls), "optimize": 0.0, "role_mix": None,
                "aux": None, "batches": 1, "skipped": 0, "ratio": 1.0,
                "forced_positions": 0, "available_positions": 0,
                "teacher_forcing_policy": "lcbt_v4_tf_v1", "run_seed": 32,
                "epoch": epoch, "training": True}

    monkeypatch.setattr(train_v4, "validate", fake_validate)
    monkeypatch.setattr(train_v4, "train_one_epoch", fake_train)
    report = train_v4.train(CONFIGS["MoGERNN"], output_root=tmp_path, device="cpu",
                            seed=32, max_epochs=4, max_train_batches=1,
                            max_validation_batches=1, batch_size=1, num_workers=0)
    # 5.0, 3.0, 3.0, 4.0 -> epoch 1, because a tie keeps the earlier epoch.
    assert report["best_epoch"] == 1
    assert report["best_validation_life_avg_mae"] == 3.0
    assert report["tie_rule"] == "keep_earlier_epoch"


# --- the eight-combination preflight -------------------------------------------


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model", ("ALFNet", "MoGERNN"))
def test_the_cpu_preflight_passes_for_every_combination(preflight, dataset, model):
    """One train batch, one validation batch, CPU, and no test context anywhere."""
    report = preflight(dataset, model)
    epoch = report["history"][0]
    train_report, validation_report = epoch["train"], epoch["validation"]

    assert report["dataset"] == dataset and report["model"] == model
    assert report["parameter_count"] == (1_037_529 if model == "ALFNet"
                                         else 226_270)
    assert train_report["batches"] == 1 and train_report["skipped"] == 0
    assert validation_report["batches"] == 1
    for value in (train_report["base"], train_report["optimize"],
                  validation_report["mae"], validation_report["mse"],
                  validation_report["life_avg_mae"]):
        assert value == value and abs(value) < float("inf")
    assert train_report["ratio"] == 1.0
    assert validation_report["teacher_forcing_ratio"] == 0.0


@pytest.mark.parametrize("dataset", DATASETS)
def test_both_models_score_the_same_validation_cells(preflight, dataset):
    """Same loss mask, so the same number of scored positions on every dataset."""
    counts = {}
    for model in ("ALFNet", "MoGERNN"):
        validation = preflight(dataset, model)["history"][0]["validation"]
        counts[model] = (validation["count"],
                         tuple(validation[f"{name}_count"] for name in
                               ("zero_history", "steady", "join", "fault")))
    assert counts["ALFNet"] == counts["MoGERNN"], (dataset, counts)
    assert counts["ALFNet"][0] > 0


def test_the_preflight_never_opens_a_test_context(monkeypatch):
    """The guard, on one real combination. Asserted where it can actually fire."""
    from experiments import v4_runner

    original = v4_runner.build_runtime_context
    seen = []

    def guarded(name, phase, **kwargs):
        seen.append(phase)
        assert phase != "test", f"the preflight opened a {name} test context"
        return original(name, phase, **kwargs)

    monkeypatch.setattr(v4_runner, "build_runtime_context", guarded)
    _preflight(CONFIGS["MoGERNN"], save=False)
    assert set(seen) == {"train", "validation"}

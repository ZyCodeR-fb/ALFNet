"""The ten outer v4 reproduction configs and their nested ALF experts.

A confirmation dataset proves nothing if the two models were given different
data, and the v3 tree shows how easily that happens without a check: every
PEMS08 config there declares `input_features: v` while the payload's target is
flow, because the file was copied from PEMS-BAY. Both are one channel wide, so
no shape ever disagreed.

So this file asserts on the real objects -- args parsed from the YAML on disk,
contexts built from the manifests, datasets built from the payloads -- rather
than on the text. Nothing is monkeypatched: a config that needs a runtime fix to
work is a config that is wrong.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from args import load_args_from_config
from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from experiments.v4_runner import CONFIG_DIR, MANIFEST_DIR, build_v4_session
from lib.lcbt_v4 import STRATEGY
from lib.runtime_context import build_runtime_context

ROOT = Path(__file__).resolve().parents[1]
# Five datasets: three flow (q), two speed (v). METR-LA joined at Stage
# 2-B2B; its 207-node payload, directed graph and real WGS84 coordinates
# come from the same canonical source PEMS-BAY does.
DATASETS = ("PEMS03", "PEMS04", "PEMS08", "METRLA", "PEMSBAY")
MODELS = ("ALFNet", "MoGERNN")
COMBINATIONS = [(dataset, model) for dataset in DATASETS for model in MODELS]

# Everything a config may differ on *between datasets*. Anything else differing
# is retuning, and retuning per dataset is what makes a transfer result
# meaningless.
DATASET_SPECIFIC = {
    "dloader_name", "data_path", "unknown_nodes_path",
    "input_features", "target_features", "config",
    # geometry frozen by the manifest, not a hyper-parameter
    "spa_patchsize",
}
# Derived identity, not declared configuration. `canonical_config` is a snapshot
# of every field, so comparing it would be circular -- it differs exactly when
# something else does, and its two hashes are meant to differ per dataset.
DERIVED_IDENTITY = {"canonical_config", "canonical_config_hash",
                    "legacy_canonical_config_hash"}


def config_path(dataset, model):
    return CONFIG_DIR / f"{dataset}_{model}.yaml"


# PEMS03 and PEMS04 build a schedule over ~15k origins. Each session is built
# once and shared; nothing below mutates a model.
_SESSIONS = {}


def session_for(dataset, model, batch_size=2):
    key = (dataset, model, batch_size)
    if key not in _SESSIONS:
        _SESSIONS[key] = build_v4_session(config_path(dataset, model),
                                          batch_size=batch_size)
    return _SESSIONS[key]


def _args(dataset, model):
    return load_args_from_config(str(config_path(dataset, model)), [])


@pytest.fixture(scope="module")
def manifests():
    import json

    return {dataset: json.loads(
        (MANIFEST_DIR / f"{dataset}_run_manifest.json").read_text("utf-8"))
        for dataset in DATASETS}


# --- the files exist and were generated ---------------------------------------


def test_every_outer_and_nested_config_exists():
    """Ten outer, ten nested: one pair per dataset per model."""
    for dataset, model in COMBINATIONS:
        assert config_path(dataset, model).exists(), (dataset, model)
    for dataset in DATASETS:
        for expert in ("ALFGraph", "ALFTemporal"):
            assert (CONFIG_DIR / "experts" / f"{dataset}_{expert}.yaml").exists()
    assert len(list(CONFIG_DIR.glob("*.yaml"))) == 2 * len(DATASETS) == 10
    assert len(list((CONFIG_DIR / "experts").glob("*.yaml"))) == 2 * len(DATASETS)


def test_the_five_datasets_carry_their_declared_units():
    """Three flow, two speed. The two are never averaged together."""
    units = {dataset: _args(dataset, "ALFNet").target_features
             for dataset in DATASETS}
    assert units == {"PEMS03": "q", "PEMS04": "q", "PEMS08": "q",
                     "METRLA": "v", "PEMSBAY": "v"}
    channels = {dataset: json.loads(
        (MANIFEST_DIR / f"{dataset}_run_manifest.json").read_text("utf-8")
    )["target"]["channel"] for dataset in DATASETS}
    assert channels == {"PEMS03": 0, "PEMS04": 0, "PEMS08": 0,
                        "METRLA": 2, "PEMSBAY": 2}


def test_metrla_keeps_its_canonical_graph_and_coordinates():
    manifest = json.loads(
        (MANIFEST_DIR / "METRLA_run_manifest.json").read_text("utf-8"))
    assert manifest["payload"]["shape"] == [34272, 207, 3]
    assert manifest["payload"]["adj_directed"] is True
    assert manifest["propagation_graph"]["directed"] is True
    assert manifest["positions"]["policy"] == "verified_wgs84"
    assert manifest["unseen"]["count"] == 50
    assert manifest["layout"]["algorithm"] == "kdtree_on_verified_wgs84"
    assert manifest["layout"]["spa_patchsize"] == 8
    assert manifest["time"]["start"].startswith("2012-03-01")
    assert manifest["time"]["end"].startswith("2012-06-27")
    assert manifest["files"] == ["raw_data.pkl", "unknown_nodes.npy"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_each_run_directory_holds_exactly_two_files(dataset):
    """A third file beside the payload is a third model input waiting to happen."""
    manifest = json.loads(
        (MANIFEST_DIR / f"{dataset}_run_manifest.json").read_text("utf-8"))
    directory = ROOT / manifest["run_directory"]
    assert sorted(p.name for p in directory.iterdir()) == \
           ["raw_data.pkl", "unknown_nodes.npy"]


def test_the_generator_is_deterministic():
    """Regenerating must be byte-identical, twice in a row."""
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_v4_configs.py"), "--check"],
            capture_output=True, text=True, cwd=str(ROOT))
        assert result.returncode == 0, result.stdout + result.stderr


# --- the target, path and layout each config declares --------------------------


@pytest.mark.parametrize("dataset, model", COMBINATIONS)
def test_the_config_declares_the_manifests_target_and_paths(manifests, dataset,
                                                            model):
    manifest, args = manifests[dataset], _args(dataset, model)
    target = manifest["target"]["name"]
    assert args.input_features == target and args.target_features == target
    assert Path(args.data_path).resolve() == (ROOT / manifest["payload"]["path"]).resolve()
    assert Path(args.unknown_nodes_path).resolve() == (
        ROOT / manifest["unseen"]["path"]).resolve()
    assert "data_v4" in str(args.data_path)
    assert args.lifecycle_training_strategy == STRATEGY
    assert (args.lcbt_train_end_ratio, args.lcbt_val_end_ratio,
            args.lcbt_test_start_ratio, args.lcbt_scaler_end_ratio) == \
           (0.60, 0.80, 0.80, 0.60)
    assert args.lcbt_zero_history_ratio == args.lcbt_steady_ratio == \
           args.lcbt_join_ratio == args.lcbt_fault_ratio == 0.25
    assert args.input_len == 12 and args.pred_len == 12
    assert args.stop_based == "val_lifecycle" and args.loss == "mse"


@pytest.mark.parametrize("dataset, model", COMBINATIONS)
def test_the_frozen_batch_size_and_learning_rate_are_explicit(dataset, model):
    args = _args(dataset, model)
    assert args.batch_size == 64, f"{dataset}/{model} batch_size={args.batch_size}"
    assert args.lr == (5e-05 if model == "ALFNet" else 0.001)


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_patch_layout_comes_from_the_manifest(manifests, dataset):
    nested = load_args_from_config(
        str(CONFIG_DIR / "experts" / f"{dataset}_ALFTemporal.yaml"), [])
    model_cfg = nested.model_family_cfg["model"]
    layout = manifests[dataset]["layout"]
    assert model_cfg["spa_patchsize"] == layout["spa_patchsize"]
    assert model_cfg["recur_times"] == layout["recur_times"]
    context = build_runtime_context(dataset, "train")
    assert context.layout_algorithm == layout["algorithm"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_nested_experts_name_the_same_dataset_and_target(manifests, dataset):
    target = manifests[dataset]["target"]["name"]
    for expert in ("ALFGraph", "ALFTemporal"):
        nested = load_args_from_config(
            str(CONFIG_DIR / "experts" / f"{dataset}_{expert}.yaml"), [])
        assert nested.dloader_name == dataset
        assert nested.input_features == target and nested.target_features == target
    outer = _args(dataset, "ALFNet")
    for spec in outer.model_family_cfg["experts"].values():
        assert "configs/paper/v4/experts/" in spec["config_path"], spec
        assert Path(spec["config_path"]).exists()


# --- the two models get the same run ------------------------------------------


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_two_models_differ_only_where_they_are_allowed_to(dataset):
    """Every shared field, compared field by field, with the diff reported."""
    alfnet, mogernn = _args(dataset, "ALFNet"), _args(dataset, "MoGERNN")
    shared = ("dloader_name", "data_path", "unknown_nodes_path", "input_features",
              "target_features", "input_len", "pred_len", "slide_step", "freq", "seed",
              "stop_based", "loss", "lifecycle_training_strategy",
              "lcbt_train_end_ratio", "lcbt_val_end_ratio", "lcbt_test_start_ratio",
              "lcbt_scaler_end_ratio", "lcbt_zero_history_ratio",
              "lcbt_steady_ratio", "lcbt_join_ratio", "lcbt_fault_ratio",
              "lcbt_validation_seed", "max_epoch", "patience",
              "raw_enc_in", "c_out")
    differing = {name: (getattr(alfnet, name), getattr(mogernn, name))
                 for name in shared
                 if getattr(alfnet, name) != getattr(mogernn, name)}
    assert not differing, f"{dataset}: models disagree on {differing}"
    # ALFNet's lifecycle channels are its method inputs; MoGERNN declares none.
    assert alfnet.enc_in == alfnet.raw_enc_in + 6
    assert mogernn.enc_in == mogernn.raw_enc_in
    assert not (mogernn.lifecycle_role_features
                or mogernn.lifecycle_history_age_feature
                or mogernn.lifecycle_observation_mask_feature)


@pytest.mark.parametrize("model", MODELS)
def test_one_model_is_the_same_across_datasets_outside_the_whitelist(model):
    """A per-dataset difference outside the whitelist is retuning."""
    baseline = _args("PEMS08", model)
    ignored = DATASET_SPECIFIC | DERIVED_IDENTITY | {"model_family_cfg"}
    for dataset in DATASETS:
        other = _args(dataset, model)
        differing = {
            name: (getattr(baseline, name), value)
            for name, value in vars(other).items()
            if name not in ignored and not name.startswith("_")
            and getattr(baseline, name, object()) != value
        }
        assert not differing, f"{model} {dataset} differs outside the whitelist: {differing}"


@pytest.mark.parametrize("dataset", DATASETS)
def test_both_models_share_the_static_assets_and_the_episode_design(dataset):
    """One dataset, one schedule: origins, scenarios, cohorts and loss masks."""
    sessions = {model: session_for(dataset, model) for model in MODELS}
    left, right = sessions["ALFNet"], sessions["MoGERNN"]
    assert left.train_context.static_asset_hash() == right.train_context.static_asset_hash()
    assert left.train_context.exposure_hash() == right.train_context.exposure_hash()
    assert left.validation_context.exposure_hash() == \
           right.validation_context.exposure_hash()
    assert left.scaler == right.scaler
    assert (left.train_dataset.origins == right.train_dataset.origins).all()
    assert left.run_seed == right.run_seed

    for epoch in (0, 2):
        design = {}
        for model, session in sessions.items():
            session.train_dataset.set_epoch(epoch)
            batch = next(iter(DataLoader(session.train_dataset, batch_size=4,
                                         shuffle=False)))
            design[model] = (batch["origin"].tolist(), list(batch["scenario"]),
                             batch["evidence_steps"].tolist(),
                             list(batch["calendar_regime"]),
                             batch["loss_mask"].clone(),
                             batch["target_cohort_mask"].clone(),
                             batch["decoder_available_mask"].clone())
        for index, name in enumerate(("origin", "scenario", "evidence_steps",
                                      "calendar_regime", "loss_mask",
                                      "target_cohort_mask", "decoder_available")):
            first, second = design["ALFNet"][index], design["MoGERNN"][index]
            if torch.is_tensor(first):
                assert torch.equal(first, second), (dataset, epoch, name)
            else:
                assert first == second, (dataset, epoch, name)


# --- fail closed ---------------------------------------------------------------


@pytest.mark.parametrize("dataset, model", COMBINATIONS)
def test_no_config_needs_a_runtime_patch(dataset, model):
    """Parsed straight off disk and handed to the real entry point."""
    session = session_for(dataset, model)
    assert session.args.lifecycle_training_strategy == STRATEGY
    assert session.model.static_asset_hash == \
           build_runtime_context(dataset, "train").static_asset_hash()
    expected = 1_037_529 if model == "ALFNet" else 226_270
    assert sum(p.numel() for p in session.model.parameters()) == expected


@pytest.mark.parametrize("dataset, wrong", [("PEMS08", "v"), ("PEMS03", "v"),
                                            ("PEMS04", "v"), ("PEMSBAY", "q")])
@pytest.mark.parametrize("model", MODELS)
def test_the_wrong_quantity_is_refused_before_the_model_is_built(dataset, wrong,
                                                                 model):
    """Both are one channel wide, so only the name can catch this."""
    import model.ALFNet.ALFNet as alfnet
    import model.comparison_model.MoGERNN.MoGERNN as mogernn

    module = alfnet if model == "ALFNet" else mogernn
    args = _args(dataset, model)
    args.device = "cpu"
    assert args.input_features != wrong
    args.input_features = args.target_features = wrong
    with pytest.raises(ValueError, match="target is"):
        module.Model.from_runtime_context(args, build_runtime_context(dataset, "train"))


@pytest.mark.parametrize("dataset, wrong_channel", [("PEMS03", 2), ("PEMS08", 2),
                                                    ("PEMSBAY", 0)])
def test_the_wrong_target_channel_is_refused(dataset, wrong_channel):
    import model.ALFNet.ALFNet as alfnet

    args = _args(dataset, "ALFNet")
    args.device = "cpu"
    context = build_runtime_context(dataset, "train")
    assert wrong_channel != context.target_channel
    args.target_channel = wrong_channel
    with pytest.raises(ValueError, match="refusing rather than silently winning"):
        alfnet.Model.from_runtime_context(args, context)


def test_a_nested_expert_naming_the_wrong_target_is_refused():
    """The nested config sizes the expert's input, so it is not documentation."""
    import model.ALFTemporal.ALFTemporal as alftemporal

    args = load_args_from_config(
        str(CONFIG_DIR / "experts" / "PEMS08_ALFTemporal.yaml"), [])
    args.device = "cpu"
    args.input_features = args.target_features = "v"
    with pytest.raises(ValueError, match="target is"):
        alftemporal.Model.from_runtime_context(
            args, build_runtime_context("PEMS08", "train"))


def test_the_v4_scaler_is_fitted_on_training_known_nodes_only():
    session = session_for("PEMS08", "ALFNet")
    assert session.scaler["fitted_on"] == "train window, training-known nodes only"
    assert session.validation_dataset.scaler is session.train_dataset.scaler
    alone = V4LifecycleDataset(session.validation_context,
                              session.train_dataset_payload
                              if hasattr(session, "train_dataset_payload")
                              else ROOT / "data_v4/PEMS08/raw_data.pkl")
    assert alone.scaler is not session.scaler

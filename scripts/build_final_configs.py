#!/usr/bin/env python
"""Generate the only active model/dataset YAML tree from v4 manifests."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import lcbt_v4  # noqa: E402
MANIFEST_DIR = ROOT / "configs/paper/v4_data_manifests"
DATASETS = ("PEMS03", "PEMS04", "PEMS08", "METRLA", "PEMSBAY")
MODELS = (
    "ALFNet", "MoGERNN", "HA", "FCLSTM", "STGCN", "STGODE",
    "GWNET", "STID", "STFiT", "SLPF", "PatchSTG", "STBP",
    "MCPST_FSL",
)
CONFIG_DIR = {
    "ALFNet": "ALFNet", "MoGERNN": "MoGERNN", "HA": "HA",
    "FCLSTM": "FC-LSTM", "STGCN": "STGCN", "STGODE": "STGODE",
    "GWNET": "GWNET", "STID": "STID", "STFiT": "ST-FiT",
    "SLPF": "SLPF", "PatchSTG": "PatchSTG", "STBP": "STBP",
    "MCPST_FSL": "MCPST-FSL",
}
TRAINING = {
    "ALFNet": (64, 5e-5), "MoGERNN": (64, 1e-3), "HA": (64, 1e-3),
    "FCLSTM": (16, 2e-3), "STGCN": (16, 2e-3),
    "STGODE": (16, 2e-3), "GWNET": (16, 1e-3),
    "STID": (16, 1e-3), "STFiT": (16, 2e-3),
    "SLPF": (16, 1e-3), "PatchSTG": (16, 2e-3),
    "STBP": (16, 1e-3), "MCPST_FSL": (16, 3e-4),
}
SOURCE_LOSS = {
    "ALFNet": "mse", "MoGERNN": "mse", "HA": "mae",
    "FCLSTM": "mae", "STGCN": "mae", "STGODE": "mae",
    "GWNET": "mae", "STID": "mae", "STFiT": "mae",
    "SLPF": "mae", "PatchSTG": "mae", "STBP": "mae",
    "MCPST_FSL": "mse",
}


def facts(dataset):
    manifest = json.loads(
        (MANIFEST_DIR / f"{dataset}_run_manifest.json").read_text("utf-8"))
    return {
        "dataset": dataset, "payload": manifest["payload"]["path"],
        "unseen": manifest["unseen"]["path"],
        "target": manifest["target"]["name"],
        "channel": int(manifest["target"]["channel"]),
        "nodes": int(manifest["propagation_graph"]["shape"][0]),
        "patch": int(manifest["layout"]["spa_patchsize"]),
        "recur_times": int(manifest["layout"]["recur_times"]),
    }


def lifecycle(enabled):
    return {
        "train_for_async_lifecycle": enabled,
        "lifecycle_role_features": enabled,
        "lifecycle_history_age_feature": enabled,
        "lifecycle_observation_mask_feature": enabled,
        "lifecycle_age_mode": "state_duration",
        "lifecycle_training_strategy": lcbt_v4.STRATEGY,
        "lcbt_validation_seed": 12026,
        "lcbt_train_end_ratio": lcbt_v4.TRAIN_END_RATIO,
        "lcbt_val_end_ratio": lcbt_v4.VAL_END_RATIO,
        "lcbt_test_start_ratio": lcbt_v4.TEST_START_RATIO,
        "lcbt_scaler_end_ratio": lcbt_v4.SCALER_END_RATIO,
        **lcbt_v4.SCENARIO_RATIOS,
    }


def common(dataset, model, *, lifecycle_features=False, t_mark=False):
    fact = facts(dataset)
    batch, lr = TRAINING[model]
    return {
        "runtime": {
            "model": model, "model_variant": f"{model.lower()}_base_v4",
            "model_recipe": "source_architecture_v4_adapter",
            "mode": "train", "t_mark": bool(t_mark),
        },
        "dataset": {
            "dloader_name": dataset, "data_path": f"./{fact['payload']}",
            "unknown_nodes_path": f"./{fact['unseen']}",
            "input_len": 12, "pred_len": 12, "look_back": 0,
            "slide_step": lcbt_v4.PRIMARY_SLIDE_STEP,
            "input_features": fact["target"],
            "target_features": fact["target"], "freq": "min",
        },
        "training": {
            "device": "cuda:0", "seed": 32, "max_epoch": 100,
            "patience": 10, "stop_based": "val_lifecycle",
            "batch_size": batch, "num_workers": 8,
            "loss": SOURCE_LOSS[model],
            "optimizer": "adamw", "scheduler": "None",
            "scheduler_patience": 1, "lr": lr, "weight_decay": 1e-4,
            "d_model": 64, "e_layers": 2, "d_layers": 1, "k_hop": 1,
        },
        "lifecycle": lifecycle(lifecycle_features),
    }


def family(model, fact):
    if model == "HA":
        return {"model": {"channel": None, "reverse": False}}
    if model == "FCLSTM":
        return {"model": {"embed_dim": 64, "hidden_dim": 64,
                           "end_dim": 12, "num_layer": 2, "dropout": 0.1},
                "recipe": {"graph_learning": False, "dynamic_graph": False,
                           "dynamic_mix": False, "aug_node": False,
                           "sim_loss": False, "fst_loss": False,
                           "is_vae": False}}
    if model in {"STGCN", "STFiT"}:
        enabled = model == "STFiT"
        return {"model": {"hidden_channels": 64, "kernel_size": 3,
                           "K": 3, "normalization": "sym", "bias": True},
                "recipe": {"graph_learning": enabled,
                           "dynamic_graph": enabled, "dynamic_mix": enabled,
                           "aug_node": enabled, "sim_loss": enabled,
                           "fst_loss": enabled, "is_vae": False,
                           "anchor_lambda": 0.5, "sparse_threshold": 0.9,
                           "gumbel_softmax": True, "gumbel_tau": 1.0,
                           "vae_loss_weight": 2.0}}
    if model == "STGODE":
        return {"model": {"window_size": 24, "horizon": 12},
                "recipe": {"semantic_graph_policy":
                           "topology_row_cosine_no_dynamic_signal"}}
    if model == "GWNET":
        return {"model": {"adj_type": "doubletransition", "adp_adj": True,
                           "dropout": 0.3, "residual_channels": 32,
                           "dilation_channels": 32, "skip_channels": 256,
                           "end_channels": 512, "kernel_size": 2,
                           "blocks": 4, "layers": 2}}
    if model == "STID":
        return {"model": {"node_dim": 32, "embed_dim": 32,
                           "num_layer": 3, "temp_dim_tid": 32,
                           "temp_dim_diw": 32, "time_of_day_size": 1440,
                           "day_of_week_size": 7, "if_T_i_D": True,
                           "if_D_i_W": False, "if_node": True}}
    if model == "SLPF":
        return {"model": {"embed_dim": 64, "num_layer": 3}}
    if model == "PatchSTG":
        return {"model": {"layers": 5, "tem_patchsize": 12,
                           "tem_patchnum": 1,
                           "recur_times": fact["recur_times"],
                           "spa_patchsize": fact["patch"], "factors": 1,
                           "input_dims": 64, "node_dims": 32,
                           "tod_dims": 32, "dow_dims": 32,
                           "tod": 288, "dow": 7}}
    if model == "STBP":
        return {"dropout": 0.0,
                "model": {"in_channel": 12, "out_channel": 12,
                           "hidden_channel": 192}}
    if model == "MCPST_FSL":
        return {"training": {"test_dataset": fact["dataset"].lower()},
                "model": {"input_dim": 1, "hidden_dim": 64,
                           "pred_len": 12, "spatial_dim": 4,
                           "num_diffusion_steps": 3,
                           "num_oscillator_steps": 4,
                           "num_eigen_vectors": 6,
                           "node_minibatch_cpu": 2048,
                           "node_minibatch_gpu": 1024}}
    return {}


def alfnet(dataset):
    cfg = common(dataset, "ALFNet", lifecycle_features=True, t_mark=True)
    cfg["runtime"].update({"model_variant": "alfnet_unified_base_v4",
                           "model_recipe": "lifecycle_fusion"})
    cfg["family"] = {
        "experts": {
            "graph": {"model": "ALFGraph",
                      "config_path": f"./configs/ALFGraph/{dataset}.yaml",
                      "checkpoint_path": "", "load_checkpoint": False},
            "patch": {"model": "ALFTemporal",
                      "config_path": f"./configs/ALFTemporal/{dataset}.yaml",
                      "checkpoint_path": "", "load_checkpoint": False},
        },
        "model": {"state_hidden_dim": 32, "state_dim": 32,
                  "gate_hidden_dim": 32, "graph_refiner_hidden_dim": 32},
        "recipe": {"graph_expert_lr": 5e-4, "patch_expert_lr": 1e-4,
                   "gate_lr": 5e-5, "fusion_mode": "film",
                   "forecast_loss_mode": "role_balanced_mix",
                   "forecast_role_balanced_beta": 0.1,
                   "graph_diffusion_steps": 1, "graph_diffusion_scale": 0.05,
                   "graph_diffusion_role_scale": {
                       "aas": 0.0, "nas": 1.0, "vs": 1.0, "fs": 0.25}},
    }
    return cfg


def mogernn(dataset):
    cfg = common(dataset, "MoGERNN")
    cfg["runtime"]["model_variant"] = "mg_v4"
    cfg["training"].update({"mean_expert": True, "weight_expert": True,
                            "max_expert": True, "min_expert": True,
                            "diffusion_expert": True,
                            "num_used_experts": 3})
    return cfg


def alfgraph(dataset):
    cfg = common(dataset, "ALFNet", lifecycle_features=True)
    cfg["runtime"].update({"model": "ALFGraph",
                           "model_variant": "alfgraph_unified_base_v4",
                           "model_recipe": "lcbt_v4_graph"})
    cfg["family"] = {"recipe": {
        "use_grin_input_imputer": False, "grin_hidden_dim": 64,
        "grin_ff_dim": 64, "grin_layers": 1, "grin_kernel_size": 2,
        "grin_decoder_order": 1, "grin_merge": "mlp",
        "grin_dropout": 0.0, "grin_ff_dropout": 0.0,
        "grin_layer_norm": False, "grin_use_extra_features": True,
        "grin_use_role_mask": True, "grin_cached_support": False,
        "decoder_output_mode": "residual_delta",
        "decoder_residual_scale": 0.3, "graph_conv": "lifecycle_moge",
        "moge_support_set": "unified_base_legacy5_v1", "moge_top_k": 1,
        "moge_router_mode": "soft", "moge_noisy_gating": True,
        "moge_load_balance_coef": 0.01}}
    return cfg


def alftemporal(dataset):
    fact = facts(dataset)
    cfg = common(dataset, "ALFNet", lifecycle_features=True, t_mark=True)
    cfg["runtime"].update({"model": "ALFTemporal", "model_variant": "cond1",
                           "model_recipe": "filtered_patch"})
    cfg["family"] = {"model": {
        "lifecycle_input": True, "emit": "features", "layers": 1,
        "tem_patchsize": 12, "tem_patchnum": 1,
        "recur_times": fact["recur_times"], "spa_patchsize": fact["patch"],
        "factors": 1, "crossformer_router_factor": 4,
        "crossformer_dropout": 0.1, "input_dims": 64, "node_dims": 32,
        "tod_dims": 32, "dow_dims": 32, "tod": 288, "dow": 7}}
    return cfg


def render():
    files = {}
    for dataset in DATASETS:
        fact = facts(dataset)
        files[ROOT / "configs/ALFNet" / f"{dataset}.yaml"] = alfnet(dataset)
        files[ROOT / "configs/MoGERNN" / f"{dataset}.yaml"] = mogernn(dataset)
        files[ROOT / "configs/ALFGraph" / f"{dataset}.yaml"] = alfgraph(dataset)
        files[ROOT / "configs/ALFTemporal" / f"{dataset}.yaml"] = alftemporal(dataset)
        for model in MODELS[2:]:
            cfg = common(dataset, model,
                         t_mark=model in {"STID", "SLPF", "PatchSTG"})
            cfg["family"] = family(model, fact)
            files[ROOT / "configs" / CONFIG_DIR[model] /
                  f"{dataset}.yaml"] = cfg
    header = ("# GENERATED by scripts/build_final_configs.py -- do not edit.\n"
              "# Active v4 protocol; source architecture with documented "
              "topology-known adaptation.\n")
    return {path: header + OmegaConf.to_yaml(OmegaConf.create(payload),
                                              resolve=True)
            for path, payload in files.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path, text in sorted(render().items()):
        if args.check:
            if not path.exists() or path.read_text("utf-8") != text:
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, "utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        raise SystemExit("generated configs differ: " + ", ".join(stale))
    if args.check:
        print(f"{len(render())} final configs hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

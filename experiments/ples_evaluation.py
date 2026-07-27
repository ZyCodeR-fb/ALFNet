import csv
import json
import os

import numpy as np
import torch

from data_provider.ples_protocol import (
    MAIN_TABLE_ROWS,
    PHASE_ORDER,
    PLES_EFFECT_NAMES,
    bootstrap_paired_effects,
    phase_for_tau_v2,
    summarize_lce_rows,
    summarize_paired_rows,
)
from utils.model_output import unpack_model_output


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value
def _paired_error_sums(treated, control, target, nodes):
    if not nodes:
        return None
    treated_part = treated[..., nodes, :]
    control_part = control[..., nodes, :]
    target_part = target[..., nodes, :]
    valid = torch.isfinite(target_part) & torch.isfinite(treated_part) & torch.isfinite(control_part)
    if not torch.any(valid):
        return None
    target_valid = target_part[valid]
    treated_diff = treated_part[valid] - target_valid
    control_diff = control_part[valid] - target_valid
    denominator = torch.clamp(torch.abs(target_valid), min=1e-6)
    return {
        "count": int(valid.sum().item()),
        "treated_ape_sum": float(torch.sum(torch.abs(treated_diff) / denominator).item()),
        "control_ape_sum": float(torch.sum(torch.abs(control_diff) / denominator).item()),
        "treated_ae_sum": float(torch.sum(torch.abs(treated_diff)).item()),
        "control_ae_sum": float(torch.sum(torch.abs(control_diff)).item()),
        "treated_se_sum": float(torch.sum(treated_diff**2).item()),
        "control_se_sum": float(torch.sum(control_diff**2).item()),
    }


def _inverse_transform(dataset, tensor):
    std = torch.as_tensor(dataset.std_[..., dataset.tar_feas], dtype=tensor.dtype, device=tensor.device)
    mean = torch.as_tensor(dataset.mean_[..., dataset.tar_feas], dtype=tensor.dtype, device=tensor.device)
    return tensor * std + mean


def _stack_samples(samples, item_index):
    return np.stack([sample[item_index] for sample in samples], axis=0)


def _predict_samples(model, samples, dataset, args, adj, pos_mark):
    x_enc = torch.as_tensor(_stack_samples(samples, 0), dtype=torch.float32, device=args.device)
    x_enc = torch.nan_to_num(x_enc, nan=0.0)
    enc_t_mark = None
    if args.t_mark:
        enc_t_mark = torch.as_tensor(_stack_samples(samples, 2), dtype=torch.float32, device=args.device)
    dec_t_mark = torch.as_tensor(_stack_samples(samples, 3), dtype=torch.float32, device=args.device)
    target = torch.as_tensor(_stack_samples(samples, 1), dtype=torch.float32, device=args.device)
    target = target[:, -args.pred_len :]
    with torch.no_grad():
        output = model(adj, x_enc, enc_t_mark, pos_mark, None, dec_t_mark, 0)
        prediction, metadata = unpack_model_output(output)
    return (
        _inverse_transform(dataset, prediction).detach().cpu(),
        _inverse_transform(dataset, target).detach().cpu(),
        metadata,
    )


def _graph_weight(metadata, sample_index, nodes):
    routing = metadata.get("route_routing")
    if not isinstance(routing, dict) or "alpha" not in routing or not nodes:
        return None
    alpha = routing["alpha"].detach()
    values = alpha[sample_index, :, nodes, :]
    values = values[torch.isfinite(values)]
    return None if not len(values) else float(values.mean().item())


def _write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def _flatten_curve_rows_v2(curves):
    rows = []
    for curve in curves:
        row = {
            "track": curve["track"],
            "tau": curve["tau"],
            "phase": curve["phase"],
            "cell_count": curve["cell_count"],
            "target_count": curve["target_count"],
            "effect_name": curve["effect_name"],
            "graph_weight": curve["graph_weight"],
            "patch_residual_weight": curve["patch_residual_weight"],
        }
        for metric in ("MAPE", "MAE", "RMSE"):
            token = metric.lower()
            row[f"treated_{token}"] = curve["treated"][metric]
            row[f"control_{token}"] = curve["control"][metric]
            row[f"effect_{token}"] = curve["effect"][metric]
        rows.append(row)
    return rows


def _load_training_trace(args):
    checkpoint_dir = getattr(args, "best_checkpoint_path", None)
    loaded_checkpoint = getattr(args, "loaded_checkpoint_metadata", None)
    if loaded_checkpoint:
        checkpoint_dir = loaded_checkpoint.get("checkpoint_dir") or checkpoint_dir
    training_summary = None
    if checkpoint_dir:
        summary_path = os.path.join(checkpoint_dir, "training_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                training_summary = json.load(handle)
    summary_hash = (training_summary or {}).get("training_protocol_hash")
    checkpoint_hash = (loaded_checkpoint or {}).get("training_protocol_hash")
    if summary_hash and checkpoint_hash and summary_hash != checkpoint_hash:
        raise ValueError(
            "Checkpoint and training summary disagree on training_protocol_hash"
        )
    return {
        "checkpoint_dir": checkpoint_dir,
        "training_summary": training_summary,
        "loaded_checkpoint": loaded_checkpoint,
        "training_protocol_hash": checkpoint_hash or summary_hash,
    }


def _format_training_trace(training_trace):
    training_summary = training_trace.get("training_summary") or {}
    loaded_checkpoint = training_trace.get("loaded_checkpoint") or {}
    finished_epoch = training_summary.get("finished_epoch", "NA")
    max_epoch = training_summary.get("max_epoch") or loaded_checkpoint.get("max_epoch")
    finished = f"{finished_epoch}/{max_epoch}" if max_epoch is not None else str(finished_epoch)
    parts = [f"finished_epoch={finished}"]
    for label, key in (
        ("stop_reason", "stop_reason"),
        ("best_val_epoch", "best_val_metric_epoch"),
        ("best_train_epoch", "best_train_metric_epoch"),
    ):
        if training_summary.get(key) is not None:
            parts.append(f"{label}={training_summary[key]}")
    if loaded_checkpoint.get("loaded_checkpoint_epoch") is not None:
        parts.append(f"loaded_checkpoint_epoch={loaded_checkpoint['loaded_checkpoint_epoch']}")
    if loaded_checkpoint.get("loaded_checkpoint_metric") is not None:
        parts.append(f"loaded_checkpoint_metric={loaded_checkpoint['loaded_checkpoint_metric']:.6g}")
    return "TrainingTrace: " + " | ".join(parts)




def _write_manifest(dataset, args, output_dir):
    manifest = dataset.get_protocol_manifest()
    root = os.path.abspath(getattr(args, "ples_manifest_root", "./protocol_manifests"))
    canonical_path = os.path.join(
        root,
        str(manifest["dataset"]["name"]),
        "lifecycle_v3",
        "protocol_manifest.json",
    )
    if not os.path.exists(canonical_path):
        raise FileNotFoundError(f"Lifecycle v3 manifest not found: {canonical_path}")
    with open(canonical_path, "r", encoding="utf-8") as handle:
        existing = json.load(handle)
    if existing.get("protocol_hash") != manifest.get("protocol_hash"):
        raise ValueError(f"Lifecycle v3 manifest mismatch at {canonical_path}")
    result_path = os.path.join(output_dir, "protocol_manifest.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(manifest), handle, ensure_ascii=False, indent=2)
    return canonical_path, result_path


def _write_metric_table(output_dir, stem, metrics, row_order, row_label):
    rows = [
        {
            row_label: name,
            "MAPE": metrics[name]["MAPE"],
            "MAE": metrics[name]["MAE"],
            "RMSE": metrics[name]["RMSE"],
        }
        for name in row_order
    ]
    _write_csv(
        os.path.join(output_dir, f"{stem}.csv"),
        rows,
        [row_label, "MAPE", "MAE", "RMSE"],
    )
    return rows




def _async_lifecycle_inference(model, dataset, args):
    manifest = dataset.get_protocol_manifest()
    if int(getattr(args, "lce_protocol_version", 0)) != 3:
        raise ValueError("The maintained evaluation protocol requires LCE v3")
    protocol_token = manifest["protocol_hash"][:10]
    output_dir = os.path.join(
        args.best_prediction_path,
        args.exp_id,
        f"lifecycle-v3-{protocol_token}",
    )
    os.makedirs(output_dir, exist_ok=True)
    canonical_manifest_path, result_manifest_path = _write_manifest(
        dataset,
        args,
        output_dir,
    )

    adj_np, pos_np = dataset.get_adj_and_pos_mark()
    adj = torch.as_tensor(adj_np, dtype=torch.float32, device=args.device)
    pos_mark = None
    if args.pos_mark and pos_np is not None:
        pos_mark = torch.as_tensor(pos_np, dtype=torch.float32, device=args.device)

    rows = []
    cohorts = manifest["cohorts"]
    pre_steps = int(manifest["requested"]["pre_steps"])
    post_steps = int(manifest["requested"]["post_steps"])
    pre_prediction_max_abs = 0.0
    model.eval()
    print("-> Start LCE v3 + PLES v3 inference...")
    for origin_index, origin in enumerate(manifest["origins"], start=1):
        for tau in range(-pre_steps, post_steps + 1):
            control_sample = dataset.build_ples_sample(
                origin,
                cohorts[0],
                "join",
                "control",
                tau,
            )
            samples = [control_sample]
            sample_index = {}
            for track in ("join", "fault"):
                for cohort in cohorts:
                    cohort_id = int(cohort["cohort_id"])
                    treatment = dataset.build_ples_sample(
                        origin,
                        cohort,
                        track,
                        "treatment",
                        tau,
                    )
                    if tau < 0 and not np.array_equal(
                        treatment[0],
                        control_sample[0],
                        equal_nan=True,
                    ):
                        raise ValueError(
                            "PLES v3 pre-event treatment/control inputs differ; "
                            f"origin={origin['origin_id']} track={track} cohort={cohort_id} tau={tau}"
                        )
                    sample_index[(track, cohort_id)] = len(samples)
                    samples.append(treatment)

            predictions, targets, metadata = _predict_samples(
                model,
                samples,
                dataset,
                args,
                adj,
                pos_mark,
            )
            control_prediction = predictions[0]
            control_target = targets[0]
            for track in ("join", "fault"):
                phase = phase_for_tau_v2(track, tau, int(args.input_len))
                for cohort in cohorts:
                    cohort_id = int(cohort["cohort_id"])
                    nodes = cohort["join_nodes"] if track == "join" else cohort["fault_nodes"]
                    treated_index = sample_index[(track, cohort_id)]
                    treated_prediction = predictions[treated_index]
                    if not torch.allclose(
                        targets[treated_index],
                        control_target,
                        equal_nan=True,
                    ):
                        raise ValueError(
                            "PLES v3 paired samples do not share the same prediction target"
                        )
                    if tau < 0 and nodes:
                        difference = torch.abs(
                            treated_prediction[..., nodes, :]
                            - control_prediction[..., nodes, :]
                        )
                        finite = difference[torch.isfinite(difference)]
                        if len(finite):
                            pre_prediction_max_abs = max(
                                pre_prediction_max_abs,
                                float(finite.max().item()),
                            )
                    sums = _paired_error_sums(
                        treated_prediction,
                        control_prediction,
                        control_target,
                        nodes,
                    )
                    if sums is None:
                        continue
                    graph_weight = _graph_weight(metadata, treated_index, nodes)
                    rows.append(
                        {
                            "dataset": manifest["dataset"]["name"],
                            "model": args.model,
                            "origin_id": int(origin["origin_id"]),
                            "calendar_day": origin.get("calendar_day"),
                            "regime": origin["regime"],
                            "event_time_index": int(origin["event_time_index"]),
                            "event_timestamp": origin["event_timestamp"],
                            "cohort_id": cohort_id,
                            "cohort_fraction": float(cohort.get("fraction", 0.0)),
                            "cohort_style": cohort.get("style"),
                            "track": track,
                            "tau": tau,
                            "phase": phase,
                            "node_count": len(nodes),
                            "graph_weight": graph_weight,
                            "patch_residual_weight": (
                                None if graph_weight is None else 1.0 - graph_weight
                            ),
                            **sums,
                        }
                    )
        print(
            f"   origin={origin_index:03d}/{len(manifest['origins']):03d} | "
            f"regime={origin['regime']} | time={origin['event_timestamp']}"
        )

    if pre_prediction_max_abs > 1e-5:
        raise ValueError(
            "PLES v3 pre-event treatment/control predictions differ: "
            f"max_abs={pre_prediction_max_abs:.6g}"
        )
    main_cohort_ids = manifest["resolved"]["main_cohort_ids"]
    lce_summary = summarize_lce_rows(
        rows,
        input_len=args.input_len,
        main_cohort_ids=main_cohort_ids,
    )
    ples_summary = summarize_paired_rows(
        rows,
        input_len=args.input_len,
        main_cohort_ids=main_cohort_ids,
    )
    pre_metric_difference = float(
        ples_summary["pre_event_max_abs_mape_difference"]
    )
    if pre_metric_difference > 1e-7:
        raise ValueError(
            "PLES v3 pre-event paired MAPE difference is non-zero: "
            f"max_abs={pre_metric_difference:.6g}"
        )
    bootstrap = bootstrap_paired_effects(
        rows,
        input_len=args.input_len,
        samples=int(getattr(args, "ples_bootstrap_samples", 2000)),
        seed=int(getattr(args, "ples_bootstrap_seed", 2026)),
        main_cohort_ids=main_cohort_ids,
        trajectory_span=pre_steps + int(args.input_len) + post_steps + int(args.pred_len),
    )
    training_trace = _load_training_trace(args)
    required_training_hash = manifest.get("training_protocol_hash")
    checkpoint_training_hash = training_trace.get("training_protocol_hash")
    formal_protocol_match = bool(
        checkpoint_training_hash
        and checkpoint_training_hash == required_training_hash
    )
    if not formal_protocol_match:
        print(
            "Warning: checkpoint is not proven to come from this frozen LCBT v3 "
            "training manifest; results are diagnostic-only and the three-seed "
            "collector will reject them."
        )
    common = {
        "schema_version": 3,
        "protocol": manifest["protocol"],
        "protocol_hash": manifest["protocol_hash"],
        "training_protocol_hash": checkpoint_training_hash,
        "required_training_protocol_hash": required_training_hash,
        "formal_protocol_match": formal_protocol_match,
        "lce_protocol_hash": manifest["lce_protocol_hash"],
        "ples_protocol_hash": manifest["ples_protocol_hash"],
        "model": args.model,
        "model_variant": getattr(args, "model_variant", "default"),
        "seed": int(getattr(args, "seed", -1)),
        "exp_id": args.exp_id,
        "dataset": manifest["dataset"],
        "canonical_manifest_path": canonical_manifest_path,
        "result_manifest_path": result_manifest_path,
        "training_trace": training_trace,
        "row_count": len(rows),
    }
    main_payload = {
        **common,
        "table": "LCE v3 main",
        "aggregation": manifest["lce"]["aggregation"],
        "metrics": lce_summary["main_metrics"],
        "large_event_metrics": lce_summary["large_event_metrics"],
        "main_cohort_ids": main_cohort_ids,
        "large_event_cohort_ids": manifest["resolved"]["large_event_cohort_ids"],
    }
    phase_payload = {
        **common,
        "table": "LCE v3 phases",
        "metrics": lce_summary["phase_metrics"],
    }
    trajectory_payload = {
        **common,
        "table": "LCE v3 evidence trajectories",
        "metrics": lce_summary["trajectory_metrics"],
    }
    effects_payload = {
        **common,
        "table": "PLES v3 paired effects",
        "paired_effects": ples_summary["paired_effects"],
        "paired_effect_intervals": bootstrap,
        "pre_event_max_abs_input_difference": 0.0,
        "pre_event_max_abs_prediction_difference": pre_prediction_max_abs,
        "pre_event_max_abs_mape_difference": pre_metric_difference,
    }

    row_fields = [
        "dataset", "model", "origin_id", "calendar_day", "regime", "event_time_index",
        "event_timestamp", "cohort_id", "cohort_fraction", "cohort_style", "track", "tau",
        "phase", "node_count", "count", "graph_weight", "patch_residual_weight",
        "treated_ape_sum", "control_ape_sum", "treated_ae_sum", "control_ae_sum",
        "treated_se_sum", "control_se_sum",
    ]
    _write_csv(os.path.join(output_dir, "ples_paired_rows.csv"), rows, row_fields)
    curve_rows = _flatten_curve_rows_v2(ples_summary["curves"])
    curve_fields = [
        "track", "tau", "phase", "cell_count", "target_count", "effect_name",
        "treated_mape", "control_mape", "effect_mape", "treated_mae", "control_mae",
        "effect_mae", "treated_rmse", "control_rmse", "effect_rmse", "graph_weight",
        "patch_residual_weight",
    ]
    _write_csv(os.path.join(output_dir, "ples_curves.csv"), curve_rows, curve_fields)
    main_rows = _write_metric_table(
        output_dir,
        "lce_main_summary",
        lce_summary["main_metrics"],
        MAIN_TABLE_ROWS,
        "group",
    )
    phase_rows = _write_metric_table(
        output_dir,
        "lce_phase_summary",
        lce_summary["phase_metrics"],
        PHASE_ORDER,
        "phase",
    )
    trajectory_rows = [
        {"track": track, **cell}
        for track, cells in lce_summary["trajectory_metrics"].items()
        for cell in cells
    ]
    _write_csv(
        os.path.join(output_dir, "lce_trajectory_summary.csv"),
        trajectory_rows,
        ["track", "k", "visibility", "MAPE", "MAE", "RMSE"],
    )
    effect_rows = [
        {
            "effect": name,
            "metric": metric,
            "point": bootstrap[metric][name]["point"],
            "ci_low": bootstrap[metric][name]["low"],
            "ci_high": bootstrap[metric][name]["high"],
            "direction": "higher_is_better" if name == "JoinLift" else "lower_is_better",
        }
        for metric in ("MAPE", "MAE")
        for name in PLES_EFFECT_NAMES
    ]
    _write_csv(
        os.path.join(output_dir, "ples_effects_summary.csv"),
        effect_rows,
        ["effect", "metric", "point", "ci_low", "ci_high", "direction"],
    )
    for filename, payload in (
        ("lce_main_summary.json", main_payload),
        ("lce_phase_summary.json", phase_payload),
        ("lce_trajectory_summary.json", trajectory_payload),
        ("ples_effects_summary.json", effects_payload),
    ):
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)

    trace_line = _format_training_trace(training_trace)
    hash_line = (
        "ProtocolHashes: "
        f"training={common['training_protocol_hash']} | "
        f"lce={common['lce_protocol_hash']} | ples={common['ples_protocol_hash']}"
    )
    main_lines = [trace_line, hash_line] + [
        f"{row['group']}: MAPE={row['MAPE']:.4f} | MAE={row['MAE']:.4f} | RMSE={row['RMSE']:.4f}"
        for row in main_rows
    ]
    phase_lines = [trace_line, hash_line] + [
        f"{row['phase']}: MAPE={row['MAPE']:.4f} | MAE={row['MAE']:.4f} | RMSE={row['RMSE']:.4f}"
        for row in phase_rows
    ]
    effect_lines = [
        trace_line,
        hash_line,
        (
            "PreEventCheck: "
            f"prediction_max_abs={pre_prediction_max_abs:.6g} | "
            f"mape_difference_max_abs={pre_metric_difference:.6g}"
        ),
    ]
    for row in effect_rows:
        label = f"{row['effect']}[{row['metric']}]"
        if row["ci_low"] is None or row["ci_high"] is None:
            effect_lines.append(f"{label}={row['point']:.4f} | 95% CI=NA")
        else:
            effect_lines.append(
                f"{label}={row['point']:.4f} | "
                f"95% CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            )
    for filename, lines in (
        ("lce_main_results.txt", main_lines),
        ("lce_phase_results.txt", phase_lines),
        ("ples_effects_results.txt", effect_lines),
    ):
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    print(f"LCE: dataset={manifest['dataset']['name']} | origins={len(manifest['origins'])} | cohorts={len(cohorts)}")
    for line in main_lines:
        print(line)
    print("PLES v3 paired effects:")
    for line in effect_lines[2:]:
        print(line)
    print(f"-> LCE/PLES v3 results saved to {output_dir}")
    return {
        **common,
        "lce": {
            "main_metrics": lce_summary["main_metrics"],
            "large_event_metrics": lce_summary["large_event_metrics"],
            "phase_metrics": lce_summary["phase_metrics"],
            "trajectory_metrics": lce_summary["trajectory_metrics"],
        },
        "ples": {
            "paired_effects": ples_summary["paired_effects"],
            "paired_effect_intervals": bootstrap,
            "curves": ples_summary["curves"],
        },
    }


def async_lifecycle_inference(model, data_loader, args, criterions=None):
    if int(args.slide_step) != 1:
        raise ValueError("LCE v3/PLES v3 requires slide_step=1")
    return _async_lifecycle_inference(model, data_loader.dataset, args)

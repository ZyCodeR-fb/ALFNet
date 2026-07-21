import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch

from data_provider.ples_protocol import PHASE_ORDER, phase_for_tau, summarize_phase_metrics
from utils.model_output import unpack_model_output


ERROR_FIELDS = ("ape_sum", "ae_sum", "se_sum")
EFFECT_NAMES = ("JoinBenefitAUC", "FaultShockAUC", "PeakFaultShock")


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


def _safe_token(value):
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_").lower()
    return token or "dataset"


def metric_from_sums(sums, prefix):
    count = int(sums.get("count", 0))
    if count <= 0:
        return None
    return {
        "MAPE": float(sums[f"{prefix}_ape_sum"]) / count * 100.0,
        "MAE": float(sums[f"{prefix}_ae_sum"]) / count,
        "RMSE": float(np.sqrt(float(sums[f"{prefix}_se_sum"]) / count)),
    }


def _sum_rows(rows):
    totals = {"count": 0}
    for prefix in ("treated", "control"):
        for field in ERROR_FIELDS:
            totals[f"{prefix}_{field}"] = 0.0
    for row in rows:
        totals["count"] += int(row["count"])
        for prefix in ("treated", "control"):
            for field in ERROR_FIELDS:
                totals[f"{prefix}_{field}"] += float(row[f"{prefix}_{field}"])
    return totals


def _curve_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["track"], int(row["tau"]), row["phase"])].append(row)

    curves = []
    for (track, tau, phase), cell_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        totals = _sum_rows(cell_rows)
        treated = metric_from_sums(totals, "treated")
        control = metric_from_sums(totals, "control")
        if treated is None or control is None:
            continue
        graph_weights = [float(row["graph_weight"]) for row in cell_rows if row.get("graph_weight") not in {None, ""}]
        curves.append(
            {
                "track": track,
                "tau": tau,
                "phase": phase,
                "count": totals["count"],
                "treated": treated,
                "control": control,
                "ACE": {key: treated[key] - control[key] for key in treated},
                "graph_weight": float(np.mean(graph_weights)) if graph_weights else None,
                "patch_residual_weight": float(1.0 - np.mean(graph_weights)) if graph_weights else None,
            }
        )
    return curves


def _paired_effects(curves):
    post = [row for row in curves if row["tau"] >= 0]
    join_ace = [row["ACE"]["MAPE"] for row in post if row["track"] == "join"]
    fault_ace = [row["ACE"]["MAPE"] for row in post if row["track"] == "fault"]
    if not join_ace or not fault_ace:
        raise ValueError("PLES paired effects require post-event Join and Fault curves")
    return {
        "JoinBenefitAUC": float(-np.mean(join_ace)),
        "FaultShockAUC": float(np.mean(fault_ace)),
        "PeakFaultShock": float(np.max(fault_ace)),
    }


def summarize_paired_rows(rows):
    phase_metrics = {}
    for phase in PHASE_ORDER:
        phase_rows = [row for row in rows if row["phase"] == phase and int(row["tau"]) >= 0]
        totals = _sum_rows(phase_rows)
        metrics = metric_from_sums(totals, "treated")
        if metrics is None:
            raise ValueError(f"PLES phase has no valid targets: {phase}")
        phase_metrics[phase] = metrics
    curves = _curve_rows(rows)
    return {
        "phase_metrics": phase_metrics,
        "main_metrics": summarize_phase_metrics(phase_metrics),
        "paired_effects": _paired_effects(curves),
        "curves": curves,
    }


def bootstrap_paired_effects(rows, samples=2000, seed=2026):
    point = summarize_paired_rows(rows)["paired_effects"]
    if samples <= 0:
        return {name: {"point": point[name], "low": None, "high": None} for name in EFFECT_NAMES}

    units = defaultdict(lambda: defaultdict(list))
    for row in rows:
        units[int(row["origin_id"])][int(row["cohort_id"])].append(row)
    origin_ids = sorted(units)
    if not origin_ids:
        raise ValueError("cannot bootstrap empty PLES rows")

    rng = np.random.default_rng(seed)
    draws = {name: np.empty(samples, dtype=np.float64) for name in EFFECT_NAMES}
    for sample_index in range(samples):
        sampled_rows = []
        sampled_origins = rng.choice(origin_ids, size=len(origin_ids), replace=True)
        for origin_id in sampled_origins:
            cohort_ids = sorted(units[int(origin_id)])
            sampled_cohorts = rng.choice(cohort_ids, size=len(cohort_ids), replace=True)
            for cohort_id in sampled_cohorts:
                sampled_rows.extend(units[int(origin_id)][int(cohort_id)])
        effects = _paired_effects(_curve_rows(sampled_rows))
        for name in EFFECT_NAMES:
            draws[name][sample_index] = effects[name]

    return {
        name: {
            "point": point[name],
            "low": float(np.percentile(draws[name], 2.5)),
            "high": float(np.percentile(draws[name], 97.5)),
        }
        for name in EFFECT_NAMES
    }


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
    routing = metadata.get("oracle_routing") or metadata.get("route_routing")
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


def _flatten_curve_rows(curves):
    rows = []
    for curve in curves:
        row = {
            "track": curve["track"],
            "tau": curve["tau"],
            "phase": curve["phase"],
            "count": curve["count"],
            "graph_weight": curve["graph_weight"],
            "patch_residual_weight": curve["patch_residual_weight"],
        }
        for metric in ("MAPE", "MAE", "RMSE"):
            token = metric.lower()
            row[f"treated_{token}"] = curve["treated"][metric]
            row[f"control_{token}"] = curve["control"][metric]
            row[f"ace_{token}"] = curve["ACE"][metric]
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
    return {
        "checkpoint_dir": checkpoint_dir,
        "training_summary": training_summary,
        "loaded_checkpoint": loaded_checkpoint,
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
    dataset_dir = os.path.join(root, _safe_token(manifest["dataset"]["name"]))
    os.makedirs(dataset_dir, exist_ok=True)
    canonical_path = os.path.join(dataset_dir, "protocol_manifest.json")
    if os.path.exists(canonical_path):
        with open(canonical_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("protocol_hash") != manifest.get("protocol_hash"):
            if not getattr(args, "ples_overwrite_manifest", False):
                raise ValueError(
                    f"PLES manifest mismatch at {canonical_path}; use --ples_overwrite_manifest only when intentionally changing the protocol"
                )
    with open(canonical_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(manifest), handle, ensure_ascii=False, indent=2)
    result_path = os.path.join(output_dir, "protocol_manifest.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(manifest), handle, ensure_ascii=False, indent=2)
    return canonical_path, result_path


def _plot_trajectory(curves_path, output_path, input_len):
    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "plot_ples_trajectory.py")
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                script_path,
                "--curves",
                curves_path,
                "--output",
                output_path,
                "--input-len",
                str(input_len),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Warning: PLES trajectory plotting failed; numeric outputs are complete. {error}")
        return False
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip().splitlines()
        reason = details[-1] if details else f"exit code {completed.returncode}"
        print(f"Warning: PLES trajectory plotting failed; numeric outputs are complete. {reason}")
        return False
    return True


def async_lifecycle_inference(model, data_loader, args, criterions=None):
    if int(args.slide_step) != 1:
        raise ValueError("PLES requires slide_step=1 for dense event-relative evaluation")
    if args.test_for_changed:
        raise ValueError("test_for_changed and test_for_async_lifecycle should not be enabled together")

    dataset = data_loader.dataset
    manifest = dataset.get_protocol_manifest()
    protocol_token = manifest["protocol_hash"][:10]
    output_dir = os.path.join(args.best_prediction_path, args.exp_id, f"ples-v1-{protocol_token}")
    os.makedirs(output_dir, exist_ok=True)
    canonical_manifest_path, result_manifest_path = _write_manifest(dataset, args, output_dir)

    adj_np, pos_np = dataset.get_adj_and_pos_mark()
    adj = torch.as_tensor(adj_np, dtype=torch.float32, device=args.device)
    pos_mark = None
    if args.pos_mark and pos_np is not None:
        pos_mark = torch.as_tensor(pos_np, dtype=torch.float32, device=args.device)

    rows = []
    cohorts = manifest["cohorts"]
    pre_steps = int(manifest["requested"]["pre_steps"])
    post_steps = int(manifest["requested"]["post_steps"])
    model.eval()
    print("-> Start Paired Lifecycle Event Study inference...")
    for origin in manifest["origins"]:
        for tau in range(-pre_steps, post_steps + 1):
            control_sample = dataset.build_ples_sample(origin, cohorts[0], "join", "control", tau)
            samples = [control_sample]
            sample_index = {}
            if tau >= 0:
                for track in ("join", "fault"):
                    for cohort in cohorts:
                        sample_index[(track, int(cohort["cohort_id"]))] = len(samples)
                        samples.append(dataset.build_ples_sample(origin, cohort, track, "treatment", tau))
            predictions, targets, metadata = _predict_samples(model, samples, dataset, args, adj, pos_mark)
            control_prediction = predictions[0]
            target = targets[0]
            for track in ("join", "fault"):
                phase = phase_for_tau(track, tau, int(args.input_len))
                for cohort in cohorts:
                    cohort_id = int(cohort["cohort_id"])
                    nodes = cohort["join_nodes"] if track == "join" else cohort["fault_nodes"]
                    treated_index = sample_index.get((track, cohort_id), 0)
                    treated_prediction = predictions[treated_index]
                    sums = _paired_error_sums(treated_prediction, control_prediction, target, nodes)
                    if sums is None:
                        continue
                    graph_weight = _graph_weight(metadata, treated_index, nodes)
                    rows.append(
                        {
                            "dataset": manifest["dataset"]["name"],
                            "model": args.model,
                            "origin_id": int(origin["origin_id"]),
                            "block_id": int(origin["block_id"]),
                            "regime": origin["regime"],
                            "event_time_index": int(origin["event_time_index"]),
                            "event_timestamp": origin["event_timestamp"],
                            "cohort_id": cohort_id,
                            "track": track,
                            "tau": tau,
                            "phase": phase,
                            "node_count": len(nodes),
                            "graph_weight": graph_weight,
                            "patch_residual_weight": None if graph_weight is None else 1.0 - graph_weight,
                            **sums,
                        }
                    )
        print(
            f"   origin={origin['origin_id']:02d}/{len(manifest['origins']):02d} | "
            f"block={origin['block_id']} | regime={origin['regime']} | time={origin['event_timestamp']}"
        )

    summary = summarize_paired_rows(rows)
    bootstrap = bootstrap_paired_effects(
        rows,
        samples=int(getattr(args, "ples_bootstrap_samples", 2000)),
        seed=int(getattr(args, "ples_bootstrap_seed", 2026)),
    )
    training_trace = _load_training_trace(args)
    result = {
        "schema_version": 1,
        "protocol": manifest["protocol"],
        "protocol_hash": manifest["protocol_hash"],
        "training_protocol_hash": getattr(
            dataset,
            "training_protocol_hash",
            None,
        ),
        "ples_protocol_hash": manifest["protocol_hash"],
        "model": args.model,
        "model_variant": getattr(args, "model_variant", "default"),
        "exp_id": args.exp_id,
        "dataset": manifest["dataset"],
        "canonical_manifest_path": canonical_manifest_path,
        "result_manifest_path": result_manifest_path,
        "phase_metrics": summary["phase_metrics"],
        "main_metrics": summary["main_metrics"],
        "paired_effects": summary["paired_effects"],
        "paired_effect_intervals": bootstrap,
        "training_trace": training_trace,
        "row_count": len(rows),
    }

    row_fields = [
        "dataset", "model", "origin_id", "block_id", "regime", "event_time_index", "event_timestamp",
        "cohort_id", "track", "tau", "phase", "node_count", "count", "graph_weight",
        "patch_residual_weight", "treated_ape_sum", "control_ape_sum", "treated_ae_sum", "control_ae_sum",
        "treated_se_sum", "control_se_sum",
    ]
    curve_rows = _flatten_curve_rows(summary["curves"])
    curve_fields = [
        "track", "tau", "phase", "count", "treated_mape", "control_mape", "ace_mape",
        "treated_mae", "control_mae", "ace_mae", "treated_rmse", "control_rmse", "ace_rmse",
        "graph_weight", "patch_residual_weight",
    ]
    _write_csv(os.path.join(output_dir, "paired_event_rows.csv"), rows, row_fields)
    curves_path = os.path.join(output_dir, "paired_event_curves.csv")
    _write_csv(curves_path, curve_rows, curve_fields)
    with open(os.path.join(output_dir, "paired_event_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(_jsonable(result), handle, ensure_ascii=False, indent=2)

    trace_line = _format_training_trace(training_trace)
    text_lines = [
        (
            f"PLES: dataset={manifest['dataset']['name']} | origins={manifest['resolved']['origin_count']} | "
            f"cohorts={manifest['resolved']['cohort_count']} | cohort_size={manifest['resolved']['cohort_size']} | "
            f"protocol_hash={manifest['protocol_hash'][:12]}"
        ),
        (
            "ProtocolHashes: "
            f"training={result['training_protocol_hash'] or 'legacy'} | "
            f"ples={result['ples_protocol_hash']}"
        ),
        trace_line,
    ]
    for phase in PHASE_ORDER:
        metrics = summary["phase_metrics"][phase]
        text_lines.append(
            f"{phase}: MAPE={metrics['MAPE']:.4f} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f}"
        )
    for name in ("Join", "Fault", "LifeMacro"):
        metrics = summary["main_metrics"][name]
        text_lines.append(
            f"{name}: MAPE={metrics['MAPE']:.4f} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f}"
        )
    text_lines.append("PairedEffects (MAPE percentage points):")
    for name in EFFECT_NAMES:
        interval = bootstrap[name]
        if interval["low"] is None or interval["high"] is None:
            text_lines.append(f"{name}={interval['point']:.4f} | 95% CI=NA")
        else:
            text_lines.append(
                f"{name}={interval['point']:.4f} | 95% CI=[{interval['low']:.4f}, {interval['high']:.4f}]"
            )
    with open(os.path.join(output_dir, "paired_event_results.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(text_lines) + "\n")
    for line in text_lines:
        print(line)

    _plot_trajectory(curves_path, os.path.join(output_dir, "paired_event_trajectory.png"), int(args.input_len))
    print(f"-> PLES results saved to {output_dir}")
    return result

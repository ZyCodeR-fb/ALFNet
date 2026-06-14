import json
import os
import pickle
import queue
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from .async_calibrators import ADCSDModule, Calibration, CorrCoefLoss, FRPlusModule, SimpleOutputAdapter
from utils.model_output import unpack_model_output

ASYNC_TTC_QUEUE_LEN = 6
ASYNC_TTC_NAS_LOSS_WEIGHT = 0.8
ASYNC_TTC_AAS_LOSS_WEIGHT = 0.2
ASYNC_TTC_LR = 5e-4

ASYNC_ADCSD_QUEUE_LEN = 12
ASYNC_ADCSD_MOVING_AVG = 5
ASYNC_ADCSD_LR = 1e-2

ASYNC_COSA_LR = 1e-3
ASYNC_COSA_WEIGHT_DECAY = 1e-4
ASYNC_COSA_STEPS = 5
ASYNC_COSA_BUFFER_CONTEXT_SIZE = 5
ASYNC_COSA_VAR_WISE_GATING = True
ASYNC_COSA_ADAPTER_LAYERS = 1
ASYNC_COSA_HIDDEN_DIM = 64
ASYNC_COSA_L2_REG = 1e-4
ASYNC_COSA_HISTORY_MAXLEN = 200

ASYNC_PETSA_LR = 1e-3
ASYNC_PETSA_STEPS = 1
ASYNC_PETSA_GATING_INIT = 0.01
ASYNC_PETSA_HIDDEN_DIM = 128
ASYNC_PETSA_GCM_VAR_WISE = True
ASYNC_PETSA_RANK = 16
ASYNC_PETSA_LOSS_ALPHA = 0.1

AVAILABILITY_STATE_ORDER = [
    "zero_history_virtual",
    "new_sensor_early",
    "new_sensor_late",
    "failed_sensor_early",
    "failed_sensor_late",
    "stable_observed",
]


def _validate_async_args(args):
    if args.input_len != 12 or args.pred_len != 12 or args.slide_step != 12:
        raise ValueError("async_lifecycle currently requires input_len=12, pred_len=12, and slide_step=12")
    if args.async_commissioning_k not in {6, 12, 24}:
        raise ValueError("async_lifecycle currently supports commissioning_k in {6, 12, 24}")
    if args.test_for_changed:
        raise ValueError("test_for_changed and test_for_async_lifecycle should not be enabled together")
    if args.async_adcsd_queue_len is not None and args.async_adcsd_queue_len <= 0:
        raise ValueError("async_adcsd_queue_len must be positive")
    if args.async_adcsd_moving_avg is not None and args.async_adcsd_moving_avg <= 0:
        raise ValueError("async_adcsd_moving_avg must be positive")
    if args.async_adcsd_moving_avg is not None and args.async_adcsd_moving_avg % 2 == 0:
        raise ValueError("async_adcsd_moving_avg must be odd to preserve prediction length")
    if args.async_adcsd_lr is not None and args.async_adcsd_lr <= 0:
        raise ValueError("async_adcsd_lr must be positive")


def _format_async_float(value):
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _get_adcsd_hparams(args):
    return {
        "queue_len": int(args.async_adcsd_queue_len or ASYNC_ADCSD_QUEUE_LEN),
        "moving_avg": int(args.async_adcsd_moving_avg or ASYNC_ADCSD_MOVING_AVG),
        "lr": float(args.async_adcsd_lr or ASYNC_ADCSD_LR),
    }


def _prepare_paths(args):
    adaptor_param_suffix = ""
    if args.async_adaptor == "adcsd":
        adcsd_hparams = _get_adcsd_hparams(args)
        adaptor_param_suffix = (
            f"-q-{adcsd_hparams['queue_len']}"
            f"-ma-{adcsd_hparams['moving_avg']}"
            f"-lr-{_format_async_float(adcsd_hparams['lr'])}"
        )
    async_suffix = (
        f"async_lifecycle-{args.async_adaptor}-events-{args.async_num_events}-"
        f"k-{args.async_commissioning_k}-seed-{args.async_event_seed}"
        f"{adaptor_param_suffix}"
    )
    output_dir = os.path.join(args.best_prediction_path, args.exp_id, async_suffix)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _fill_missing_with_zero(x_enc):
    observed_mask = ~torch.isnan(x_enc)
    x_filled = torch.nan_to_num(x_enc, nan=0.0)
    return x_filled, observed_mask


def _apply_frp(FRP, tensor):
    batch_size, time_steps, num_nodes, channels = tensor.shape
    yb = tensor.permute(0, 3, 2, 1).reshape(batch_size * channels, 1, num_nodes, time_steps)
    yc = FRP(yb)
    return yc.reshape(batch_size, channels, num_nodes, time_steps).permute(0, 3, 2, 1)


def _merge_raw_with_extra(raw_tensor, source_tensor, raw_enc_in):
    if source_tensor.shape[-1] == raw_enc_in:
        return raw_tensor
    return torch.cat([raw_tensor, source_tensor[..., raw_enc_in:]], dim=-1)


def _get_raw_enc_in(dataset, fallback):
    if hasattr(dataset, "in_feas"):
        return len(dataset.in_feas)
    return fallback


def _prepare_st_ttc_input(x_enc_filled, observed_mask, FRP, raw_enc_in):
    raw_x = x_enc_filled[..., :raw_enc_in]
    raw_observed_mask = observed_mask[..., :raw_enc_in]
    x_calibrated = _apply_frp(FRP, raw_x)
    raw_model_input = x_calibrated * raw_observed_mask.to(dtype=x_calibrated.dtype)
    return _merge_raw_with_extra(raw_model_input, x_enc_filled, raw_enc_in)


def _forward_model_bundle(model, adj, model_input, enc_t_mark, pos_mark, dec_t_mark):
    return unpack_model_output(model(adj, model_input, enc_t_mark, pos_mark, None, dec_t_mark, 0))


def _forward_model(model, adj, model_input, enc_t_mark, pos_mark, dec_t_mark):
    y_hat, _ = _forward_model_bundle(model, adj, model_input, enc_t_mark, pos_mark, dec_t_mark)
    return y_hat


def _compute_loss(criterion, pred, target):
    target_mask = ~torch.isnan(target)
    if target_mask.sum() == 0:
        return None
    if pred.shape[-1] != target.shape[-1]:
        y_omi = torch.isnan(target)
        return criterion(pred, target, y_omi)
    return criterion(pred[target_mask], target[target_mask])


def _slice_role_tensors(pred, target, nodes):
    if not nodes:
        return None, None
    pred_part = pred[..., nodes, :]
    target_part = target[..., nodes, :]
    target_mask = ~torch.isnan(target_part)
    if target_mask.sum() == 0:
        return None, None
    return pred_part, target_part


def _compute_weighted_st_ttc_loss(criterion, pred, target, aas_nodes, nas_nodes):
    weighted_losses = []
    normalizer = 0.0

    pred_nas, target_nas = _slice_role_tensors(pred, target, nas_nodes)
    if pred_nas is not None:
        loss_nas = _compute_loss(criterion, pred_nas, target_nas)
        if loss_nas is not None:
            weighted_losses.append(ASYNC_TTC_NAS_LOSS_WEIGHT * loss_nas)
            normalizer += ASYNC_TTC_NAS_LOSS_WEIGHT

    pred_aas, target_aas = _slice_role_tensors(pred, target, aas_nodes)
    if pred_aas is not None:
        loss_aas = _compute_loss(criterion, pred_aas, target_aas)
        if loss_aas is not None:
            weighted_losses.append(ASYNC_TTC_AAS_LOSS_WEIGHT * loss_aas)
            normalizer += ASYNC_TTC_AAS_LOSS_WEIGHT

    if not weighted_losses or normalizer == 0:
        return None
    return sum(weighted_losses) / normalizer


def _compute_metrics(pred, target):
    target_mask = (~torch.isnan(target)) & torch.isfinite(target) & torch.isfinite(pred)
    if target_mask.sum() == 0:
        return None
    pred_valid = pred[target_mask]
    target_valid = target[target_mask]
    diff = pred_valid - target_valid
    denom = torch.where(torch.abs(target_valid) < 1e-6, torch.full_like(target_valid, 1e-6), target_valid)
    return {
        "MAE": torch.mean(torch.abs(diff)).item(),
        "RMSE": torch.sqrt(torch.mean(diff**2)).item(),
        "MAPE": torch.mean(torch.abs(diff / denom)).item() * 100,
    }


def _inverse_transform_with_grad(dataset, tensor):
    std = torch.as_tensor(dataset.std_[..., dataset.tar_feas], dtype=tensor.dtype, device=tensor.device)
    mean = torch.as_tensor(dataset.mean_[..., dataset.tar_feas], dtype=tensor.dtype, device=tensor.device)
    return tensor * std + mean


def _concat_or_none(tensors):
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def _mean_metric_dicts(metric_dicts):
    if not metric_dicts:
        return None
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metric[key] for metric in metric_dicts])) for key in keys}


def _append_state_buffer(state_buffers, state_name, pred, target):
    if pred is None:
        return
    # Availability-state groups aggregate windows/events with different node counts.
    # Flatten the variable node dimension before concatenating in _summarize_state_buffers.
    state_buffers[state_name]["pred"].append(pred.reshape(-1, pred.shape[-1]))
    state_buffers[state_name]["true"].append(target.reshape(-1, target.shape[-1]))


def _summarize_state_buffers(state_buffers):
    state_metrics = {}
    for state_name, buffers in state_buffers.items():
        pred = _concat_or_none(buffers["pred"])
        target = _concat_or_none(buffers["true"])
        state_metrics[state_name] = None if pred is None else _compute_metrics(pred, target)

    valid_metrics = [metrics for metrics in state_metrics.values() if metrics is not None]
    macro_metrics = _mean_metric_dicts(valid_metrics)
    worst_state = None
    if valid_metrics:
        worst_name, worst_metrics = max(
            ((state_name, metrics) for state_name, metrics in state_metrics.items() if metrics is not None),
            key=lambda item: item[1]["MAPE"],
        )
        worst_state = {"name": worst_name, "metrics": worst_metrics}
    return state_metrics, macro_metrics, worst_state


def _summarize_event_age_curve(event_metrics):
    curves = {role: {} for role in ["NAS", "FS", "AAS", "VS"]}
    for event_result in event_metrics.values():
        for role_name, role_result in event_result.items():
            for window_metric in role_result["window_metrics"]:
                offset = str(window_metric["window_offset"])
                curves[role_name].setdefault(offset, []).append(
                    {key: window_metric[key] for key in ["MAPE", "MAE", "RMSE"]}
                )
    return {
        role_name: {
            offset: _mean_metric_dicts(metrics)
            for offset, metrics in sorted(offset_metrics.items(), key=lambda item: int(item[0]))
        }
        for role_name, offset_metrics in curves.items()
    }


def _empty_routing_diag_buffers():
    return {
        role: {
            "alpha": [],
            "graph_win": [],
            "gate_match": [],
            "graph_pred": [],
            "patch_pred": [],
            "fused_pred": [],
            "true": [],
        }
        for role in ["NAS", "FS", "AAS", "VS"]
    }


def _append_routing_diagnostics(buffers, output_meta, roles, fused_pred, target, dataset):
    routing = output_meta.get("oracle_routing") or output_meta.get("route_routing")
    if not routing:
        return
    if not all(key in routing for key in ["alpha", "y_graph", "y_patch"]):
        return
    alpha = routing["alpha"].detach()
    graph_pred = _inverse_transform_with_grad(dataset, routing["y_graph"].detach())
    patch_pred = _inverse_transform_with_grad(dataset, routing["y_patch"].detach())
    for role_name, nodes in roles.items():
        if not nodes:
            continue
        alpha_part = alpha[..., nodes, :]
        graph_part = graph_pred[..., nodes, :]
        patch_part = patch_pred[..., nodes, :]
        fused_part = fused_pred[..., nodes, :]
        target_part = target[..., nodes, :]
        valid_mask = (~torch.isnan(target_part)) & torch.isfinite(target_part) & torch.isfinite(graph_part) & torch.isfinite(patch_part)
        if valid_mask.sum() == 0:
            continue
        graph_err = torch.abs(graph_part - target_part)
        patch_err = torch.abs(patch_part - target_part)
        graph_win = (graph_err <= patch_err).to(dtype=graph_part.dtype)
        gate_graph = (alpha_part >= 0.5).to(dtype=graph_part.dtype)
        gate_match = (gate_graph == graph_win).to(dtype=graph_part.dtype)
        buffers[role_name]["alpha"].append(alpha_part[valid_mask].detach().cpu())
        buffers[role_name]["graph_win"].append(graph_win[valid_mask].detach().cpu())
        buffers[role_name]["gate_match"].append(gate_match[valid_mask].detach().cpu())
        buffers[role_name]["graph_pred"].append(graph_part.reshape(-1, graph_part.shape[-1]).detach().cpu())
        buffers[role_name]["patch_pred"].append(patch_part.reshape(-1, patch_part.shape[-1]).detach().cpu())
        buffers[role_name]["fused_pred"].append(fused_part.reshape(-1, fused_part.shape[-1]).detach().cpu())
        buffers[role_name]["true"].append(target_part.reshape(-1, target_part.shape[-1]).detach().cpu())


def _summarize_routing_diagnostics(buffers):
    summary = {}
    for role_name, role_buffers in buffers.items():
        alpha = _concat_or_none(role_buffers["alpha"])
        if alpha is None:
            summary[role_name] = None
            continue
        graph_win = _concat_or_none(role_buffers["graph_win"])
        gate_match = _concat_or_none(role_buffers["gate_match"])
        true = _concat_or_none(role_buffers["true"])
        graph_pred = _concat_or_none(role_buffers["graph_pred"])
        patch_pred = _concat_or_none(role_buffers["patch_pred"])
        fused_pred = _concat_or_none(role_buffers["fused_pred"])
        graph_metrics = None if graph_pred is None or true is None else _compute_metrics(graph_pred, true)
        patch_metrics = None if patch_pred is None or true is None else _compute_metrics(patch_pred, true)
        fused_metrics = None if fused_pred is None or true is None else _compute_metrics(fused_pred, true)
        summary[role_name] = {
            "alpha_mean": float(alpha.mean().item()),
            "graph_win_rate": None if graph_win is None else float(graph_win.mean().item()),
            "patch_win_rate": None if graph_win is None else float(1.0 - graph_win.mean().item()),
            "gate_match_rate": None if gate_match is None else float(gate_match.mean().item()),
            "graph_metrics": graph_metrics,
            "patch_metrics": patch_metrics,
            "fused_metrics": fused_metrics,
        }
    return summary


def _compute_role_gap(summary_metrics):
    valid = [metrics for metrics in summary_metrics.values() if metrics is not None]
    if not valid:
        return None
    return {
        "MAPE": max(metric["MAPE"] for metric in valid) - min(metric["MAPE"] for metric in valid),
        "MAE": max(metric["MAE"] for metric in valid) - min(metric["MAE"] for metric in valid),
        "RMSE": max(metric["RMSE"] for metric in valid) - min(metric["RMSE"] for metric in valid),
    }


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {key: _to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(value) for value in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _freeze_model_params(model):
    requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)
    return requires_grad


def _restore_model_params(model, requires_grad):
    if requires_grad is None:
        return
    for param, flag in zip(model.parameters(), requires_grad):
        param.requires_grad_(flag)


def _reshape_cosa_tensor(tensor):
    batch_size, pred_len, num_nodes, channels = tensor.shape
    return tensor.reshape(batch_size, pred_len, num_nodes * channels), num_nodes, channels


def _restore_cosa_tensor(tensor, num_nodes, channels):
    batch_size, pred_len, _ = tensor.shape
    return tensor.reshape(batch_size, pred_len, num_nodes, channels)


def _apply_cosa_adapter(output_adapter, pred, context_data):
    pred_reshaped, num_nodes, channels = _reshape_cosa_tensor(pred)
    adapted_pred = output_adapter(pred_reshaped, context_data)
    return _restore_cosa_tensor(adapted_pred, num_nodes, channels)


def _freeze_all_model_params(module):
    for param in module.parameters():
        param.requires_grad_(False)


def _unfreeze_adapter_params(module):
    for param in module.parameters():
        param.requires_grad_(True)


def _update_memory_buffer(adaptor_state, targets, prediction, calibration_nodes):
    pred_part, target_part = _slice_role_tensors(prediction, targets, calibration_nodes)
    if pred_part is None:
        return None

    seq_mse = F.mse_loss(pred_part, target_part).item()
    batch_info = {
        "time_idx": adaptor_state["current_time_idx"],
        "target_mean": target_part.mean().item(),
        "seq_mse": seq_mse,
    }
    adaptor_state["sample_history"].append(batch_info)
    adaptor_state["current_time_idx"] += targets.shape[0]
    adaptor_state["step_count"] += 1
    adaptor_state["adapters_enabled"] = True
    return seq_mse


def _get_individual_context_for_batch(adaptor_state, batch_size):
    if len(adaptor_state["sample_history"]) == 0:
        return torch.zeros(batch_size, adaptor_state["buffer_context_size"], device=adaptor_state["device"])

    history_size = min(adaptor_state["buffer_context_size"], len(adaptor_state["sample_history"]))
    context_values = [adaptor_state["sample_history"][-(i + 1)]["target_mean"] for i in range(history_size)]
    if len(context_values) < adaptor_state["buffer_context_size"]:
        last_val = context_values[-1] if context_values else 0.0
        context_values.extend([last_val] * (adaptor_state["buffer_context_size"] - len(context_values)))

    context_tensor = torch.tensor(context_values, dtype=torch.float32, device=adaptor_state["device"])
    return context_tensor.unsqueeze(0).expand(batch_size, -1)


def _compute_petsa_loss(corrcoef_loss, pred, target, calibration_nodes):
    pred_part, target_part = _slice_role_tensors(pred, target, calibration_nodes)
    if pred_part is None:
        return None

    loss_feq = (torch.fft.rfft(pred_part, dim=1) - torch.fft.rfft(target_part, dim=1)).abs().mean()
    loss_tmp = F.huber_loss(pred_part, target_part, delta=0.5)
    loss = loss_tmp + loss_feq * ASYNC_PETSA_LOSS_ALPHA

    coss = corrcoef_loss(pred_part, target_part)
    sf_pred = F.log_softmax(pred_part - pred_part.mean(dim=1, keepdim=True), dim=1)
    sf_gt = F.softmax(target_part - target_part.mean(dim=1, keepdim=True), dim=1)
    loss_var = F.kl_div(sf_pred, sf_gt, reduction="batchmean")
    loss_mean = F.l1_loss(pred_part.mean(dim=1, keepdim=True), target_part.mean(dim=1, keepdim=True))
    total_loss = loss + coss + loss_var + loss_mean
    if not torch.isfinite(total_loss):
        return None
    return total_loss


def _build_async_adaptor(args, dataset, model, event_plan):
    adaptor_name = args.async_adaptor
    adaptor_state = {
        "name": adaptor_name,
        "module": None,
        "optimizer": None,
        "queue": None,
        "active_event_index": None,
        "update_count_per_event": {str(event["event_id"]): 0 for event in event_plan},
        "total_update_count": 0,
        "has_input_calibration": False,
        "has_output_calibration": False,
        "metadata": {},
        "device": args.device,
        "model_param_requires_grad": None,
    }

    if adaptor_name == "base":
        return adaptor_state

    adaptor_state["model_param_requires_grad"] = _freeze_model_params(model)
    output_channels = len(dataset.tar_feas)

    if adaptor_name == "st_ttc":
        freq_bins = args.input_len // 2 + 1
        module = FRPlusModule(dataset.num_nodes, freq_bins, groups=4).to(args.device)
        optimizer = torch.optim.Adam(module.parameters(), lr=ASYNC_TTC_LR)
        adaptor_state.update(
            {
                "module": module,
                "optimizer": optimizer,
                "has_input_calibration": True,
                "metadata": {
                    "queue_len": ASYNC_TTC_QUEUE_LEN,
                    "lr": ASYNC_TTC_LR,
                    "groups": 4,
                    "loss_weights": {"NAS": ASYNC_TTC_NAS_LOSS_WEIGHT, "AAS": ASYNC_TTC_AAS_LOSS_WEIGHT},
                    "shared_across_nodes": True,
                },
            }
        )
        module.eval()
        return adaptor_state

    if adaptor_name == "adcsd":
        adcsd_hparams = _get_adcsd_hparams(args)
        module = ADCSDModule(output_dim=output_channels, output_window=args.pred_len, num_nodes=dataset.num_nodes, moving_avg=adcsd_hparams["moving_avg"]).to(args.device)
        optimizer = torch.optim.Adam(module.parameters(), lr=adcsd_hparams["lr"], eps=1e-8, weight_decay=0, amsgrad=False)
        adaptor_state.update(
            {
                "module": module,
                "optimizer": optimizer,
                "has_output_calibration": True,
                "metadata": {
                    "queue_len": adcsd_hparams["queue_len"],
                    "moving_avg": adcsd_hparams["moving_avg"],
                    "lr": adcsd_hparams["lr"],
                },
            }
        )
        module.eval()
        return adaptor_state

    if adaptor_name == "cosa":
        module = SimpleOutputAdapter(
            pred_len=args.pred_len,
            buffer_context_size=ASYNC_COSA_BUFFER_CONTEXT_SIZE,
            n_vars=dataset.num_nodes * output_channels,
            var_wise_gating=ASYNC_COSA_VAR_WISE_GATING,
            num_layers=ASYNC_COSA_ADAPTER_LAYERS,
            hidden_dim=ASYNC_COSA_HIDDEN_DIM,
        ).to(args.device)
        _freeze_all_model_params(module)
        _unfreeze_adapter_params(module)
        optimizer = torch.optim.Adam(module.parameters(), lr=ASYNC_COSA_LR, weight_decay=ASYNC_COSA_WEIGHT_DECAY)
        adaptor_state.update(
            {
                "module": module,
                "optimizer": optimizer,
                "has_output_calibration": True,
                "sample_history": deque(maxlen=ASYNC_COSA_HISTORY_MAXLEN),
                "current_time_idx": 0,
                "step_count": 0,
                "adapters_enabled": False,
                "buffer_context_size": ASYNC_COSA_BUFFER_CONTEXT_SIZE,
                "adapt_steps": ASYNC_COSA_STEPS,
                "metadata": {
                    "lr": ASYNC_COSA_LR,
                    "steps": ASYNC_COSA_STEPS,
                    "buffer_context_size": ASYNC_COSA_BUFFER_CONTEXT_SIZE,
                    "var_wise_gating": ASYNC_COSA_VAR_WISE_GATING,
                    "adapter_layers": ASYNC_COSA_ADAPTER_LAYERS,
                    "hidden_dim": ASYNC_COSA_HIDDEN_DIM,
                    "paas": False,
                    "fast_adaptation": False,
                    "adaptive_lr": False,
                },
            }
        )
        module.eval()
        return adaptor_state

    if adaptor_name == "petsa":
        module = Calibration(
            seq_len=args.input_len,
            pred_len=args.pred_len,
            n_var=dataset.num_nodes * output_channels,
            hidden_dim=ASYNC_PETSA_HIDDEN_DIM,
            gating_init=ASYNC_PETSA_GATING_INIT,
            var_wise=ASYNC_PETSA_GCM_VAR_WISE,
            low_rank=ASYNC_PETSA_RANK,
        ).to(args.device)
        optimizer = torch.optim.Adam(module.parameters(), lr=ASYNC_PETSA_LR, weight_decay=1e-4)
        corrcoef_loss = CorrCoefLoss().to(args.device)
        adaptor_state.update(
            {
                "module": module,
                "optimizer": optimizer,
                "corrcoef_loss": corrcoef_loss,
                "has_input_calibration": True,
                "has_output_calibration": True,
                "metadata": {
                    "lr": ASYNC_PETSA_LR,
                    "steps": ASYNC_PETSA_STEPS,
                    "gating_init": ASYNC_PETSA_GATING_INIT,
                    "hidden_dim": ASYNC_PETSA_HIDDEN_DIM,
                    "gcm_var_wise": ASYNC_PETSA_GCM_VAR_WISE,
                    "rank": ASYNC_PETSA_RANK,
                    "loss_alpha": ASYNC_PETSA_LOSS_ALPHA,
                    "paas": False,
                    "adjust_pred": False,
                },
            }
        )
        module.eval()
        return adaptor_state

    raise ValueError(f"Unsupported async adaptor: {adaptor_name}")


def _predict_window(adaptor_state, model, adj, x_enc_filled, observed_mask, enc_t_mark, pos_mark, dec_t_mark, dataset):
    adaptor_name = adaptor_state["name"]
    module = adaptor_state["module"]

    if adaptor_name == "base":
        with torch.no_grad():
            y_hat, output_meta = _forward_model_bundle(model, adj, x_enc_filled, enc_t_mark, pos_mark, dec_t_mark)
            final_pred = _inverse_transform_with_grad(dataset, y_hat)
        return {"final_pred": final_pred, "output_meta": output_meta}

    if adaptor_name == "st_ttc":
        with torch.no_grad():
            raw_enc_in = _get_raw_enc_in(dataset, x_enc_filled.shape[-1])
            model_input = _prepare_st_ttc_input(x_enc_filled, observed_mask, module, raw_enc_in)
            y_hat = _forward_model(model, adj, model_input, enc_t_mark, pos_mark, dec_t_mark)
            final_pred = _inverse_transform_with_grad(dataset, y_hat)
        return {"final_pred": final_pred}

    if adaptor_name == "adcsd":
        with torch.no_grad():
            y_hat = _forward_model(model, adj, x_enc_filled, enc_t_mark, pos_mark, dec_t_mark)
            y_hat = module(y_hat)
            final_pred = _inverse_transform_with_grad(dataset, y_hat)
        return {"final_pred": final_pred}

    if adaptor_name == "cosa":
        with torch.no_grad():
            y_hat = _forward_model(model, adj, x_enc_filled, enc_t_mark, pos_mark, dec_t_mark)
            original_pred = _inverse_transform_with_grad(dataset, y_hat)
            if adaptor_state["adapters_enabled"]:
                context_data = _get_individual_context_for_batch(adaptor_state, original_pred.shape[0])
                final_pred = _apply_cosa_adapter(module, original_pred, context_data)
                if not torch.isfinite(final_pred).all():
                    final_pred = original_pred
            else:
                final_pred = original_pred
        return {"final_pred": final_pred, "original_pred": original_pred}

    if adaptor_name == "petsa":
        with torch.no_grad():
            raw_enc_in = _get_raw_enc_in(dataset, x_enc_filled.shape[-1])
            raw_input, _, _, _ = module.input_calibration((x_enc_filled[..., :raw_enc_in], enc_t_mark, None, dec_t_mark))
            raw_input = raw_input * observed_mask[..., :raw_enc_in].to(dtype=raw_input.dtype)
            model_input = _merge_raw_with_extra(raw_input, x_enc_filled, raw_enc_in)
            y_hat = _forward_model(model, adj, model_input, enc_t_mark, pos_mark, dec_t_mark)
            base_pred = _inverse_transform_with_grad(dataset, y_hat)
            final_pred = module.output_calibration(base_pred)
            if not torch.isfinite(final_pred).all():
                final_pred = base_pred
        return {"final_pred": final_pred}

    raise ValueError(f"Unsupported async adaptor: {adaptor_name}")


def _run_commissioning_update(adaptor_state, event, model, adj, x_enc_filled, observed_mask, y, enc_t_mark, pos_mark, dec_t_mark, dataset, criterion, event_index):
    adaptor_name = adaptor_state["name"]
    event_id = str(event["event_id"])
    module = adaptor_state["module"]
    optimizer = adaptor_state["optimizer"]

    if adaptor_name == "base":
        return

    if adaptor_name in {"st_ttc", "adcsd"}:
        queue_len = adaptor_state["metadata"]["queue_len"]
        if adaptor_state["active_event_index"] != event_index:
            adaptor_state["active_event_index"] = event_index
            adaptor_state["queue"] = queue.Queue(maxsize=queue_len)
        adaptor_state["queue"].put(
            (
                x_enc_filled.detach().clone(),
                observed_mask.detach().clone(),
                y.detach().clone(),
                enc_t_mark.detach().clone() if enc_t_mark is not None else None,
                dec_t_mark.detach().clone(),
            )
        )
        if not adaptor_state["queue"].full():
            return

    if adaptor_name == "st_ttc":
        x_o, observed_mask_o, y_o, enc_t_mark_o, dec_t_mark_o = adaptor_state["queue"].get()
        module.train()
        optimizer.zero_grad()
        raw_enc_in = _get_raw_enc_in(dataset, x_o.shape[-1])
        model_input = _prepare_st_ttc_input(x_o, observed_mask_o, module, raw_enc_in)
        pred_o = _forward_model(model, adj, model_input, enc_t_mark_o, pos_mark, dec_t_mark_o)
        pred_o = _inverse_transform_with_grad(dataset, pred_o)
        y_o = _inverse_transform_with_grad(dataset, y_o)
        loss = _compute_weighted_st_ttc_loss(criterion, pred_o, y_o, event["aas_nodes"], event["nas_nodes"])
        if loss is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adaptor_state["update_count_per_event"][event_id] += 1
            adaptor_state["total_update_count"] += 1
        module.eval()
        return

    if adaptor_name == "adcsd":
        x_o, _, y_o, enc_t_mark_o, dec_t_mark_o = adaptor_state["queue"].get()
        with torch.no_grad():
            y_hat_o = _forward_model(model, adj, x_o, enc_t_mark_o, pos_mark, dec_t_mark_o)
        module.train()
        optimizer.zero_grad()
        pred_o = module(y_hat_o)
        pred_o = _inverse_transform_with_grad(dataset, pred_o)
        y_o = _inverse_transform_with_grad(dataset, y_o)
        pred_part, target_part = _slice_role_tensors(pred_o, y_o, event["calibration_nodes"])
        loss = None if pred_part is None else _compute_loss(criterion, pred_part, target_part)
        if loss is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adaptor_state["update_count_per_event"][event_id] += 1
            adaptor_state["total_update_count"] += 1
        module.eval()
        return

    if adaptor_name == "cosa":
        if not adaptor_state["adapters_enabled"]:
            return
        context_data = _get_individual_context_for_batch(adaptor_state, y.shape[0])
        target = _inverse_transform_with_grad(dataset, y)
        original_pred = adaptor_state["latest_original_pred"]
        module.train()
        for _ in range(adaptor_state["adapt_steps"]):
            optimizer.zero_grad()
            adapted_pred = _apply_cosa_adapter(module, original_pred, context_data)
            pred_part, target_part = _slice_role_tensors(adapted_pred, target, event["calibration_nodes"])
            if pred_part is None:
                break
            loss = F.mse_loss(pred_part, target_part)
            adapter_params = list(module.parameters())
            l2_reg = sum(param.pow(2).sum() for param in adapter_params if param.requires_grad)
            loss = loss + ASYNC_COSA_L2_REG * l2_reg
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=0.1)
            optimizer.step()
            adaptor_state["update_count_per_event"][event_id] += 1
            adaptor_state["total_update_count"] += 1
        module.eval()
        return

    if adaptor_name == "petsa":
        target = _inverse_transform_with_grad(dataset, y)
        module.train()
        for _ in range(ASYNC_PETSA_STEPS):
            optimizer.zero_grad()
            raw_enc_in = _get_raw_enc_in(dataset, x_enc_filled.shape[-1])
            raw_input, _, _, _ = module.input_calibration((x_enc_filled[..., :raw_enc_in], enc_t_mark, None, dec_t_mark))
            raw_input = raw_input * observed_mask[..., :raw_enc_in].to(dtype=raw_input.dtype)
            model_input = _merge_raw_with_extra(raw_input, x_enc_filled, raw_enc_in)
            pred = _forward_model(model, adj, model_input, enc_t_mark, pos_mark, dec_t_mark)
            pred = _inverse_transform_with_grad(dataset, pred)
            pred = module.output_calibration(pred)
            loss = _compute_petsa_loss(adaptor_state["corrcoef_loss"], pred, target, event["calibration_nodes"])
            if loss is None:
                optimizer.zero_grad()
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=0.1)
            optimizer.step()
            adaptor_state["update_count_per_event"][event_id] += 1
            adaptor_state["total_update_count"] += 1
        module.eval()
        return

    raise ValueError(f"Unsupported async adaptor: {adaptor_name}")


def _format_adaptor_text(adaptor_state):
    adaptor_name = adaptor_state["name"]
    if adaptor_name == "base":
        return None

    metadata = adaptor_state["metadata"]
    if adaptor_name == "st_ttc":
        return (
            f"ST-TTC: queue_len={metadata['queue_len']} | "
            f"lr={metadata['lr']:.0e} | "
            f"loss_w=(NAS:{metadata['loss_weights']['NAS']:.1f},AAS:{metadata['loss_weights']['AAS']:.1f}) | "
            f"total_updates={adaptor_state['total_update_count']} | "
            f"per_event={adaptor_state['update_count_per_event']}"
        )
    if adaptor_name == "adcsd":
        return (
            f"ADCSD: queue_len={metadata['queue_len']} | "
            f"moving_avg={metadata['moving_avg']} | "
            f"lr={metadata['lr']:.0e} | "
            f"total_updates={adaptor_state['total_update_count']} | "
            f"per_event={adaptor_state['update_count_per_event']}"
        )
    if adaptor_name == "cosa":
        return (
            f"COSA: steps={metadata['steps']} | "
            f"buffer_context_size={metadata['buffer_context_size']} | "
            f"lr={metadata['lr']:.0e} | "
            f"total_updates={adaptor_state['total_update_count']} | "
            f"per_event={adaptor_state['update_count_per_event']}"
        )
    if adaptor_name == "petsa":
        return (
            f"PETSA: steps={metadata['steps']} | "
            f"rank={metadata['rank']} | "
            f"loss_alpha={metadata['loss_alpha']:.2f} | "
            f"lr={metadata['lr']:.0e} | "
            f"total_updates={adaptor_state['total_update_count']} | "
            f"per_event={adaptor_state['update_count_per_event']}"
        )
    return None


def async_lifecycle_inference(model, data_loader, args, criterions):
    _validate_async_args(args)
    dataset = data_loader.dataset
    bundle = dataset.get_async_lifecycle_bundle()
    event_plan = bundle["event_plan"]
    window_stage = bundle["window_stage"]
    window_event_index = bundle["window_event_index"]
    output_dir = _prepare_paths(args)
    output_path = os.path.join(output_dir, "async_lifecycle_results.pkl")
    summary_path = os.path.join(output_dir, "async_lifecycle_summary.json")
    text_path = os.path.join(output_dir, "async_lifecycle_results.txt")

    model.eval()
    adj, pos_mark = dataset.get_adj_and_pos_mark()
    adj = torch.from_numpy(adj).float().to(args.device)
    if not args.pos_mark:
        pos_mark = None
    elif pos_mark is not None:
        pos_mark = torch.from_numpy(pos_mark).float().to(args.device)

    adaptor_state = _build_async_adaptor(args, dataset, model, event_plan)

    role_buffers = {}
    for event in event_plan:
        role_buffers[event["event_id"]] = {
            role: {"pred": [], "true": [], "window_metrics": []}
            for role in ["NAS", "FS", "AAS", "VS"]
        }
    state_buffers = {
        state_name: {"pred": [], "true": []}
        for state_name in AVAILABILITY_STATE_ORDER
    }
    routing_diag_buffers = _empty_routing_diag_buffers()

    print("-> Start async lifecycle inference...")
    for batch_ix, batch in enumerate(data_loader):
        batch = [x.float().to(args.device) for x in batch]
        x_enc, x_dec, enc_t_mark, dec_t_mark = batch
        if not args.t_mark:
            enc_t_mark = None
        y = x_dec[:, -args.pred_len :].clone()
        x_enc_filled, observed_mask = _fill_missing_with_zero(x_enc)

        prediction_bundle = _predict_window(adaptor_state, model, adj, x_enc_filled, observed_mask, enc_t_mark, pos_mark, dec_t_mark, dataset)
        final_pred = prediction_bundle["final_pred"]
        inverse_y = _inverse_transform_with_grad(dataset, y)

        stage = window_stage[batch_ix]
        event_index = window_event_index[batch_ix]
        event = event_plan[event_index] if event_index >= 0 else None

        if adaptor_state["name"] == "cosa" and event is not None:
            adaptor_state["latest_original_pred"] = prediction_bundle["original_pred"].detach()
            _update_memory_buffer(adaptor_state, inverse_y.detach(), adaptor_state["latest_original_pred"], event["calibration_nodes"])

        if stage == "commissioning" and event is not None:
            _run_commissioning_update(
                adaptor_state,
                event,
                model,
                adj,
                x_enc_filled,
                observed_mask,
                y,
                enc_t_mark,
                pos_mark,
                dec_t_mark,
                dataset,
                criterions[args.loss],
                event_index,
            )
        elif stage == "evaluation" and event is not None:
            inverse_y_cpu = inverse_y.detach().cpu()
            inverse_y_hat_cpu = final_pred.detach().cpu()
            roles = {
                "NAS": event["nas_nodes"],
                "FS": event["fs_nodes"],
                "AAS": event["aas_nodes"],
                "VS": event["vs_nodes"],
            }
            _append_routing_diagnostics(
                routing_diag_buffers,
                prediction_bundle.get("output_meta", {}),
                roles,
                final_pred,
                inverse_y,
                dataset,
            )
            for role_name, nodes in roles.items():
                pred_part, target_part = _slice_role_tensors(inverse_y_hat_cpu, inverse_y_cpu, nodes)
                if pred_part is None:
                    continue
                role_buffers[event["event_id"]][role_name]["pred"].append(pred_part)
                role_buffers[event["event_id"]][role_name]["true"].append(target_part)
                metrics = _compute_metrics(pred_part, target_part)
                if metrics is not None:
                    role_buffers[event["event_id"]][role_name]["window_metrics"].append(
                        {
                            "window_index": batch_ix,
                            "window_offset": batch_ix - event["evaluation_window_range"][0],
                            **metrics,
                        }
                    )
            evaluation_start, evaluation_end = event["evaluation_window_range"]
            evaluation_len = max(1, evaluation_end - evaluation_start + 1)
            early_horizon = max(1, min(6, evaluation_len // 4))
            window_offset = batch_ix - evaluation_start
            for role_name, nodes in roles.items():
                pred_part, target_part = _slice_role_tensors(inverse_y_hat_cpu, inverse_y_cpu, nodes)
                if pred_part is None:
                    continue
                if role_name == "VS":
                    state_name = "zero_history_virtual"
                elif role_name == "AAS":
                    state_name = "stable_observed"
                elif role_name == "NAS":
                    state_name = "new_sensor_early" if window_offset < early_horizon else "new_sensor_late"
                else:
                    state_name = "failed_sensor_early" if window_offset < early_horizon else "failed_sensor_late"
                _append_state_buffer(state_buffers, state_name, pred_part, target_part)

    event_metrics = {}
    summary_metrics = {role: [] for role in ["NAS", "FS", "AAS", "VS"]}
    for event in event_plan:
        event_id = event["event_id"]
        event_metrics[event_id] = {}
        for role_name, buffers in role_buffers[event_id].items():
            pred = _concat_or_none(buffers["pred"])
            target = _concat_or_none(buffers["true"])
            metrics = None if pred is None else _compute_metrics(pred, target)
            event_metrics[event_id][role_name] = {
                "metrics": metrics,
                "window_metrics": buffers["window_metrics"],
            }
            if metrics is not None:
                summary_metrics[role_name].append(metrics)

    summary_metrics = {role: _mean_metric_dicts(metrics) for role, metrics in summary_metrics.items()}
    state_metrics, state_macro_metrics, worst_state = _summarize_state_buffers(state_buffers)
    event_age_curve = _summarize_event_age_curve(event_metrics)
    role_gap = _compute_role_gap(summary_metrics)
    routing_diagnostics = _summarize_routing_diagnostics(routing_diag_buffers)
    results = {
        "config": {
            "async_adaptor": args.async_adaptor,
            "async_num_events": args.async_num_events,
            "async_commissioning_k": args.async_commissioning_k,
            "async_event_seed": args.async_event_seed,
            "input_len": args.input_len,
            "pred_len": args.pred_len,
            "slide_step": args.slide_step,
        },
        "adaptor_name": adaptor_state["name"],
        "adaptor_metadata": adaptor_state["metadata"],
        "adaptor_update_count_per_event": adaptor_state["update_count_per_event"],
        "adaptor_total_update_count": adaptor_state["total_update_count"],
        "adaptor_has_input_calibration": adaptor_state["has_input_calibration"],
        "adaptor_has_output_calibration": adaptor_state["has_output_calibration"],
        "event_plan": _to_builtin(event_plan),
        "summary_metrics": summary_metrics,
        "availability_state_metrics": state_metrics,
        "availability_state_macro_metrics": state_macro_metrics,
        "worst_availability_state": worst_state,
        "event_age_curve": event_age_curve,
        "role_gap": role_gap,
        "routing_diagnostics": routing_diagnostics,
        "event_metrics": event_metrics,
        "initial_aas_nodes": bundle["initial_aas_nodes"],
        "initial_vs_nodes": bundle["initial_vs_nodes"],
    }

    _restore_model_params(model, adaptor_state["model_param_requires_grad"])

    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_to_builtin(results), handle, ensure_ascii=False, indent=2)
    with open(text_path, "w", encoding="utf-8") as handle:
        adaptor_line = _format_adaptor_text(adaptor_state)
        if adaptor_line is not None:
            handle.write(adaptor_line + "\n")
            print(adaptor_line)
        for role_name in ["VS", "AAS", "NAS", "FS"]:
            metrics = summary_metrics[role_name]
            if metrics is None:
                continue
            line = (
                f"{role_name}: "
                f"MAPE={metrics['MAPE']:.4f} | "
                f"MAE={metrics['MAE']:.4f} | "
                f"RMSE={metrics['RMSE']:.4f}"
            )
            handle.write(line + "\n")
            print(line)
        if state_macro_metrics is not None:
            line = (
                f"AvailabilityMacro: "
                f"MAPE={state_macro_metrics['MAPE']:.4f} | "
                f"MAE={state_macro_metrics['MAE']:.4f} | "
                f"RMSE={state_macro_metrics['RMSE']:.4f}"
            )
            handle.write(line + "\n")
            print(line)
        for state_name in AVAILABILITY_STATE_ORDER:
            metrics = state_metrics[state_name]
            if metrics is None:
                continue
            line = (
                f"{state_name}: "
                f"MAPE={metrics['MAPE']:.4f} | "
                f"MAE={metrics['MAE']:.4f} | "
                f"RMSE={metrics['RMSE']:.4f}"
            )
            handle.write(line + "\n")
            print(line)
        if worst_state is not None:
            metrics = worst_state["metrics"]
            line = f"WorstAvailabilityState: {worst_state['name']} | MAPE={metrics['MAPE']:.4f}"
            handle.write(line + "\n")
            print(line)
        if role_gap is not None:
            line = (
                f"RoleGap: "
                f"MAPE={role_gap['MAPE']:.4f} | "
                f"MAE={role_gap['MAE']:.4f} | "
                f"RMSE={role_gap['RMSE']:.4f}"
            )
            handle.write(line + "\n")
            print(line)
        if routing_diagnostics and any(value is not None for value in routing_diagnostics.values()):
            handle.write("RoutingDiagnostics:\n")
            print("RoutingDiagnostics:")
            for role_name in ["VS", "AAS", "NAS", "FS"]:
                diagnostics = routing_diagnostics.get(role_name)
                if diagnostics is None:
                    continue
                graph_metrics = diagnostics.get("graph_metrics") or {}
                patch_metrics = diagnostics.get("patch_metrics") or {}
                fused_metrics = diagnostics.get("fused_metrics") or {}
                line = (
                    f"{role_name}: "
                    f"alpha={diagnostics['alpha_mean']:.4f} | "
                    f"graph_win={diagnostics['graph_win_rate']:.4f} | "
                    f"gate_match={diagnostics['gate_match_rate']:.4f} | "
                    f"G_MAPE={graph_metrics.get('MAPE', float('nan')):.4f} | "
                    f"P_MAPE={patch_metrics.get('MAPE', float('nan')):.4f} | "
                    f"F_MAPE={fused_metrics.get('MAPE', float('nan')):.4f}"
                )
                handle.write(line + "\n")
                print(line)
    print(f"-> Async lifecycle results saved to {output_dir}")

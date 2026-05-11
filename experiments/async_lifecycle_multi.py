import json
import os
import pickle
import queue
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from .async_calibrators import ADCSDModule, Calibration, CorrCoefLoss, FRPlusModule, SimpleOutputAdapter

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


def _validate_async_args(args):
    if args.input_len != 12 or args.pred_len != 12 or args.slide_step != 12:
        raise ValueError("async_lifecycle currently requires input_len=12, pred_len=12, and slide_step=12")
    if args.async_commissioning_k not in {6, 12, 24}:
        raise ValueError("async_lifecycle currently supports commissioning_k in {6, 12, 24}")
    if args.test_for_changed:
        raise ValueError("test_for_changed and test_for_async_lifecycle should not be enabled together")


def _prepare_paths(args):
    async_suffix = (
        f"async_lifecycle-{args.async_adaptor}-events-{args.async_num_events}-"
        f"k-{args.async_commissioning_k}-seed-{args.async_event_seed}"
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


def _prepare_st_ttc_input(x_enc_filled, observed_mask, FRP):
    x_calibrated = _apply_frp(FRP, x_enc_filled)
    return x_calibrated * observed_mask.to(dtype=x_calibrated.dtype)


def _forward_model(model, adj, model_input, enc_t_mark, pos_mark, dec_t_mark):
    y_hat = model(adj, model_input, enc_t_mark, pos_mark, None, dec_t_mark, 0)
    if isinstance(y_hat, tuple):
        y_hat = y_hat[-1]
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
        module = ADCSDModule(output_dim=output_channels, output_window=args.pred_len, num_nodes=dataset.num_nodes, moving_avg=ASYNC_ADCSD_MOVING_AVG).to(args.device)
        optimizer = torch.optim.Adam(module.parameters(), lr=ASYNC_ADCSD_LR, eps=1e-8, weight_decay=0, amsgrad=False)
        adaptor_state.update(
            {
                "module": module,
                "optimizer": optimizer,
                "has_output_calibration": True,
                "metadata": {
                    "queue_len": ASYNC_ADCSD_QUEUE_LEN,
                    "moving_avg": ASYNC_ADCSD_MOVING_AVG,
                    "lr": ASYNC_ADCSD_LR,
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
            y_hat = _forward_model(model, adj, x_enc_filled, enc_t_mark, pos_mark, dec_t_mark)
            final_pred = _inverse_transform_with_grad(dataset, y_hat)
        return {"final_pred": final_pred}

    if adaptor_name == "st_ttc":
        with torch.no_grad():
            model_input = _prepare_st_ttc_input(x_enc_filled, observed_mask, module)
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
            model_input, _, _, _ = module.input_calibration((x_enc_filled, enc_t_mark, None, dec_t_mark))
            model_input = model_input * observed_mask.to(dtype=model_input.dtype)
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
        model_input = _prepare_st_ttc_input(x_o, observed_mask_o, module)
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
            model_input, _, _, _ = module.input_calibration((x_enc_filled, enc_t_mark, None, dec_t_mark))
            model_input = model_input * observed_mask.to(dtype=model_input.dtype)
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
    print(f"-> Async lifecycle results saved to {output_dir}")

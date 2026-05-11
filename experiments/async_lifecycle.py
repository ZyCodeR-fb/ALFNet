import json
import os
import pickle
import queue

import numpy as np
import torch
import torch.nn as nn

ASYNC_TTC_QUEUE_LEN = 6
ASYNC_TTC_NAS_LOSS_WEIGHT = 0.8
ASYNC_TTC_AAS_LOSS_WEIGHT = 0.2
ASYNC_TTC_LR = 5e-4


class FRPlusModule(nn.Module):
    """Amplitude/phase calibration module copied from ST-TTC."""

    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        self.group_size = freq_bins // groups
        self.lambda_amp = nn.Parameter(torch.zeros(groups, 1, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, 1, 1))

    def forward(self, y_pred):
        # y_pred: [B,1,N,T]
        _, _, _, T = y_pred.shape
        y = y_pred[:, 0]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf)
        P = torch.angle(Yf)

        Yf_corr = torch.zeros_like(Yf)
        for g in range(self.groups):
            start = g * self.group_size
            end = T // 2 + 1 if g == self.groups - 1 else (g + 1) * self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)
            lam_p = self.lambda_phi[g].unsqueeze(0)
            A_g = A[:, :, start:end] * (1 + lam_a)
            P_g = P[:, :, start:end] + lam_p
            Yf_corr[:, :, start:end] = A_g * torch.exp(1j * P_g)

        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.unsqueeze(1)


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


def _prepare_model_input(x_enc_filled, observed_mask, FRP=None):
    if FRP is None:
        return x_enc_filled
    x_calibrated = _apply_frp(FRP, x_enc_filled)
    return x_calibrated * observed_mask.to(dtype=x_calibrated.dtype)


def _forward_model(model, adj, x_enc_filled, observed_mask, enc_t_mark, pos_mark, dec_t_mark, FRP=None):
    model_input = _prepare_model_input(x_enc_filled, observed_mask, FRP)
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


def _compute_weighted_calibration_loss(criterion, pred, target, aas_nodes, nas_nodes):
    weighted_losses = []

    pred_nas, target_nas = _slice_role_tensors(pred, target, nas_nodes)
    if pred_nas is not None:
        loss_nas = _compute_loss(criterion, pred_nas, target_nas)
        if loss_nas is not None:
            weighted_losses.append(ASYNC_TTC_NAS_LOSS_WEIGHT * loss_nas)

    pred_aas, target_aas = _slice_role_tensors(pred, target, aas_nodes)
    if pred_aas is not None:
        loss_aas = _compute_loss(criterion, pred_aas, target_aas)
        if loss_aas is not None:
            weighted_losses.append(ASYNC_TTC_AAS_LOSS_WEIGHT * loss_aas)

    if not weighted_losses:
        return None
    normalizer = 0.0
    if pred_nas is not None:
        normalizer += ASYNC_TTC_NAS_LOSS_WEIGHT
    if pred_aas is not None:
        normalizer += ASYNC_TTC_AAS_LOSS_WEIGHT
    return sum(weighted_losses) / normalizer


def _slice_role_tensors(pred, target, nodes):
    if not nodes:
        return None, None
    pred_part = pred[..., nodes, :]
    target_part = target[..., nodes, :]
    target_mask = ~torch.isnan(target_part)
    if target_mask.sum() == 0:
        return None, None
    return pred_part, target_part


def _compute_metrics(pred, target):
    target_mask = ~torch.isnan(target)
    if target_mask.sum() == 0:
        return None
    pred_valid = pred[target_mask]
    target_valid = target[target_mask]
    diff = pred_valid - target_valid
    return {
        "MAE": torch.mean(torch.abs(diff)).item(),
        "RMSE": torch.sqrt(torch.mean(diff**2)).item(),
        "MAPE": torch.mean(torch.abs(diff / target_valid)).item() * 100,
    }


def _inverse_transform_tensor(dataset, tensor):
    transformed = dataset.inverse_transform(tensor)
    if isinstance(transformed, np.ndarray):
        transformed = torch.from_numpy(transformed)
    return transformed


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

    FRP = None
    optim = None
    model_param_requires_grad = None
    if args.async_adaptor == "st_ttc":
        freq_bins = args.input_len // 2 + 1
        FRP = FRPlusModule(dataset.num_nodes, freq_bins, groups=4).to(args.device)
        optim = torch.optim.Adam(FRP.parameters(), lr=ASYNC_TTC_LR)
        FRP.eval()
        model_param_requires_grad = [param.requires_grad for param in model.parameters()]
        for param in model.parameters():
            param.requires_grad_(False)

    role_buffers = {}
    for event in event_plan:
        role_buffers[event["event_id"]] = {
            role: {"pred": [], "true": [], "window_metrics": []}
            for role in ["NAS", "FS", "AAS", "VS"]
        }

    active_queue = None
    active_event_index = None
    ttc_queue_len = ASYNC_TTC_QUEUE_LEN if args.async_adaptor == "st_ttc" else None
    ttc_update_count_per_event = {
        str(event["event_id"]): 0
        for event in event_plan
    }
    ttc_total_update_count = 0

    print("-> Start async lifecycle inference...")
    for batch_ix, batch in enumerate(data_loader):
        batch = [x.float().to(args.device) for x in batch]
        x_enc, x_dec, enc_t_mark, dec_t_mark = batch
        if not args.t_mark:
            enc_t_mark = None
        y = x_dec[:, -args.pred_len :].clone()
        x_enc_filled, observed_mask = _fill_missing_with_zero(x_enc)

        with torch.no_grad():
            y_hat = _forward_model(model, adj, x_enc_filled, observed_mask, enc_t_mark, pos_mark, dec_t_mark, FRP)

        y_cpu = y.detach().cpu()
        y_hat_cpu = y_hat.detach().cpu()
        inverse_y = _inverse_transform_tensor(dataset, y_cpu)
        inverse_y_hat = _inverse_transform_tensor(dataset, y_hat_cpu)

        stage = window_stage[batch_ix]
        event_index = window_event_index[batch_ix]
        if stage == "commissioning" and args.async_adaptor == "st_ttc":
            if active_event_index != event_index:
                active_event_index = event_index
                active_queue = queue.Queue(maxsize=ASYNC_TTC_QUEUE_LEN)
            active_queue.put(
                (
                    x_enc_filled.detach().clone(),
                    observed_mask.detach().clone(),
                    y.detach().clone(),
                    enc_t_mark.detach().clone() if enc_t_mark is not None else None,
                    dec_t_mark.detach().clone(),
                )
            )
            if active_queue.full():
                event = event_plan[event_index]
                x_o, observed_mask_o, y_o, enc_t_mark_o, dec_t_mark_o = active_queue.get()
                FRP.train()
                optim.zero_grad()
                yc_o = _forward_model(model, adj, x_o, observed_mask_o, enc_t_mark_o, pos_mark, dec_t_mark_o, FRP)
                yc_o = _inverse_transform_with_grad(dataset, yc_o)
                y_o = _inverse_transform_with_grad(dataset, y_o)
                loss = _compute_weighted_calibration_loss(
                    criterions[args.loss],
                    yc_o,
                    y_o,
                    event["aas_nodes"],
                    event["nas_nodes"],
                )
                if loss is not None:
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    ttc_update_count_per_event[str(event["event_id"])] += 1
                    ttc_total_update_count += 1
                FRP.eval()
        elif stage == "evaluation":
            event = event_plan[event_index]
            roles = {
                "NAS": event["nas_nodes"],
                "FS": event["fs_nodes"],
                "AAS": event["aas_nodes"],
                "VS": event["vs_nodes"],
            }
            for role_name, nodes in roles.items():
                pred_part, target_part = _slice_role_tensors(inverse_y_hat, inverse_y, nodes)
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
        "ttc_queue_len": ttc_queue_len,
        "ttc_update_count_per_event": ttc_update_count_per_event,
        "ttc_total_update_count": ttc_total_update_count,
        "ttc_lr": ASYNC_TTC_LR if args.async_adaptor == "st_ttc" else None,
        "ttc_loss_weights": {
            "NAS": ASYNC_TTC_NAS_LOSS_WEIGHT if args.async_adaptor == "st_ttc" else None,
            "AAS": ASYNC_TTC_AAS_LOSS_WEIGHT if args.async_adaptor == "st_ttc" else None,
        },
        "ttc_shared_across_nodes": True if args.async_adaptor == "st_ttc" else None,
        "event_plan": _to_builtin(event_plan),
        "summary_metrics": summary_metrics,
        "event_metrics": event_metrics,
        "initial_aas_nodes": bundle["initial_aas_nodes"],
        "initial_vs_nodes": bundle["initial_vs_nodes"],
    }

    if model_param_requires_grad is not None:
        for param, requires_grad in zip(model.parameters(), model_param_requires_grad):
            param.requires_grad_(requires_grad)

    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_to_builtin(results), handle, ensure_ascii=False, indent=2)
    with open(text_path, "w", encoding="utf-8") as handle:
        if args.async_adaptor == "st_ttc":
            ttc_line = (
                f"TTC: queue_len={ttc_queue_len} | "
                f"lr={ASYNC_TTC_LR:.0e} | "
                f"loss_w=(NAS:{ASYNC_TTC_NAS_LOSS_WEIGHT:.1f},AAS:{ASYNC_TTC_AAS_LOSS_WEIGHT:.1f}) | "
                f"total_updates={ttc_total_update_count} | "
                f"per_event={ttc_update_count_per_event}"
            )
            handle.write(ttc_line + "\n")
            print(ttc_line)
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


from .async_lifecycle_multi import async_lifecycle_inference

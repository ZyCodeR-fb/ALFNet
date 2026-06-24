import torch
from utils.logtool import detail_metrics
import random
import torch.nn.functional as F
from utils.model_output import sum_aux_losses, unpack_model_output

ROLE_CHANNEL_COUNT = 4


def _extract_last_role_from_input(x_enc, args):
    if not getattr(args, "lifecycle_role_features", False):
        return None
    role_start = int(getattr(args, "raw_enc_in", 1))
    role_end = role_start + ROLE_CHANNEL_COUNT
    if x_enc.size(-1) < role_end:
        return None
    return x_enc[:, -1, :, role_start:role_end]


def _masked_loss(y_hat, y, mask, loss_name):
    if mask.sum() <= 0:
        return None
    safe_y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    safe_diff = torch.where(mask, y_hat - safe_y, torch.zeros_like(y_hat))
    if loss_name == "mse":
        return safe_diff.pow(2)[mask].mean()
    if loss_name == "rmse":
        return torch.sqrt(safe_diff.pow(2)[mask].mean())
    if loss_name == "mae":
        return safe_diff.abs()[mask].mean()
    if loss_name == "huber":
        per_element = F.huber_loss(y_hat, safe_y, reduction="none")
        return per_element[mask].mean()
    return None


def _element_loss(y_hat, y, loss_name):
    safe_y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if loss_name == "mse" or loss_name == "rmse":
        return (y_hat - safe_y).pow(2)
    if loss_name == "mae":
        return torch.abs(y_hat - safe_y)
    if loss_name == "huber":
        return F.huber_loss(y_hat, safe_y, reduction="none")
    return None


def _role_balanced_average(element_loss, valid_mask, last_role):
    if last_role is None:
        if valid_mask.sum() <= 0:
            return None
        return element_loss[valid_mask].mean()
    role_losses = []
    for role_idx in range(ROLE_CHANNEL_COUNT):
        role_mask = last_role[..., role_idx].unsqueeze(1).unsqueeze(-1) > 0.5
        role_mask = role_mask.expand_as(valid_mask)
        mask = valid_mask & role_mask
        if mask.any():
            role_losses.append(element_loss[mask].mean())
    if not role_losses:
        return None
    return torch.stack(role_losses).mean()


def _compute_focal_forecast_loss(y_hat, y, y_omi, last_role, loss_name, gamma=1.0, max_weight=5.0, role_balanced=False):
    element_loss = _element_loss(y_hat, y, loss_name)
    if element_loss is None:
        return None
    valid_mask = (~y_omi) & torch.isfinite(y_hat) & torch.isfinite(y)
    if not valid_mask.any():
        return None
    valid_loss = element_loss[valid_mask].detach()
    scale = valid_loss.mean().clamp_min(1e-6)
    weights = (element_loss.detach() / scale).clamp_min(0.0).pow(float(gamma))
    weights = weights.clamp(max=float(max_weight))
    weighted_loss = element_loss * weights
    if role_balanced:
        return _role_balanced_average(weighted_loss, valid_mask, last_role)
    return weighted_loss[valid_mask].sum() / weights[valid_mask].sum().clamp_min(1e-6)


def _compute_role_balanced_forecast_loss(y_hat, y, y_omi, last_role, loss_name):
    if last_role is None:
        return None
    valid_mask = (~y_omi) & torch.isfinite(y_hat) & torch.isfinite(y)
    role_losses = []
    for role_idx in range(ROLE_CHANNEL_COUNT):
        role_mask = last_role[..., role_idx].unsqueeze(1).unsqueeze(-1) > 0.5
        role_mask = role_mask.expand_as(valid_mask)
        loss = _masked_loss(y_hat, y, valid_mask & role_mask, loss_name)
        if loss is not None:
            role_losses.append(loss)
    if not role_losses:
        return None
    return torch.stack(role_losses).mean()


def _compute_expert_role_loss(expert_training, y_for_loss, y_omi, last_role, loss_name):
    lambda_expert_role = float(expert_training.get("lambda_expert_role", 0.0))
    if lambda_expert_role <= 0 or last_role is None:
        return None
    graph_weight = float(expert_training.get("expert_role_graph_weight", 1.0))
    patch_weight = float(expert_training.get("expert_role_patch_weight", 1.0))
    y_graph = expert_training["y_graph"]
    y_patch = expert_training["y_patch"]
    valid_mask = (~y_omi) & torch.isfinite(y_for_loss) & torch.isfinite(y_graph) & torch.isfinite(y_patch)
    if not valid_mask.any():
        return None

    graph_role_mask = (last_role[..., 1:2] + last_role[..., 2:3]).unsqueeze(1) > 0.5
    patch_role_mask = (last_role[..., 0:1] + last_role[..., 3:4]).unsqueeze(1) > 0.5
    graph_role_mask = graph_role_mask.expand_as(valid_mask)
    patch_role_mask = patch_role_mask.expand_as(valid_mask)

    losses = []
    weights = []
    graph_loss = _masked_loss(y_graph, y_for_loss, valid_mask & graph_role_mask, loss_name)
    patch_loss = _masked_loss(y_patch, y_for_loss, valid_mask & patch_role_mask, loss_name)
    if graph_loss is not None and graph_weight > 0:
        losses.append(graph_weight * graph_loss)
        weights.append(graph_weight)
    if patch_loss is not None and patch_weight > 0:
        losses.append(patch_weight * patch_loss)
        weights.append(patch_weight)
    if not losses:
        return None
    return lambda_expert_role * torch.stack(losses).sum() / max(sum(weights), 1e-6)


def _compute_patch_residual_delta_loss(patch_residual_training, y_for_loss, y_omi, last_role):
    lambda_patch_delta = float(patch_residual_training.get("lambda_patch_delta", 0.0))
    if lambda_patch_delta <= 0:
        return None
    residual = patch_residual_training["residual"]
    y_graph = patch_residual_training["y_graph"]
    valid_mask = (~y_omi) & torch.isfinite(y_for_loss) & torch.isfinite(y_graph) & torch.isfinite(residual)
    if not valid_mask.any():
        return None

    target_delta = torch.where(valid_mask, y_for_loss - y_graph, torch.zeros_like(residual)).detach()
    safe_residual = torch.where(valid_mask, residual, torch.zeros_like(residual))
    delta_loss = F.huber_loss(safe_residual, target_delta, reduction="none")
    if last_role is None:
        return lambda_patch_delta * delta_loss[valid_mask].mean()

    role_weight = patch_residual_training.get("role_weight", None)
    if role_weight is None:
        weights = torch.ones_like(delta_loss)
    else:
        role_weight = role_weight.to(device=delta_loss.device, dtype=delta_loss.dtype)
        weights = torch.sum(last_role.to(dtype=delta_loss.dtype) * role_weight.view(1, 1, ROLE_CHANNEL_COUNT), dim=-1, keepdim=True)
        weights = weights.unsqueeze(1).expand_as(delta_loss)
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    if weights.sum() <= 0:
        return None
    return lambda_patch_delta * (delta_loss * weights).sum() / weights.sum()


def _compute_regret_loss(y_hat, y, y_omi, y_graph, y_patch, last_role):
    valid_mask = (~y_omi) & torch.isfinite(y_hat) & torch.isfinite(y_graph) & torch.isfinite(y_patch) & torch.isfinite(y)
    if not valid_mask.any():
        return None
    fusion_error = torch.abs(y_hat - y)
    best_expert_error = torch.minimum(torch.abs(y_graph - y), torch.abs(y_patch - y)).detach()
    regret = torch.relu(fusion_error - best_expert_error)
    return _role_balanced_average(regret, valid_mask, last_role)


def _compute_weighted_train_loss(y_hat, y, y_omi, sample_role_weights, args):
    if sample_role_weights is None:
        return None
    valid_mask_bool = (~y_omi) & torch.isfinite(y_hat) & torch.isfinite(y)
    valid_mask = valid_mask_bool.to(dtype=y_hat.dtype)
    weights = sample_role_weights.to(device=y_hat.device, dtype=y_hat.dtype) * valid_mask
    if weights.sum() <= 0:
        return None
    safe_y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    safe_diff = torch.where(valid_mask_bool, y_hat - safe_y, torch.zeros_like(y_hat))
    if args.loss == "mse":
        per_element = safe_diff.pow(2)
        return (per_element * weights).sum() / weights.sum()
    if args.loss == "rmse":
        per_element = safe_diff.pow(2)
        mse = (per_element * weights).sum() / weights.sum()
        return torch.sqrt(mse)
    if args.loss == "mae":
        per_element = safe_diff.abs()
        return (per_element * weights).sum() / weights.sum()
    if args.loss == "huber":
        per_element = F.huber_loss(y_hat, safe_y, reduction="none")
        per_element = torch.where(valid_mask_bool, per_element, torch.zeros_like(per_element))
        return (per_element * weights).sum() / weights.sum()
    return None


def _compute_testam_routing_loss(testam_routing, y_for_loss, y_omi, epoch, last_role=None):
    # Adapted from references/TESTAM/engine.py::get_label with the same
    # worst-avoidance + best-choice pseudo-label routing idea.
    lambda_testam = float(testam_routing.get("lambda_testam", 0.0))
    warmup_epoch = int(testam_routing.get("warmup_epoch", 0))
    if lambda_testam <= 0 or epoch <= warmup_epoch:
        return None

    gate = testam_routing["gate"].clamp(1e-6, 1.0 - 1e-6)
    expert_preds = testam_routing["expert_preds"]
    if gate.dim() == y_for_loss.dim():
        gate = gate.unsqueeze(-2)
    if gate.size(-2) == 1 and y_for_loss.size(-1) != 1:
        gate = gate.expand(*gate.shape[:-2], y_for_loss.size(-1), gate.size(-1))

    valid_mask = (~y_omi) & torch.isfinite(y_for_loss) & torch.isfinite(expert_preds).all(dim=-1)
    if not valid_mask.any():
        return None

    ind_loss = torch.abs(expert_preds - y_for_loss.unsqueeze(-1))
    empty_val = ~valid_mask.unsqueeze(-1).expand_as(gate)

    max_error = ind_loss.argmax(dim=-1, keepdim=True)
    cur_expert = gate.argmax(dim=-1, keepdim=True)
    incorrect = max_error == cur_expert
    selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0)
    scaling = torch.ones_like(gate) * ind_loss.detach()
    scaling = scaling.scatter(-1, max_error, 0.0)
    scaling = scaling / scaling.sum(dim=-1, keepdim=True).clamp_min(1e-6) * (1.0 - selected)
    l_worst_avoidance = torch.where(incorrect, scaling, selected)
    l_worst_avoidance = torch.where(empty_val, torch.zeros_like(gate), l_worst_avoidance).detach()

    min_error = ind_loss.argmin(dim=-1, keepdim=True)
    correct = min_error == cur_expert
    scaling = torch.zeros_like(gate).scatter(-1, min_error, 1.0)
    l_best_choice = torch.where(correct, selected, scaling)
    l_best_choice = torch.where(empty_val, torch.zeros_like(gate), l_best_choice).detach()

    routing_loss = -0.5 * l_worst_avoidance * torch.log(gate) - 0.5 * l_best_choice * torch.log(gate)
    if bool(testam_routing.get("role_balanced", False)):
        element_loss = routing_loss.sum(dim=-1)
        balanced_loss = _role_balanced_average(element_loss, valid_mask, last_role)
        return None if balanced_loss is None else lambda_testam * balanced_loss
    return lambda_testam * routing_loss[~empty_val].mean()


def _compute_lifecycle_route_loss(route_routing, y_for_loss, y_omi, epoch, last_role=None):
    lambda_route = float(route_routing.get("lambda_route", 0.0))
    warmup_epoch = int(route_routing.get("warmup_epoch", 0))
    if lambda_route <= 0 or epoch <= warmup_epoch:
        return None

    alpha = route_routing["alpha"].clamp(1e-6, 1.0 - 1e-6)
    y_graph = route_routing["y_graph"]
    y_patch = route_routing["y_patch"]
    valid_mask = (~y_omi) & torch.isfinite(y_for_loss) & torch.isfinite(y_graph) & torch.isfinite(y_patch)
    if not valid_mask.any():
        return None

    graph_err = torch.abs(y_graph - y_for_loss)
    patch_err = torch.abs(y_patch - y_for_loss)
    if graph_err.size(-1) != 1:
        graph_err = graph_err.mean(dim=-1, keepdim=True)
        patch_err = patch_err.mean(dim=-1, keepdim=True)
        valid_mask = valid_mask.all(dim=-1, keepdim=True)
    if alpha.shape != graph_err.shape:
        alpha = alpha.expand_as(graph_err)

    target_type = str(route_routing.get("target", "hard_bce")).lower()
    if target_type in {"soft_kl", "soft"}:
        temperature = max(float(route_routing.get("temperature", 1.0)), 1e-6)
        pair_err = torch.cat([graph_err, patch_err], dim=-1)
        pair_valid = valid_mask.expand_as(pair_err)
        err_scale = pair_err[pair_valid].detach().mean().clamp_min(1e-6)
        route_logits = -torch.nan_to_num(pair_err / err_scale, nan=0.0, posinf=0.0, neginf=0.0) / temperature
        target = torch.softmax(route_logits, dim=-1).detach()
        gate = torch.cat([alpha, 1.0 - alpha], dim=-1)
        route_loss = -(target * torch.log(gate)).sum(dim=-1, keepdim=True)
    else:
        target = (graph_err <= patch_err).to(dtype=alpha.dtype).detach()
        route_loss = F.binary_cross_entropy(alpha, target, reduction="none")

    if bool(route_routing.get("role_balanced", False)):
        balanced_loss = _role_balanced_average(route_loss, valid_mask, last_role)
        return None if balanced_loss is None else lambda_route * balanced_loss
    return lambda_route * route_loss[valid_mask].mean()


def shared_step(model, dataloader, optimizer, criterions, args, mode, train, epoch):
    # Save step-wise metrics
    inverse_transform = dataloader.dataset.inverse_transform
    metrics = {}
    for metric_name, criterion in criterions.items():
        if metric_name == "split_phy_loss":
            for sub_metric_name in criterion.sub_metrics:
                metrics[sub_metric_name] = 0.0
        else:
            metrics[metric_name] = 0.0
    metrics2 = metrics.copy()
    metrics3 = metrics.copy()
    metrics4 = metrics.copy()
    metrics_role = metrics.copy()
    if mode == "train" and train == True:
        model.train()
        model.zero_grad()
        grad_enabled = True
        if hasattr(model, "on_train_epoch_start"):
            model.on_train_epoch_start(epoch)
    else:
        model.eval()
        grad_enabled = False

    # Save predictions
    total_y_true = []
    total_y_pred = []
    total_y_true_trans = []
    total_y_pred_trans = []
    part_nodes = dict(zip(("always_known_nodes", "never_known_nodes ", "new_add_nodes     ", "failed_nodes      "), dataloader.dataset.get_know_unknow_nodes()))

    with torch.set_grad_enabled(grad_enabled):
        model.to(args.device)
        max_bacth_num = min(600, len(dataloader)) if mode == "train" else len(dataloader)
        for batch_ix, batch in enumerate(dataloader):
            if batch_ix >= max_bacth_num:
                break
            batch = [x.float().to(args.device) for x in batch]
            sample_role_weights = None
            if len(batch) == 5:
                x_enc, x_dec, enc_t_mark, dec_t_mark, sample_role_weights = batch
            else:
                x_enc, x_dec, enc_t_mark, dec_t_mark = batch
            if not args.t_mark:
                enc_t_mark = None
            y = x_dec[:, -args.pred_len :].clone()
            x_dec = x_dec[:, -args.pred_len :]
            adj, pos_mark = dataloader.dataset.get_adj_and_pos_mark()
            if not args.pos_mark:
                pos_mark = None
            adj = torch.from_numpy(adj).float().to(args.device)
            if pos_mark is not None:
                pos_mark = torch.from_numpy(pos_mark).float().to(args.device)
            raw_nan_mask = torch.isnan(x_enc[..., :args.raw_enc_in]).clone()
            x_enc[torch.isnan(x_enc)] = 0  # 在源数据集中，有些异常值被标注为0，这些异常值不参加任何评价计算
            x_dec[torch.isnan(x_dec)] = 0
            if mode != "test":
                # 每次去掉一些节点(10%)（从图中直接拿掉）#模拟传感器退出和新增，且减少计算复杂度
                # batch_known_nodes = list(random.sample(range(0, x_enc.size(2)),int(x_enc.size(2) * 0.9)))
                batch_known_nodes = list(range(0, x_enc.size(2)))
                x_enc = x_enc[..., batch_known_nodes, :]
                raw_nan_mask = raw_nan_mask[..., batch_known_nodes, :]
                adj = adj[batch_known_nodes, :][:, batch_known_nodes]
                if pos_mark is not None:
                    pos_mark = pos_mark[batch_known_nodes, :]
                x_dec = x_dec[..., batch_known_nodes, :]
                y = y[..., batch_known_nodes, :]
                if sample_role_weights is not None:
                    sample_role_weights = sample_role_weights[..., batch_known_nodes, :]
                # 每次还得给一些节点(25%)赋值为0，保留图结构
                should_apply_random_mask = (not args.train_for_async_lifecycle) or args.lifecycle_keep_random_mask
                if should_apply_random_mask:
                    batch_mask = random.sample(range(0, len(batch_known_nodes)), int(len(batch_known_nodes) * 0.25))
                else:
                    batch_mask = torch.where(raw_nan_mask.any(dim=(0, 1, 3)))[0].cpu().tolist()
                batch_unmask = list(set(range(0, len(batch_known_nodes))) - set(batch_mask))
                if should_apply_random_mask and batch_mask:
                    x_enc[..., batch_mask, :args.raw_enc_in] = 0
            if mode != "train":
                x_dec = None
            model_output = model(adj, x_enc, enc_t_mark, pos_mark, x_dec, dec_t_mark, epoch)
            y_hat, output_meta = unpack_model_output(model_output)
            y_for_loss = output_meta.get("target", y)
            last_role = _extract_last_role_from_input(x_enc, args)
            if sample_role_weights is not None and sample_role_weights.shape != y_hat.shape:
                sample_role_weights = None
            y_omi = torch.isnan(y_for_loss)
            if (~y_omi).sum() == 0:
                continue
            loss = criterions[args.loss](y_hat[~y_omi], y_for_loss[~y_omi])
            optimize_loss = loss
            if grad_enabled:
                forecast_loss_mode = str(output_meta.get("forecast_loss_mode", "standard")).lower()
                role_balanced_modes = {"role_balanced", "role_balanced_forecast"}
                role_balanced_mix_modes = {"role_balanced_mix", "mixed_role_balanced", "mix_role_balanced"}
                focal_modes = {"focal", "focal_forecast", "role_focal", "role_balanced_focal"}
                focal_mix_modes = {"focal_mix", "mixed_focal"}
                if forecast_loss_mode in role_balanced_modes | role_balanced_mix_modes:
                    role_balanced_loss = _compute_role_balanced_forecast_loss(y_hat, y_for_loss, y_omi, last_role, args.loss)
                    if role_balanced_loss is not None:
                        if forecast_loss_mode in role_balanced_mix_modes:
                            beta = float(output_meta.get("forecast_role_balanced_beta", 0.5))
                            beta = max(0.0, min(1.0, beta))
                            optimize_loss = (1.0 - beta) * optimize_loss + beta * role_balanced_loss
                        else:
                            optimize_loss = role_balanced_loss
                if forecast_loss_mode in focal_modes | focal_mix_modes:
                    focal_loss = _compute_focal_forecast_loss(
                        y_hat,
                        y_for_loss,
                        y_omi,
                        last_role,
                        args.loss,
                        gamma=float(output_meta.get("forecast_focal_gamma", 1.0)),
                        max_weight=float(output_meta.get("forecast_focal_max_weight", 5.0)),
                        role_balanced=forecast_loss_mode in {"role_focal", "role_balanced_focal"},
                    )
                    if focal_loss is not None:
                        if forecast_loss_mode in focal_mix_modes:
                            beta = float(output_meta.get("forecast_focal_beta", 0.5))
                            beta = max(0.0, min(1.0, beta))
                            optimize_loss = (1.0 - beta) * optimize_loss + beta * focal_loss
                        else:
                            optimize_loss = focal_loss
                weighted_loss = _compute_weighted_train_loss(y_hat, y_for_loss, y_omi, sample_role_weights, args)
                non_standard_forecast = role_balanced_modes | role_balanced_mix_modes | focal_modes | focal_mix_modes
                if weighted_loss is not None and forecast_loss_mode not in non_standard_forecast:
                    optimize_loss = weighted_loss
                aux_loss = sum_aux_losses(output_meta.get("aux_losses"), device=y_hat.device)
                if aux_loss is not None:
                    optimize_loss = optimize_loss + aux_loss
                oracle_routing = output_meta.get("oracle_routing")
                if oracle_routing:
                    alpha = oracle_routing["alpha"].clamp(1e-6, 1.0 - 1e-6)
                    y_graph = oracle_routing["y_graph"]
                    y_patch = oracle_routing["y_patch"]
                    valid_mask = (~y_omi) & torch.isfinite(y_graph) & torch.isfinite(y_patch) & torch.isfinite(y_for_loss)
                    if valid_mask.any():
                        graph_err = torch.abs(y_graph - y_for_loss)
                        patch_err = torch.abs(y_patch - y_for_loss)
                        oracle_target = (graph_err <= patch_err).to(dtype=alpha.dtype)
                        oracle_bce = F.binary_cross_entropy(alpha, oracle_target, reduction="none")
                        oracle_loss = (oracle_bce * valid_mask.to(dtype=oracle_bce.dtype)).sum() / valid_mask.to(dtype=oracle_bce.dtype).sum()
                        optimize_loss = optimize_loss + float(oracle_routing.get("lambda_oracle", 0.1)) * oracle_loss
                    lambda_role_balanced = float(oracle_routing.get("lambda_role_balanced", 0.0))
                    if lambda_role_balanced > 0:
                        role_balanced_loss = _compute_role_balanced_forecast_loss(y_hat, y_for_loss, y_omi, last_role, args.loss)
                        if role_balanced_loss is not None:
                            optimize_loss = optimize_loss + lambda_role_balanced * role_balanced_loss
                    lambda_regret = float(oracle_routing.get("lambda_regret", 0.0))
                    if lambda_regret > 0:
                        regret_loss = _compute_regret_loss(y_hat, y_for_loss, y_omi, y_graph, y_patch, last_role)
                        if regret_loss is not None:
                            optimize_loss = optimize_loss + lambda_regret * regret_loss
                testam_routing = output_meta.get("testam_routing")
                if testam_routing:
                    testam_loss = _compute_testam_routing_loss(testam_routing, y_for_loss, y_omi, epoch, last_role)
                    if testam_loss is not None:
                        optimize_loss = optimize_loss + testam_loss
                route_routing = output_meta.get("route_routing")
                if route_routing:
                    route_loss = _compute_lifecycle_route_loss(route_routing, y_for_loss, y_omi, epoch, last_role)
                    if route_loss is not None:
                        optimize_loss = optimize_loss + route_loss
                expert_training = output_meta.get("expert_training")
                if expert_training:
                    expert_role_loss = _compute_expert_role_loss(expert_training, y_for_loss, y_omi, last_role, args.loss)
                    if expert_role_loss is not None:
                        optimize_loss = optimize_loss + expert_role_loss
                patch_residual_training = output_meta.get("patch_residual_training")
                if patch_residual_training:
                    patch_delta_loss = _compute_patch_residual_delta_loss(patch_residual_training, y_for_loss, y_omi, last_role)
                    if patch_delta_loss is not None:
                        optimize_loss = optimize_loss + patch_delta_loss
            if not torch.isfinite(optimize_loss):
                if grad_enabled:
                    optimizer.zero_grad()
                continue
            if grad_enabled:
                optimize_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            role_metric_loss = None
            if mode != "test":
                with torch.no_grad():
                    detached_role = last_role.detach() if last_role is not None else None
                    role_metric_loss = _compute_role_balanced_forecast_loss(
                        y_hat.detach(),
                        y_for_loss.detach(),
                        y_omi.detach(),
                        detached_role,
                        args.loss,
                    )
                    if role_metric_loss is None:
                        role_metric_loss = loss.detach()
            y = y_for_loss.detach().cpu()
            y_hat = y_hat.detach().cpu()
            y_omi = y_omi.detach().cpu()
            inverse_y = inverse_transform(y)
            inverse_y_hat = inverse_transform(y_hat)
            total_y_true.append(y)
            total_y_pred.append(y_hat)
            total_y_true_trans.append(inverse_y)
            total_y_pred_trans.append(inverse_y_hat)
            if mode != "test":
                metrics = detail_metrics(
                    dataloader, criterions, args, mode + "_total ", metrics, y_omi, loss, inverse_y, inverse_y_hat, max_bacth_num, grad_enabled, batch_ix
                )
                metrics_role = detail_metrics(
                    dataloader,
                    criterions,
                    args,
                    mode + "_role  ",
                    metrics_role,
                    y_omi,
                    role_metric_loss,
                    inverse_y,
                    inverse_y_hat,
                    max_bacth_num,
                    grad_enabled,
                    batch_ix,
                )
                for i, mask in enumerate([batch_mask, batch_unmask]):
                    if len(mask) == 0:
                        continue
                    inverse_y_hat_part = inverse_y_hat[..., mask, :]
                    inverse_y_part = inverse_y[..., mask, :]
                    y_hat_part = y_hat[..., mask, :]
                    y_part = y[..., mask, :]
                    y_omi_part = torch.isnan(inverse_y_part)
                    if (~y_omi_part).sum() == 0:
                        continue
                    loss = criterions[args.loss](y_hat_part[~y_omi_part], y_part[~y_omi_part])
                    if i == 0:
                        metrics2 = detail_metrics(
                            dataloader,
                            criterions,
                            args,
                            mode + "_mask  ",
                            metrics2,
                            y_omi_part,
                            loss,
                            inverse_y_part,
                            inverse_y_hat_part,
                            max_bacth_num,
                            grad_enabled,
                            batch_ix,
                        )
                    else:
                        metrics3 = detail_metrics(
                            dataloader,
                            criterions,
                            args,
                            mode + "_unmask",
                            metrics3,
                            y_omi_part,
                            loss,
                            inverse_y_part,
                            inverse_y_hat_part,
                            max_bacth_num,
                            grad_enabled,
                            batch_ix,
                        )
    if not total_y_true:
        raise RuntimeError(
            f"No valid batches were collected in {mode} at epoch {epoch}. "
            "All losses/predictions were non-finite or all targets were missing; "
            "check the current loss configuration and learning rates."
        )

    total_y = {
        "true": torch.cat(total_y_true, dim=0),
        "pred": torch.cat(total_y_pred, dim=0),
        "true_trans": torch.cat(total_y_true_trans, dim=0),
        "pred_trans": torch.cat(total_y_pred_trans, dim=0),
    }

    if mode != "test":
        for metric_name in metrics.keys():
            metrics[metric_name] /= batch_ix + 1  # average over batches
        for metric_name in metrics2.keys():
            metrics2[metric_name] /= batch_ix + 1
        for metric_name in metrics3.keys():
            metrics3[metric_name] /= batch_ix + 1
        for metric_name in metrics_role.keys():
            metrics_role[metric_name] /= batch_ix + 1

    else:
        for name, nodes in part_nodes.items():
            if len(nodes) == 0:
                continue
            inverse_y_hat_part = total_y["pred_trans"][..., nodes, :]
            inverse_y_part = total_y["true_trans"][..., nodes, :]
            y_hat_part = total_y["pred"][..., nodes, :]
            y_part = total_y["true"][..., nodes, :]
            y_omi_part = torch.isnan(inverse_y_part)
            loss = criterions[args.loss](y_hat_part[~y_omi_part], y_part[~y_omi_part])
            if name == "always_known_nodes":
                metrics = detail_metrics(
                    dataloader,
                    criterions,
                    args,
                    mode + f"_{name}",
                    metrics,
                    y_omi_part,
                    loss,
                    inverse_y_part,
                    inverse_y_hat_part,
                    max_bacth_num,
                    grad_enabled,
                    batch_ix,
                )
            elif name == "never_known_nodes ":
                metrics2 = detail_metrics(
                    dataloader,
                    criterions,
                    args,
                    mode + f"_{name}",
                    metrics2,
                    y_omi_part,
                    loss,
                    inverse_y_part,
                    inverse_y_hat_part,
                    max_bacth_num,
                    grad_enabled,
                    batch_ix,
                )
            elif name == "new_add_nodes     ":
                metrics3 = detail_metrics(
                    dataloader,
                    criterions,
                    args,
                    mode + f"_{name}",
                    metrics3,
                    y_omi_part,
                    loss,
                    inverse_y_part,
                    inverse_y_hat_part,
                    max_bacth_num,
                    grad_enabled,
                    batch_ix,
                )
            else:
                metrics4 = detail_metrics(
                    dataloader,
                    criterions,
                    args,
                    mode + f"_{name}",
                    metrics4,
                    y_omi_part,
                    loss,
                    inverse_y_part,
                    inverse_y_hat_part,
                    max_bacth_num,
                    grad_enabled,
                    batch_ix,
                )
    return model, (metrics, metrics2, metrics3, metrics4, metrics_role), total_y

import torch
from utils.logtool import detail_metrics
import random
import torch.nn.functional as F


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
    if mode == "train" and train == True:
        model.train()
        model.zero_grad()
        grad_enabled = True
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
            y_hat = model(adj, x_enc, enc_t_mark, pos_mark, x_dec, dec_t_mark, epoch)
            y_omi = torch.isnan(y)
            if (~y_omi).sum() == 0:
                continue
            loss = criterions[args.loss](y_hat[~y_omi], y[~y_omi])
            optimize_loss = loss
            if grad_enabled:
                weighted_loss = _compute_weighted_train_loss(y_hat, y, y_omi, sample_role_weights, args)
                if weighted_loss is not None:
                    optimize_loss = weighted_loss
            if not torch.isfinite(optimize_loss):
                if grad_enabled:
                    optimizer.zero_grad()
                continue
            if grad_enabled:
                optimize_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            y = y.detach().cpu()
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
    return model, (metrics, metrics2, metrics3, metrics4), total_y

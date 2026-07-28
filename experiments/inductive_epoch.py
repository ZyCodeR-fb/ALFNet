import torch
import torch.nn.functional as F

from data_provider.calendar_strata import STRATA, stratum_weights
from layers.evidence_router import routing_diagnostics
from data_provider.lcbt_protocol import (
    accumulate_lifecycle_mae,
    finalize_lifecycle_validation_mae,
)
from utils.model_output import sum_aux_losses, unpack_model_output


ROLE_CHANNEL_COUNT = 4
ROLE_AAS = 0


def _last_role(x_enc, args):
    role_start = int(args.raw_enc_in)
    role_end = role_start + ROLE_CHANNEL_COUNT
    if x_enc.size(-1) < role_end:
        raise ValueError("LCBT v3 input is missing the four availability channels")
    return x_enc[:, -1, :, role_start:role_end]


def _element_loss(y_hat, y, loss_name):
    safe_y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if loss_name in {"mse", "rmse"}:
        return (y_hat - safe_y).pow(2)
    if loss_name == "mae":
        return torch.abs(y_hat - safe_y)
    if loss_name == "huber":
        return F.huber_loss(y_hat, safe_y, reduction="none")
    raise ValueError(f"Unsupported training loss: {loss_name}")


def _role_balanced_loss(y_hat, y, valid_mask, last_role, loss_name):
    element_loss = _element_loss(y_hat, y, loss_name)
    losses = []
    for role_index in range(ROLE_CHANNEL_COUNT):
        role_mask = last_role[..., role_index].unsqueeze(1).unsqueeze(-1) > 0.5
        selected = valid_mask & role_mask.expand_as(valid_mask)
        if selected.any():
            losses.append(element_loss[selected].mean())
    return torch.stack(losses).mean() if losses else None


def _context_loss(y_hat, y, cohort_valid, last_role, loss_name):
    """Ordinary forecasting on the steadily-online nodes outside the cohort."""
    steady = last_role[..., ROLE_AAS].unsqueeze(1).unsqueeze(-1) > 0.5
    selected = (
        steady.expand_as(cohort_valid)
        & ~cohort_valid
        & torch.isfinite(y)
        & torch.isfinite(y_hat)
    )
    if not selected.any():
        return None
    return _element_loss(y_hat, y, loss_name)[selected].mean()


def _stratum_balanced_loss(y_hat, y, valid_mask, calendar_ids, loss_name):
    """Re-weight the batch onto the *evaluation* calendar-stratum weights.

    LCBT draws training windows from whatever the schedule offers, so the
    empirical stratum mix does not match the equal-over-tod / 5:7 day-type
    weighting LCE and PLES score with: measured training-to-evaluation weight
    ratios run 0.62 (AM/PM peak, under-weighted) to 1.55 (EarlyAM,
    over-weighted). This makes the training objective an unbiased estimate of
    the reported one. It is off by default because turning it on changes the
    optimisation target, which is a model decision, not a protocol fix.
    """
    element_loss = _element_loss(y_hat, y, loss_name)
    weights = stratum_weights()
    terms, total = [], 0.0
    ids = calendar_ids.reshape(-1)
    for index, name in enumerate(STRATA):
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        sample_mask = (ids == index).view(-1, *([1] * (valid_mask.dim() - 1)))
        selected = valid_mask & sample_mask.expand_as(valid_mask)
        if selected.any():
            terms.append(weight * element_loss[selected].mean())
            total += weight
    if not terms or total <= 0:
        return None
    return torch.stack(terms).sum() / total


def _route_loss(route, y, valid_mask, last_role, epoch):
    weight = float(route.get("lambda_route", 0.0))
    if weight <= 0 or epoch <= int(route.get("warmup_epoch", 0)):
        return None
    alpha = route["alpha"].clamp(1e-6, 1.0 - 1e-6)
    graph_error = torch.abs(route["y_graph"] - y)
    patch_error = torch.abs(route["y_patch"] - y)
    if graph_error.size(-1) != 1:
        graph_error = graph_error.mean(dim=-1, keepdim=True)
        patch_error = patch_error.mean(dim=-1, keepdim=True)
        valid_mask = valid_mask.all(dim=-1, keepdim=True)
    alpha = alpha.expand_as(graph_error)
    target = (graph_error <= patch_error).to(alpha.dtype).detach()
    per_element = F.binary_cross_entropy(alpha, target, reduction="none")
    if bool(route.get("role_balanced", False)):
        losses = []
        for role_index in range(ROLE_CHANNEL_COUNT):
            role_mask = last_role[..., role_index].unsqueeze(1).unsqueeze(-1) > 0.5
            selected = valid_mask & role_mask.expand_as(valid_mask)
            if selected.any():
                losses.append(per_element[selected].mean())
        return None if not losses else weight * torch.stack(losses).mean()
    return weight * per_element[valid_mask].mean()


def _regret_loss(regret, y_hat, y, valid_mask, last_role, loss_name):
    weight = float(regret.get("lambda_regret", 0.0))
    if weight <= 0:
        return None
    fused_error = torch.abs(y_hat - y)
    best_error = torch.minimum(
        torch.abs(regret["y_graph"] - y),
        torch.abs(regret["y_patch"] - y),
    ).detach()
    per_element = torch.relu(fused_error - best_error)
    losses = []
    for role_index in range(ROLE_CHANNEL_COUNT):
        role_mask = last_role[..., role_index].unsqueeze(1).unsqueeze(-1) > 0.5
        selected = valid_mask & role_mask.expand_as(valid_mask)
        if selected.any():
            losses.append(per_element[selected].mean())
    return None if not losses else weight * torch.stack(losses).mean()


def _criterion_metrics(criterions, args, loss, y_omi, inverse_y, inverse_y_hat):
    output = {}
    for name, criterion in criterions.items():
        if name == args.loss:
            output[name] = float(loss.detach().cpu())
        elif name == "split_phy_loss":
            values = criterion(
                y_pred=inverse_y_hat,
                y_true=inverse_y,
                miss_indicator=y_omi,
            )
            output.update({key: float(value) for key, value in values.items()})
    return output


def shared_step(model, dataloader, optimizer, criterions, args, mode, train, epoch):
    if args.lifecycle_training_strategy != "lcbt_v3":
        raise ValueError("The streamlined trainer requires lcbt_v3")
    training = bool(train and mode == "train")
    model.train(training)
    model.to(args.device)
    if training:
        optimizer.zero_grad()
        if hasattr(model, "on_train_epoch_start"):
            model.on_train_epoch_start(epoch)

    inverse_transform = dataloader.dataset.inverse_transform
    metric_sums = {}
    valid_batches = 0
    lifecycle_cells = {}
    routing_sums, routing_batches = {}, 0

    with torch.set_grad_enabled(training):
        for batch in dataloader:
            if len(batch) != 10:
                raise ValueError(
                    f"lcbt_v3 batches must contain 10 tensors, got {len(batch)}"
                )
            batch = [tensor.to(args.device) for tensor in batch]
            (
                x_enc,
                x_dec,
                enc_t_mark,
                dec_t_mark,
                _target_weights,
                phase_ids,
                group_ids,
                target_cohort_mask,
                calendar_ids,
                evidence_steps,
            ) = batch
            x_enc = x_enc.float()
            x_dec = x_dec.float()
            enc_t_mark = enc_t_mark.float()
            dec_t_mark = dec_t_mark.float()
            y = x_dec[:, -args.pred_len:].clone()
            model_x_dec = x_dec[:, -args.pred_len:].clone() if training else None
            x_enc = torch.nan_to_num(x_enc, nan=0.0, posinf=0.0, neginf=0.0)
            if model_x_dec is not None:
                model_x_dec = torch.nan_to_num(model_x_dec, nan=0.0, posinf=0.0, neginf=0.0)

            adj, pos_mark = dataloader.dataset.get_adj_and_pos_mark()
            adj = torch.as_tensor(adj, dtype=torch.float32, device=args.device)
            if args.pos_mark and pos_mark is not None:
                pos_mark = torch.as_tensor(pos_mark, dtype=torch.float32, device=args.device)
            else:
                pos_mark = None
            if not args.t_mark:
                enc_t_mark = dec_t_mark = None

            model_output = model(
                adj,
                x_enc,
                enc_t_mark,
                pos_mark,
                model_x_dec,
                dec_t_mark,
                epoch,
            )
            y_hat, meta = unpack_model_output(model_output)
            y_for_loss = meta.get("target", y)
            cohort_valid = (target_cohort_mask[:, None, :, None] > 0.5).expand_as(y_for_loss)
            valid_mask = cohort_valid & torch.isfinite(y_for_loss) & torch.isfinite(y_hat)
            if not valid_mask.any():
                continue

            base_loss = criterions[args.loss](y_hat[valid_mask], y_for_loss[valid_mask])
            optimize_loss = base_loss
            # Off by default (0.0): the training objective stays exactly the
            # unweighted batch mean unless the run explicitly opts in.
            stratum_weight = float(getattr(args, "lcbt_stratum_balanced_loss", 0.0))
            if stratum_weight > 0:
                stratum_loss = _stratum_balanced_loss(
                    y_hat,
                    y_for_loss,
                    valid_mask,
                    calendar_ids,
                    args.loss,
                )
                if stratum_loss is not None:
                    optimize_loss = (
                        1.0 - stratum_weight
                    ) * optimize_loss + stratum_weight * stratum_loss
            last_role = _last_role(x_enc, args)
            # Supervising only the event cohort leaves ~6% of nodes carrying the
            # gradient, so ordinary forecasting on the rest of the graph is never
            # trained. A small term on the visible non-cohort nodes restores it
            # without letting the bulk of the graph drown out the event nodes.
            context_weight = float(getattr(args, "lcbt_context_loss_weight", 0.0))
            if context_weight > 0:
                context_loss = _context_loss(
                    y_hat,
                    y_for_loss,
                    cohort_valid,
                    last_role,
                    args.loss,
                )
                if context_loss is not None:
                    optimize_loss = optimize_loss + context_weight * context_loss
            if meta.get("forecast_loss_mode") == "role_balanced_mix":
                role_loss = _role_balanced_loss(
                    y_hat,
                    y_for_loss,
                    valid_mask,
                    last_role,
                    args.loss,
                )
                if role_loss is not None:
                    beta = float(meta.get("forecast_role_balanced_beta", 0.1))
                    optimize_loss = (1.0 - beta) * optimize_loss + beta * role_loss

            aux_loss = sum_aux_losses(meta.get("aux_losses"), device=y_hat.device)
            if aux_loss is not None:
                optimize_loss = optimize_loss + aux_loss
            route_loss = _route_loss(
                meta.get("route_routing", {}),
                y_for_loss,
                valid_mask,
                last_role,
                epoch,
            )
            if route_loss is not None:
                optimize_loss = optimize_loss + route_loss
            regret_loss = _regret_loss(
                meta.get("regret_routing", {}),
                y_hat,
                y_for_loss,
                valid_mask,
                last_role,
                args.loss,
            )
            if regret_loss is not None:
                optimize_loss = optimize_loss + regret_loss
            if not torch.isfinite(optimize_loss):
                if training:
                    optimizer.zero_grad()
                continue
            if training:
                optimize_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            detached_y = y_for_loss.detach().cpu()
            detached_pred = y_hat.detach().cpu()
            inverse_y = inverse_transform(detached_y)
            inverse_pred = inverse_transform(detached_pred)
            y_omi = ~valid_mask.detach().cpu()
            batch_metrics = _criterion_metrics(
                criterions,
                args,
                base_loss,
                y_omi,
                inverse_y,
                inverse_pred,
            )
            for name, value in batch_metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + value
            valid_batches += 1

            routing = meta.get("route_routing")
            if mode == "val" and isinstance(routing, dict) and "alpha" in routing:
                # Collapse is only visible if it is measured: a router that
                # always picks one branch shows a flat beta across availability
                # groups and a route accuracy pinned at the base rate.
                report = routing_diagnostics(
                    1.0 - routing["alpha"],
                    routing["y_graph"],
                    routing["y_patch"],
                    y_for_loss,
                    valid_mask,
                    last_role,
                )
                for name, value in report.items():
                    if value == value:                       # skip NaN
                        routing_sums[name] = routing_sums.get(name, 0.0) + value
                routing_batches += 1

            if mode == "val":
                accumulate_lifecycle_mae(
                    lifecycle_cells,
                    inverse_y.numpy(),
                    inverse_pred.numpy(),
                    group_ids.detach().cpu().numpy(),
                    phase_ids.detach().cpu().numpy(),
                    target_cohort_mask.detach().cpu().numpy(),
                    calendar_ids.detach().cpu().numpy(),
                    evidence_steps.detach().cpu().numpy(),
                )

    if valid_batches == 0:
        raise RuntimeError(
            f"No finite {args.lifecycle_training_strategy} batches were collected "
            f"for {mode} at epoch {epoch}"
        )
    metrics = {name: value / valid_batches for name, value in metric_sums.items()}
    lifecycle = (
        finalize_lifecycle_validation_mae(lifecycle_cells, input_len=args.input_len)
        if mode == "val"
        else {}
    )
    routing = (
        {name: value / routing_batches for name, value in routing_sums.items()}
        if routing_batches
        else {}
    )
    return model, {"total": metrics, "lifecycle": lifecycle, "routing": routing}

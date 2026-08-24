"""The two forecast-loss primitives used by the v4 runner."""

import torch
import torch.nn.functional as F

from data_provider.lifecycle_roles import ROLE_CHANNEL_COUNT


def element_loss(prediction, target, name):
    safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    if name in {"mse", "rmse"}:
        return (prediction - safe_target).square()
    if name == "mae":
        return (prediction - safe_target).abs()
    if name == "huber":
        return F.huber_loss(prediction, safe_target, reduction="none")
    raise ValueError(f"unsupported v4 training loss: {name}")


def role_balanced_loss(prediction, target, valid, last_role, name):
    elements = element_loss(prediction, target, name)
    terms = []
    for role in range(ROLE_CHANNEL_COUNT):
        role_mask = last_role[..., role].unsqueeze(1).unsqueeze(-1) > 0.5
        selected = valid & role_mask.expand_as(valid)
        if selected.any():
            terms.append(elements[selected].mean())
    return torch.stack(terms).mean() if terms else None

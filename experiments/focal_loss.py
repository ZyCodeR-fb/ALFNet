"""The selected Full+MAE-Focal objective, in common normalized units."""
import math

import torch

NAME = 'mae_focal_raw02'


def focal_elements(prediction, target, scaler):
    if scaler is None or not math.isfinite(float(scaler['std'])) or scaler['std'] <= 0:
        raise ValueError('focal requires the positive train-only scaler')
    safe = torch.nan_to_num(target, nan=0., posinf=0., neginf=0.)
    error = (prediction - safe).abs()
    # Raw error sets the focal weight; normalize the loss by train std.
    # Gamma=1 and no detach, exactly as in the selected experiments.
    return error * (2. * torch.sigmoid(.2 * float(scaler['std']) * error) - 1.)

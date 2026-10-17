"""Study-scoped unit-invariant Focal. Original production path is unchanged."""
import math
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

NAME = 'mae_focal_standardized2'


def focal_elements(prediction, target, scaler):
    if scaler is None or not math.isfinite(float(scaler['std'])) or scaler['std'] <= 0:
        raise ValueError('positive train-only scaler required')
    safe = torch.nan_to_num(target, nan=0., posinf=0., neginf=0.)
    error = (prediction - safe).abs()
    # Both tensors already use the common train-K scalar standardization.
    # Fixed coefficient 2 -> tanh(error). No target magnitude reweighting,
    # no clipping/metric change, no detach, and no test-derived scale.
    return error * torch.tanh(error)


@contextmanager
def training_adapter():
    from experiments import v4_runner
    from scripts import train_v4
    base_contract = train_v4.model_method_contract

    def contract(session):
        result = base_contract(session)
        if session.args.loss != NAME or session.model_name != 'ALFNet':
            raise ValueError('flow repair adapter must only run its declared ALFNet configs')
        recipe = session.args.model_family_cfg['recipe']
        result['flow_repair_20261002'] = dict(
            loss=NAME, normalized_focal_coefficient=2.,
            role_balance_beta=float(recipe['forecast_role_balanced_beta']),
            refiner_scale=float(recipe['graph_diffusion_scale']))
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(v4_runner, 'FOCAL_NAME', NAME))
        stack.enter_context(patch.object(v4_runner, 'focal_elements', focal_elements))
        stack.enter_context(patch.object(train_v4, 'model_method_contract', contract))
        yield

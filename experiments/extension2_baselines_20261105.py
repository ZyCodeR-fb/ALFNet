"""Study-only registration of the two 2026 extension baselines added 2026-11-05.

MCPST-FSL (arXiv 2602.01936) and MetaDG (AAAI 2026). Like
experiments/extension_baselines_20261103.py, production v4_runner/train_v4 are
not edited; ``extension2_context()`` adds both models with their released
recipes on top of the first study's context. Released recipes (structure from
the source code, budget from the common protocol: at most 100 epochs, patience
10, validation 33-cell LifeAvg MAE):

* MCPST   Adam lr 3e-4 (source_lr) wd 1e-4, clip 1.0, ReduceLROnPlateau(.7, 10,
  min 1e-6) on the selection metric, batch 8 (the released legacy step:
  ``500 steps x task batch_size``; the active minibatch_size=2048 path encodes
  node sequences one at a time). Objective: 2*MSE(flow) + .01*uncertainty MSE +
  .03*physics + .015*consistency, the released PhysicsInformedLoss terms that need
  no unavailable label; its spatial/temporal pseudo-targets are built from every
  node's future values and are not used.
* MetaDG  Adam lr 1e-3 eps 1e-3, released per-block weight decay, clip 5,
  ExponentialLR(.97) per epoch, batch 16; HuberLoss (flow) / masked MAE (speed)
  on the original scale (lib/losses.loss_select).
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

ENTRY_POINTS = {'MCPST': 'model.comparison_model.MCPST', 'MetaDG': 'model.comparison_model.MetaDG'}
DECODER_POLICY = {'MetaDG': 'source_curriculum_legal_target_teacher_input'}
RECIPES = {
    'MCPST': dict(optimizer='Adam', lr=3e-4, weight_decay=1e-4, clip=1.0, batch_size=8,
                  scheduler=('ReduceLROnPlateau', dict(mode='min', factor=.7, patience=10, min_lr=1e-6))),
    'MetaDG': dict(optimizer='Adam', lr=1e-3, eps=1e-3, clip=5.0, batch_size=16,
                   weight_decay={'PEMS04': 1e-4, 'PEMS08': 0.0, 'METRLA': 0.0, 'PEMSBAY': 1e-4},
                   scheduler=('ExponentialLR', dict(gamma=.97))),
}
MCPST_WEIGHTS = dict(flow=2.0, uncertainty=0.01, physics=0.03, consistency=0.015)


def clipped(optimizer_class, max_norm):
    class Clipped(optimizer_class):
        def step(self, closure=None):
            params = [p for g in self.param_groups for p in g['params'] if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params, max_norm)
            return super().step(closure)
    Clipped.__name__ = f'Clipped{optimizer_class.__name__}'
    return Clipped


def weight_decay(session):
    value = RECIPES[session.model_name]['weight_decay']
    return value[session.dataset] if isinstance(value, dict) else value


def build_optimizer(session, fallback):
    recipe = RECIPES.get(session.model_name)
    if recipe is None:
        return fallback(session)
    assert float(session.args.lr) == recipe['lr'] and float(session.args.weight_decay) == weight_decay(session)
    kwargs = dict(lr=recipe['lr'], weight_decay=weight_decay(session))
    if 'eps' in recipe:
        kwargs['eps'] = recipe['eps']
    params = [p for p in session.model.parameters() if p.requires_grad]
    return clipped(getattr(torch.optim, recipe['optimizer']), recipe['clip'])([{'params': params}], **kwargs)


def build_scheduler(session, optimizer, fallback):
    recipe = RECIPES.get(session.model_name)
    if recipe is None:
        return fallback(session, optimizer)
    name, kwargs = recipe['scheduler']
    return getattr(torch.optim.lr_scheduler, name)(optimizer, **kwargs)


def _masked(values, valid):
    return values[valid].mean()


def compute_losses(model_output, batch, loss_name='mse', scaler=None, fallback=None):
    from experiments.v4_runner import RunnerRefused, valid_positions
    from utils.model_output import unpack_model_output
    prediction, meta = unpack_model_output(model_output)
    objective = meta.get('source_objective')
    if objective not in ('mcpst_physics_informed', 'metadg_released_loss'):
        return fallback(model_output, batch, loss_name, scaler=scaler)
    valid = valid_positions(batch, prediction)
    empty = {'prediction': prediction, 'valid': valid, 'base': None, 'optimize': None,
             'role_mix': None, 'scale_mix': None, 'synthetic_unseen_mix': None,
             'low_flow_bin_mix': None, 'aux': None, 'spectral_mix': None, 'meta': meta}
    if not bool(valid.any()):
        return empty
    target = batch['target']
    if objective == 'mcpst_physics_informed':
        if loss_name != 'mse':
            raise RunnerRefused('MCPST uses its released MSE flow loss')
        error = prediction - target
        flow = _masked(error.square(), valid)
        uncertainty = _masked((meta['mcpst_uncertainty'] - error.abs().detach()).square(), valid)
        physics = (meta['mcpst_order_parameter'] - 0.5).square().mean() + \
            torch.clamp(meta['mcpst_spectral_gap'], 0, 1).square().mean() * 0.1
        consistency = (meta['mcpst_physics_weights'].sum() - 1.0).square()
        aux = MCPST_WEIGHTS['uncertainty'] * uncertainty + MCPST_WEIGHTS['physics'] * physics + \
            MCPST_WEIGHTS['consistency'] * consistency
        return dict(empty, base=flow, optimize=MCPST_WEIGHTS['flow'] * flow + aux, aux=aux)
    if scaler is None or loss_name != 'mae':
        raise RunnerRefused('MetaDG uses its released loss on the original scale')
    raw = (prediction - target) * float(scaler['std'])
    if meta['metadg_loss'] == 'huber':
        loss = torch.nn.functional.huber_loss(raw[valid], torch.zeros_like(raw[valid]), delta=1.0)
    else:
        loss = raw[valid].abs().mean()
    return dict(empty, base=raw[valid].abs().mean(), optimize=loss)


def method_contract(session, fallback):
    report = fallback(session)
    if session.model_name in RECIPES:
        model = session.model
        report['extension_baseline'] = dict(
            model=session.model_name, recipe={k: v for k, v in RECIPES[session.model_name].items()},
            weight_decay=weight_decay(session),
            node_identity=getattr(model, 'node_identity', 'none in released model'),
            released_block=getattr(model, 'released_block', None), lifecycle_inputs=False)
    return report


@contextmanager
def extension2_context():
    from experiments import v4_runner
    from experiments.extension_baselines_20261103 import extension_context
    from scripts import train_v4
    with ExitStack() as stack:
        stack.enter_context(extension_context())
        base_optimizer, base_scheduler = train_v4.build_optimizer, train_v4.build_scheduler
        base_losses, base_contract = v4_runner.compute_losses, train_v4.model_method_contract
        stack.enter_context(patch.dict(v4_runner.MODEL_ENTRY_POINTS, ENTRY_POINTS))
        stack.enter_context(patch.dict(v4_runner.DECODER_TRAINING_POLICIES, DECODER_POLICY))
        stack.enter_context(patch.object(train_v4, 'build_optimizer',
                                         lambda s: build_optimizer(s, base_optimizer)))
        stack.enter_context(patch.object(train_v4, 'build_scheduler',
                                         lambda s, o: build_scheduler(s, o, base_scheduler)))
        stack.enter_context(patch.object(
            v4_runner, 'compute_losses',
            lambda o, b, loss_name='mse', scaler=None: compute_losses(o, b, loss_name, scaler, base_losses)))
        stack.enter_context(patch.object(train_v4, 'model_method_contract',
                                         lambda s: method_contract(s, base_contract)))
        yield

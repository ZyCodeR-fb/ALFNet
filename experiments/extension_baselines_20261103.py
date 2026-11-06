"""Study-only registration of the four 2026-11-03 extension baselines.

experiments/v4_runner.py and scripts/train_v4.py are not edited. Inside
``extension_context()`` the four models are added to the runner's entry points
and each receives its released optimiser, scheduler, gradient clipping and (for
STONE) objective; every other registered model keeps the production behaviour.
Merlin's two released stages are driven by ``train_merlin``.

Released recipes (structure from the source code, budget from the common
protocol: at most 100 epochs, patience 10, validation 33-cell LifeAvg MAE):

* HD-TTS  AdamW lr 1e-3 wd 1e-2, ReduceLROnPlateau(.5, 10, 1e-4, min 1e-6) on the
  selection metric, clip 5, batch 32 (config/default.yaml, lr_scheduler/redplat).
* STONE   Adam lr 1e-3 wd 1e-4 for the model and for each graph editor,
  StepLR(10, .95), clip 5 on model parameters only, batch 64
  (experiments/stone/main.py, src/utils/args.py).
* STOP    Adam lr 2e-3 wd 1e-5 eps 1e-8, MultiStepLR([30,80,160,250], .5), clip 5,
  batch 64 (TrafficStream/experiments/stop/main.py).
* Merlin  per stage Adam lr 2e-4 wd 1e-4, clip 5, batch 16; teacher milestones
  [1,10,25,50,75,90,100,125,150,190], student threshold-shifted milestones
  [4,13,28,53,78,93,103], gamma .5 (train_teacher.py, train_student_merlin.py).
"""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json

import torch

ENTRY_POINTS = {
    'Merlin': 'model.comparison_model.Merlin',
    'HDTTS': 'model.comparison_model.HDTTS',
    'STONE': 'model.comparison_model.STONE',
    'STOP': 'model.comparison_model.STOP',
}
CLIP_NORM = 5.
MERLIN_MASK_SEED_INDEX = 1_103_000  # training_runtime sub-stream, disjoint from epochs
RECIPES = {
    'HDTTS': dict(optimizer='AdamW', lr=1e-3, weight_decay=1e-2, batch_size=32,
                  scheduler=('ReduceLROnPlateau', dict(mode='min', factor=.5, patience=10,
                                                       threshold=1e-4, min_lr=1e-6))),
    'STONE': dict(optimizer='Adam', lr=1e-3, weight_decay=1e-4, batch_size=64,
                  scheduler=('StepLR', dict(step_size=10, gamma=.95))),
    'STOP': dict(optimizer='Adam', lr=2e-3, weight_decay=1e-5, eps=1e-8, batch_size=64,
                 scheduler=('MultiStepLR', dict(milestones=[30, 80, 160, 250], gamma=.5))),
    'Merlin': dict(optimizer='Adam', lr=2e-4, weight_decay=1e-4, batch_size=16,
                   teacher_milestones=[1, 10, 25, 50, 75, 90, 100, 125, 150, 190],
                   student_milestones=[4, 13, 28, 53, 78, 93, 103], gamma=.5),
}


def clipped(optimizer_class, clip_groups):
    """Same optimiser and state_dict; clips the listed groups before each step."""
    class Clipped(optimizer_class):
        def step(self, closure=None):
            params = [p for i in clip_groups for p in self.param_groups[i]['params']
                      if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
            return super().step(closure)
    Clipped.__name__ = f'Clipped{optimizer_class.__name__}'
    return Clipped


def build_optimizer(session, fallback):
    recipe = RECIPES.get(session.model_name)
    if recipe is None:
        return fallback(session)
    if session.model_name == 'Merlin':
        raise RuntimeError('Merlin optimisers are stage-specific; use train_merlin')
    assert float(session.args.lr) == recipe['lr'] and float(session.args.weight_decay) == recipe['weight_decay']
    kwargs = dict(lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    if 'eps' in recipe:
        kwargs['eps'] = recipe['eps']
    if session.model_name == 'STONE':
        groups = [{'params': list(session.model.core.parameters())},
                  {'params': session.model.editor_parameters()}]
        assert sum(len(g['params']) for g in groups) == len(list(session.model.parameters()))
    else:
        groups = [{'params': list(session.model.parameters())}]
    return clipped(getattr(torch.optim, recipe['optimizer']), (0,))(groups, **kwargs)


def build_scheduler(session, optimizer, fallback):
    recipe = RECIPES.get(session.model_name)
    if recipe is None:
        return fallback(session, optimizer)
    name, kwargs = recipe['scheduler']
    return getattr(torch.optim.lr_scheduler, name)(optimizer, **kwargs)


def compute_losses(model_output, batch, loss_name='mse', scaler=None, fallback=None):
    """STONE's released train_batch objective; every other model is untouched."""
    from experiments.v4_runner import RunnerRefused, valid_positions
    from utils.model_output import unpack_model_output
    prediction, meta = unpack_model_output(model_output)
    if meta.get('source_objective') != 'stone_environments':
        return fallback(model_output, batch, loss_name, scaler=scaler)
    if loss_name != 'mae' or scaler is None:
        raise RunnerRefused('STONE uses its released masked MAE on the original scale')
    environments = meta['stone_environments']
    valid = valid_positions(batch, prediction)
    empty = {'prediction': prediction, 'valid': valid, 'base': None, 'optimize': None,
             'role_mix': None, 'scale_mix': None, 'synthetic_unseen_mix': None,
             'low_flow_bin_mix': None, 'aux': None, 'spectral_mix': None, 'meta': meta}
    if not bool(valid.any()):
        return empty
    every = valid.unsqueeze(0).expand_as(environments) & torch.isfinite(environments)
    errors = (environments - batch['target'].unsqueeze(0)).abs()[every] * float(scaler['std'])
    # masked_mae_train: mean and variance of the absolute errors over all
    # environments; the variance is taken over supervised positions only
    # because the common loss mask is sparse (released labels are dense).
    loss, variance = errors.mean(), errors.var()
    log_p = meta['stone_log_p']
    editors = -(log_p[0] + log_p[1]) * variance.detach()
    return dict(empty, base=loss, optimize=loss + variance + editors, aux=variance + editors)


def method_contract(session, fallback):
    report = fallback(session)
    name = session.model_name
    if name in RECIPES:
        model = session.model
        report['extension_baseline'] = dict(
            model=name, recipe={k: v for k, v in RECIPES[name].items()},
            node_identity='StructuralNodeEncoder' if name in ('Merlin', 'HDTTS') else 'none in released model',
            lifecycle_inputs=False,
            hierarchy_sha256=getattr(model, 'hierarchy_sha256', None),
            hierarchy_sizes=getattr(model, 'hierarchy_sizes', None),
            source_import_side_effects=getattr(model, 'source_import_side_effects', None))
    return report


@contextmanager
def extension_context():
    from experiments import v4_runner
    from scripts import train_v4
    base_optimizer, base_scheduler = train_v4.build_optimizer, train_v4.build_scheduler
    base_losses, base_contract = v4_runner.compute_losses, train_v4.model_method_contract
    with ExitStack() as stack:
        stack.enter_context(patch.dict(v4_runner.MODEL_ENTRY_POINTS, ENTRY_POINTS))
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


# --- Merlin's two released stages ------------------------------------------


class TeacherView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.core = model

    def forward_v4(self, batch, runtime_context, epoch=0, teacher_forcing_mask=None):
        return self.core.teacher_forward_v4(batch, runtime_context)


def merlin_stage(train_fn, config, output_root, plan, stage, teacher=None, device='cuda:0',
                 smoke=False, resume=False, max_train_batches=None, max_validation_batches=None):
    """One Merlin stage through the common train(); returns its run report."""
    from experiments import v4_runner as runner
    from lib.v4_seeds import derive_seed
    from model.comparison_model.Merlin.adapter import (
        keep_masks_for_batch, source_point_masks, teacher_history_for_batch)
    from scripts import train_v4 as common
    from scripts.run_metrla_hparams import sha
    assert stage in ('teacher', 'student') and (stage == 'teacher') == (teacher is None)
    recipe = RECIPES['Merlin']
    original_build, original_contract = runner.build_v4_session, common.model_method_contract
    original_collect = common.collect_validation_table
    teacher_sha = None if teacher is None else sha(teacher)
    state = {}

    def build(*args, **kwargs):
        s = original_build(*args, **kwargs)
        assert s.model_name == 'Merlin'
        assert (s.args.max_epoch, s.args.patience, s.args.early_stopping_start_epoch) == (100, 10, 0)
        assert float(s.args.lr) == recipe['lr'] and float(s.args.weight_decay) == recipe['weight_decay']
        if teacher is not None:
            cp = torch.load(teacher, map_location='cpu', weights_only=False)
            assert cp['plan_sha256'] == plan['plan_sha256'] and cp['config_sha256'] == sha(config)
            assert cp['model_method_contract']['merlin_stage'] == 'teacher'
            assert cp['train_exposure_hash'] == s.train_context.exposure_hash()
            assert cp['scaler'] == s.scaler
            s.model.load_state_dict(cp['model_state'], strict=True)
            s.model.freeze_teacher()
            seed = derive_seed(s.run_seed, 'training_runtime', index=MERLIN_MASK_SEED_INDEX)
            state['bank'] = source_point_masks(s.train_dataset.natural.shape[0], s.model.num_nodes, seed)
            state['bank_seed'] = seed
        return s

    def contract(s):
        report = original_contract(s)
        report.update(merlin_stage=stage, teacher_sha256=teacher_sha,
                      source_objective=('masked teacher MAE' if stage == 'teacher' else
                                        '2*sum4 MAE + 2*decay*(sum4 hidden KD + sum4 response KD) + decay*CL'),
                      teacher_history='train-K natural history only',
                      point_masks='released per-node whole-series masks, rates .25/.5/.75/.9, fixed',
                      point_mask_seed=state.get('bank_seed'))
        return report

    def optimizer(s):
        network = s.model.teacher if stage == 'teacher' else s.model.student
        params = list(network.parameters())
        trainable = [p for p in params if p.requires_grad]
        return clipped(torch.optim.Adam, (0,))([{'params': trainable}], lr=recipe['lr'],
                                                weight_decay=recipe['weight_decay'])

    def scheduler(s, opt):
        milestones = recipe['teacher_milestones'] if stage == 'teacher' else recipe['student_milestones']
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=recipe['gamma'])

    def epoch_train(s, opt, epoch, max_batches=None, loss_name='mae'):
        s.set_epoch(epoch)
        s.model.train()
        totals, batches = {}, 0
        for i, raw in enumerate(s.train_loader):
            if max_batches is not None and i >= max_batches:
                break
            history = teacher_history_for_batch(s.train_dataset, raw).to(s.device)
            batch = runner.to_device(raw, s.device)
            opt.zero_grad(set_to_none=True)
            if stage == 'teacher':
                terms = {'loss': s.model.teacher_objective(batch, s.train_context, history)}
            else:
                keep = keep_masks_for_batch(state['bank'], raw, s.model.input_len)
                terms = s.model.student_objective(batch, s.train_context, history, keep, epoch + 1)
            loss = terms['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite Merlin objective; no batch is skipped')
            loss.backward()
            opt.step()
            batches += 1
            for k, v in terms.items():
                totals[k] = totals.get(k, 0.) + float(v.detach())
        assert batches
        report = dict(batches=batches, skipped=0,
                      base=totals.get('forecast', totals['loss']) / batches,
                      optimize=totals['loss'] / batches,
                      aux=None if stage == 'teacher' else totals['loss'] / batches,
                      role_mix=None, scale_mix=None, synthetic_unseen_mix=None,
                      low_flow_bin_mix=None, spectral_mix=None, forced_positions=0,
                      available_positions=0, synthetic_unseen_positions=0,
                      merlin_terms={k: v / batches for k, v in totals.items()})
        report.update(runner.decoder_training_metadata(s, epoch, True))
        report['run_seed'] = s.run_seed
        return report

    def validate(s, **kwargs):
        return runner.validate(replace(s, model=TeacherView(s.model)) if stage == 'teacher' else s, **kwargs)

    def collect(s, **kwargs):
        return original_collect(replace(s, model=TeacherView(s.model)) if stage == 'teacher' else s, **kwargs)

    with ExitStack() as stack:
        for name, fn in [('build_v4_session', build), ('model_method_contract', contract),
                         ('build_optimizer', optimizer), ('build_scheduler', scheduler),
                         ('train_one_epoch', epoch_train), ('validate', validate),
                         ('collect_validation_table', collect)]:
            stack.enter_context(patch.object(common, name, fn))
        return train_fn(config, output_root, device=device, plan=plan, resume=resume, smoke=smoke,
                        max_train_batches=max_train_batches, max_validation_batches=max_validation_batches)


def train_merlin(train_fn, config, output_root, plan, device='cuda:0', smoke=False, resume=False,
                 max_train_batches=None, max_validation_batches=None):
    """Teacher stage, then the student from the validation-selected teacher."""
    import tempfile
    from scripts.run_metrla_hparams import sha
    root = Path(output_root) if output_root is not None else Path(tempfile.mkdtemp(prefix='merlin-smoke-'))
    kwargs = dict(device=device, smoke=smoke, max_train_batches=max_train_batches,
                  max_validation_batches=max_validation_batches)
    reports = list((root / 'teacher').rglob('run_report.json'))
    if resume and reports:
        assert len(reports) == 1 and reports[0].with_name('COMPLETE').exists()
        teacher_report = json.loads(reports[0].read_text())
        assert teacher_report['plan_sha256'] == plan['plan_sha256'] and teacher_report['config_sha256'] == sha(config)
    else:
        teacher_report = merlin_stage(train_fn, config, root / 'teacher', plan, 'teacher', resume=resume, **kwargs)
    teacher = Path(teacher_report['run_directory']) / 'best.pt'
    if smoke:
        # A smoke run saves no teacher checkpoint; the launcher preflight exercises
        # the student objective separately (merlin_student_smoke).
        return teacher_report
    result = merlin_stage(train_fn, config, root / 'student', plan, 'student', teacher=teacher,
                          resume=resume, **kwargs)
    result['teacher_report'] = str(Path(teacher_report['run_directory']) / 'run_report.json')
    result['teacher_checkpoint_sha256'] = sha(teacher)
    Path(result['run_directory'], 'run_report.json').write_text(
        json.dumps(result, indent=1, default=str, allow_nan=False) + '\n')
    return result


def merlin_student_smoke(session, batches=10):
    """Student objective, backward and released optimiser step on real batches."""
    from experiments import v4_runner as runner
    from lib.v4_seeds import derive_seed
    from model.comparison_model.Merlin.adapter import (
        keep_masks_for_batch, source_point_masks, teacher_history_for_batch)
    recipe = RECIPES['Merlin']
    model = session.model
    model.freeze_teacher()
    seed = derive_seed(session.run_seed, 'training_runtime', index=MERLIN_MASK_SEED_INDEX)
    bank = source_point_masks(session.train_dataset.natural.shape[0], model.num_nodes, seed)
    trainable = [q for q in model.student.parameters() if q.requires_grad]
    opt = clipped(torch.optim.Adam, (0,))([{'params': trainable}], lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    model.train()
    losses = []
    for i, raw in enumerate(session.train_loader):
        if i >= batches:
            break
        history = teacher_history_for_batch(session.train_dataset, raw).to(session.device)
        batch = runner.to_device(raw, session.device)
        opt.zero_grad(set_to_none=True)
        terms = model.student_objective(batch, session.train_context, history,
                                        keep_masks_for_batch(bank, raw, model.input_len), epoch=i + 1)
        terms['loss'].backward()
        assert all(q.grad is None for q in model.teacher.parameters())
        opt.step()
        losses.append({k: float(v.detach()) for k, v in terms.items()})
    model.eval()
    with torch.no_grad():
        for i, raw in enumerate(session.validation_loader):
            if i >= 2:
                break
            out = model.forward_v4(runner.to_device(raw, session.device), session.validation_context)['pred']
            assert torch.isfinite(out).all()
    kept = [float(bank[r].mean()) for r in range(bank.shape[0])]
    return dict(student_batches=len(losses), last_terms=losses[-1], keep_fraction_by_rate=kept, mask_seed=seed)


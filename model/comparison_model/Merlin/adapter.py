"""Merlin (KDD 2025): released STID teacher/student, protocol adapter only.

The four files under ``source/`` are byte copies of references/Merlin (see
SOURCE_MANIFEST.json). This module only reshapes the common v4 batch into the
released ``[value, tod, dow]`` input, restricts every training term to the
training-known nodes K, and replaces the per-node identity table with the shared
structural encoder used by the other inductive baselines.

Training is two-stage and cannot go through the generic single loss: the
teacher is trained with ``teacher_objective``, selected on validation, frozen,
and the student is then trained with ``student_objective``. The launcher in
experiments/extension_baselines_20261103.py drives both stages.
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from layers.structural_node_encoder import StructuralNodeEncoder
from lib.lcbt_v4 import assert_v4_strategy
from lib.v4_batch import bind_static_assets, read_common
from .source.Loss import CL_loss
from .source.stid_arch import STID, STID_CL

# Released train_teacher.py / train_student_merlin.py settings.
ARCHITECTURE = dict(node_dim=64, input_dim=3, embed_dim=64, num_layer=3,
                    temp_dim_tid=64, temp_dim_diw=64, time_of_day_size=288,
                    day_of_week_size=7, if_T_i_D=True, if_D_i_W=True, if_node=True)
TEACHER_CL_HIDDEN, STUDENT_CL_HIDDEN = 4, 16
MISSING_RATES = (.25, .50, .75, .90)
EPOCH_THRESHOLD, WEIGHT_PRE, WEIGHT_KD, WEIGHT_CL, TEMPERATURE = 3, 2., 2., 1., 1.
STEPS_PER_DAY = 288


class StructuralNodeTable(nn.Module):
    """Parametrization: the source ``node_emb`` table becomes a structural function."""

    def __init__(self, context, width):
        super().__init__()
        self.encoder = StructuralNodeEncoder(context.adjacency, context.positions, out_dim=width)

    def forward(self, original):
        return self.encoder()


def source_features(raw, marks):
    """``[value, tod, dow]`` exactly as the released preprocessing builds them.

    Released preprocessing stores tod = step/288 and dow = day/7 as fractions;
    stid_arch.py indexes ``tod*288`` but indexes dow without the factor 7, so
    its day-of-week embedding always reads row 0. That released behaviour is
    kept rather than silently repaired.
    """
    if raw.ndim != 4 or raw.shape[-1] != 1 or marks.shape[:2] != raw.shape[:2]:
        raise ValueError('Merlin requires [B,L,N,1] traffic and matching calendar marks')
    minute, hour, dow = (marks[..., i].double() for i in range(3))
    if not ((minute >= 0) & (minute < 60) & (hour >= 0) & (hour < 24) & (dow >= 0) & (dow < 7)).all():
        raise ValueError('invalid raw calendar marks')
    step = hour * (STEPS_PER_DAY // 24) + torch.floor(minute / (1440 // STEPS_PER_DAY))
    calendar = torch.stack([step / STEPS_PER_DAY, dow / 7.], -1).to(raw.dtype)
    return torch.cat([raw, calendar.unsqueeze(2).expand(-1, -1, raw.shape[2], -1)], -1)


def masked_mean(values, valid):
    valid = valid.bool()
    if not valid.any():
        raise ValueError('empty legal supervision')
    selected = values[valid]
    if not torch.isfinite(selected).all():
        raise ValueError('nonfinite Merlin objective term')
    return selected.mean()


def source_point_masks(num_steps, num_nodes, seed):
    """Released ``random_mask_matric_all``: per node, int(T*rate) points of the
    whole series are dropped; the four rates are drawn independently, in order,
    once, and reused every epoch. Returns keep masks ``[rates, T, N]``."""
    rng = np.random.RandomState(int(seed))
    keep = np.ones((len(MISSING_RATES), num_steps, num_nodes), dtype=bool)
    for r, rate in enumerate(MISSING_RATES):
        count = int(num_steps * rate)
        for node in range(num_nodes):
            keep[r, rng.choice(num_steps, size=count, replace=False), node] = False
    return keep


class Model(nn.Module):
    def __init__(self, configs, runtime_context):
        super().__init__()
        if int(getattr(configs, 'raw_enc_in', 1)) != 1:
            raise ValueError('Merlin source recipe uses one traffic channel')
        self.raw_enc_in = 1
        self.num_nodes = runtime_context.num_nodes
        self.input_len, self.pred_len = int(configs.input_len), int(configs.pred_len)
        self.register_buffer('known', torch.as_tensor(runtime_context.known_nodes, dtype=torch.long))
        self.register_buffer('teacher_ready', torch.tensor(False))
        common = dict(num_nodes=self.num_nodes, input_len=self.input_len,
                      output_len=self.pred_len, **ARCHITECTURE)
        self.teacher = STID(cl_hidden=TEACHER_CL_HIDDEN, **common)
        self.student = STID_CL(cl_hidden=STUDENT_CL_HIDDEN, **common)
        for network in (self.teacher, self.student):
            network.node_emb.requires_grad_(False)
            parametrize.register_parametrization(
                network, 'node_emb', StructuralNodeTable(runtime_context, ARCHITECTURE['node_dim']))
        # The student CL projection is a Conv2d spanning the node axis; only
        # K may contribute during training.
        self.student.FC_layer.register_forward_pre_hook(self._scope_projection)

    @classmethod
    def from_runtime_context(cls, configs, runtime_context):
        assert_v4_strategy(configs, 'Merlin')
        return bind_static_assets(cls(configs, runtime_context), configs, runtime_context, 'Merlin')

    def known_mask(self, device):
        allowed = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
        allowed[self.known.to(device)] = True
        return allowed

    def _scope_projection(self, module, inputs):
        if not self.training:
            return inputs
        hidden = inputs[0]
        return (torch.where(self.known_mask(hidden.device)[None, None, :, None], hidden, 0.),)

    def train(self, mode=True):
        super().train(mode)
        if self.teacher_ready.item():
            self.teacher.eval()
            self.teacher.requires_grad_(False)
        return self

    def freeze_teacher(self):
        """Call after restoring the validation-selected teacher state."""
        self.teacher_ready.fill_(True)
        self.teacher.eval().requires_grad_(False)

    def _common(self, batch, context):
        return read_common(batch, context, self.static_asset_hash, 1, 'Merlin',
                           self._v4_adjacency_cache, require_target=False)

    def forward_v4(self, batch, runtime_context, epoch=0, teacher_forcing_mask=None):
        if self.training:
            raise RuntimeError('Merlin training requires its two-stage objectives')
        common = self._common(batch, runtime_context)
        prediction, _, _ = self.student(source_features(common['base_visible_input'],
                                                        common['encoder_time_marks']))
        return {'pred': prediction.unsqueeze(-1).contiguous()}

    def teacher_forward_v4(self, batch, runtime_context):
        """Teacher selection reads the same common visible input as the student."""
        if self.training:
            raise RuntimeError('teacher evaluation requires eval mode')
        common = self._common(batch, runtime_context)
        prediction, _, _ = self.teacher(source_features(common['base_visible_input'],
                                                        common['encoder_time_marks']), None)
        return {'pred': prediction.unsqueeze(-1).contiguous()}

    def _training_inputs(self, batch, context, teacher_history):
        common = self._common(batch, context)
        if context.phase != 'train' or not self.training:
            raise ValueError('training objectives require the train context and train mode')
        base = common['base_visible_input']
        if teacher_history.shape != base.shape or not torch.isfinite(teacher_history).all():
            raise ValueError('teacher history must be finite and match the common input')
        allowed = self.known_mask(base.device)
        if torch.count_nonzero(teacher_history[:, :, ~allowed]) or torch.count_nonzero(base[:, :, ~allowed]):
            raise ValueError('held-out dynamic signal supplied to training')
        valid = batch['loss_mask'].bool()
        if valid.ndim != 3 or valid.shape != batch['target'].shape[:-1] or valid[:, :, ~allowed].any():
            raise ValueError('supervision must be the canonical [B,H,N] train-K mask')
        return common, valid, allowed

    def teacher_objective(self, batch, context, teacher_history):
        if self.teacher_ready.item():
            raise RuntimeError('teacher is frozen')
        common, valid, _ = self._training_inputs(batch, context, teacher_history)
        pred, _, _ = self.teacher(source_features(teacher_history, common['encoder_time_marks']), None)
        return masked_mean((pred - batch['target'][..., 0]).abs(), valid)

    def student_objective(self, batch, context, teacher_history, keep_masks, epoch):
        """Released ``Merlin_Loss_add``; ``epoch`` is one-based as in the source loop.

        ``keep_masks``: ``[4, B, L, N]`` windows of the fixed released point masks.
        """
        if not self.teacher_ready.item():
            raise RuntimeError('select and freeze the teacher before training the student')
        common, valid, allowed = self._training_inputs(batch, context, teacher_history)
        if keep_masks.shape != (len(MISSING_RATES),) + tuple(common['base_visible_input'].shape[:3]):
            raise ValueError('student keep masks must be [rates, B, L, N]')
        self.teacher.eval().requires_grad_(False)
        with torch.no_grad():
            teacher_pred, teacher_hidden, _ = self.teacher(
                source_features(teacher_history, common['encoder_time_marks']), None)
        x = source_features(common['base_visible_input'], common['encoder_time_marks'])
        students = []
        for keep in keep_masks.to(x.device):
            view = x.clone()
            # tensor_mask: value * mask, calendar channels untouched.
            view[..., 0] = torch.where(keep, x[..., 0], torch.zeros_like(x[..., 0]))
            students.append(self.student(view))
        target = batch['target'][..., 0]
        horizon_known = allowed[None, None, :].expand_as(valid)
        pre = sum(masked_mean((s[0] - target).abs(), valid) for s in students)
        rd = sum(masked_mean((s[0] - teacher_pred).square(), horizon_known) for s in students)
        hd = sum((s[1][:, :, allowed] - teacher_hidden[:, :, allowed]).square().mean() for s in students)
        pairs = [(i, j) for i in range(len(students)) for j in range(i + 1, len(students))]
        cl = 2 * sum(CL_loss(students[i][2], students[j][2], TEMPERATURE) for i, j in pairs) / (
            len(students) * (len(students) - 1))
        decay = 1. if epoch <= EPOCH_THRESHOLD else 1. / (epoch - EPOCH_THRESHOLD)
        loss = WEIGHT_PRE * pre + WEIGHT_KD * decay * (hd + rd) + WEIGHT_CL * decay * cl
        return {'loss': loss, 'forecast': pre, 'hidden_distillation': hd,
                'response_distillation': rd, 'contrastive': cl}


def teacher_history_for_batch(dataset, batch):
    """Clean train-K history at registered training origins; never V/T values.

    Missing natural entries use the common train-mean fill (normalized zero).
    """
    if dataset.context.phase != 'train':
        raise ValueError('teacher history is training-only')
    origins = batch['origin'].reshape(-1).tolist()
    registered = set(int(o) for o in dataset.origins.tolist())
    known = np.asarray(dataset.context.known_nodes, dtype=np.int64)
    history = torch.zeros_like(batch['base_visible_input'])
    for i, origin in enumerate(origins):
        if int(origin) not in registered:
            raise ValueError('origin is outside the training split')
        end = int(origin) + dataset.input_len
        raw = torch.as_tensor(dataset.normalised[int(origin):end][:, known].copy())
        natural = torch.as_tensor(dataset.natural[int(origin):end][:, known].copy())
        history[i, :, torch.as_tensor(known), 0] = torch.where(natural, raw, torch.zeros_like(raw)).to(history)
    return history


def keep_masks_for_batch(bank, batch, input_len):
    """Slice the fixed ``[rates, T, N]`` bank at each sample's input window."""
    origins = batch['origin'].reshape(-1).tolist()
    windows = np.stack([bank[:, int(o):int(o) + input_len] for o in origins], axis=1)
    return torch.from_numpy(windows)

"""What `lcbt_v4` fixes, and the one policy that decides teacher forcing.

Two things live here because both need exactly one definition.

**The strategy name.** v3 and v4 differ on the split (50/10/40 against
60/20/20), on the node scope and on what the training computation graph may
touch. Borrowing `lcbt_v3` for a v4 run would leave every checkpoint, manifest
and table unable to say which protocol produced it, and the two are not
comparable. So `lcbt_v4` is an explicit third value: the v3 entry points still
accept only `lcbt_v3`, and a model built through a RuntimeDataContext requires
`lcbt_v4`.

**The teacher-forcing schedule.** The source decoders drew one `torch.rand` per
step. That is not reproducible across a batch-size change, a reshuffle, a
different worker count, or two models that consume different amounts of RNG --
and ALFNet and MoGERNN consume very different amounts. Under v4 the decision is
a pure function of stable facts about the *sample*, hashed:

    teacher_seed = H(run_seed, "teacher_forcing")
    u = sha256(policy | teacher_seed | epoch | origin | horizon_step) -> [0, 1)
    force = u < ratio,  ratio = clip(1 - epoch / 40, 0, 1)

`origin` is the window's start index in the payload, so the same sample gets the
same decision whichever batch it lands in and whichever model reads it. SHA256
rather than Python's `hash`, which is salted per process unless PYTHONHASHSEED
is pinned.

The per-step decision is then broadcast across nodes and intersected with
`decoder_available_mask`. Broadcasting keeps the source's semantics -- one
decoder step is either teacher-forced or not -- while the intersection is what
makes it safe: a node the episode still hides at the last observed step has no
present state to carry into the horizon, and its future must not be read.

`ratio` reproduces the v3 schedule `1 - epoch / 40` exactly, so this changes
when a step is forced, not how often.
"""

import hashlib

import numpy as np
import torch

STRATEGY = "lcbt_v4"

# The v4 time split. `V4LifecycleDataset` defaults to these and `args.py`
# refuses a config that declares anything else under this strategy.
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
TRAIN_END_RATIO = 0.60
VAL_END_RATIO = 0.80
TEST_START_RATIO = 0.80
SCALER_END_RATIO = 0.60
# Every valid five-minute origin is used.  This is a protocol constant, not a
# model hyper-parameter: a value of 12 would retain only one hourly offset and
# discard roughly 92% of otherwise valid windows.
PRIMARY_SLIDE_STEP = 1
SCENARIO_RATIOS = {
    "lcbt_zero_history_ratio": 0.25,
    "lcbt_steady_ratio": 0.25,
    "lcbt_join_ratio": 0.25,
    "lcbt_fault_ratio": 0.25,
}

TEACHER_FORCING_POLICY = "lcbt_v4_tf_v3_effective_h_minus_1"
TEACHER_FORCING_DECAY_EPOCHS = 40.0

# The training shuffle, keyed on (run seed, epoch) rather than carried in a
# generator. A `DataLoader(shuffle=True, generator=...)` advances that generator
# once per epoch, so epoch 7's permutation depends on six epochs having been
# drawn in this process -- which makes a resumed run see a different epoch 7
# from an uninterrupted one, silently. Hashing the epoch instead makes the
# permutation a function of the run, not of the process that produced it.
SHUFFLE_POLICY = "lcbt_v4_shuffle_v2_named_subseed"


def shuffle_seed(shuffle_seed_base, epoch):
    key = f"{SHUFFLE_POLICY}|{int(shuffle_seed_base)}|{int(epoch)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def assert_v4_strategy(configs, who):
    """A v4 model runs under the v4 protocol or it does not run."""
    declared = getattr(configs, "lifecycle_training_strategy", None)
    if declared != STRATEGY:
        raise ValueError(
            f"{who}: a model built from a RuntimeDataContext requires "
            f"lifecycle_training_strategy={STRATEGY!r}, config declares "
            f"{declared!r}; v3 and v4 differ on the split, the node scope and "
            "the training computation graph, and their numbers may not share a "
            "table")
    stride = getattr(configs, "slide_step", None)
    if stride is None or int(stride) != PRIMARY_SLIDE_STEP:
        raise ValueError(
            f"{who}: active v4 requires slide_step={PRIMARY_SLIDE_STEP}, "
            f"config declares {stride!r}; stride changes the evaluated origin "
            "population and is not selected per model")


def teacher_forcing_ratio(epoch, mode="linear_decay",
                          decay_epochs=TEACHER_FORCING_DECAY_EPOCHS):
    """Return the declared decoder-input policy for this training epoch."""
    if mode == "free_run":
        return 0.0
    if mode != "linear_decay":
        raise ValueError(f"unknown teacher_forcing_mode={mode!r}")
    decay_epochs = float(decay_epochs)
    if not np.isfinite(decay_epochs) or decay_epochs <= 0:
        raise ValueError(f"decay_epochs must be finite and positive, got {decay_epochs}")
    return float(min(max(1.0 - float(epoch) / decay_epochs, 0.0), 1.0))


def _uniform01(policy, teacher_seed, epoch, origin, horizon_step):
    """A stable [0, 1) draw from the sample's identity. No process state."""
    key = f"{policy}|{int(teacher_seed)}|{int(epoch)}|{int(origin)}|{int(horizon_step)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def teacher_forcing_eligible_mask(batch):
    """Positions that can affect a later decoder output.

    A horizon-H autoregressive decoder makes only H-1 input choices after an
    output: the choice made after output H is never consumed.  Keeping that
    final position false makes both the mask and its reported denominator
    describe effective teacher-forcing decisions rather than H random draws.
    """
    available = batch["decoder_available_mask"].to(torch.bool)
    if available.ndim != 3 or available.shape[1] < 1:
        raise ValueError(
            "decoder_available_mask must have shape [batch, horizon, nodes] "
            "with a positive horizon")
    eligible = available.clone()
    eligible[:, -1, :] = False
    return eligible


def build_v4_teacher_forcing_mask(batch, teacher_seed, epoch, training,
                                  mode="linear_decay",
                                  decay_epochs=TEACHER_FORCING_DECAY_EPOCHS):
    """`[B, H, N]` bool: where a decoder may read the real future.

    Validation and test get an all-false mask, always. There is no argument that
    turns that off, because a validation metric that depended on the target
    would not be a validation metric.
    """
    available = teacher_forcing_eligible_mask(batch)
    if not training:
        return torch.zeros_like(available)
    if "origin" not in batch:
        raise ValueError(
            f"{TEACHER_FORCING_POLICY} needs the sample's origin; without it the "
            "decision would depend on the position in the batch")
    ratio = teacher_forcing_ratio(epoch, mode=mode, decay_epochs=decay_epochs)
    batch_size, horizon, _ = available.shape
    origins = batch["origin"].reshape(-1).tolist()
    if len(origins) != batch_size:
        raise ValueError(
            f"{TEACHER_FORCING_POLICY}: {len(origins)} origins for a batch of "
            f"{batch_size}")
    draws = np.empty((batch_size, horizon), dtype=np.float64)
    for index, origin in enumerate(origins):
        for step in range(horizon):
            draws[index, step] = _uniform01(
                TEACHER_FORCING_POLICY, teacher_seed, epoch, origin, step)
    # `<` and not `<=`: at ratio 0 nothing is forced even though a draw can be
    # exactly 0.0, and at ratio 1 everything is, since draws live in [0, 1).
    decided = torch.from_numpy(draws < ratio).to(available.device)
    return decided.unsqueeze(-1).expand_as(available) & available


def teacher_forcing_metadata(teacher_seed, epoch, training,
                             mode="linear_decay",
                             decay_epochs=TEACHER_FORCING_DECAY_EPOCHS):
    """What a manifest or checkpoint has to record to reproduce the schedule."""
    return {
        "teacher_forcing_policy": TEACHER_FORCING_POLICY,
        "teacher_forcing_mode": str(mode),
        "teacher_forcing_decay_epochs": int(decay_epochs),
        "teacher_forcing_seed": int(teacher_seed),
        "epoch": int(epoch),
        "ratio": (teacher_forcing_ratio(epoch, mode=mode,
                                         decay_epochs=decay_epochs)
                  if training else 0.0),
        "training": bool(training),
    }

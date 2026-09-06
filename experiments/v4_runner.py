"""One v4 train/validation runner shared by every registered model.

This runner contains only the active v4 protocol and objective.

The construction order matters and is enforced by `build_v4_session`:

    train context -> validation context -> (same static assets, different
    exposure) -> train dataset, which fits the scaler -> validation dataset on
    *that* scaler -> loaders -> model, from the train context's static assets

The data resolves the graph first, and the model is built to fit it.

No test loader is constructed, and `build_v4_session` refuses `phase="test"`.
The confirmation splits are spent once; nothing in a training runner has any
business opening one.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from args import load_args_from_config
from data_provider.lifecycle_roles import ROLE_CHANNEL_COUNT, ROLE_VS
from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from experiments.v4_losses import (
    element_loss,
    role_balanced_loss, role_balanced_elements,
)
from lib.lcbt_v4 import (
    SHUFFLE_POLICY,
    assert_v4_strategy,
    build_v4_teacher_forcing_mask,
    shuffle_seed,
    teacher_forcing_eligible_mask,
    teacher_forcing_metadata,
)
from lib.runtime_context import build_runtime_context
from lib.v4_seeds import (
    EpochWorkerSeeder,
    derive_seed,
    seed_bundle,
    seed_process,
)
from utils.model_output import sum_aux_losses, unpack_model_output

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "configs/paper/v4_data_manifests"
CONFIG_DIR = ROOT / "configs"

SCENARIOS = ("zero_history", "steady", "join", "fault")
MODEL_ENTRY_POINTS = {
    "ALFNet": "model.ALFNet.ALFNet",
    "MoGERNN": "model.comparison_model.MoGERNN.MoGERNN",
    "HA": "model.comparison_model.HA",
    "FCLSTM": "model.comparison_model.FCLSTM",
    "STGCN": "model.comparison_model.STGCN",
    "STGODE": "model.comparison_model.stgode",
    "GWNET": "model.comparison_model.GWNet",
    "STID": "model.comparison_model.STID",
    "STFiT": "model.comparison_model.STFiT",
    "SLPF": "model.comparison_model.SLPF",
    "PatchSTG": "model.comparison_model.PatchSTG",
    "STBP": "model.comparison_model.STBP",
    "MCPST_FSL": "model.comparison_model.MCPST_FSL",
}

# Future-target access is a model contract, not a convenience inherited from
# the common runner.  Only the two genuinely autoregressive decoders consume
# the externally generated lifecycle-safe teacher mask.  ST-FiT owns a
# different, source-native use of train targets (its node-augmentation VAE) in
# its wrapper; every other comparison model is target-blind while predicting.
DECODER_TRAINING_POLICIES = {
    "ALFNet": "lifecycle_masked_autoregressive_teacher_forcing",
    "MoGERNN": "source_masked_autoregressive_teacher_forcing",
    "STFiT": "source_future_node_augmentation_not_teacher_forcing",
}
TARGET_BLIND_DECODER_POLICY = "target_blind_non_autoregressive"


def decoder_training_policy(model_name, model=None):
    """Return the frozen future-target contract for one registered model."""
    if model_name not in MODEL_ENTRY_POINTS:
        raise RunnerRefused(f"{model_name!r} has no registered decoder policy")
    declared = getattr(model, "decoder_training_policy", None)
    if declared is not None:
        if model_name != "ALFNet":
            raise RunnerRefused(
                f"{model_name} may not override its registered decoder policy")
        if declared not in {
                "lifecycle_masked_autoregressive_teacher_forcing",
                "target_blind_direct_multi_horizon"}:
            raise RunnerRefused(
                f"ALFNet declared unsupported decoder policy {declared!r}")
        return declared
    return DECODER_TRAINING_POLICIES.get(
        model_name, TARGET_BLIND_DECODER_POLICY)


def uses_external_teacher_mask(model_name, model=None):
    return decoder_training_policy(model_name, model).endswith("teacher_forcing")


def decoder_training_metadata(session, epoch, training=True):
    """Report what the model may read, without calling every future use TF."""
    model = getattr(session, "model", None)
    policy = decoder_training_policy(session.model_name, model)
    if uses_external_teacher_mask(session.model_name, model):
        report = teacher_forcing_metadata(
            session.seed_contract["derived"]["teacher_forcing"], epoch, training,
            mode=session.args.teacher_forcing_mode,
            decay_epochs=session.args.teacher_forcing_decay_epochs)
    else:
        report = {
            "teacher_forcing_policy": "none",
            "teacher_forcing_mode": "not_applicable",
            "teacher_forcing_decay_epochs": None,
            "teacher_forcing_seed": None,
            "epoch": int(epoch),
            "ratio": 0.0,
            "training": bool(training),
        }
    report["decoder_training_policy"] = policy
    return report


def build_model_teacher_mask(session, batch, epoch, training):
    """Build a real teacher mask only for a decoder that declares one."""
    eligible = teacher_forcing_eligible_mask(batch)
    if not uses_external_teacher_mask(
            session.model_name, getattr(session, "model", None)):
        return torch.zeros_like(eligible)
    return build_v4_teacher_forcing_mask(
        batch, session.seed_contract["derived"]["teacher_forcing"], epoch,
        training, mode=session.args.teacher_forcing_mode,
        decay_epochs=session.args.teacher_forcing_decay_epochs)


# The confirmation splits are spent once. A training runner has no business
# opening one, so the refusal is a named rule rather than the incidental
# consequence of nobody having asked yet.
FORBIDDEN_PHASES = ("test",)


class RunnerRefused(ValueError):
    """The requested session or step is not something this runner will do."""


class EpochSeededSampler(Sampler):
    """A training permutation that is a function of (run seed, epoch).

    The DistributedSampler pattern, for the same reason it exists there: the
    order must not depend on how many epochs this particular process has already
    drawn. A run resumed at epoch 7 gets exactly the permutation an uninterrupted
    run would have had at epoch 7.
    """

    policy = SHUFFLE_POLICY

    def __init__(self, data_source, run_seed):
        self.length = len(data_source)
        self.run_seed = int(run_seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        return self

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(shuffle_seed(self.run_seed, self.epoch))
        yield from torch.randperm(self.length, generator=generator).tolist()

    def __len__(self):
        return self.length


def resolve_context(dataset, phase, seed):
    if phase in FORBIDDEN_PHASES:
        raise RunnerRefused(
            f"the v4 training runner will not build a {phase!r} context; the "
            "confirmation splits are read once, by the evaluator, and never "
            "during training or model selection")
    return build_runtime_context(dataset, phase, seed=seed)


@dataclass
class V4Session:
    """Everything one (dataset, model) v4 run needs, built in the right order."""

    dataset: str
    model_name: str
    args: object
    model: torch.nn.Module
    train_context: object
    validation_context: object
    train_dataset: V4LifecycleDataset
    validation_dataset: V4LifecycleDataset
    train_loader: DataLoader
    validation_loader: DataLoader
    run_seed: int
    seed_contract: dict
    train_worker_seeder: object = field(default=None, repr=False)
    validation_worker_seeder: object = field(default=None, repr=False)
    device: str = "cpu"
    scaler: dict = field(default=None)

    def set_epoch(self, epoch):
        """Move the schedule and the shuffle together. Both are (seed, epoch)."""
        self.train_dataset.set_epoch(epoch)
        self.validation_dataset.set_epoch(epoch)
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if self.train_worker_seeder is not None:
            self.train_worker_seeder.set_epoch(epoch)
        # Validation remains one frozen design, including worker state.
        if self.validation_worker_seeder is not None:
            self.validation_worker_seeder.set_epoch(0)
        return self


def resolve_payload_path(dataset, args):
    """The payload the manifest declares, cross-checked against the config."""
    manifest = json.loads(
        (MANIFEST_DIR / f"{dataset}_run_manifest.json").read_text("utf-8"))
    declared = ROOT / manifest["payload"]["path"]
    configured = getattr(args, "data_path", None)
    if configured and Path(configured).resolve() != declared.resolve():
        raise RunnerRefused(
            f"the config reads {configured}, the {dataset} manifest declares "
            f"{manifest['payload']['path']}; one of them is wrong and it is not "
            "for this runner to pick")
    return declared


def build_v4_session(config_path, device="cpu", run_seed=None, batch_size=None,
                     num_workers=None, payload_path=None, init_seed=None):
    """Build one v4 session from a v4 config. Data first, model second.

    Everything but the four run-level knobs comes from the config. The strategy
    is *checked*, never rewritten: the previous form loaded a v3 YAML and set
    `lifecycle_training_strategy` to v4 in code, which meant the file on disk and
    the run it produced disagreed, and only the code knew.
    """
    args = load_args_from_config(str(config_path), [])
    try:
        assert_v4_strategy(args, "v4 runner")
    except ValueError as error:
        raise RunnerRefused(str(error)) from error
    model_name = str(args.model)
    if model_name not in MODEL_ENTRY_POINTS:
        raise RunnerRefused(
            f"{model_name} has no v4 entry point; this runner takes "
            f"{sorted(MODEL_ENTRY_POINTS)}")
    dataset = str(args.dloader_name)
    args.device = device
    run_seed = int(args.seed if run_seed is None else run_seed)
    args.seed = run_seed
    seeds = seed_bundle(run_seed)
    if init_seed is not None and int(init_seed) != seeds["derived"]["model_init"]:
        raise RunnerRefused(
            "v4 model initialisation is derived from run_seed; an independent "
            "init_seed would make the advertised run seed incomplete")
    args.model_init_seed = int(seeds["derived"]["model_init"])
    validation_seed = int(args.lcbt_validation_seed)
    if validation_seed != int(seeds["evaluation_design_seed"]):
        raise RunnerRefused(
            f"v4 evaluation design seed is frozen at "
            f"{seeds['evaluation_design_seed']}, config declares "
            f"{validation_seed}")
    batch_size = int(args.batch_size if batch_size is None else batch_size)
    num_workers = int(args.num_workers if num_workers is None else num_workers)
    payload_path = payload_path or resolve_payload_path(dataset, args)

    # 1-3. Both phases, and they must agree on the graph and differ on exposure.
    train_context = resolve_context(dataset, "train", run_seed)
    validation_context = resolve_context(dataset, "validation", run_seed)
    if train_context.static_asset_hash() != validation_context.static_asset_hash():
        raise RunnerRefused(
            "the train and validation contexts resolved different static graph "
            "assets; one static model cannot serve both")
    if train_context.exposure_hash() == validation_context.exposure_hash():
        raise RunnerRefused(
            "the train and validation contexts share an exposure hash; "
            "validation would be reading the training scope")

    # 4-5. The scaler is fitted once, on the training window and training-known
    #      nodes, and *handed* to validation. Rebuilding it there would produce
    #      the same numbers today and silently stop doing so the moment the
    #      validation window changed.
    train_dataset = V4LifecycleDataset(
        train_context, payload_path, input_len=args.input_len,
        pred_len=args.pred_len, slide_step=args.slide_step, freq=args.freq,
        seed=seeds["derived"]["train_schedule"], epoch=0,
        validation_seed=validation_seed)
    validation_dataset = V4LifecycleDataset(
        validation_context, payload_path, input_len=args.input_len,
        pred_len=args.pred_len, slide_step=args.slide_step, freq=args.freq,
        seed=seeds["derived"]["train_schedule"], epoch=0,
        validation_seed=validation_seed,
        scaler=train_dataset.scaler)
    if validation_dataset.scaler is not train_dataset.scaler:
        raise RunnerRefused("validation did not receive the training scaler")

    # 6. Loaders. The training order is drawn from (run seed, epoch) so it
    #    survives a resume; the teacher-forcing policy is keyed on the sample's
    #    origin and cannot see the order at all.
    train_sampler = EpochSeededSampler(train_dataset, seeds["derived"]["shuffle"])
    train_worker_seeder = EpochWorkerSeeder(run_seed, "train")
    validation_worker_seeder = EpochWorkerSeeder(run_seed, "validation")
    train_generator = torch.Generator().manual_seed(
        derive_seed(run_seed, "dataloader_worker", index=0))
    validation_generator = torch.Generator().manual_seed(
        derive_seed(run_seed, "dataloader_worker", index=1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=num_workers,
                              drop_last=False,
                              worker_init_fn=train_worker_seeder,
                              generator=train_generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers,
                                   worker_init_fn=validation_worker_seeder,
                                   generator=validation_generator)

    # 7-8. One static model, built on the train context's assets, which the
    #      validation context shares. It then accepts either per batch.
    module = __import__(MODEL_ENTRY_POINTS[model_name], fromlist=["Model"])
    seed_process(seeds["derived"]["model_init"])
    model = module.Model.from_runtime_context(args, train_context).to(device)
    # Training-time randomness (dropout and any future stochastic operation)
    # starts from its own stream, independent of how many numbers construction
    # consumed.
    seed_process(seeds["derived"]["training_runtime"])
    return V4Session(
        dataset=dataset, model_name=model_name, args=args, model=model,
        train_context=train_context, validation_context=validation_context,
        train_dataset=train_dataset, validation_dataset=validation_dataset,
        train_loader=train_loader, validation_loader=validation_loader,
        run_seed=run_seed, seed_contract=seeds,
        train_worker_seeder=train_worker_seeder,
        validation_worker_seeder=validation_worker_seeder,
        device=str(device), scaler=train_dataset.scaler)


# --- the loss ------------------------------------------------------------------


def valid_positions(batch, prediction):
    """Where the objective may be evaluated. Exactly the loader's `loss_mask`.

    Not re-derived from the cohort, the phase or a non-zero test. The loader
    already combined natural availability, the episode's cohort and the phase's
    loss scope; a second derivation here would be a second protocol.
    """
    mask = batch["loss_mask"].unsqueeze(-1)
    return mask & torch.isfinite(prediction) & torch.isfinite(batch["target"])


def base_loss(prediction, target, valid, loss_name="mse"):
    """The one common objective, identical for both models."""
    if not bool(valid.any()):
        return None
    return element_loss(prediction, target, loss_name)[valid].mean()


def _alfnet_role_mix(prediction, batch, valid, meta, loss_name):
    """ALFNet's declared `role_balanced_mix`, at its own beta.

    Reused from the v3 step rather than reimplemented, and it applies only when
    the model itself asks for it. MoGERNN reports no `forecast_loss_mode` and
    gets no substitute term invented on its behalf.
    """
    if meta.get("forecast_loss_mode") != "role_balanced_mix":
        return None, None
    last_role = batch["lifecycle_context"][:, -1, :, :ROLE_CHANNEL_COUNT]
    role_loss = role_balanced_loss(prediction, batch["target"], valid, last_role,
                                   loss_name)
    if role_loss is None:
        return None, None
    return role_loss, float(meta.get("forecast_role_balanced_beta", 0.1))


def _alfnet_scale_loss(prediction, target, batch, valid, meta, scaler):
    """Stable low-value emphasis without a target-dependent denominator.

    The signed log transform is smooth on both sides of zero and compresses
    large flow values.  Its scale is derived only from the training scaler, so
    neither validation/test values nor ``1/|target|`` weighting can leak into
    the objective.  Huber limits the influence of a few large transformed
    residuals; lifecycle roles are aggregated by the same contract as BASE.
    """
    if meta.get("forecast_scale_loss_mode") != "signed_log_huber":
        return None, None
    if scaler is None:
        raise RunnerRefused("signed_log_huber requires the train-only scaler")
    weight = float(meta.get("forecast_scale_loss_weight", 0.0))
    fraction = float(meta.get("forecast_scale_loss_scale_fraction", 0.05))
    if not 0.0 <= weight <= 1.0 or not fraction > 0.0:
        raise RunnerRefused("invalid signed_log_huber weight/scale fraction")
    scale = max(1.0, abs(float(scaler["std"])) * fraction)
    mean, std = float(scaler["mean"]), float(scaler["std"])
    predicted_raw = prediction * std + mean
    target_raw = target * std + mean
    transform = lambda value: torch.sign(value) * torch.log1p(value.abs() / scale)
    elements = F.smooth_l1_loss(
        transform(predicted_raw), transform(target_raw), reduction="none")
    last_role = batch["lifecycle_context"][:, -1, :, :ROLE_CHANNEL_COUNT]
    loss = role_balanced_elements(elements, valid, last_role)
    return loss, weight


def _alfnet_synthetic_unseen_loss(
        prediction, target, batch, meta, loss_name):
    """Forecast loss on train-only nodes hidden from the entire input window."""
    if meta.get("synthetic_unseen_training_mode") is None:
        return None, None
    if meta.get("synthetic_unseen_training_mode") != \
            "observed_node_mask_reconstruction":
        raise RunnerRefused("unsupported synthetic-unseen loss mode")
    if "synthetic_unseen_aux_mask" not in batch:
        raise RunnerRefused(
            "model declared synthetic-unseen training without an augmented batch")
    weight = float(meta.get("synthetic_unseen_loss_weight", 0.0))
    if not 0.0 < weight < 1.0:
        raise RunnerRefused("synthetic-unseen loss weight must be in (0, 1)")
    valid = (batch["synthetic_unseen_aux_mask"].unsqueeze(-1)
             & torch.isfinite(prediction) & torch.isfinite(target))
    if not bool(valid.any()):
        return None, weight
    return element_loss(prediction, target, loss_name)[valid].mean(), weight


def _alfnet_train_scale_bin_loss(
        prediction, target, valid, meta, scaler):
    """Equal-weight fixed train-scale magnitude bins without 1/target weights."""
    if meta.get("low_flow_calibration_mode") is None:
        return None, None
    if meta.get("low_flow_calibration_mode") != \
            "role_horizon_residual_train_scale_bins":
        raise RunnerRefused("unsupported low-flow calibration mode")
    if scaler is None:
        raise RunnerRefused("train-scale bin loss requires the training scaler")
    weight = float(meta.get("low_flow_bin_loss_weight", 0.0))
    if not 0.0 < weight < 1.0:
        raise RunnerRefused("low-flow bin loss weight must be in (0, 1)")
    mean, std = float(scaler["mean"]), abs(float(scaler["std"]))
    target_raw = (target * std + mean).abs()
    # Fixed from the training standard deviation, never from validation/test or
    # the current target denominator.  Every populated magnitude band gets the
    # same influence instead of a near-zero example receiving unbounded weight.
    boundaries = prediction.new_tensor(
        [0.0, 0.05 * std, 0.10 * std, 0.20 * std, 0.50 * std,
         float("inf")])
    elements = F.smooth_l1_loss(prediction, target, reduction="none")
    terms = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = valid & (target_raw >= lower) & (target_raw < upper)
        if bool(selected.any()):
            terms.append(elements[selected].mean())
    return (torch.stack(terms).mean() if terms else None), weight


def compute_losses(model_output, batch, loss_name="mse", scaler=None):
    """Split the objective into what is shared and what the model declared.

    Returns `base`, the model's own terms, and the tensor actually optimised.
    Keeping them apart is the point: `base` is the model's declared forecast
    loss and must not quietly absorb an ALFNet-only or source-specific term.
    """
    prediction, meta = unpack_model_output(model_output)
    target = meta.get("target", batch["target"])
    if target.shape != prediction.shape:
        raise RunnerRefused(
            f"model target override shape {tuple(target.shape)} does not match "
            f"prediction {tuple(prediction.shape)}")
    if "target" in meta:
        valid = torch.isfinite(prediction) & torch.isfinite(target)
    else:
        valid = valid_positions(batch, prediction)
    base = base_loss(prediction, target, valid, loss_name)
    if base is None:
        return {"prediction": prediction, "valid": valid, "base": None,
                "optimize": None, "role_mix": None, "scale_mix": None,
                "synthetic_unseen_mix": None, "low_flow_bin_mix": None,
                "aux": None,
                "meta": meta}

    optimize = base
    role_loss, beta = _alfnet_role_mix(prediction, batch, valid, meta, loss_name)
    if role_loss is not None:
        optimize = (1.0 - beta) * optimize + beta * role_loss
    scale_loss, scale_weight = _alfnet_scale_loss(
        prediction, target, batch, valid, meta, scaler)
    if scale_loss is not None:
        optimize = (1.0 - scale_weight) * optimize + scale_weight * scale_loss
    synthetic_unseen_loss, synthetic_unseen_weight = \
        _alfnet_synthetic_unseen_loss(
            prediction, target, batch, meta, loss_name)
    if synthetic_unseen_loss is not None:
        optimize = ((1.0 - synthetic_unseen_weight) * optimize
                    + synthetic_unseen_weight * synthetic_unseen_loss)
    low_flow_bin_loss, low_flow_bin_weight = _alfnet_train_scale_bin_loss(
        prediction, target, valid, meta, scaler)
    if low_flow_bin_loss is not None:
        optimize = ((1.0 - low_flow_bin_weight) * optimize
                    + low_flow_bin_weight * low_flow_bin_loss)
    aux = sum_aux_losses(meta.get("aux_losses"), device=prediction.device)
    if aux is not None:
        optimize = optimize + aux
    return {"prediction": prediction, "valid": valid, "base": base,
            "optimize": optimize, "role_mix": role_loss,
            "scale_mix": scale_loss,
            "synthetic_unseen_mix": synthetic_unseen_loss,
            "low_flow_bin_mix": low_flow_bin_loss,
            "aux": aux, "meta": meta}


# --- the two passes ------------------------------------------------------------


def to_device(batch, device):
    """Move the batch's tensors to the model's device, metadata untouched.

    The model is moved with `.to(device)` at construction; the batch comes off a
    DataLoader on the CPU. Every test up to here ran on the CPU, where the two
    agree by accident, so the first GPU call was the first time it mattered --
    which is what the pre-flight health check exists to find.

    `context_identity` is metadata that never reaches a module, so it stays as
    it is; moving it would only make the exposure check compare CUDA tensors.
    """
    if str(device).startswith("cpu"):
        return batch
    moved = {}
    for key, value in batch.items():
        moved[key] = (value.to(device, non_blocking=True)
                      if torch.is_tensor(value) and key != "context_identity"
                      else value)
    return moved


def _synthetic_unseen_seed(run_seed, epoch, sample_index):
    """Stable per-sample stream, independent of batch order and CUDA RNG."""
    key = (f"lcbt_v4_observed_node_mask_v1|{int(run_seed)}|{int(epoch)}|"
           f"{int(sample_index)}")
    return int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % (2 ** 63)


def apply_synthetic_unseen_training(session, batch, epoch, training=True):
    """Hide observed K nodes and expose their future only to an auxiliary loss.

    This is ALFNet-only, train-only augmentation.  Selection uses the run seed,
    epoch, sample id, input availability and the already-visible K scope; it
    never reads a future target value.  The original lifecycle cohort/loss mask
    is preserved, so the paper objective remains identifiable beside the
    synthetic reconstruction term.
    """
    model = getattr(session, "model", None)
    mode = getattr(model, "synthetic_unseen_training_mode", None)
    if not training or mode is None:
        return batch
    if session.model_name != "ALFNet":
        raise RunnerRefused("synthetic-unseen training is an ALFNet method term")
    if mode != "observed_node_mask_reconstruction":
        raise RunnerRefused(f"unsupported synthetic-unseen mode {mode!r}")
    fraction = float(model.synthetic_unseen_fraction)
    if not 0.0 < fraction < 1.0:
        raise RunnerRefused("synthetic-unseen fraction must be in (0, 1)")

    required = {
        "base_visible_input", "input_available_mask",
        "decoder_available_mask", "target_available_mask",
        "target_cohort_mask", "lifecycle_role_ids",
        "lifecycle_state_age", "lifecycle_context", "sample_index",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise RunnerRefused(
            f"synthetic-unseen batch is missing fields {missing}")
    augmented = dict(batch)
    mutable = (
        "base_visible_input", "input_available_mask",
        "decoder_available_mask", "lifecycle_role_ids",
        "lifecycle_state_age", "lifecycle_context",
    )
    for key in mutable:
        augmented[key] = batch[key].clone()
    batch_size, _, node_count = batch["input_available_mask"].shape
    selected = torch.zeros((batch_size, node_count), dtype=torch.bool)
    scope = torch.zeros(node_count, dtype=torch.bool)
    scope[torch.as_tensor(
        session.train_context.signal_scope, dtype=torch.long)] = True
    target_count = max(1, int(round(fraction * int(scope.sum()))))

    for position in range(batch_size):
        input_seen = batch["input_available_mask"][position].any(dim=0)
        original_cohort = batch["target_cohort_mask"][position] > 0.5
        eligible = torch.nonzero(
            scope & input_seen & ~original_cohort, as_tuple=False).flatten()
        if eligible.numel() == 0:
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_synthetic_unseen_seed(
            session.run_seed, epoch, batch["sample_index"][position]))
        take = min(target_count, int(eligible.numel()))
        nodes = eligible[torch.randperm(
            int(eligible.numel()), generator=generator)[:take]]
        selected[position, nodes] = True

    if not bool(selected.any()):
        raise RunnerRefused("synthetic-unseen training selected no observed node")
    node_mask = selected[:, None, :]
    augmented["base_visible_input"].masked_fill_(
        node_mask.unsqueeze(-1), 0.0)
    augmented["input_available_mask"].masked_fill_(node_mask, False)
    augmented["decoder_available_mask"].masked_fill_(node_mask, False)
    augmented["lifecycle_role_ids"].masked_fill_(node_mask, ROLE_VS)
    augmented["lifecycle_state_age"].masked_fill_(node_mask, 0.0)
    lifecycle = augmented["lifecycle_context"]
    lifecycle[..., :ROLE_CHANNEL_COUNT].masked_fill_(
        node_mask.unsqueeze(-1), 0.0)
    lifecycle[..., ROLE_VS].masked_fill_(node_mask, 1.0)
    lifecycle[..., ROLE_CHANNEL_COUNT:].masked_fill_(
        node_mask.unsqueeze(-1), 0.0)
    augmented["synthetic_unseen_node_mask"] = selected
    augmented["synthetic_unseen_aux_mask"] = (
        batch["target_available_mask"] & selected[:, None, :])
    return augmented


def _forward(session, batch, context, epoch, teacher_forcing_mask):
    return session.model.forward_v4(batch, context, epoch=epoch,
                                    teacher_forcing_mask=teacher_forcing_mask)


def train_one_epoch(session, optimizer, epoch, max_batches=None, loss_name="mse"):
    """One training pass. Deterministic given the session, seed and epoch."""
    session.model.train()
    epoch_hook = getattr(session.model, "on_train_epoch_start", None)
    if callable(epoch_hook):
        epoch_hook(epoch)
    session.set_epoch(epoch)
    totals = {"batches": 0, "skipped": 0, "base": 0.0, "optimize": 0.0,
              "role_mix": 0.0, "scale_mix": 0.0,
              "synthetic_unseen_mix": 0.0, "low_flow_bin_mix": 0.0,
              "aux": 0.0,
              "forced_positions": 0,
              "available_positions": 0,
              "synthetic_unseen_positions": 0}
    # Counted separately from the sums: a model that declares no role-mix term
    # must report `None`, not the 0.0 that an absent term and a zero-valued one
    # would otherwise share. MoGERNN declares none and gets no stand-in.
    contributed = {"base": 0, "optimize": 0, "role_mix": 0, "scale_mix": 0,
                   "synthetic_unseen_mix": 0, "low_flow_bin_mix": 0,
                   "aux": 0}
    for index, batch in enumerate(session.train_loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = apply_synthetic_unseen_training(
            session, batch, epoch, training=True)
        if "synthetic_unseen_aux_mask" in batch:
            totals["synthetic_unseen_positions"] += int(
                batch["synthetic_unseen_aux_mask"].sum())
        batch = to_device(batch, session.device)
        teacher = build_model_teacher_mask(session, batch, epoch, True)
        totals["forced_positions"] += int(teacher.sum())
        if uses_external_teacher_mask(session.model_name, session.model):
            totals["available_positions"] += int(
                teacher_forcing_eligible_mask(batch).sum())
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(_forward(session, batch, session.train_context,
                                         epoch, teacher), batch, loss_name,
                                scaler=session.scaler)
        if losses["optimize"] is None or not torch.isfinite(losses["optimize"]):
            # Counted, never silently averaged into a zero.
            totals["skipped"] += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        losses["optimize"].backward()
        optimizer.step()
        totals["batches"] += 1
        for key in ("base", "optimize", "role_mix", "scale_mix",
                    "synthetic_unseen_mix", "low_flow_bin_mix", "aux"):
            if losses[key] is not None:
                totals[key] += float(losses[key].detach())
                contributed[key] += 1
    if totals["batches"] == 0:
        raise RunnerRefused(
            f"epoch {epoch} produced no batch with a valid loss position; "
            "refusing to report a zero")
    report = {key: (totals[key] / contributed[key] if contributed[key] else None)
              for key in ("base", "optimize", "role_mix", "scale_mix",
                          "synthetic_unseen_mix", "low_flow_bin_mix", "aux")}
    report.update({"batches": totals["batches"], "skipped": totals["skipped"],
                   "forced_positions": totals["forced_positions"],
                   "available_positions": totals["available_positions"],
                   "synthetic_unseen_positions":
                       totals["synthetic_unseen_positions"]})
    report.update(decoder_training_metadata(session, epoch, True))
    report["run_seed"] = session.run_seed
    return report


@torch.no_grad()
def validate(session, epoch=0, max_batches=None, loss_name="mse",
             return_cell_table=False):
    """One validation pass, optionally returning the 33-cell table.

    The former training path ran this loop once for early stopping and a second
    time whenever the best epoch changed.  A collector can now share the exact
    predictions from this pass, so checkpoint selection and the diagnostic
    table cannot drift and validation costs one forward pass per epoch.
    """
    session.model.eval()
    scaler = session.scaler
    collector = None
    if return_cell_table:
        from experiments.v4_validation_collector import ValidationCellCollector
        collector = ValidationCellCollector(session.validation_context, scaler)
    per_scenario = {name: {"absolute": 0.0, "squared": 0.0, "count": 0}
                    for name in SCENARIOS}
    overall = {"absolute": 0.0, "squared": 0.0, "count": 0}
    batches, skipped = 0, 0
    for index, batch in enumerate(session.validation_loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = to_device(batch, session.device)
        teacher = build_model_teacher_mask(session, batch, epoch, False)
        if bool(teacher.any()):
            raise RunnerRefused("validation produced a non-empty teacher mask")
        prediction, _ = unpack_model_output(
            _forward(session, batch, session.validation_context, epoch, None))
        if collector is not None:
            collector.update(batch, prediction)
        valid = valid_positions(batch, prediction)
        if not bool(valid.any()):
            skipped += 1
            continue
        batches += 1
        # Metrics in the original scale. `inverse = z * std + mean` on both
        # sides; the offset cancels in the error but the scale does not.
        error = (prediction - batch["target"]) * float(scaler["std"])
        absolute, squared = error.abs(), error.pow(2)
        overall["absolute"] += float(absolute[valid].sum())
        overall["squared"] += float(squared[valid].sum())
        overall["count"] += int(valid.sum())
        for position, scenario in enumerate(batch["scenario"]):
            sample = valid[position]
            if not bool(sample.any()):
                continue
            bucket = per_scenario[str(scenario)]
            bucket["absolute"] += float(absolute[position][sample].sum())
            bucket["squared"] += float(squared[position][sample].sum())
            bucket["count"] += int(sample.sum())
    if overall["count"] == 0:
        raise RunnerRefused(
            "validation found no valid loss position in any batch; refusing to "
            "report a zero")
    report = {
        "mae": overall["absolute"] / overall["count"],
        "mse": overall["squared"] / overall["count"],
        "count": overall["count"], "batches": batches, "skipped": skipped,
    }
    for name in SCENARIOS:
        bucket = per_scenario[name]
        report[f"{name}_mae"] = (bucket["absolute"] / bucket["count"]
                                 if bucket["count"] else None)
        report[f"{name}_count"] = bucket["count"]
    # LifeAvg: the equal-weight mean over the scenarios that actually produced a
    # valid sample. Scoring an absent scenario as zero would reward a schedule
    # that failed to draw it.
    present = [report[f"{name}_mae"] for name in SCENARIOS
               if report[f"{name}_mae"] is not None]
    if not present:
        raise RunnerRefused("no scenario produced a valid sample")
    report["life_avg_mae"] = sum(present) / len(present)
    report["scenarios_present"] = len(present)
    report["teacher_forcing_ratio"] = 0.0
    if return_cell_table:
        return report, collector.table()
    return report


def early_stopping_metric(validation_report, cell_table):
    """The reported 33-cell main/LifeAvg/MAE, from this same forward pass."""
    try:
        value = float(cell_table["main"]["LifeAvg"]["MAE"])
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerRefused(
            "checkpoint selection requires validation 33-cell "
            "main/LifeAvg/MAE") from error
    if not torch.isfinite(torch.tensor(value)):
        raise RunnerRefused("validation 33-cell main/LifeAvg/MAE is not finite")
    if cell_table.get("split") != "validation":
        raise RunnerRefused("checkpoint selection table is not validation")
    # Retain the raw micro-aggregated diagnostic, but never silently confuse it
    # with the calendar/k/tau-balanced paper endpoint used for selection.
    validation_report["micro_life_avg_mae"] = validation_report["life_avg_mae"]
    validation_report["checkpoint_life_avg_mae"] = value
    return value

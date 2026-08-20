"""One v4 train/validation runner, shared by ALFNet and MoGERNN.

Deliberately a new file rather than another branch inside `shared_step`. The v3
step unpacks a ten-tensor tuple positionally and carries eight optional
objective knobs; threading a second protocol through it would put v3 and v4 one
`if` apart on every line, and the two are supposed to be separable.

The construction order matters and is enforced by `build_v4_session`:

    train context -> validation context -> (same static assets, different
    exposure) -> train dataset, which fits the scaler -> validation dataset on
    *that* scaler -> loaders -> model, from the train context's static assets

The v3 order builds the model first and lets the loader resolve its own graph
later, which is how a model and its data ended up describing different node
sets. Here the data decides, and the model is built to fit it.

No test loader is constructed, and `build_v4_session` refuses `phase="test"`.
The confirmation splits are spent once; nothing in a training runner has any
business opening one.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from args import load_args_from_config
from data_provider.lifecycle_roles import ROLE_CHANNEL_COUNT
from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from experiments.inductive_epoch import _element_loss, _role_balanced_loss
from lib.lcbt_v4 import (
    STRATEGY,
    build_v4_teacher_forcing_mask,
    teacher_forcing_metadata,
)
from lib.runtime_context import build_runtime_context
from utils.model_output import sum_aux_losses, unpack_model_output

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "configs/paper/v4_data_manifests"
CONFIG_DIR = ROOT / "configs/paper/v4"

SCENARIOS = ("zero_history", "steady", "join", "fault")
MODEL_ENTRY_POINTS = {
    "ALFNet": "model.ALFNet.ALFNet",
    "MoGERNN": "model.comparison_model.MoGERNN.MoGERNN",
}


# The confirmation splits are spent once. A training runner has no business
# opening one, so the refusal is a named rule rather than the incidental
# consequence of nobody having asked yet.
FORBIDDEN_PHASES = ("test",)


class RunnerRefused(ValueError):
    """The requested session or step is not something this runner will do."""


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
    device: str = "cpu"
    scaler: dict = field(default=None)

    def set_epoch(self, epoch):
        self.train_dataset.set_epoch(epoch)
        self.validation_dataset.set_epoch(epoch)
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
                     num_workers=None, payload_path=None, init_seed=1234):
    """Build one v4 session from a v4 config. Data first, model second.

    Everything but the four run-level knobs comes from the config. The strategy
    is *checked*, never rewritten: the previous form loaded a v3 YAML and set
    `lifecycle_training_strategy` to v4 in code, which meant the file on disk and
    the run it produced disagreed, and only the code knew.
    """
    args = load_args_from_config(str(config_path), [])
    if args.lifecycle_training_strategy != STRATEGY:
        raise RunnerRefused(
            f"{config_path} declares lifecycle_training_strategy="
            f"{args.lifecycle_training_strategy!r}; the v4 runner takes "
            f"{STRATEGY!r} configs only and does not rewrite one")
    model_name = str(args.model)
    if model_name not in MODEL_ENTRY_POINTS:
        raise RunnerRefused(
            f"{model_name} has no v4 entry point; this runner takes "
            f"{sorted(MODEL_ENTRY_POINTS)}")
    dataset = str(args.dloader_name)
    args.device = device
    run_seed = int(args.seed if run_seed is None else run_seed)
    args.seed = run_seed
    validation_seed = int(args.lcbt_validation_seed)
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
        train_context, payload_path, seed=run_seed, epoch=0,
        validation_seed=validation_seed)
    validation_dataset = V4LifecycleDataset(
        validation_context, payload_path, seed=run_seed, epoch=0,
        validation_seed=validation_seed, scaler=train_dataset.scaler)
    if validation_dataset.scaler is not train_dataset.scaler:
        raise RunnerRefused("validation did not receive the training scaler")

    # 6. Loaders. Shuffling is the loader's business; the teacher-forcing policy
    #    is keyed on the sample's origin and cannot see it.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False,
                              generator=torch.Generator().manual_seed(run_seed))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)

    # 7-8. One static model, built on the train context's assets, which the
    #      validation context shares. It then accepts either per batch.
    module = __import__(MODEL_ENTRY_POINTS[model_name], fromlist=["Model"])
    torch.manual_seed(init_seed)
    model = module.Model.from_runtime_context(args, train_context).to(device)
    return V4Session(
        dataset=dataset, model_name=model_name, args=args, model=model,
        train_context=train_context, validation_context=validation_context,
        train_dataset=train_dataset, validation_dataset=validation_dataset,
        train_loader=train_loader, validation_loader=validation_loader,
        run_seed=run_seed, device=str(device), scaler=train_dataset.scaler)


# --- the loss ------------------------------------------------------------------


def valid_positions(batch, prediction):
    """Where the objective may be evaluated. Exactly the loader's `loss_mask`.

    Not re-derived from the cohort, the phase or a non-zero test. The loader
    already combined natural availability, the episode's cohort and the phase's
    loss scope; a second derivation here would be a second protocol.
    """
    mask = batch["loss_mask"].unsqueeze(-1)
    return mask & torch.isfinite(prediction) & torch.isfinite(batch["target"])


def base_loss(prediction, batch, valid, loss_name="mse"):
    """The one common objective, identical for both models."""
    if not bool(valid.any()):
        return None
    return _element_loss(prediction, batch["target"], loss_name)[valid].mean()


def _alfnet_role_mix(prediction, batch, valid, meta, loss_name):
    """ALFNet's declared `role_balanced_mix`, at its own beta.

    Reused from the v3 step rather than reimplemented, and it applies only when
    the model itself asks for it. MoGERNN reports no `forecast_loss_mode` and
    gets no substitute term invented on its behalf.
    """
    if meta.get("forecast_loss_mode") != "role_balanced_mix":
        return None, None
    last_role = batch["lifecycle_context"][:, -1, :, :ROLE_CHANNEL_COUNT]
    role_loss = _role_balanced_loss(prediction, batch["target"], valid, last_role,
                                    loss_name)
    if role_loss is None:
        return None, None
    return role_loss, float(meta.get("forecast_role_balanced_beta", 0.1))


def compute_losses(model_output, batch, loss_name="mse"):
    """Split the objective into what is shared and what the model declared.

    Returns `base`, the model's own terms, and the tensor actually optimised.
    Keeping them apart is the point: `base` is the only quantity the two models
    can be compared on, and it must not quietly absorb an ALFNet-only term.
    """
    prediction, meta = unpack_model_output(model_output)
    valid = valid_positions(batch, prediction)
    base = base_loss(prediction, batch, valid, loss_name)
    if base is None:
        return {"prediction": prediction, "valid": valid, "base": None,
                "optimize": None, "role_mix": None, "aux": None, "meta": meta}

    optimize = base
    role_loss, beta = _alfnet_role_mix(prediction, batch, valid, meta, loss_name)
    if role_loss is not None:
        optimize = (1.0 - beta) * optimize + beta * role_loss
    aux = sum_aux_losses(meta.get("aux_losses"), device=prediction.device)
    if aux is not None:
        optimize = optimize + aux
    return {"prediction": prediction, "valid": valid, "base": base,
            "optimize": optimize, "role_mix": role_loss, "aux": aux, "meta": meta}


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


def _forward(session, batch, context, epoch, teacher_forcing_mask):
    return session.model.forward_v4(batch, context, epoch=epoch,
                                    teacher_forcing_mask=teacher_forcing_mask)


def train_one_epoch(session, optimizer, epoch, max_batches=None, loss_name="mse"):
    """One training pass. Deterministic given the session, seed and epoch."""
    session.model.train()
    session.train_dataset.set_epoch(epoch)
    totals = {"batches": 0, "skipped": 0, "base": 0.0, "optimize": 0.0,
              "role_mix": 0.0, "aux": 0.0, "forced_positions": 0,
              "available_positions": 0}
    # Counted separately from the sums: a model that declares no role-mix term
    # must report `None`, not the 0.0 that an absent term and a zero-valued one
    # would otherwise share. MoGERNN declares none and gets no stand-in.
    contributed = {"base": 0, "optimize": 0, "role_mix": 0, "aux": 0}
    for index, batch in enumerate(session.train_loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = to_device(batch, session.device)
        teacher = build_v4_teacher_forcing_mask(batch, session.run_seed, epoch, True)
        totals["forced_positions"] += int(teacher.sum())
        totals["available_positions"] += int(batch["decoder_available_mask"].sum())
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(_forward(session, batch, session.train_context,
                                         epoch, teacher), batch, loss_name)
        if losses["optimize"] is None or not torch.isfinite(losses["optimize"]):
            # Counted, never silently averaged into a zero.
            totals["skipped"] += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        losses["optimize"].backward()
        optimizer.step()
        totals["batches"] += 1
        for key in ("base", "optimize", "role_mix", "aux"):
            if losses[key] is not None:
                totals[key] += float(losses[key].detach())
                contributed[key] += 1
    if totals["batches"] == 0:
        raise RunnerRefused(
            f"epoch {epoch} produced no batch with a valid loss position; "
            "refusing to report a zero")
    report = {key: (totals[key] / contributed[key] if contributed[key] else None)
              for key in ("base", "optimize", "role_mix", "aux")}
    report.update({"batches": totals["batches"], "skipped": totals["skipped"],
                   "forced_positions": totals["forced_positions"],
                   "available_positions": totals["available_positions"]})
    report.update(teacher_forcing_metadata(session.run_seed, epoch, True))
    return report


@torch.no_grad()
def validate(session, epoch=0, max_batches=None, loss_name="mse"):
    """One validation pass, in the original scale, with no teacher forcing."""
    session.model.eval()
    scaler = session.scaler
    per_scenario = {name: {"absolute": 0.0, "squared": 0.0, "count": 0}
                    for name in SCENARIOS}
    overall = {"absolute": 0.0, "squared": 0.0, "count": 0}
    batches, skipped = 0, 0
    for index, batch in enumerate(session.validation_loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = to_device(batch, session.device)
        teacher = build_v4_teacher_forcing_mask(batch, session.run_seed, epoch, False)
        if bool(teacher.any()):
            raise RunnerRefused("validation produced a non-empty teacher mask")
        prediction, _ = unpack_model_output(
            _forward(session, batch, session.validation_context, epoch, None))
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
    return report


def early_stopping_metric(validation_report):
    """The only candidate this runner offers. It never reads a test split."""
    return validation_report["life_avg_mae"]

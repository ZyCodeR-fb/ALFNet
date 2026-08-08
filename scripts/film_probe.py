"""Is the ungated FiLM path what costs ALFNet its no-history nodes?

`ALFNet._run_experts` applies the FiLM `(gamma, beta)` from the temporal branch
to the graph encoder state unconditionally, whereas `_evidence_fusion` gates its
self-residual by `e_self`. The hypothesis under test is that a node with no own
history still has its graph representation rewritten by a conditioner that read
nothing, and that this is what costs the Unseen group.

The hypothesis makes two falsifiable predictions on a trained checkpoint:

  1. |gamma| and |beta| are not near identity on the nodes with `e_self == 0`;
  2. zeroing FiLM measurably moves the forecast on exactly those nodes.

If FiLM is near-identity at `e_self == 0`, or zeroing it changes nothing there,
the hypothesis is wrong and the Unseen deficit comes from somewhere else. This
probe is read-only: one validation batch, one forward, no optimiser step, no
write to any checkpoint, manifest or protocol artefact.

Run it from the active project environment (server: ReaCH, local Windows: COMG):

    python scripts/film_probe.py \
        --run outputs/ALFNet/gs-r1-s32 --config configs/ALFNet/gs.yaml
"""

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROLE_NAMES = {0: "Steady(AAS)", 1: "Join(NAS)", 2: "Unseen(VS)", 3: "Fault(FS)"}
TIMEOUT_SECONDS = 300


def _watchdog(stage):
    """Hard timeout. A hung DataLoader must report where it hung, not sit."""

    def kill():
        print(
            f"\n!! TIMEOUT after {TIMEOUT_SECONDS}s -- stuck at: {stage['at']}",
            file=sys.stderr,
            flush=True,
        )
        # os._exit: a hung worker pool will not unwind through sys.exit.
        import os

        os._exit(3)

    timer = threading.Timer(TIMEOUT_SECONDS, kill)
    timer.daemon = True
    timer.start()
    return timer


def _summarise(values):
    """mean / median / p95 of a 1-D tensor, as plain floats."""
    if values.numel() == 0:
        return None
    flat = values.flatten().tolist()
    flat_sorted = sorted(flat)
    p95 = flat_sorted[min(len(flat_sorted) - 1, int(0.95 * len(flat_sorted)))]
    return {
        "n": len(flat),
        "mean": float(statistics.fmean(flat)),
        "median": float(statistics.median(flat)),
        "p95": float(p95),
        "max": float(flat_sorted[-1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/ALFNet/gs-r1-s32")
    parser.add_argument("--config", default="configs/ALFNet/gs.yaml")
    # Must match the run's training flag: batch_size enters the canonical config
    # hash, so a different value fails provenance. It costs nothing here anyway --
    # the validation loader hard-codes 256 regardless (data_factory.py).
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--allow_legacy_checkpoint", action="store_true")
    probe_args = parser.parse_args()

    stage = {"at": "startup"}
    _watchdog(stage)
    started = time.time()

    run_dir = (ROOT / probe_args.run).resolve()
    checkpoint_path = run_dir / "checkpoints" / "best_val_checkpoint.pth"
    if not checkpoint_path.exists():
        raise SystemExit(f"no checkpoint at {checkpoint_path}")

    # args parsing reads sys.argv; hand it the run's own training flags so the
    # canonical config hash it derives is the one the checkpoint was saved with.
    stage["at"] = "initialize_args"
    sys.argv = [
        "main.py",
        # Verbatim, not resolved to an absolute path: the config path enters the
        # canonical config hash, so absolutising it fails provenance.
        "-c", probe_args.config,
        "--batch_size", str(probe_args.batch_size),
        "--output_root", "./outputs",
        "--run_id", run_dir.name,
    ]
    from args import apply_output_root, initialize_args

    args = initialize_args()
    apply_output_root(args)
    args.device = "cpu"
    # Windows spawns workers by re-importing; a probe needs none of that, and a
    # worker pool is exactly what hung the first attempt.
    args.num_workers = 0

    stage["at"] = "build model"
    from model.regist_model import load_model_class

    model = load_model_class(args.model)(args).float()
    if getattr(model, "film_head", None) is None:
        raise SystemExit(f"{args.model} has no film_head -- nothing to probe")

    stage["at"] = "load checkpoint"
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = blob.get("model_state_dict", blob.get("state_dict", blob))

    # Provenance lives under `config`, not at the top level -- reading it from
    # the top level yields None for every field and turns each check into a
    # vacuous pass.
    stored_config = blob.get("config") or {}
    provenance = {
        key: stored_config.get(key)
        for key in ("model", "model_variant", "router_mode",
                    "canonical_config_hash", "training_protocol_hash")
    }
    provenance["epoch"] = blob.get("epoch")
    live_hash = getattr(args, "canonical_config_hash", None)
    stored_hash = provenance.get("canonical_config_hash")
    if stored_hash is None or live_hash is None:
        raise SystemExit(
            "cannot verify canonical_config_hash "
            f"(checkpoint={stored_hash}, live={live_hash}) -- refusing to probe "
            "a checkpoint whose provenance cannot be established"
        )
    config_match = stored_hash == live_hash
    if not config_match and not probe_args.allow_legacy_checkpoint:
        raise SystemExit(
            "canonical_config_hash mismatch -- this checkpoint was not trained "
            f"from {probe_args.config}\n  checkpoint {stored_hash}\n  config     {live_hash}"
        )

    # The real validator, not a reimplementation: a probe that checks provenance
    # its own way can pass a checkpoint the trainer would refuse.
    import main as main_module

    main_module._validate_structural_fingerprints(blob, model)
    structure_verified = True

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected\n"
            f"  missing[:5]={list(missing)[:5]}\n  unexpected[:5]={list(unexpected)[:5]}"
        )
    model.eval()

    stage["at"] = "build validation loader"
    from data_provider.data_factory import data_provider

    val_loader = data_provider(args, "val")

    stage["at"] = "fetch one validation batch"
    batch = next(iter(val_loader))
    if len(batch) != 10:
        raise SystemExit(f"expected a 10-tensor lcbt_v3 batch, got {len(batch)}")
    (x_enc, x_dec, enc_t_mark, dec_t_mark, _tw, _phase, _group,
     _cohort, _cal, _evid) = [t.to("cpu") for t in batch]
    x_enc = torch.nan_to_num(x_enc.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x_dec = x_dec.float()
    enc_t_mark = enc_t_mark.float()
    dec_t_mark = dec_t_mark.float()

    adj, pos_mark = val_loader.dataset.get_adj_and_pos_mark()
    adj = torch.as_tensor(adj, dtype=torch.float32)
    pos_mark = (torch.as_tensor(pos_mark, dtype=torch.float32)
                if (args.pos_mark and pos_mark is not None) else None)
    if not args.t_mark:
        enc_t_mark = dec_t_mark = None

    # Validation path: x_dec is withheld from the model exactly as in shared_step.
    forward_args = (adj, x_enc, enc_t_mark, pos_mark, None, dec_t_mark, 0)

    # Same quantities the model itself computes, so the buckets are the model's
    # own notion of evidence rather than a re-derivation.
    stage["at"] = "compute evidence buckets"
    from layers.evidence_router import self_evidence

    mask_index = int(model.mask_index)
    observation_mask = x_enc[..., mask_index:mask_index + 1]
    e_self = self_evidence(observation_mask).squeeze(-1)          # [B, N]
    role_start = int(args.raw_enc_in)
    last_role = x_enc[:, -1, :, role_start:role_start + 4].argmax(dim=-1)  # [B, N]

    captured = {}

    def hook(_module, _inputs, output):
        gamma, beta = output.detach().chunk(2, dim=-1)
        captured["gamma"] = gamma
        captured["beta"] = beta

    handle = model.film_head.register_forward_hook(hook)

    stage["at"] = "forward (FiLM active)"
    from utils.model_output import unpack_model_output

    with torch.no_grad():
        y_film, _meta = unpack_model_output(model(*forward_args))
    handle.remove()

    gamma, beta = captured["gamma"], captured["beta"]
    # [B, N, H] -> per-node magnitude over the hidden dimension
    gamma_node = gamma.abs().reshape(*e_self.shape, -1).mean(dim=-1)
    beta_node = beta.abs().reshape(*e_self.shape, -1).mean(dim=-1)

    stage["at"] = "forward (FiLM zeroed)"
    # Zero the head's output layer so the FiLM factor is exactly (1, 0): the
    # identity it is initialised at. Restored immediately; nothing is written.
    out_layer = model.film_head[-1]
    saved_w = out_layer.weight.detach().clone()
    saved_b = out_layer.bias.detach().clone()
    encoder_states = {}
    try:
        with torch.no_grad():
            out_layer.weight.zero_()
            out_layer.bias.zero_()
            y_off, _meta_off = unpack_model_output(model(*forward_args))
            encoder_states["off"] = model._last_encoder_state
    finally:
        with torch.no_grad():
            out_layer.weight.copy_(saved_w)
            out_layer.bias.copy_(saved_b)

    with torch.no_grad():
        _ = unpack_model_output(model(*forward_args))
        encoder_states["on"] = model._last_encoder_state

    stage["at"] = "aggregate"
    # y is [B, T, N, C]; collapse everything but the node axis.
    delta_pred = (y_film - y_off).abs().mean(dim=(1, 3))          # [B, N]
    # A raw delta would be larger on Unseen nodes if their forecasts are simply
    # larger in magnitude, which would explain the pattern without FiLM being
    # the cause. Normalising by the FiLM-off forecast scale removes that.
    pred_scale = y_off.abs().mean(dim=(1, 3)).clamp_min(1e-6)
    delta_pred_rel = delta_pred / pred_scale
    delta_state = None
    if encoder_states["on"] is not None and encoder_states["off"] is not None:
        on, off = encoder_states["on"], encoder_states["off"]
        if on.shape == off.shape:
            flat_on = on.reshape(e_self.shape[0], e_self.shape[1], -1)
            flat_off = off.reshape(e_self.shape[0], e_self.shape[1], -1)
            delta_state = (flat_on - flat_off).abs().mean(dim=-1)  # [B, N]

    buckets = {
        "e_self == 0 (no own history)": e_self == 0,
        "0 < e_self <= 0.25 (low evidence)": (e_self > 0) & (e_self <= 0.25),
        "0.25 < e_self < 0.95 (partial)": (e_self > 0.25) & (e_self < 0.95),
        "e_self >= 0.95 (full history)": e_self >= 0.95,
    }
    for role_id, name in ROLE_NAMES.items():
        buckets[f"role {name}"] = last_role == role_id

    report = {
        "run": str(run_dir),
        "config": probe_args.config,
        "checkpoint": str(checkpoint_path),
        "provenance": provenance,
        "canonical_config_hash_live": live_hash,
        "canonical_config_hash_match": config_match,
        "structural_provenance_verified": structure_verified,
        # The val loader hard-codes 256 (data_provider/data_factory.py); the
        # --batch_size flag only reaches the train loader.
        "batch_shape": list(x_enc.shape),
        "film_head_output_layer": {
            "weight_absmax": float(saved_w.abs().max()),
            "weight_rms": float(saved_w.pow(2).mean().sqrt()),
            "bias_absmax": float(saved_b.abs().max()),
        },
        "buckets": {},
    }
    for name, selector in buckets.items():
        entry = {
            "abs_gamma": _summarise(gamma_node[selector]),
            "abs_beta": _summarise(beta_node[selector]),
            "abs_pred_delta_film_on_minus_off": _summarise(delta_pred[selector]),
            "rel_pred_delta_film_on_minus_off": _summarise(delta_pred_rel[selector]),
        }
        if delta_state is not None:
            entry["abs_encoder_state_delta"] = _summarise(delta_state[selector])
        report["buckets"][name] = entry
    report["elapsed_seconds"] = round(time.time() - started, 2)

    stage["at"] = "write report"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "film_probe.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "FiLM conditioning probe -- is the ungated FiLM path active on no-history nodes?",
        f"run        : {run_dir}",
        f"checkpoint : {checkpoint_path.name}",
        f"provenance : model={provenance.get('model')} variant={provenance.get('model_variant')} "
        f"router={provenance.get('router_mode')} epoch={provenance.get('epoch')}",
        f"             canonical_config_hash match = {config_match}",
        f"             training_protocol_hash      = {provenance.get('training_protocol_hash')}",
        f"             structural provenance       = "
        f"{'verified' if structure_verified else 'UNVERIFIED'}",
        f"batch      : {list(x_enc.shape)} (val loader hard-codes 256)   "
        f"elapsed {report['elapsed_seconds']}s",
        "",
        f"{'bucket':<36}{'n':>7}{'|gamma|':>9}{'|beta|':>9}{'d_state':>9}"
        f"{'d_pred':>9}{'d_pred/y':>10}",
        "-" * 89,
    ]
    for name, entry in report["buckets"].items():
        g, b = entry["abs_gamma"], entry["abs_beta"]
        d = entry.get("abs_encoder_state_delta")
        p, r = (entry["abs_pred_delta_film_on_minus_off"],
                entry["rel_pred_delta_film_on_minus_off"])
        if g is None:
            lines.append(f"{name:<36}{0:>7}{'-':>9}{'-':>9}{'-':>9}{'-':>9}{'-':>10}")
            continue
        lines.append(
            f"{name:<36}{g['n']:>7}{g['mean']:>9.5f}{b['mean']:>9.5f}"
            f"{(d['mean'] if d else float('nan')):>9.5f}{p['mean']:>9.5f}{r['mean']:>10.5f}"
        )
    lines += [
        "",
        "mean shown; median/p95/max are in film_probe.json.",
        "d_state / d_pred = |FiLM on - FiLM zeroed| on the encoder state and the forecast.",
        "d_pred/y normalises by the FiLM-off forecast magnitude, so a larger raw",
        "delta on nodes that simply forecast larger values cannot masquerade as an effect.",
        "Read it against the hypothesis: near-zero |gamma|,|beta| and d_pred on the",
        "'e_self == 0' row would refute the ungated-FiLM explanation of the Unseen deficit.",
    ]
    text = "\n".join(lines)
    (reports_dir / "film_probe.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

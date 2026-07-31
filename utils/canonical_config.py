"""An immutable identity for a run's configuration.

Why not `vars(args)`
--------------------
`args` is a live namespace: the trainer writes `best_val_metric`,
`validation_group_history`, `exp_id`, resolved output paths and per-epoch
bookkeeping onto it. Hashing it after training therefore hashes the *run*, not
the *configuration*, and two identical experiments produce different hashes
while a genuinely edited YAML can produce the same one if the edit lands before
the fields are read.

So the hash is frozen the moment parsing finishes, from an explicit allowlist of
what defines the experiment:

  * every parsed scalar argument that is not a runtime or path field;
  * the nested `model_family_cfg`, resolved -- including the expert YAMLs it
    points at, which is where `moge_router_mode` and the graph capacity live;
  * the protocol versions;
  * the data, adjacency and coordinate fingerprints.

Excluded, deliberately: `output_root`, `run_id`, `device`, `num_workers`,
manifest roots, `exp_id`, wandb settings and anything the trainer writes back.
Relocating a finished run on disk, or re-running it on another GPU, must not
change its identity.
"""

import hashlib
import json
import os

import yaml


# Fields that describe *where* or *how* a run executes, never *what* it is.
RUNTIME_FIELDS = frozenset({
    "output_root", "run_id", "device", "num_workers", "exp_id",
    "best_checkpoint_path", "best_prediction_path", "best_fig_path",
    "best_metrics_path", "log_dir", "run_output_dir", "profile_dir",
    "lcbt_manifest_root", "ples_manifest_root", "paths_are_run_scoped",
    "wandb", "wandb_project", "wandb_entity", "verbose", "mode",
    "pretrained_model_path", "train_init_checkpoint_path",
    "lcbt_overwrite_manifest", "ples_overwrite_manifest",
    "num_params", "loaded_checkpoint_metadata", "allow_legacy_checkpoint",
    # Which *mode* a configuration is being executed in. `train` and
    # `test_async` are two runs of the same experiment, and the evaluation pass
    # has to be able to load the checkpoint the training pass produced -- so a
    # flag that flips between them cannot be part of the configuration's
    # identity. Everything that defines the model, the data and the protocol is
    # unaffected and still enters the hash.
    "test_for_async_lifecycle", "train_for_async_lifecycle",
})

# Anything the trainer writes back onto `args` while running. Listed by prefix
# because the set grows; a stray runtime field silently changing the hash is the
# failure this guards against.
RUNTIME_PREFIXES = ("best_val", "validation_", "training_protocol_hash",
                    "lce_protocol_hash", "ples_protocol_hash", "last_routing",
                    "best_routing", "canonical_config", "_")


def _is_runtime(key):
    return key in RUNTIME_FIELDS or key.startswith(RUNTIME_PREFIXES)


def _jsonable(value):
    """A stable textual form for anything argparse or YAML can produce."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_nested_configs(family_cfg, root="."):
    """Inline every expert YAML the family config points at.

    `family.experts.graph.config_path` is where the router mode and the graph
    capacity actually live, so a hash that only saw the top-level file would
    call `fg1`, `gs` and `g1` the same configuration.
    """
    resolved = {}
    experts = (family_cfg or {}).get("experts") or {}
    for name, spec in sorted(experts.items()):
        path = (spec or {}).get("config_path")
        if not path:
            continue
        candidate = os.path.join(root, str(path).lstrip("./")) if not os.path.isabs(path) else path
        if not os.path.exists(candidate):
            resolved[name] = {"config_path": str(path), "resolved": "missing"}
            continue
        with open(candidate, "r", encoding="utf-8") as handle:
            resolved[name] = {
                "config_path": str(path),
                "resolved": _jsonable(yaml.safe_load(handle)),
            }
    return resolved


def build_canonical_config(args, data_fingerprints=None, root="."):
    """The immutable dict that defines this experiment."""
    scalars = {
        key: _jsonable(value)
        for key, value in vars(args).items()
        if not _is_runtime(key) and key != "model_family_cfg"
    }
    return {
        "schema": 1,
        "scalars": dict(sorted(scalars.items())),
        "model_family_cfg": _jsonable(getattr(args, "model_family_cfg", {}) or {}),
        "nested_expert_configs": _resolve_nested_configs(
            getattr(args, "model_family_cfg", {}) or {}, root=root
        ),
        "protocol_versions": {
            "lcbt": int(getattr(args, "lcbt_protocol_version", 0) or 0),
            "lce": int(getattr(args, "lce_protocol_version", 0) or 0),
            "ples": int(getattr(args, "ples_protocol_version", 0) or 0),
        },
        "data_fingerprints": _jsonable(data_fingerprints or {}),
    }


def canonical_config_hash(canonical):
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_canonical_config(args, data_fingerprints=None, root="."):
    """Compute and attach the canonical config and its hash, once.

    Called immediately after parsing. Attaching it to `args` is safe because
    both fields are excluded from the hash by `_is_runtime`.
    """
    canonical = build_canonical_config(args, data_fingerprints, root=root)
    args.canonical_config = canonical
    args.canonical_config_hash = canonical_config_hash(canonical)
    return args.canonical_config_hash

"""The one in-memory description of what a model is allowed to see.

Every model runs on the full static node axis N and sees the full topology --
that is the topology-known setting this work assumes. What differs by split is
the *dynamic signal*: training may read measurements from the training-known
nodes K only, validation from K union V, test from K union T. Nodes outside the
current signal scope still exist, still participate in propagation and still
carry structure; they simply contribute no measurement.

Two things this file is deliberately not:

* It is not a file. Nothing here is written next to the payload. A node mask on
  disk beside `raw_data.pkl` is a third model input waiting to be read, and the
  run directory is asserted to hold exactly two files.
* It is not a place to decide the split. The V/T partition is derived here so
  the plumbing can be built and tested, and it is marked provisional. Freezing
  it is Stage 2-B, and `training_ready` stays false until then.

Masking uses `torch.where`, never `value * mask`. An unavailable measurement may
be NaN or Inf in the source; multiplying it by zero yields NaN, which then
propagates through every downstream reduction and silently poisons a batch that
was supposed to be blind to that node.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lib.graph_native_layout import LAYOUT_VERSION, resolve_layout
from lib.v4_payload import load_v4_payload

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "configs/paper/v4_data_manifests"
PHASES = ("train", "validation", "test")


class ContextRefused(ValueError):
    """The requested context cannot be built, or does not match the model."""


def _sha256_array(array):
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_unseen_pool(unseen, seed=32):
    """Partition the master unseen pool into disjoint validation and test sets.

    Provisional. Stage 2-B freezes this; until then it exists so the exposure
    plumbing can be built and tested against a concrete, reproducible split.
    Deterministic given the pool and the seed, and the two halves never overlap
    -- a node that appeared in both would make the validation metric a partial
    read of the test split.
    """
    pool = np.asarray(sorted(set(int(node) for node in unseen)), dtype=np.int64)
    order = np.random.default_rng(seed).permutation(pool.size)
    cut = pool.size // 2
    validation = np.sort(pool[order[:cut]])
    test = np.sort(pool[order[cut:]])
    assert not set(validation.tolist()) & set(test.tolist())
    return validation, test


@dataclass(frozen=True)
class RuntimeDataContext:
    """What one (dataset, phase) pair exposes. Built once, shared by everything."""

    dataset: str
    phase: str
    target: str
    target_channel: int
    payload_sha256: str
    node_ids: np.ndarray
    node_ids_sha256: str
    adjacency: torch.Tensor
    adjacency_sha256: str
    positions: torch.Tensor | None
    layout_parts: tuple
    layout_sha256: str
    layout_algorithm: str
    known_nodes: np.ndarray
    validation_unseen: np.ndarray
    test_unseen: np.ndarray
    signal_scope: np.ndarray
    readiness: dict
    provisional_split: bool = True
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def num_nodes(self):
        return int(self.node_ids.size)

    # --- identity -------------------------------------------------------------
    #
    # Two identities, deliberately separate. A model's static graph assets do not
    # change between train, validation and test, so binding a model to an
    # identity that carries `phase` would make one architecture look like three
    # different models -- and the only ways out are to reuse a model whose
    # identity check no longer means anything, or to loosen the check until it
    # stops catching a swapped graph. Exposure is what varies by phase, and it is
    # checked per batch, not per constructor.

    STATIC_FIELDS = ("dataset", "target", "target_channel", "payload_sha256",
                     "node_ids_sha256", "adjacency_sha256", "positions_policy",
                     "positions_sha256", "layout_algorithm", "layout_version",
                     "layout_sha256", "full_num_nodes")

    def static_asset_identity(self):
        """Everything a constructor may bind to. No phase, no split, no scope."""
        identity = {
            "dataset": self.dataset,
            "target": self.target,
            "target_channel": int(self.target_channel),
            "payload_sha256": self.payload_sha256,
            "node_ids_sha256": self.node_ids_sha256,
            "adjacency_sha256": self.adjacency_sha256,
            "positions_policy": ("verified_wgs84" if self.positions is not None
                                 else "graph_native_no_coordinates"),
            "positions_sha256": (None if self.positions is None
                                 else _sha256_array(self.positions.numpy())),
            "layout_algorithm": self.layout_algorithm,
            "layout_version": int(LAYOUT_VERSION),
            "layout_sha256": self.layout_sha256,
            "full_num_nodes": self.num_nodes,
        }
        missing = [f for f in self.STATIC_FIELDS if f not in identity]
        if missing:
            raise ContextRefused(f"static identity is missing {missing}")
        for field in self.STATIC_FIELDS:
            if identity[field] is None and field != "positions_sha256":
                raise ContextRefused(f"static identity field {field} is None")
        return identity

    def static_asset_hash(self):
        return hashlib.sha256(
            json.dumps(self.static_asset_identity(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def exposure_identity(self):
        """What varies by phase. Checked per batch, never bound to a constructor."""
        return {
            "static_asset_hash": self.static_asset_hash(),
            "phase": self.phase,
            "known_nodes_sha256": _sha256_array(self.known_nodes),
            "validation_unseen_sha256": _sha256_array(self.validation_unseen),
            "test_unseen_sha256": _sha256_array(self.test_unseen),
            "signal_scope_sha256": _sha256_array(self.signal_scope),
            "provisional_split": bool(self.provisional_split),
        }

    def exposure_hash(self):
        return hashlib.sha256(
            json.dumps(self.exposure_identity(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def identity(self):
        """Deprecated alias for the exposure identity.

        Kept so existing callers keep working. Model construction must use
        `static_asset_identity`; binding a constructor to this would pin the
        model to one phase.
        """
        return self.exposure_identity()

    @staticmethod
    def _refuse_differences(mine, theirs, who, what):
        differing = {key: (value, dict(theirs).get(key))
                     for key, value in mine.items()
                     if dict(theirs).get(key) != value}
        if differing:
            raise ContextRefused(
                f"{who} does not share this run's {what}: " + ", ".join(
                    f"{key} expected {expected!r} got {found!r}"
                    for key, (expected, found) in sorted(differing.items())))

    def assert_static_asset_matches(self, other, who):
        """Refuse a module that resolved a different graph, order or layout."""
        self._refuse_differences(self.static_asset_identity(), other, who,
                                 "static graph assets")

    def assert_exposure_matches(self, other, who):
        self._refuse_differences(self.exposure_identity(), other, who,
                                 "exposure permissions")

    def assert_matches(self, other, who):
        """Deprecated alias; compares exposure."""
        self.assert_exposure_matches(other, who)

    # --- node scopes ----------------------------------------------------------

    def signal_scope_mask(self):
        """(N,) bool: whose measurements this phase may read."""
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[torch.from_numpy(self.signal_scope)] = True
        return mask

    def loss_scope_mask(self):
        """(N,) bool: whose targets this phase may score.

        Training scores K only. Validation and test score their own held-out
        nodes as well -- that is the measurement the experiment exists to make.
        """
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        if self.phase == "train":
            mask[torch.from_numpy(self.known_nodes)] = True
        else:
            mask[torch.from_numpy(self.signal_scope)] = True
        return mask

    # --- the five explicit fields --------------------------------------------

    def visible_signal(self, raw, observed):
        """The only raw signal a model may receive, plus its availability.

        `raw` is (B, T, N, C) and `observed` is (B, T, N) natural availability.
        Nodes outside the signal scope are replaced, not multiplied: an
        out-of-scope value may be NaN or Inf, and `NaN * 0` is NaN.
        """
        if raw.shape[2] != self.num_nodes:
            raise ContextRefused(
                f"raw signal has {raw.shape[2]} nodes, context has {self.num_nodes}; "
                "the model node axis must be the full static network")
        scope = self.signal_scope_mask().to(raw.device)
        available = observed.to(torch.bool) & scope.view(1, 1, -1)
        visible = torch.where(available.unsqueeze(-1),
                              raw, torch.zeros((), dtype=raw.dtype, device=raw.device))
        if not torch.isfinite(visible).all():
            raise ContextRefused(
                "visible signal is not finite after masking: an in-scope, "
                "available measurement is NaN or Inf")
        return visible, available

    def decoder_available_mask(self, observed_future):
        """Which future targets may be teacher-forced into a decoder.

        Only in-scope nodes. Feeding a held-out node's future value to the
        decoder hands the model the answer for a node it is being tested on,
        and scheduled sampling would do it silently.
        """
        scope = self.signal_scope_mask().to(observed_future.device)
        return observed_future.to(torch.bool) & scope.view(1, 1, -1)

    def target_available_mask(self, observed_future):
        """Natural availability of the target, before any scope is applied."""
        return observed_future.to(torch.bool)

    def loss_mask(self, observed_future):
        """Where the objective may be evaluated: available and in loss scope."""
        scope = self.loss_scope_mask().to(observed_future.device)
        return observed_future.to(torch.bool) & scope.view(1, 1, -1)

    def masked_decoder_input(self, future_target, observed_future):
        """Teacher-forcing input with forbidden futures replaced by zero."""
        mask = self.decoder_available_mask(observed_future)
        return torch.where(mask.unsqueeze(-1), future_target,
                           torch.zeros((), dtype=future_target.dtype,
                                       device=future_target.device))


def build_runtime_context(dataset, phase, seed=32, manifest_dir=None):
    """Assemble the context from the canonical manifest and the v4 payload."""
    if phase not in PHASES:
        raise ContextRefused(f"unknown phase {phase!r}, expected one of {PHASES}")
    directory = Path(manifest_dir) if manifest_dir else MANIFEST_DIR
    manifest_path = directory / f"{dataset}_run_manifest.json"
    if not manifest_path.exists():
        raise ContextRefused(
            f"no canonical run manifest for {dataset}; v4 fails closed rather "
            "than falling back to a v3 payload")
    manifest = json.loads(manifest_path.read_text("utf-8"))

    payload_path = ROOT / manifest["payload"]["path"]
    actual = _sha256_file(payload_path)
    if actual != manifest["payload"]["sha256"]:
        raise ContextRefused(
            f"{dataset} payload hash {actual} does not match the manifest "
            f"{manifest['payload']['sha256']}")
    payload = load_v4_payload(payload_path)

    adjacency = torch.from_numpy(np.ascontiguousarray(payload["adj"]))
    adjacency_sha = _sha256_array(payload["adj"])
    if adjacency_sha != manifest["propagation_graph"]["sha256"]:
        raise ContextRefused(f"{dataset} adjacency does not match the manifest")

    num_nodes = int(adjacency.shape[0])
    node_ids = np.arange(num_nodes, dtype=np.int64)
    settings = manifest["layout"]
    positions_array = payload["pos_stamp"]
    layout = resolve_layout(payload["adj"], positions_array, int(settings["recur_times"]))
    layout_sha = layout["membership_sha256"]
    if layout_sha != settings["membership_sha256"]:
        raise ContextRefused(f"{dataset} layout does not match the manifest")
    if layout["algorithm"] != settings["algorithm"]:
        raise ContextRefused(
            f"{dataset} layout algorithm {layout['algorithm']} does not match the "
            f"manifest {settings['algorithm']}")
    parts = layout["parts"]

    unseen = np.load(ROOT / manifest["unseen"]["path"])
    validation_unseen, test_unseen = split_unseen_pool(unseen, seed=seed)
    known = np.setdiff1d(node_ids, np.asarray(unseen, dtype=np.int64))
    scope = {"train": known,
             "validation": np.union1d(known, validation_unseen),
             "test": np.union1d(known, test_unseen)}[phase]

    positions = (None if payload["pos_stamp"] is None
                 else torch.from_numpy(np.ascontiguousarray(payload["pos_stamp"])))
    return RuntimeDataContext(
        dataset=dataset, phase=phase,
        target=manifest["target"]["name"],
        target_channel=int(manifest["target"]["channel"]),
        payload_sha256=manifest["payload"]["sha256"],
        node_ids=node_ids, node_ids_sha256=_sha256_array(node_ids),
        adjacency=adjacency, adjacency_sha256=adjacency_sha,
        positions=positions,
        layout_parts=tuple(tuple(int(n) for n in part) for part in parts),
        layout_algorithm=layout["algorithm"],
        layout_sha256=layout_sha,
        known_nodes=known, validation_unseen=validation_unseen,
        test_unseen=test_unseen, signal_scope=np.asarray(scope, dtype=np.int64),
        readiness=dict(manifest["readiness"]),
    )

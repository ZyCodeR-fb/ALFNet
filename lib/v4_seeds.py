"""One reproducible random-seed contract for the active v4 experiment.

Users choose one ``run_seed`` (32, 42 or 52).  Every stochastic component is
then assigned a named sub-stream.  The node split deliberately keeps the
already-published direct mapping ``node_split_seed == run_seed``: changing it
after test exposure would redraw held-out node identities.  All training-time
streams are independently hashed, so an added random module cannot shift data
order, initialisation or teacher forcing.

The fixed validation-design seed is deliberately not part of this bundle.  It
defines the evaluation instrument and stays identical across runs and epochs;
it is not training randomness.
"""

import hashlib
import random

import numpy as np
import torch


SEED_POLICY = "lcbt_v4_named_subseeds_v1_frozen_direct_node_split"
NAMESPACES = (
    "node_split",
    "model_init",
    "train_schedule",
    "shuffle",
    "teacher_forcing",
    "dataloader_worker",
    "training_runtime",
)


def derive_seed(run_seed, namespace, index=None, bits=32):
    """Stable non-negative seed derived without Python's salted ``hash``."""
    if namespace not in NAMESPACES:
        raise ValueError(f"unknown v4 seed namespace {namespace!r}")
    if bits not in (32, 63):
        raise ValueError("v4 derived seeds support 32 or 63 bits")
    if namespace == "node_split":
        if index is not None:
            raise ValueError("the frozen node split has no indexed sub-stream")
        return int(run_seed) % (2 ** bits)
    suffix = "" if index is None else f"|{int(index)}"
    key = f"{SEED_POLICY}|{int(run_seed)}|{namespace}{suffix}"
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return value % (2 ** bits)


def seed_bundle(run_seed):
    """JSON-safe provenance for all stochastic training components."""
    return {
        "policy": SEED_POLICY,
        "run_seed": int(run_seed),
        "derived": {name: derive_seed(run_seed, name) for name in NAMESPACES},
        "evaluation_design_seed": 12026,
    }


def seed_process(seed):
    """Seed process-global RNGs used by initialisation/dropout/runtime code."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class EpochWorkerSeeder:
    """Resume-stable DataLoader worker seeds keyed by run, epoch and worker."""

    def __init__(self, run_seed, phase="train"):
        self.run_seed = int(run_seed)
        if phase not in ("train", "validation"):
            raise ValueError("worker seed phase must be train or validation")
        self.phase = phase
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __call__(self, worker_id):
        phase_offset = 0 if self.phase == "train" else 1_000_000_000_000
        base = derive_seed(
            self.run_seed,
            "dataloader_worker",
            index=phase_offset + self.epoch * 1_000_003 + int(worker_id),
            bits=32,
        )
        random.seed(base)
        np.random.seed(base)
        torch.manual_seed(base)

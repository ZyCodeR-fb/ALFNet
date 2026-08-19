"""The v4 full-N lifecycle dataset.

A node's existence and its measurements are different things. Every v4 batch
runs on the full static node axis N with the full topology, in all three phases;
what changes is which nodes' *dynamic signal* the phase may read -- K for
training, K union V for validation, K union T for test. Nodes outside that scope
are still present, still propagate, and simply contribute no measurement.

Deliberately a separate class rather than a mode flag inside the v3 dataset. The
v3 path returns a ten-tensor tuple that a dozen call sites unpack positionally,
and threading a second node axis through it would put the two protocols one
`if` apart everywhere. v4 returns a named dict; nothing else reads it yet.

Nothing here guesses. The dataset is constructed from a RuntimeDataContext or
not at all -- there is no sniffing of shapes or filenames, and no fallback to v3
when a v4 build fails, because a silent downgrade is how an experiment ends up
reporting the wrong protocol.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from data_provider.calendar_strata import calendar_stratum
from data_provider.lcbt_protocol import build_lcbt_schedule
from lib.runtime_context import ContextRefused
from lib.v4_payload import load_v4_payload

ROLE_UNSEEN, ROLE_STEADY, ROLE_JOIN, ROLE_FAULT = 0, 1, 2, 3


class V4LifecycleDataset(Dataset):
    """Full-N windows with sample-level exposure masks."""

    def __init__(self, context, payload_path, input_len=12, pred_len=12,
                 seed=32, epoch=0, split_fractions=(0.6, 0.2, 0.2), scaler=None):
        if context is None:
            raise ContextRefused(
                "v4 requires a RuntimeDataContext; it is not inferred from the "
                "payload and never falls back to the v3 loader")
        self.context = context
        self.input_len, self.pred_len = int(input_len), int(pred_len)
        self.num_nodes = context.num_nodes
        payload = load_v4_payload(payload_path)

        raw = np.asarray(payload["data"], dtype=np.float32)
        channel = context.target_channel
        signal = raw[:, :, channel].astype(np.float64)
        # Existing project semantics: a non-positive reading is not a
        # measurement of zero flow, it is an absent measurement.
        natural = np.isfinite(signal) & (signal > 0)
        signal = np.where(natural, signal, np.nan)

        steps = signal.shape[0]
        train_end = int(steps * split_fractions[0])
        validation_end = train_end + int(steps * split_fractions[1])
        self.boundaries = {"train": (0, train_end),
                           "validation": (train_end, validation_end),
                           "test": (validation_end, steps)}

        # 3. The scaler sees training time and training-known nodes only. V and T
        #    values never enter a mean, a std, or any other reduction.
        self.scaler = scaler if scaler is not None else self._fit_scaler(
            signal, natural, train_end, context.known_nodes)
        # 4. Normalise the whole array; masking happens afterwards, in normalised
        #    space, so an unavailable slot becomes 0 == the training mean rather
        #    than the normalised image of a physical zero.
        self.normalised = ((signal - self.scaler["mean"]) / self.scaler["std"]).astype(np.float32)
        self.natural = natural

        stamps = np.asarray(payload["time_stamp"])
        self.stamps = stamps
        self._build_origins()
        self._build_schedule(seed, epoch, stamps)

    # --- construction ---------------------------------------------------------

    @staticmethod
    def _fit_scaler(signal, natural, train_end, known_nodes):
        window = signal[:train_end][:, known_nodes]
        valid = window[natural[:train_end][:, known_nodes]]
        if valid.size == 0:
            raise ContextRefused("no valid training-known values to fit the scaler")
        std = float(valid.std())
        return {"mean": float(valid.mean()), "std": std if std > 1e-6 else 1.0,
                "fitted_on": "train window, training-known nodes only",
                "n_values": int(valid.size)}

    def _build_origins(self):
        """Window starts whose guard reads only in-scope nodes."""
        start, end = self.boundaries[self.context.phase]
        span = self.input_len + self.pred_len
        scope = self.context.signal_scope
        origins = []
        for origin in range(start, end - span + 1):
            window = self.natural[origin:origin + span][:, scope]
            # A window is usable if the in-scope nodes carry any measurement at
            # all. Out-of-scope columns are not consulted, so their values and
            # their missingness cannot move which origins exist.
            if window.any():
                origins.append(origin)
        self.origins = np.asarray(origins, dtype=np.int64)

    def _build_schedule(self, seed, epoch, stamps):
        regimes = [calendar_stratum(stamps[origin + self.input_len - 1])
                   for origin in self.origins]
        adjacency = self.context.adjacency.numpy()
        known = self.context.known_nodes
        # Simulated lifecycle cohorts are drawn from K in every phase: a held-out
        # node is already unseen and cannot also be a training join event.
        self.schedule = build_lcbt_schedule(
            num_samples=len(self.origins), input_len=self.input_len,
            adjacency=adjacency, calendar_regimes=regimes, seed=seed, epoch=epoch,
            scenario_node_pools={scenario: known for scenario in
                                 ("zero_history", "steady", "join", "fault")},
            topology_adjacency=adjacency, node_ids=np.arange(self.num_nodes),
            cohort_size_base=self.num_nodes)

    def __len__(self):
        return int(self.origins.size)

    # --- the five masks -------------------------------------------------------

    def _availability(self, entry, natural_input):
        """(T, N) input availability, and the (N,) per-node scope it came from.

        Three independent reasons a measurement may be absent, combined and
        never conflated: the sensor recorded nothing, the phase may not read
        that node, or the lifecycle episode hides it for this sample.
        """
        scope = np.zeros(self.num_nodes, dtype=bool)
        scope[self.context.signal_scope] = True

        lifecycle = np.ones((self.input_len, self.num_nodes), dtype=bool)
        hidden = np.asarray(entry["hidden_nodes"], dtype=np.int64)
        changed = np.asarray(entry["changed_nodes"], dtype=np.int64)
        evidence = int(entry["evidence_steps"])
        if hidden.size:
            lifecycle[:, hidden] = False
        if changed.size:
            if entry["scenario"] == "join":
                # A joining node has no history and then the last k steps of it.
                lifecycle[:, changed] = False
                if evidence > 0:
                    lifecycle[-evidence:, changed] = True
            elif entry["scenario"] == "fault":
                # A failing node is observed until the last k steps go dark.
                lifecycle[:, changed] = True
                if evidence > 0:
                    lifecycle[-evidence:, changed] = False

        available = natural_input & scope[None, :] & lifecycle
        return available, scope, lifecycle

    def __getitem__(self, index):
        origin = int(self.origins[index])
        entry = self.schedule[index]
        end = origin + self.input_len
        stop = end + self.pred_len

        natural_input = self.natural[origin:end]
        natural_target = self.natural[end:stop]
        available, scope, lifecycle = self._availability(entry, natural_input)

        # 6. Select in normalised space. `value * mask` would turn a NaN into a
        #    NaN, not into a zero.
        normalised = torch.from_numpy(self.normalised[origin:end].copy())
        base_visible = torch.where(torch.from_numpy(available),
                                   normalised, torch.zeros((), dtype=torch.float32))
        if not torch.isfinite(base_visible).all():
            raise ContextRefused("base_visible_input is not finite after masking")

        target = torch.from_numpy(
            np.nan_to_num(self.normalised[end:stop], nan=0.0).copy()).unsqueeze(-1)
        # Scoped, not raw natural availability. An unscoped version leaked: a
        # forbidden node's *missingness* is itself a measurement, and poisoning
        # V/T raw values moved this mask while every other field held.
        target_available = torch.from_numpy(natural_target & scope[None, :])

        # Sample-level, not phase-level: a node the episode still hides at the
        # last observed step has no present state to carry into the horizon, so
        # its future value must not be teacher-forced.
        last_observed = torch.from_numpy(available[-1].copy())
        decoder_available = (last_observed[None, :].expand(self.pred_len, -1)
                             & target_available)

        cohort = torch.zeros(self.num_nodes, dtype=torch.float32)
        cohort[np.asarray(entry["changed_nodes"], dtype=np.int64)] = 1.0
        if entry["scenario"] in ("zero_history", "steady"):
            cohort[np.asarray(entry["hidden_nodes"], dtype=np.int64)] = 1.0
            if entry["scenario"] == "steady":
                cohort = torch.from_numpy(scope.astype(np.float32))
        loss_scope = self.context.loss_scope_mask()
        loss_mask = target_available & (cohort > 0.5)[None, :] & loss_scope[None, :]

        roles, ages = self._lifecycle_state(entry, available, scope)
        return {
            "base_visible_input": base_visible.unsqueeze(-1),
            "input_available_mask": torch.from_numpy(available),
            "decoder_available_mask": decoder_available,
            "target_available_mask": target_available,
            "target_cohort_mask": cohort,
            "loss_mask": loss_mask,
            "target": target,
            "lifecycle_role_ids": roles,
            "lifecycle_state_age": ages,
            # ALFNet's method inputs, kept out of base_visible_input so every
            # baseline receives the identical common tensor. The next stage's
            # adapter concatenates these for ALFNet only.
            "lifecycle_context": torch.cat([
                torch.nn.functional.one_hot(roles, 4).float(),
                ages.unsqueeze(-1),
                torch.from_numpy(available[-1].astype(np.float32)).unsqueeze(-1),
            ], dim=-1),
            "origin": origin,
            "phase_id": entry["phase_id"],
            "scenario": entry["scenario"],
            "evidence_steps": int(entry["evidence_steps"]),
            "calendar_regime": entry["calendar_regime"],
            "context_identity": self.context.identity(),
        }

    def _lifecycle_state(self, entry, available, scope):
        """Roles and age, from present and past availability only."""
        roles = np.full(self.num_nodes, ROLE_STEADY, dtype=np.int64)
        roles[~scope] = ROLE_UNSEEN
        never = ~available.any(axis=0)
        roles[never & scope] = ROLE_UNSEEN
        changed = np.asarray(entry["changed_nodes"], dtype=np.int64)
        if changed.size:
            roles[changed] = ROLE_JOIN if entry["scenario"] == "join" else (
                ROLE_FAULT if entry["scenario"] == "fault" else ROLE_UNSEEN)
        age = available.sum(axis=0).astype(np.float32) / float(self.input_len)
        return torch.from_numpy(roles), torch.from_numpy(age)

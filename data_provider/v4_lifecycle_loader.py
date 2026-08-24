"""The v4 full-N lifecycle dataset.

A node's existence and its measurements are different things. Every v4 batch
runs on the full static node axis N with the full topology, in all three phases;
what changes is which nodes' *dynamic signal* the phase may read -- K for
training, K union V for validation, K union T for test. Nodes outside that scope
are still present, still propagate, and simply contribute no measurement.

The loader returns a named dict; positional legacy batches are unsupported.

Nothing here guesses. The dataset is constructed from a RuntimeDataContext or
not at all -- there is no sniffing of shapes or filenames and no fallback
when a v4 build fails, because a silent downgrade is how an experiment ends up
reporting the wrong protocol.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from data_provider.calendar_strata import calendar_stratum
from data_provider.v4_schedule import build_lcbt_schedule
from lib.runtime_context import ContextRefused
from data_provider.lifecycle_roles import (
    ROLE_AAS, ROLE_CHANNEL_COUNT, ROLE_FS, ROLE_NAS, ROLE_VS,
)
from lib.lcbt_v4 import SPLIT_FRACTIONS
from lib.v4_payload import load_v4_payload
from utils.timefeatures import time_features


class V4LifecycleDataset(Dataset):
    """Full-N windows with sample-level exposure masks."""

    def __init__(self, context, payload_path, input_len=12, pred_len=12,
                 seed=32, epoch=0, split_fractions=SPLIT_FRACTIONS, scaler=None,
                 freq="min", validation_seed=12026, slide_step=1):
        if context is None:
            raise ContextRefused(
                "v4 requires a RuntimeDataContext; it is not inferred from the "
                "payload and never falls back to the v3 loader")
        self.context = context
        self.input_len, self.pred_len = int(input_len), int(pred_len)
        self.slide_step = int(slide_step)
        if self.input_len <= 0 or self.pred_len <= 0:
            raise ContextRefused("input_len and pred_len must be positive")
        if self.slide_step <= 0:
            raise ContextRefused("slide_step must be a positive integer")
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
        # Calendar features, from the payload's own timestamps and through the
        # same `time_features` the v3 loader used, at the same `freq`. At
        # freq="min" that is exactly [minute-of-hour, hour-of-day, day-of-week]
        # -- the three columns ALFTemporal indexes positionally. Deriving them
        # here rather than in a model keeps one definition for both models and
        # all three phases, and adds no feature the v3 semantics did not have.
        self.freq = str(freq)
        self.time_marks = np.ascontiguousarray(
            time_features(stamps, freq=self.freq).astype(np.float32))
        if self.time_marks.shape[0] != steps:
            raise ContextRefused(
                f"time marks cover {self.time_marks.shape[0]} steps, the signal "
                f"has {steps}")
        # Two seeds, deliberately. Training redraws its schedule every epoch
        # from `seed`; every other phase draws once from `validation_seed` and
        # never moves, so a validation number from epoch 7 is comparable with
        # one from epoch 3. Sharing one seed would make the validation schedule
        # a function of the training epoch counter.
        self.seed = int(seed)
        self.validation_seed = int(validation_seed)
        self.epoch = int(epoch)
        self._build_origins()
        self._build_schedule(stamps)

    # --- construction ---------------------------------------------------------

    @staticmethod
    def _fit_scaler(signal, natural, train_end, known_nodes):
        window = signal[:train_end][:, known_nodes]
        valid = window[natural[:train_end][:, known_nodes]]
        if valid.size == 0:
            raise ContextRefused("no valid training-known values to fit the scaler")
        positive = valid[valid > 0]
        if positive.size == 0:
            raise ContextRefused(
                "no positive training-known values for the safe-relative floor")
        std = float(valid.std())
        return {"mean": float(valid.mean()), "std": std if std > 1e-6 else 1.0,
                "fitted_on": "train window, training-known nodes only",
                "n_values": int(valid.size),
                # A robust denominator fixed before validation/test is opened.
                # q10 is deliberately data-scale aware: one vehicle and one mph
                # should not mean the same thing across unrelated targets.
                "safe_relative_floor": float(np.quantile(positive, 0.10)),
                "safe_relative_floor_policy":
                    "positive_target_q10_on_train_window_training_known_nodes",
                "safe_relative_floor_n_values": int(positive.size)}

    def _build_origins(self):
        """Window starts whose guard reads only in-scope nodes."""
        start, end = self.boundaries[self.context.phase]
        span = self.input_len + self.pred_len
        scope = self.context.signal_scope
        origins = []
        for origin in range(start, end - span + 1, self.slide_step):
            window = self.natural[origin:origin + span][:, scope]
            # A window is usable if the in-scope nodes carry any measurement at
            # all. Out-of-scope columns are not consulted, so their values and
            # their missingness cannot move which origins exist.
            if window.any():
                origins.append(origin)
        self.origins = np.asarray(origins, dtype=np.int64)

    def _schedule_draw(self):
        """(seed, epoch) for this phase's schedule."""
        if self.context.phase == "train":
            return self.seed, self.epoch
        return self.validation_seed, 0

    def _build_schedule(self, stamps=None):
        if stamps is None:
            stamps = self.stamps
        regimes = [calendar_stratum(stamps[origin + self.input_len - 1])
                   for origin in self.origins]
        adjacency = self.context.adjacency.numpy()
        seed, epoch = self._schedule_draw()
        self.schedule = build_lcbt_schedule(
            num_samples=len(self.origins), input_len=self.input_len,
            adjacency=adjacency, calendar_regimes=regimes, seed=seed, epoch=epoch,
            scenario_node_pools=self._scenario_pools(),
            topology_adjacency=adjacency, node_ids=np.arange(self.num_nodes),
            cohort_size_base=self.num_nodes)

    def set_epoch(self, epoch):
        """Redraw the training schedule for `epoch`. Deterministic in (seed, epoch).

        Only the schedule moves. The origins, the scaler, the K/V/T sets and the
        time boundaries are facts about the dataset and are not recomputed --
        re-fitting the scaler per epoch would make the normalisation a function
        of the epoch counter.

        A no-op on validation and test by construction: `_schedule_draw` returns
        the fixed validation seed and epoch 0 for them, so the rebuild would
        produce the identical schedule. It is skipped rather than repeated so
        the object identity holds too, which is what the test asserts.
        """
        self.epoch = int(epoch)
        if self.context.phase == "train":
            self._build_schedule()
        return self

    def _scenario_pools(self):
        """Which nodes each scenario may draw, per phase.

        Training simulates its no-history and join tracks on K, because K is all
        it may read. Validation and test run those two tracks on their *own*
        held-out set -- that is the measurement the experiment exists to make,
        and drawing them from K instead would quietly turn Unseen and Join into
        another Steady row. Steady and Fault stay on K in every phase: they need
        history, and a held-out node has none.
        """
        known = self.context.known_nodes
        held_out = {"train": known,
                    "validation": self.context.validation_unseen,
                    "test": self.context.test_unseen}[self.context.phase]
        return {"zero_history": held_out, "join": held_out,
                "steady": known, "fault": known}

    def __len__(self):
        return int(self.origins.size)

    # --- the five masks -------------------------------------------------------

    def _availability(self, entry, natural_input):
        """(L, N) input availability, plus the scope and trajectory behind it.

        Three independent reasons a measurement may be absent, combined here and
        never conflated anywhere else: the sensor recorded nothing, the phase may
        not read that node, or the episode hides it for this sample.
        """
        scope = np.zeros(self.num_nodes, dtype=bool)
        scope[self.context.signal_scope] = True
        roles, lifecycle, age = self._lifecycle_state(entry, scope)
        return natural_input & scope[None, :] & lifecycle, scope, roles, age

    def __getitem__(self, index):
        return self._sample_from_entry(index, self.schedule[index])

    def counterfactual(self, index):
        """Matched control for a Join or Fault schedule entry.

        The origin, cohort, calendar regime and target are identical. Join is
        compared with zero-history on the same held-out cohort; Fault with
        steady operation on the same known cohort. This is an evaluator-only
        construction and never changes the training schedule.
        """
        treated = self.schedule[index]
        scenario = str(treated["scenario"])
        controls = {"join": "zero_history", "fault": "steady"}
        if scenario not in controls:
            raise ContextRefused(
                f"PLES needs a join or fault entry, got {scenario!r}")
        control = dict(treated)
        control["scenario"] = controls[scenario]
        control["phase_id"] = -1
        control["evidence_steps"] = 0
        return self._sample_from_entry(index, control)

    def _sample_from_entry(self, index, entry):
        origin = int(self.origins[index])
        end = origin + self.input_len
        stop = end + self.pred_len

        natural_input = self.natural[origin:end]
        natural_target = self.natural[end:stop]
        available, scope, roles, ages = self._availability(entry, natural_input)

        # 6. Select in normalised space. `value * mask` would turn a NaN into a
        #    NaN, not into a zero.
        normalised = torch.from_numpy(self.normalised[origin:end].copy())
        base_visible = torch.where(torch.from_numpy(available),
                                   normalised, torch.zeros((), dtype=torch.float32))
        if not torch.isfinite(base_visible).all():
            raise ContextRefused("base_visible_input is not finite after masking")

        # Scoped, not raw natural availability. An unscoped version leaked: a
        # forbidden node's *missingness* is itself a measurement, and poisoning
        # V/T raw values moved this mask while every other field held.
        target_available = torch.from_numpy(natural_target & scope[None, :])
        # The exposed target is masked, not merely accompanied by a mask that
        # says "do not use". Leaving the real future value in the batch means any
        # consumer with a wrong mask reads the answer, and no counterfactual on
        # the masks alone would catch it.
        target = torch.where(
            target_available.unsqueeze(-1),
            torch.from_numpy(np.nan_to_num(self.normalised[end:stop], nan=0.0).copy())
                 .unsqueeze(-1),
            torch.zeros((), dtype=torch.float32))

        # Sample-level, not phase-level: a node the episode still hides at the
        # last observed step has no present state to carry into the horizon, so
        # its future value must not be teacher-forced.
        last_observed = torch.from_numpy(available[-1].copy())
        decoder_available = (last_observed[None, :].expand(self.pred_len, -1)
                             & target_available)

        # Exactly the entry's changed_nodes, in every scenario. Widening steady
        # to the whole scope, or zero_history to the whole hidden pool, makes the
        # row a different measurement from the one the schedule declared.
        cohort = torch.zeros(self.num_nodes, dtype=torch.float32)
        cohort[np.asarray(entry["changed_nodes"], dtype=np.int64)] = 1.0
        loss_scope = self.context.loss_scope_mask()
        loss_mask = target_available & (cohort > 0.5)[None, :] & loss_scope[None, :]

        return {
            "base_visible_input": base_visible.unsqueeze(-1),
            "input_available_mask": torch.from_numpy(available),
            "decoder_available_mask": decoder_available,
            "target_available_mask": target_available,
            "target_cohort_mask": cohort,
            "loss_mask": loss_mask,
            "target": target,
            "lifecycle_role_ids": torch.from_numpy(roles),
            "lifecycle_state_age": torch.from_numpy(ages),
            # Calendar only. No node id, no holiday flag, and identical for
            # every node, so nothing here can carry a forbidden measurement.
            "encoder_time_marks": torch.from_numpy(
                self.time_marks[origin:end].copy()),
            "decoder_time_marks": torch.from_numpy(
                self.time_marks[end:stop].copy()),
            # ALFNet's method inputs, kept out of base_visible_input so every
            # baseline receives the identical common tensor. The next stage's
            # adapter concatenates these for ALFNet only.
            "lifecycle_context": torch.cat([
                torch.nn.functional.one_hot(
                    torch.from_numpy(roles), ROLE_CHANNEL_COUNT).float(),
                torch.from_numpy(ages).unsqueeze(-1),
                torch.from_numpy(available.astype(np.float32)).unsqueeze(-1),
            ], dim=-1),
            "origin": origin,
            "sample_index": int(index),
            "phase_id": entry["phase_id"],
            "scenario": entry["scenario"],
            "evidence_steps": int(entry["evidence_steps"]),
            "calendar_regime": entry["calendar_regime"],
            "context_identity": self.context.identity(),
        }

    def _lifecycle_state(self, entry, scope):
        """(L, N) roles, lifecycle mask and age -- the frozen v3 trajectory.

        Join and Fault are trajectories, not end states: collapsing them to the
        last step throws away the whole thing the two tracks measure. Roles are
        decided by the phase scope and the episode only. Natural missingness is
        expressed by the observation mask and must never move a role, or a
        sensor that dropped one reading would be reported as newly attached.
        """
        length = self.input_len
        roles = np.full((length, self.num_nodes), ROLE_AAS, dtype=np.int64)
        lifecycle = np.ones((length, self.num_nodes), dtype=bool)
        age = np.ones((length, self.num_nodes), dtype=np.float32)

        def to_virtual(nodes):
            if len(nodes):
                roles[:, nodes] = ROLE_VS
                lifecycle[:, nodes] = False
                age[:, nodes] = 0.0

        # Nodes this phase may not read carry no history at all.
        to_virtual(np.nonzero(~scope)[0])
        to_virtual(np.asarray(entry["hidden_nodes"], dtype=np.int64))

        changed = np.asarray(entry["changed_nodes"], dtype=np.int64)
        scenario = str(entry["scenario"])
        evidence = int(entry.get("evidence_steps", 0))
        cutoff = length - evidence
        ramp = (np.arange(1, evidence + 1, dtype=np.float32)[:, None]
                / float(length)) if evidence else None

        if scenario == "zero_history":
            to_virtual(changed)
        elif scenario == "steady":
            if changed.size:
                roles[:, changed] = ROLE_AAS
                lifecycle[:, changed] = True
                age[:, changed] = 1.0
        elif scenario == "join":
            to_virtual(changed)
            if changed.size and evidence:
                roles[cutoff:, changed] = ROLE_NAS
                lifecycle[cutoff:, changed] = True
                age[cutoff:, changed] = ramp
        elif scenario == "fault":
            if changed.size:
                roles[:, changed] = ROLE_AAS
                lifecycle[:, changed] = True
                age[:, changed] = 1.0
                if evidence:
                    roles[cutoff:, changed] = ROLE_FS
                    lifecycle[cutoff:, changed] = False
                    age[cutoff:, changed] = ramp
        else:
            raise ContextRefused(f"unsupported scenario: {scenario}")
        return roles, lifecycle, age

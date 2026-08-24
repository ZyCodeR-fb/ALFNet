"""Validation/test mechanism diagnostics for the v4 lifecycle protocol.

This module is deliberately outside the training objective.  It reports:

* a k-study: absolute Join and Fault error at every evidence length 1..L;
* PLES: matched treated/control contrasts on the same origin and cohort.

PLES is descriptive mechanism evidence, not a causal or significance claim.
Every effect is accompanied by both absolute arms so a better-looking contrast
cannot be manufactured by degrading its control.
"""

from collections import defaultdict

import numpy as np
import torch

from data_provider.v4_metrics import (
    calendar_balanced_metrics,
    tau_balanced_metrics,
)

SCHEMA_VERSION = 1
METRICS = ("MAE", "MAPE")
TRACKS = ("join", "fault")
EFFECTS = {
    "JoinLift": {
        "track": "join",
        "formula": "control(Unseen) - treated(Join)",
        "higher_is_better": True,
    },
    "FaultPenalty": {
        "track": "fault",
        "formula": "treated(Fault) - control(Steady)",
        "higher_is_better": False,
    },
    "OutageFaultPenalty": {
        "track": "fault",
        "formula": "treated(Fault,k=L) - control(Steady,k=L)",
        "higher_is_better": False,
    },
}


class MechanismRefused(ValueError):
    """A diagnostic input is not an exact matched v4 pair."""


def _error_sums(prediction, target, valid, mean, std):
    predicted = prediction[valid] * std + mean
    actual = target[valid] * std + mean
    difference = predicted - actual
    denominator = torch.clamp(actual.abs(), min=1e-6)
    return {
        "count": int(valid.sum()),
        "ae_sum": float(difference.abs().sum()),
        "se_sum": float(difference.square().sum()),
        "ape_sum": float((difference.abs() / denominator).sum()),
    }


def _prefixed(sums, prefix):
    return {
        "count": sums["count"],
        f"{prefix}_ae_sum": sums["ae_sum"],
        f"{prefix}_se_sum": sums["se_sum"],
        f"{prefix}_ape_sum": sums["ape_sum"],
    }


def _finite_metrics(metrics, where):
    result = {name: float(metrics[name]) for name in ("MAE", "RMSE", "MAPE")}
    if not all(np.isfinite(value) for value in result.values()):
        raise MechanismRefused(f"{where} contains a non-finite metric")
    return result


class V4MechanismCollector:
    """Accumulate one model's matched Join/Unseen and Fault/Steady pairs."""

    def __init__(self, context, scaler, input_len):
        if context.phase not in ("validation", "test"):
            raise MechanismRefused("mechanism evaluation is validation/test only")
        self.context = context
        self.mean = float(scaler["mean"])
        self.std = float(scaler["std"])
        self.input_len = int(input_len)
        self.rows = defaultdict(list)

    def update(self, treated_batch, treated_prediction,
               control_batch, control_prediction):
        if treated_prediction.shape != control_prediction.shape:
            raise MechanismRefused("treated/control prediction shapes differ")
        for key in ("origin", "sample_index", "calendar_regime"):
            if list(treated_batch[key]) != list(control_batch[key]):
                raise MechanismRefused(f"treated/control {key} identity differs")
        if not torch.equal(treated_batch["target"], control_batch["target"]):
            raise MechanismRefused("treated/control targets differ")
        if not torch.equal(treated_batch["loss_mask"], control_batch["loss_mask"]):
            raise MechanismRefused("treated/control loss masks differ")

        valid = (treated_batch["loss_mask"].unsqueeze(-1)
                 & torch.isfinite(treated_prediction)
                 & torch.isfinite(control_prediction)
                 & torch.isfinite(treated_batch["target"]))
        for index, scenario in enumerate(treated_batch["scenario"]):
            track = str(scenario)
            if track not in TRACKS:
                raise MechanismRefused(f"treated scenario {track!r} is not an event")
            sample = valid[index]
            if not bool(sample.any()):
                continue
            treated = _error_sums(treated_prediction[index],
                                  treated_batch["target"][index], sample,
                                  self.mean, self.std)
            control = _error_sums(control_prediction[index],
                                  control_batch["target"][index], sample,
                                  self.mean, self.std)
            if treated["count"] != control["count"]:
                raise MechanismRefused("matched arms have different counts")
            row = {
                "origin": int(treated_batch["origin"][index]),
                "tau": int(treated_batch["evidence_steps"][index]),
                "regime": str(treated_batch["calendar_regime"][index]),
                **_prefixed(treated, "treated"),
                **{key: value for key, value in _prefixed(control, "control").items()
                   if key != "count"},
            }
            self.rows[track].append(row)

    def _arm(self, rows, prefix, settled=False):
        if settled:
            rows = [row for row in rows if int(row["tau"]) == self.input_len]
            metrics = calendar_balanced_metrics(rows, prefix)
        else:
            metrics = tau_balanced_metrics(rows, prefix)
        if metrics is None:
            raise MechanismRefused(f"{prefix} arm has no evaluable rows")
        return _finite_metrics(metrics, prefix)

    def report(self):
        k_study = {}
        for track in TRACKS:
            k_study[track] = {}
            for evidence in range(1, self.input_len + 1):
                rows = [row for row in self.rows[track]
                        if int(row["tau"]) == evidence]
                metrics = calendar_balanced_metrics(rows, "treated")
                if metrics is None:
                    raise MechanismRefused(f"{track} k={evidence} is missing")
                k_study[track][str(evidence)] = _finite_metrics(
                    metrics, f"{track}/k={evidence}")

        effects = {}
        for name, definition in EFFECTS.items():
            rows = self.rows[definition["track"]]
            settled = name == "OutageFaultPenalty"
            treated = self._arm(rows, "treated", settled)
            control = self._arm(rows, "control", settled)
            effect = {}
            for metric in METRICS:
                if definition["higher_is_better"]:
                    effect[metric] = control[metric] - treated[metric]
                else:
                    effect[metric] = treated[metric] - control[metric]
            effects[name] = {
                **definition,
                "treated_absolute_error": treated,
                "control_absolute_error": control,
                "effect": effect,
                "pair_count": len([row for row in rows
                                   if not settled or int(row["tau"]) == self.input_len]),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "split": self.context.phase,
            "descriptive_only": True,
            "causal_claim": False,
            "input_len": self.input_len,
            "k_study": k_study,
            "ples": effects,
            "guard": ("An effect is never interpreted without its treated and "
                      "control absolute-error arms."),
        }


def compare_model_reports(left, right, left_name="ALFNet", right_name="MoGERNN"):
    """Model deltas with an explicit algebra check for every PLES endpoint."""
    records = {}
    for effect, definition in EFFECTS.items():
        records[effect] = {}
        for metric in METRICS:
            left_effect = float(left["ples"][effect]["effect"][metric])
            right_effect = float(right["ples"][effect]["effect"][metric])
            treated_delta = (
                float(left["ples"][effect]["treated_absolute_error"][metric])
                - float(right["ples"][effect]["treated_absolute_error"][metric]))
            control_delta = (
                float(left["ples"][effect]["control_absolute_error"][metric])
                - float(right["ples"][effect]["control_absolute_error"][metric]))
            contrast_delta = left_effect - right_effect
            expected = (control_delta - treated_delta
                        if definition["higher_is_better"]
                        else treated_delta - control_delta)
            residual = contrast_delta - expected
            if not np.isfinite(residual) or abs(residual) > 1e-9:
                raise MechanismRefused(
                    f"{effect}/{metric} algebra residual is {residual}")
            records[effect][metric] = {
                "models": {left_name: left_effect, right_name: right_effect},
                "contrast_delta_left_minus_right": contrast_delta,
                "treated_delta_left_minus_right": treated_delta,
                "control_delta_left_minus_right": control_delta,
                "algebra_residual": residual,
            }
    return {"schema_version": SCHEMA_VERSION, "descriptive_only": True,
            "models": [left_name, right_name], "records": records}

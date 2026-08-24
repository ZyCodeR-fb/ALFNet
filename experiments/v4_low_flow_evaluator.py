"""Low-flow diagnosis for flow datasets, outside headline evaluation.

The frozen headline remains MAE/RMSE/MAPE.  This collector adds target bins,
WMAPE and signed bias only to explain percentage-error reversals.  It never
participates in training, early stopping or model selection.
"""

from collections import defaultdict

import numpy as np
import torch


SCHEMA_VERSION = 1
BIN_DEFINITIONS = (
    ("flow_0_5", 0.0, 5.0),
    ("flow_5_10", 5.0, 10.0),
    ("flow_10_20", 10.0, 20.0),
    ("flow_20_50", 20.0, 50.0),
    ("flow_gt_50", 50.0, float("inf")),
)
MAPE_FLOOR = 1e-6


class LowFlowRefused(ValueError):
    """The diagnostic input is not the frozen PEMS flow task."""


class _Accumulator:
    def __init__(self):
        self.count = 0
        self.ae_sum = 0.0
        self.se_sum = 0.0
        self.ape_sum = 0.0
        self.target_sum = 0.0
        self.error_sum = 0.0
        self.over_count = 0
        self.target_min = float("inf")
        self.target_max = float("-inf")

    def add(self, predicted, actual):
        predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
        actual = np.asarray(actual, dtype=np.float64).reshape(-1)
        if predicted.size != actual.size or predicted.size == 0:
            raise LowFlowRefused("prediction/target bin is empty or misaligned")
        if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
            raise LowFlowRefused("low-flow diagnostic received non-finite values")
        if not np.all(actual > 0):
            raise LowFlowRefused("an evaluated flow target is non-positive")
        error = predicted - actual
        absolute = np.abs(error)
        self.count += int(actual.size)
        self.ae_sum += float(absolute.sum())
        self.se_sum += float(np.square(error).sum())
        self.ape_sum += float((absolute / np.maximum(np.abs(actual), MAPE_FLOOR)).sum())
        self.target_sum += float(np.abs(actual).sum())
        self.error_sum += float(error.sum())
        self.over_count += int((error > 0).sum())
        self.target_min = min(self.target_min, float(actual.min()))
        self.target_max = max(self.target_max, float(actual.max()))

    def report(self, total_count):
        if self.count <= 0 or self.target_sum <= 0:
            return None
        result = {
            "count": self.count,
            "share": self.count / int(total_count),
            "target_mean": self.target_sum / self.count,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "MAE": self.ae_sum / self.count,
            "RMSE": float(np.sqrt(self.se_sum / self.count)),
            "MAPE": self.ape_sum / self.count * 100.0,
            "WMAPE": self.ae_sum / self.target_sum * 100.0,
            "signed_bias": self.error_sum / self.count,
            "overprediction_rate": self.over_count / self.count,
        }
        if not all(np.isfinite(value) for value in result.values()):
            raise LowFlowRefused("a low-flow metric is non-finite")
        return result


def _bin_name(values):
    names = np.empty(values.shape, dtype=object)
    names[:] = None
    for name, low, high in BIN_DEFINITIONS:
        names[(values > low) & (values <= high)] = name
    if any(value is None for value in names.reshape(-1)):
        raise LowFlowRefused("a positive finite target did not match a frozen bin")
    return names


class LowFlowCollector:
    """Accumulate Unseen errors by frozen target-flow bin and horizon."""

    def __init__(self, context, scaler, pred_len):
        if context.phase not in ("validation", "test"):
            raise LowFlowRefused("low-flow evaluation is validation/test only")
        if context.target != "q":
            raise LowFlowRefused("low-flow bins apply to flow target q only")
        self.context = context
        self.mean = float(scaler["mean"])
        self.std = float(scaler["std"])
        self.pred_len = int(pred_len)
        self.overall = _Accumulator()
        self.bins = defaultdict(_Accumulator)
        self.horizons = {
            step: {"overall": _Accumulator(),
                   "bins": defaultdict(_Accumulator)}
            for step in range(self.pred_len)
        }
        self.origin_count = set()
        self.sample_count = 0

    def update(self, batch, prediction):
        if prediction.shape != batch["target"].shape:
            raise LowFlowRefused("prediction and target shapes differ")
        if prediction.shape[1] != self.pred_len:
            raise LowFlowRefused("prediction horizon differs from the frozen length")
        for index, scenario in enumerate(batch["scenario"]):
            if str(scenario) != "zero_history":
                continue
            valid = (batch["loss_mask"][index].unsqueeze(-1)
                     & torch.isfinite(prediction[index])
                     & torch.isfinite(batch["target"][index]))
            if not bool(valid.any()):
                continue
            self.sample_count += 1
            self.origin_count.add(int(batch["origin"][index]))
            for step in range(self.pred_len):
                step_valid = valid[step]
                if not bool(step_valid.any()):
                    continue
                predicted = (prediction[index, step][step_valid] * self.std
                             + self.mean).detach().cpu().numpy()
                actual = (batch["target"][index, step][step_valid] * self.std
                          + self.mean).detach().cpu().numpy()
                names = _bin_name(actual)
                self.overall.add(predicted, actual)
                self.horizons[step]["overall"].add(predicted, actual)
                for name, _, _ in BIN_DEFINITIONS:
                    selected = names == name
                    if bool(selected.any()):
                        self.bins[name].add(predicted[selected], actual[selected])
                        self.horizons[step]["bins"][name].add(
                            predicted[selected], actual[selected])

    @staticmethod
    def _definitions():
        return {
            name: {"lower_exclusive": low,
                   "upper_inclusive": None if np.isinf(high) else high}
            for name, low, high in BIN_DEFINITIONS
        }

    def report(self):
        total = self.overall.count
        if total <= 0:
            raise LowFlowRefused("no Unseen flow targets were evaluated")
        bins = {name: self.bins[name].report(total)
                for name, _, _ in BIN_DEFINITIONS}
        horizons = {}
        for step in range(self.pred_len):
            slot = self.horizons[step]
            step_total = slot["overall"].count
            horizons[str(step + 1)] = {
                "overall": slot["overall"].report(step_total),
                "bins": {name: slot["bins"][name].report(step_total)
                         for name, _, _ in BIN_DEFINITIONS},
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "split": self.context.phase,
            "group": "Unseen",
            "diagnostic_only": True,
            "selection_metric": False,
            "headline_metric": False,
            "bin_source": "predeclared domain cutoffs; not selected on test",
            "bin_definitions": self._definitions(),
            "mape_denominator": f"max(abs(target), {MAPE_FLOOR})",
            "wmape_definition": "100 * sum(abs(error)) / sum(abs(target))",
            "origin_count": len(self.origin_count),
            "sample_count": self.sample_count,
            "overall": self.overall.report(total),
            "bins": bins,
            "horizons": horizons,
        }


def compare_reports(left, right, left_name="ALFNet", right_name="MoGERNN"):
    """Descriptive left-minus-right deltas, with sample identity checks."""
    for field in ("split", "group", "origin_count", "sample_count",
                  "bin_definitions"):
        if left[field] != right[field]:
            raise LowFlowRefused(f"model reports differ on {field}")
    metrics = ("MAE", "RMSE", "MAPE", "WMAPE", "signed_bias",
               "overprediction_rate")
    records = {}
    for name, _, _ in BIN_DEFINITIONS:
        lbin, rbin = left["bins"][name], right["bins"][name]
        if (lbin is None) != (rbin is None):
            raise LowFlowRefused(f"models disagree on evaluability of {name}")
        if lbin is None:
            records[name] = None
            continue
        if lbin["count"] != rbin["count"]:
            raise LowFlowRefused(f"models have different counts in {name}")
        records[name] = {
            "count": lbin["count"], "share": lbin["share"],
            "models": {left_name: {key: lbin[key] for key in metrics},
                       right_name: {key: rbin[key] for key in metrics}},
            "delta_left_minus_right": {
                key: float(lbin[key]) - float(rbin[key]) for key in metrics},
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "models": [left_name, right_name],
        "split": left["split"],
        "bins": records,
    }

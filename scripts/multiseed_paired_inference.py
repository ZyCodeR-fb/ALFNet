"""Three-seed hierarchical paired inference, ALFNet against MoGERNN.

``scripts/paired_model_bootstrap.py`` compares one seed against one seed. It
says so itself -- it asserts the two runs share a seed and stamps
``single_seed_diagnostic_only`` into its output -- but that leaves the formal
comparison resting on a single training run, and training randomness is a real
source of variance here: on METR-LA the seed ordering reverses on most cells.

This adds the level that was missing. Seed is the outermost resampling unit, so
the interval covers training randomness as well as evaluation sampling:

    for each bootstrap sample
        draw 3 seeds with replacement from {32, 42, 52}
        draw one global cohort multiplicity, replayed at every drawn seed
        within each drawn seed, draw its overlap blocks stratified by day type
        both models get the identical draw
        statistic = mean over drawn seeds of (ALFNet - MoGERNN)

The point estimate is the mean over the three real seeds of the per-seed
difference, which equals the difference of the two three-seed means the
collector publishes -- the two are the same number because the mean is linear,
and a test asserts it.

**On what three seeds can support.** Three is enough to stop a single run
speaking for the model and not enough for a precise variance estimate. The
outer draw has only 3^3 = 27 distinct seed multisets, so the seed level of this
interval is coarse; read it as "does the sign survive resampling the training
runs", not as a calibrated width. The output records this rather than implying
more.

    python scripts/multiseed_paired_inference.py --dataset METRLA \\
        --output-dir ./outputs/reports/final/multiseed
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_lifecycle_results import _load_run  # noqa: E402
from scripts.final_runs import load_manifest, verify_eval  # noqa: E402
from scripts.paired_model_bootstrap import (  # noqa: E402
    RowTensor,
    _bootstrap_p_value,
    _holm,
    _main_cohort_ids,
    _pair_keys,
    _read_rows,
    _trajectory_span,
    assert_paired_bundles,
    assert_pre_event_identity,
    compute_statistics,
)

MODEL_A, MODEL_B = "ALFNet", "MoGERNN"

# Frozen multiple-testing families. Anything outside these is exploratory and
# carries no adjusted p-value: correcting across a set chosen after seeing the
# results is not a correction.
PRIMARY_LCE = tuple(("main", row, "MAE") for row in ("Unseen", "Steady", "Join", "Fault"))
PRIMARY_PLES = tuple(
    ("ples", effect, metric)
    for effect in ("JoinLift", "FaultPenalty", "OutageFaultPenalty")
    for metric in ("MAPE", "MAE")
)
FAMILIES = {"lce_primary": PRIMARY_LCE, "ples": PRIMARY_PLES}

# JoinLift is a benefit; everything else here is an error or a penalty.
HIGHER_IS_BETTER = {("ples", "JoinLift", "MAPE"), ("ples", "JoinLift", "MAE")}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["METRLA", "PEMSBAY", "PEMS08"])
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--practical-margin-mae", type=float, default=0.05,
                        help="smallest ALFNet-minus-MoGERNN MAE difference treated as "
                             "practically meaningful; must be frozen before the run")
    parser.add_argument("--practical-margin-mape", type=float, default=0.25)
    return parser.parse_args()


def load_seed_bundles(dataset, output_root):
    """Exactly the manifest's runs for this dataset. Nothing found on disk."""
    payload = load_manifest()
    expected_seeds = sorted(payload["seeds"])
    wanted = defaultdict(dict)
    for run in payload["runs"]:
        if run["dataset"] != dataset:
            continue
        if run["seed"] in wanted[run["model"]]:
            raise SystemExit(f"manifest lists {run['model']} seed {run['seed']} twice")
        wanted[run["model"]][run["seed"]] = run

    problems = []
    for model in (MODEL_A, MODEL_B):
        seeds = sorted(wanted.get(model, {}))
        if seeds != expected_seeds:
            problems.append(f"{model}: seeds {seeds}, manifest declares {expected_seeds}")
    if problems:
        raise SystemExit("REFUSING:\n" + "\n".join(f"  - {p}" for p in problems))

    bundles = {}
    for seed in expected_seeds:
        per_model = {}
        for model in (MODEL_A, MODEL_B):
            run = wanted[model][seed]
            run_problems, directory = verify_eval(run, output_root)
            if run_problems:
                problems.append(f"{run['run_id']}: " + "; ".join(run_problems))
                continue
            per_model[model] = (run, Path(directory))
        bundles[seed] = per_model
    if problems:
        raise SystemExit("REFUSING:\n" + "\n".join(f"  - {p}" for p in problems))
    return expected_seeds, bundles


def build_tensors(seeds, bundles):
    """One RowTensor per (seed, model), with the pairing checks per seed."""
    tensors, provenance = {}, {}
    for seed in seeds:
        run_a, dir_a = bundles[seed][MODEL_A]
        run_b, dir_b = bundles[seed][MODEL_B]
        (rows_a, _path_a), (rows_b, _path_b) = _read_rows(dir_a), _read_rows(dir_b)
        loaded_a, loaded_b = _load_run(dir_a), _load_run(dir_b)

        # The manifest says which seed this directory should be; the result file
        # says which seed it is. Checking both is the point -- a directory moved
        # or relabelled by hand passes one and fails the other.
        for loaded, run in ((loaded_a, run_a), (loaded_b, run_b)):
            if int(loaded["seed"]) != int(run["seed"]):
                raise SystemExit(
                    f"{run['run_id']}: result says seed {loaded['seed']}, "
                    f"manifest says {run['seed']}"
                )
            if loaded["model"] != run["model"]:
                raise SystemExit(
                    f"{run['run_id']}: result says model {loaded['model']}, "
                    f"manifest says {run['model']}"
                )

        assert_paired_bundles(loaded_a, loaded_b, rows_a, rows_b)

        # Put B's rows in A's order. Every statistic is computed from a shared
        # weight vector indexed by row position, so the two tensors must agree
        # row for row or the pairing is silently broken.
        order = {key: index for index, key in enumerate(_pair_keys(rows_a))}
        rows_b = sorted(rows_b, key=lambda row: order[
            (int(row["origin_id"]), int(row["cohort_id"]),
             str(row["track"]), int(row["tau"]))
        ])
        assert_pre_event_identity(rows_a, MODEL_A)
        assert_pre_event_identity(rows_b, MODEL_B)

        manifest = json.loads((dir_a / "protocol_manifest.json").read_text("utf-8"))
        input_len = int(manifest["resolved"]["input_len"])
        span_a = _trajectory_span(dir_a, input_len)
        span_b = _trajectory_span(dir_b, input_len)
        if span_a != span_b:
            raise SystemExit(f"seed {seed}: PLES spans differ, {span_a} vs {span_b}")
        cohort_ids = _main_cohort_ids(dir_a)

        # Saturated cohorts are outside the pooled tables, so they are dropped
        # before the tensor is built rather than filtered afterwards.
        wanted = {int(value) for value in cohort_ids}
        main_a = [row for row in rows_a if int(row["cohort_id"]) in wanted]
        main_b = [row for row in rows_b if int(row["cohort_id"]) in wanted]
        main_keys = _pair_keys(main_a)
        tensors[seed] = {
            MODEL_A: RowTensor(main_a, main_keys, input_len, span_a),
            MODEL_B: RowTensor(main_b, main_keys, input_len, span_a),
        }
        span = span_a
        provenance[seed] = {
            "protocol_hash": loaded_a["hashes"]["protocol_hash"],
            "training_protocol_hash": loaded_a["hashes"].get("training_protocol_hash"),
            "input_len": input_len,
            "trajectory_span": span,
            "main_cohort_ids": cohort_ids,
            MODEL_A: run_a["run_id"],
            MODEL_B: run_b["run_id"],
        }

    hashes = {entry["protocol_hash"] for entry in provenance.values()}
    if len(hashes) != 1:
        raise SystemExit(f"seeds disagree on the evaluation protocol hash: {sorted(hashes)}")
    return tensors, provenance


def _cohort_multiplicity(tensor, rng):
    multiplicity = np.zeros(int(max(tensor.cohort_ids)) + 1, dtype=np.float64)
    for cohort_id in rng.choice(tensor.cohort_ids, size=len(tensor.cohort_ids), replace=True):
        multiplicity[int(cohort_id)] += 1.0
    return multiplicity


def _weights(tensor, rng, cohort_multiplicity):
    """Stratified overlap-block draw crossed with a cohort draw made elsewhere.

    The cohort draw is passed in rather than made here because a bootstrap
    sample replays one global cohort multiplicity at every seed it drew; drawing
    cohorts independently per seed would treat the frozen cohort set as if it
    were resampled three times.
    """
    block_multiplicity = defaultdict(float)
    for _stratum, blocks in tensor.blocks_by_stratum.items():
        for block in rng.choice(blocks, size=len(blocks), replace=True):
            block_multiplicity[str(block)] += 1.0
    weights = np.zeros(tensor.n_rows, dtype=np.float64)
    for block, repeats in block_multiplicity.items():
        indices = tensor.block_rows[block]
        weights[indices] += repeats * cohort_multiplicity[tensor.row_cohort[indices]]
    return weights


def per_seed_deltas(tensors, seeds):
    """Observed ALFNet - MoGERNN at each seed, on the full (unresampled) rows."""
    out = {}
    for seed in seeds:
        ones_a = np.ones(tensors[seed][MODEL_A].n_rows, dtype=np.float64)
        ones_b = np.ones(tensors[seed][MODEL_B].n_rows, dtype=np.float64)
        stats_a = compute_statistics(tensors[seed][MODEL_A], ones_a)
        stats_b = compute_statistics(tensors[seed][MODEL_B], ones_b)
        out[seed] = {key: stats_a[key] - stats_b[key] for key in stats_a}
    return out


def hierarchical_bootstrap(tensors, seeds, samples, seed):
    rng = np.random.default_rng(seed)
    keys = None
    draws = None
    for index in range(samples):
        drawn_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        # one global cohort draw per bootstrap sample, replayed at every seed
        cohorts = _cohort_multiplicity(tensors[seeds[0]][MODEL_A], rng)
        accumulated = defaultdict(float)
        for drawn in drawn_seeds:
            drawn = int(drawn)
            tensor_a = tensors[drawn][MODEL_A]
            tensor_b = tensors[drawn][MODEL_B]
            weights = _weights(tensor_a, rng, cohorts)
            stats_a = compute_statistics(tensor_a, weights)
            stats_b = compute_statistics(tensor_b, weights)
            for key in stats_a:
                accumulated[key] += (stats_a[key] - stats_b[key]) / len(drawn_seeds)
        if draws is None:
            keys = sorted(accumulated)
            draws = {key: np.empty(samples, dtype=np.float64) for key in keys}
        for key in keys:
            draws[key][index] = accumulated[key]
    return draws


def _verdict(delta, adjusted_p, higher_is_better, margin):
    """Direction, adjusted significance and practical size, together.

    An unadjusted interval alone was deciding this before. All three have to
    agree: a significant difference smaller than the pre-frozen margin is
    reported as significant-but-negligible rather than as a win.
    """
    if not np.isfinite(delta) or not np.isfinite(adjusted_p):
        return "undetermined"
    favours_a = delta > 0 if higher_is_better else delta < 0
    if adjusted_p >= 0.05:
        return "no difference"
    if abs(delta) < margin:
        return f"significant but below the {margin:g} practical margin"
    return f"{MODEL_A} better" if favours_a else f"{MODEL_B} better"


def build_records(point, draws, seed_deltas, seeds, margins):
    family_of = {key: name for name, keys in FAMILIES.items() for key in keys}
    records = []
    for key in sorted(draws):
        table, row, metric = key
        delta_draws = draws[key]
        finite = np.isfinite(delta_draws)
        low, high = (
            (float(np.percentile(delta_draws[finite], 2.5)),
             float(np.percentile(delta_draws[finite], 97.5)))
            if finite.any() else (float("nan"), float("nan"))
        )
        higher_is_better = key in HIGHER_IS_BETTER
        directions = [seed_deltas[s][key] for s in seeds]
        agreeing = sum(
            1 for value in directions
            if (value > 0) == higher_is_better and value != 0
        )
        records.append({
            "table": table, "row": row, "metric": metric,
            "family": family_of.get(key),
            "delta": float(point[key]),
            "ci_low": low, "ci_high": high,
            "p_bootstrap": _bootstrap_p_value(delta_draws),
            "higher_is_better": higher_is_better,
            "per_seed_delta": {str(s): float(seed_deltas[s][key]) for s in seeds},
            "seeds_favouring_alfnet": agreeing,
            "seed_count": len(seeds),
            "seed_unanimous": agreeing in (0, len(seeds)),
        })

    # Holm inside each frozen family, independently.
    for family, keys in FAMILIES.items():
        members = [r for r in records if r["family"] == family]
        if not members:
            continue
        adjusted = _holm([r["p_bootstrap"] for r in members])
        for record, value in zip(members, adjusted):
            record["p_holm"] = float(value)
            margin = margins["MAE"] if record["metric"] == "MAE" else margins["MAPE"]
            record["practical_margin"] = margin
            record["verdict"] = _verdict(
                record["delta"], value, record["higher_is_better"], margin)
    for record in records:
        if record.get("family") is None:
            record["p_holm"] = None
            record["verdict"] = "exploratory (no correction applied)"
    return records


def main():
    args = parse_args()
    seeds, bundles = load_seed_bundles(args.dataset, args.output_root)
    tensors, provenance = build_tensors(seeds, bundles)

    seed_deltas = per_seed_deltas(tensors, seeds)
    point = {
        key: float(np.mean([seed_deltas[s][key] for s in seeds]))
        for key in seed_deltas[seeds[0]]
    }
    draws = hierarchical_bootstrap(tensors, seeds, args.bootstrap_samples,
                                   args.bootstrap_seed)
    margins = {"MAE": args.practical_margin_mae, "MAPE": args.practical_margin_mape}
    records = build_records(point, draws, seed_deltas, seeds, margins)

    print(f"MultiSeedPairedInference: {MODEL_A} vs {MODEL_B} | {args.dataset}")
    print(f"  seeds={seeds} | samples={args.bootstrap_samples} | "
          f"protocol={provenance[seeds[0]]['protocol_hash'][:12]}")
    print(f"  {'family':<12}{'row':<19}{'metric':<6}{'delta':>10}{'95% CI':>22}"
          f"{'p_holm':>9}  seeds  verdict")
    for record in records:
        if record["family"] is None:
            continue
        interval = f"[{record['ci_low']:+.4f},{record['ci_high']:+.4f}]"
        print(f"  {record['family']:<12}{record['row']:<19}{record['metric']:<6}"
              f"{record['delta']:>+10.4f}{interval:>22}{record['p_holm']:>9.3f}"
              f"   {record['seeds_favouring_alfnet']}/{record['seed_count']}"
              f"   {record['verdict']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{args.dataset}_multiseed_paired.json"
    destination.write_text(json.dumps({
        "analysis": "three-seed hierarchical paired bootstrap",
        "dataset": args.dataset,
        "model_a": MODEL_A, "model_b": MODEL_B,
        "seeds": seeds,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "outer_unit": "training seed, drawn with replacement",
            "inner_unit": "overlap-connected origin block, stratified by day type",
            "cohort_draw": "one global multiplicity per sample, replayed at every drawn seed",
            "paired_across_models": True,
        },
        "point_estimate": "mean over seeds of (ALFNet - MoGERNN); equals the "
                          "difference of the two published three-seed means",
        "families": {name: [list(k) for k in keys] for name, keys in FAMILIES.items()},
        "practical_margins": margins,
        "power_note": "n_seed=3. The outer draw has 27 distinct seed multisets, so "
                      "the seed level of these intervals is coarse. Read the sign "
                      "and the seed agreement, not the exact width.",
        "provenance": provenance,
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()

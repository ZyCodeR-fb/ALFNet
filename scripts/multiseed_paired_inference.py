"""Three-seed hierarchical paired inference, ALFNet against MoGERNN.

``scripts/paired_model_bootstrap.py`` compares one seed against one seed. It
says so itself -- it asserts the two runs share a seed and stamps
``single_seed_diagnostic_only`` into its output -- but that leaves the formal
comparison resting on a single training run, and training randomness is a real
source of variance here: on METR-LA the seed ordering reverses on most cells.

This adds the seed level. Seeds, overlap blocks and cohorts are **crossed**, not
nested: every seed replays the identical frozen evaluation design, verified by a
row-design hash that must match across all seeds and both models or the script
refuses to run. So each factor is drawn once per replicate:

    for each bootstrap replicate
        draw a seed multiplicity over {32, 42, 52}
        draw one cohort multiplicity
        draw one stratified overlap-block sample
        apply that single block/cohort weighting to every drawn seed and to
        both models; a seed drawn twice is weighted twice, not resampled
        statistic = seed-weighted mean of (ALFNet - MoGERNN)

The point estimate is the mean over the three real seeds of the per-seed
difference, which equals the difference of the two three-seed means the
collector publishes -- the same number because the mean is linear, and a test
asserts it.

**This output is descriptive, not confirmatory.** Three reasons, all recorded in
the JSON as ``post_hoc_diagnostic_only`` and
``inferential_status = descriptive_not_formal``:

1. The seed factor cannot reach significance at n=3. The exact two-sided
   sign-flip null over three paired seeds has minimum p = 2/2^3 = 0.25.
2. The seed draw is coarse: three draws with replacement from three seeds give
   27 ordered sequences but only **10** distinct multiplicity vectors.
3. The practical margins were chosen *after* the v3 test split had been read, so
   they are post-hoc and cannot be presented as pre-registered thresholds.

Holm is still computed inside the frozen families, but as a descriptive ordering
statistic only -- adjusting an uncalibrated bootstrap tail area does not make it
a test, and no verdict here consumes it. Formal seed-level inference needs a
pre-registered analysis plan and more seeds; if only three are affordable, it
stays descriptive.

    python scripts/multiseed_paired_inference.py --dataset METRLA \\
        --output-dir ./outputs/reports/final/multiseed
"""

import argparse
import hashlib
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
    parser.add_argument("--post-hoc-margin-mae", type=float, default=0.05,
                        help="MAE difference below which a direction is called negligible. "
                             "POST-HOC: chosen after the v3 test split was read, so it is "
                             "a reporting aid, not an acceptance threshold")
    parser.add_argument("--post-hoc-margin-mape", type=float, default=0.25)
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
            "row_design_hash": _row_design_hash(main_a),
            "row_count": len(main_a),
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

    # The crossed resampler builds one weight vector and applies it to every
    # seed's tensor, so the row order must be identical everywhere. Verified,
    # not assumed: if a seed ever carried a different design the weights would
    # silently address different rows and the pairing would be meaningless.
    designs = {seed: entry["row_design_hash"] for seed, entry in provenance.items()}
    if len(set(designs.values())) != 1:
        raise SystemExit(
            "seeds do not share one evaluation row design, so seed/block/cohort "
            f"are not crossed and this estimator does not apply: {designs}"
        )
    return tensors, provenance


DESIGN_FIELDS = ("origin_id", "cohort_id", "track", "tau", "phase", "regime",
                 "event_time_index", "calendar_day", "cohort_fraction",
                 "cohort_style", "node_count", "count")


def _row_design_hash(rows):
    """Digest of the evaluation design, in row order, excluding any model output.

    Only protocol-determined fields go in -- no treated/control sums -- so two
    models on one seed must produce the same digest, and so must two seeds. It
    covers row *order* as well as content because the resampler indexes rows by
    position.
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update("|".join(str(row[field]) for field in DESIGN_FIELDS).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


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


def crossed_bootstrap(tensors, seeds, samples, seed):
    """Seeds, blocks and cohorts are crossed factors, so each is drawn once.

    Every seed replays the *same* frozen evaluation design -- identical origins,
    cohorts, tracks, taus, regimes and event times, verified by a row-design
    hash before this runs. So the three seeds are three model fits measured on
    one set of test events, not three independent samples of events.

    An earlier version drew a fresh block sample inside each drawn seed. That
    treats one set of physical test events as if it had been sampled once per
    seed, which inflates the effective sample size and narrows the interval. Here
    one block draw and one cohort draw are made per replicate and reused across
    every drawn seed and both models; drawing a seed twice only doubles its
    weight in the average, it does not resample the events again.
    """
    rng = np.random.default_rng(seed)
    reference = tensors[seeds[0]][MODEL_A]
    keys, draws = None, None
    for index in range(samples):
        seed_multiplicity = defaultdict(float)
        for drawn in rng.choice(seeds, size=len(seeds), replace=True):
            seed_multiplicity[int(drawn)] += 1.0
        # one draw each, shared by every seed and both models in this replicate
        cohorts = _cohort_multiplicity(reference, rng)
        weights = _weights(reference, rng, cohorts)

        accumulated = defaultdict(float)
        total = sum(seed_multiplicity.values())
        for drawn, repeats in seed_multiplicity.items():
            stats_a = compute_statistics(tensors[drawn][MODEL_A], weights)
            stats_b = compute_statistics(tensors[drawn][MODEL_B], weights)
            for key in stats_a:
                accumulated[key] += repeats * (stats_a[key] - stats_b[key]) / total
        if draws is None:
            keys = sorted(accumulated)
            draws = {key: np.empty(samples, dtype=np.float64) for key in keys}
        for key in keys:
            draws[key][index] = accumulated[key]
    return draws


def _descriptive_verdict(delta, low, high, higher_is_better, margin, seeds_favouring,
                         seed_count):
    """A described direction, never a significance claim.

    Deliberately not a test. With three training seeds the exact two-sided
    sign-flip null has minimum p = 2/2^3 = 0.25, so no arrangement of three
    seeds can reach 0.05 on the seed factor; a bootstrap tail area computed
    across three fits is not a calibrated p-value for training randomness, and
    Holm-adjusting an uncalibrated p does not repair it. The wording therefore
    describes the interval and the seed agreement rather than declaring
    significance, and the margins are post-hoc (see the module docstring), which
    is a second reason no verdict here may be read as confirmatory.
    """
    if not np.isfinite(delta) or not np.isfinite(low) or not np.isfinite(high):
        return "undetermined"
    favours_a = delta > 0 if higher_is_better else delta < 0
    excludes_zero = low > 0 or high < 0
    unanimous = seeds_favouring in (0, seed_count)
    direction = MODEL_A if favours_a else MODEL_B
    if not excludes_zero:
        return "interval covers zero"
    if abs(delta) < margin:
        return f"below the {margin:g} post-hoc margin"
    if not unanimous:
        return f"favours {direction}, seeds split {seeds_favouring}/{seed_count}"
    return f"favours {direction}, all {seed_count} seeds agree"


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

    # Holm inside each frozen family. Reported as a descriptive ordering
    # statistic only -- it is applied to an uncalibrated bootstrap tail area, so
    # it does not convert these into tested results and no verdict consumes it.
    for family, keys in FAMILIES.items():
        members = [r for r in records if r["family"] == family]
        if not members:
            continue
        adjusted = _holm([r["p_bootstrap"] for r in members])
        for record, value in zip(members, adjusted):
            record["p_holm_descriptive"] = float(value)
    for record in records:
        record.setdefault("p_holm_descriptive", None)
        margin = margins["MAE"] if record["metric"] == "MAE" else margins["MAPE"]
        record["post_hoc_margin"] = margin
        record["verdict"] = _descriptive_verdict(
            record["delta"], record["ci_low"], record["ci_high"],
            record["higher_is_better"], margin,
            record["seeds_favouring_alfnet"], record["seed_count"])
        if record["family"] is None:
            record["verdict"] += " (exploratory)"
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
    draws = crossed_bootstrap(tensors, seeds, args.bootstrap_samples,
                                   args.bootstrap_seed)
    margins = {"MAE": args.post_hoc_margin_mae, "MAPE": args.post_hoc_margin_mape}
    records = build_records(point, draws, seed_deltas, seeds, margins)

    print(f"MultiSeedPairedInference: {MODEL_A} vs {MODEL_B} | {args.dataset}")
    print(f"  seeds={seeds} | samples={args.bootstrap_samples} | "
          f"protocol={provenance[seeds[0]]['protocol_hash'][:12]}")
    print(f"  {'family':<12}{'row':<19}{'metric':<6}{'delta':>10}{'95% CI':>22}"
          f"{'holm*':>9}  seeds  verdict")
    for record in records:
        if record["family"] is None:
            continue
        interval = f"[{record['ci_low']:+.4f},{record['ci_high']:+.4f}]"
        print(f"  {record['family']:<12}{record['row']:<19}{record['metric']:<6}"
              f"{record['delta']:>+10.4f}{interval:>22}"
              f"{record['p_holm_descriptive']:>9.3f}"
              f"   {record['seeds_favouring_alfnet']}/{record['seed_count']}"
              f"   {record['verdict']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{args.dataset}_multiseed_paired.json"
    destination.write_text(json.dumps({
        "analysis": "three-seed crossed paired bootstrap, v3 post-hoc",
        "post_hoc_diagnostic_only": True,
        "inferential_status": "descriptive_not_formal",
        "dataset": args.dataset,
        "model_a": MODEL_A, "model_b": MODEL_B,
        "seeds": seeds,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "design": "seeds, overlap blocks and cohorts are crossed, not nested",
            "design_reason": "every seed replays one frozen evaluation design; the "
                             "row-design hash is asserted equal across seeds and models",
            "seed_draw": "multiplicity over the three seeds; a repeated seed is "
                         "weighted again, not resampled",
            "block_draw": "one stratified overlap-block sample per replicate, "
                          "shared by every drawn seed and both models",
            "cohort_draw": "one global multiplicity per replicate, likewise shared",
            "paired_across_models": True,
        },
        "point_estimate": "mean over seeds of (ALFNet - MoGERNN); equals the "
                          "difference of the two published three-seed means",
        "families": {name: [list(k) for k in keys] for name, keys in FAMILIES.items()},
        "post_hoc_margins": margins,
        "margin_status": "chosen after the v3 test split was read; post-hoc, not "
                         "pre-registered, and not usable as an acceptance threshold",
        "power_note": "n_seed=3. The exact two-sided sign-flip null over three "
                      "paired seeds has minimum p=0.25, so the seed factor cannot "
                      "reach 0.05 by any arrangement. Three draws with replacement "
                      "from three seeds give 27 ordered sequences but only 10 "
                      "distinct multiplicity vectors. Read the sign and the seed "
                      "agreement, not the width, and never as a formal test.",
        "holm_status": "descriptive ordering statistic only; applied to an "
                       "uncalibrated bootstrap tail area and consumed by no verdict",
        "provenance": provenance,
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()

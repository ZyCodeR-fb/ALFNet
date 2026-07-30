# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo actually is now

It started as the official MoGERNN implementation (see `README.md`, TR-C 2025). It has since become the
development repo for **ALFNet**, a lifecycle-aware traffic predictor evaluated against MoGERNN and a dozen
other baselines under three frozen protocols. MoGERNN is now the headline baseline, not the subject.

## Research objective

Real sensing networks are not a fixed node set under stable observation conditions. A sensor joins,
accumulates history, runs steadily, degrades and goes dark; at any instant a node may have zero history,
partial history, or history polluted by a long gap. ALFNet targets that regime with **two anchors and two
trajectories** — static `Unseen` and `Steady`; a join trajectory `Cold → Warm → Ready`; a fault trajectory
`Onset → Decay → Outage`.

The design intent: **graph propagation carries the low-evidence and no-history nodes, and a light
spatio-temporal conditioner corrects the graph representation once a node has enough of its own history.**

Hard constraints on the deliverable model:

* trained **from random initialisation, single-stage, end-to-end**;
* no teacher, no distillation, no frozen anchor, no fg1-initialised weights, no multi-stage schedule.

36/36 cells beating MoGERNN is an engineering target, never a reason to change a protocol or drop a
dataset. METR-LA has been inspected repeatedly and is a **development** set; it stays in the paper
regardless of the result. Final claims are confirmed on a second, un-inspected dataset (PEMS-BAY).

## Environment and commands

The conda env in `environment.yml` is named `moge`, but the working env on this machine is **`COMG`**.

```bash
conda run -n COMG python -m pytest tests -q --no-header        # full suite (~2-7 min)
conda run -n COMG python -m pytest tests/test_input_contract.py -q
conda run -n COMG python -m pytest "tests/test_abcd_switches.py::test_combined_switches_stay_finite" -q
```

`references/` holds vendored third-party source and is **not** collected by pytest — never add it.

**Tests must be self-contained.** No test may depend on a checkpoint or result directory outside the
working tree. Seven teacher tests used to load a checkpoint from `run_v5/` and broke the moment that
gitignored tree was cleaned up; build fixtures in `tmp_path`, or assert absence behaviourally.

Training and evaluation both go through `main.py`:

```bash
# train
python main.py -c configs/ALFNet/fg1.yaml --batch_size 64 --output_root ./outputs --run_id fg1-s32
# frozen-protocol test; repeat the *exact* training flags or exp_id will not resolve the checkpoint
python main.py -c configs/ALFNet/fg1.yaml --batch_size 64 --mode test_async \
  --output_root ./outputs --run_id fg1-s32
```

Post-hoc analysis:

```bash
python scripts/paired_model_bootstrap.py --model_a <predictions/lifecycle-v3-*/> \
  --model_b <...> --output_dir ./outputs/reports/a_vs_b
python scripts/collect_lifecycle_results.py <result roots> --output-dir <dir>   # expects seeds 32/42/52
python scripts/build_ablation_configs.py                                       # regenerates combination YAMLs
```

## Artefact layout

`--output_root` derives every path (`args.apply_output_root`): `outputs/<model>/<run_id>/{checkpoints,
predictions,manifests,logs,reports,figures,profiles}`. When it is set, `train.py` and the evaluator skip
appending `exp_id` again — the run folder already identifies the run, and `exp_id` is ~176 characters.
Without it the legacy flat layout (`check_points/`, `predictions/`, …) applies.

Everything result-shaped is gitignored. Git carries only source, YAML, scripts and tests. Old scattered
result directories (`run_v5/`, `_smoke/`, `outputs/`) stay on disk but must never re-enter the index.

## The three frozen protocols

Defined in `data_provider/lcbt_protocol.py` (training) and `data_provider/ples_protocol.py`
(**LCE v3** main/phase/trajectory tables, **PLES v3** paired event study). They are frozen; do not change
grouping, weighting or aggregation to move a metric.

**Naming.** `lifecycle_training_strategy: lcbt_v3` is the *config identifier* only. The formal name is
**Lifecycle Coverage-Balanced Training v3.1**, and the internal protocol number is
`LCBT_PROTOCOL_VERSION = 4` (`data_provider/lcbt_protocol.py:14`). All three differ on purpose; do not
"fix" one to match the others.

* Time split: train 0–50%, validation 50–60%, test 60–100%. The scaler is fitted on the training window
  and training-known nodes only. `validation_unseen` and `test_unseen` are **disjoint** subsets of the
  master unseen set (METR-LA: 20 and 30 nodes).
* Four training groups at 25% each: `zero_history, steady, join, fault`. Join/Fault cover evidence
  length `k = 1..L` uniformly with `L = 12`; `k1 = Cold/Onset`, `k2..11 = Warm/Decay`, `k12 = Ready/Outage`.
  The draw also balances calendar stratum, the 2.5/5/10% cohorts, clustered vs scattered, and degree strata.
* Six evaluation phases: `Cold/Warm/Ready` (join) and `Onset/Decay/Outage` (fault). Main rows:
  `Unseen, Steady, Join, Fault, LifeAvg, WorstGroup`. Each row reports MAPE/MAE/RMSE → 36 main+phase cells.
* `M_effective = M_natural & M_lifecycle`. The base forecast loss applies only on `target_cohort_mask`.
  The six phase labels are evaluation metadata and are never fed to the model.
* A `training_protocol_hash` is written into every manifest and checkpoint summary. Changing anything that
  enters `_manifest_payload` changes it and invalidates cross-run comparability — the loader refuses a
  mismatched manifest unless `--lcbt_overwrite_manifest`. **The hash is per dataset.** METR-LA's current
  value is `59c55e6ee4e59d2b237ba12ca1c449dd87a8b5ef71838fa523bdf6f936d0406b`; another dataset must derive
  its own and is *expected* to differ.
* Training re-draws windows every epoch (`_build_schedule(..., epoch=epoch)`); **validation uses a fixed
  schedule** (`epoch=0`). This is why offline caching of training targets does not pay.
* PLES resampling unit is the **overlap-connected origin block** (`overlap_blocks`), not the calendar day —
  adjacent origins share trajectory data. `trajectory_span = pre + input_len + post + pred_len`; it travels
  in the manifest and in `ples_effects_summary.json`, and readers refuse to fall back to per-origin draws.
  Pre-event windows are an implementation consistency check, not a reported effect.

**Checkpoint selection** is `stop_based: val_lifecycle`: `val_lifecycle_mae` is the equal-weight mean of
Unseen/Steady/Join/Fault, with worst-group MAE as the tie-break. It is part of the frozen protocol — do
not change it, and never change grouping, weighting, aggregation or checkpoint rules on the basis of a
test result.

## Porting to a second dataset

The loaders are dataset-generic: boundaries, hashes, node counts, unseen strata and cohort sizes are all
derived from the payload (`lcbt_protocol.py`), and `PEMSBAY` is already a valid `--dloader_name`. So a
second dataset is a config job, and its configs must be *twins* — identical outside the `dataset:` block,
because a confirmation dataset proves nothing if capacity or learning rates were retuned for it.
`tests/test_pemsbay_configs.py` enforces that.

PEMS-BAY files: `configs/ALFNet/fg1_bay.yaml`, `configs/ALFGraph/PEMSBAY_alfgraph_k1.yaml`,
`configs/ALFTemporal/PEMSBAY_cond1.yaml`, `configs/MoGERNN/PEMSBAY_lcbt_v3.yaml`.

One value is genuinely dataset-dependent and is **geometry, not tuning**: ALFTemporal's `spa_patchsize`.
`_reorder_data` pads a KD-tree part up to the patch size but never splits one, so the patch must hold the
largest part. At `recur_times: 5` METR-LA's 207 nodes give parts of 6–7 and PEMS-BAY's 325 give 10–11; the
convention is **largest part + 1** (so every part gets at least one augmented neighbour), which is 8 and 12
respectively. ALFTemporal now raises at construction with the required value; before that it built fine and
failed inside `WindowAttBlock`'s reshape.

## Input contract (fairness rule)

Every model receives the same 7-channel tensor: `[raw, role×4, age, obs_mask]` (built by
`compose_effective_lifecycle_features`). The lifecycle channels are **ALFNet's method inputs**. Published
baselines must reduce to their source-native inputs via
`model.comparison_model.family_utils.source_native_input`, and their config builders must size themselves
with `source_native_dim(args)` — not `args.enc_in`.

`tests/test_input_contract.py` enforces this **behaviourally**: it poisons the non-raw channels and requires
bit-identical output. A source read alone is insufficient — five baselines were consuming lifecycle channels
purely through a config-derived `input_dim` with no visible slice anywhere. The test also resets the seed per
call, because STFiT carries a VAE and a stochastic model otherwise looks like one that reads the channels.
`experiments/ples_evaluation.py::MODEL_INPUT_CONTRACT` records the audited state and is written into the
per-run result manifest.

A baseline that is *given* the lifecycle channels is a different model and must be reported under a
separate `+LC` name, never as the published baseline.

## ALFNet structure

`model/ALFNet/ALFNet.py` composes two experts built from their own YAML (`family.experts.{graph,patch}`):

* `ALFGraph` — Residual-Delta DCRNN over a Lifecycle-MoGE routed graph conv (`layers/lifecycle_moge.py`),
  `graph_conv=lifecycle_moge`, 5 propagation supports, top-k=1 sparse routing, decoder predicts a delta at
  residual scale 0.3. **This is the prediction backbone** — the final forecast always comes from its
  decoder. It exposes a detached `encoder_state` for downstream conditioning. Its internal GRIN is off, so
  recovery happens once in the shared imputer.
* `ALFTemporal` — a **light spatio-temporal conditioner**, not an independent forecaster. Simplified from
  PatchSTG to one layer with TimeFilter off and `emit: features`, so it produces no forecast of its own.
  With `tem_patchsize=12, tem_patchnum=1` the stride-12 embedding conv collapses the whole input window
  into a single time token, so its attention is **spatial** despite the name. Its output FiLM-modulates
  ALFGraph's encoder hidden state.
* One `shared_input_imputer` (GRIN) feeds both.

Behaviour is selected by `family.recipe` keys, all optional and all defaulting to the historical path:
`fusion_mode` (`residual_gate` | `film`), `evidence_router`, `observed_nlinear`, `temporal_residual`
(`none`|`tcn`|`dlinear`), `companion_loss_weight`, `freeze_anchor`. Patch-side switches live in the
ALFTemporal YAML: `spatial_tokenizer`, `mixer`, `use_timefilter`, `emit`.

`ProtectedTemporalResidual` (`layers/protected_residual.py`) is **shared**, not distillation-specific: the
frozen ablations `rt/rd/rtg` and the end-to-end candidate `jr` all use it, and `companion_loss_weight` /
`freeze_anchor` are theirs. Only the teacher was removed.

Two invariants worth knowing before editing:

* **Optional modules are constructed inside `_isolated_rng`** with a fixed per-module offset. Built on the
  shared stream they shift every later module's init — enabling one switch moved `graph_refiner.input_proj`
  by up to 1.01 under the same seed, which silently turns an ablation into two unrelated models.
* **Gates that protect a group must be hard zero, not learned.** A node with no own-history has nothing for
  a temporal branch to read; `protection = clamp(e_self/floor, 0, 1)` keeps the Unseen path bit-identical to
  the anchor, and tests assert `torch.equal`.

Training-only knobs live on the CLI: `--lcbt_context_loss_weight`, `--lcbt_stratum_balanced_loss`,
`--lcbt_tail_weight`, `--lcbt_worst_group_weight`. All are folded into the experiment signature
(`main.TRAINING_OBJECTIVE_KNOBS`) **only when non-default**, so historical `cfg` hashes stay byte-identical
while two different objectives can never share a checkpoint directory.

## Measurement pitfalls (learned the expensive way)

* **Do not rank architectures by validation LifeAvg.** Checkpoint selection still follows the frozen
  `val_lifecycle` rule — that is not in question. But for *choosing between candidates*, METR-LA's
  `validation_unseen` (20 nodes) is small and disjoint from `test_unseen` (30 nodes), and its ordering has
  not transferred; since LifeAvg is a quarter Unseen, LifeAvg inherits the problem. Rank on **validation
  Steady + Join + Fault**, and keep Unseen as a non-degradation constraint and a diagnostic — it is a real
  signal about the no-history path, just not a reliable cross-architecture ordering on this dataset.
  Worked example on disk: `ad-s32` beat `fg1-s32` on validation LifeAvg (4.637 vs 4.823) purely through
  validation Unseen (5.92 vs 7.09), lost on validation S+J+F (mean 4.211 vs 4.066), and then lost on the
  test set as S+J+F predicted (LifeAvg MAE 5.147 vs 4.927).
* **Validation jitter is candidate-dependent — read it from that run's `training_summary.json`.**
  `validation_group_volatility` records per-group std and max epoch-to-epoch jump for every run. `fg1-s32`
  sits at std 0.24–0.45 / max jump 0.52–1.08; a frozen-anchor run like `kt-s32` is near zero because its
  protected groups cannot move. Do not carry one run's numbers over to another. Whatever the run's own
  figure is, candidate effects of 0.01–0.25 MAE from a single seed and a single checkpoint are not evidence.
* **METR-LA's test split has been inspected many times and is now a development set.** Final claims need a
  second dataset. Never tune on it, and never run it to break a validation tie.
* Local GPU step-time measurements are unstable (6× outliers, a different variant each run) and bs64 hits
  this card's memory ceiling. Profile at bs16, include an unchanged control variant in every profiling run,
  and treat server `s/epoch` as the authoritative speed signal. Parallel server runs cannot be compared on
  wall-clock at all.
* Auxiliary loss terms must be scaled against the measured base loss, not guessed. Base MSE is ~0.157 while
  the tail term is ~1.18 — a weight of 0.1 is *not* small, it is 75% of the objective.

## Current state

`fg1` (`configs/ALFNet/fg1.yaml`) is the mainline and satisfies the end-to-end constraint: FiLM-conditioned
graph anchor, 1,041,561 params, **32/36** main+phase cells beating MoGERNN on METR-LA (verifiable from
`outputs/{ALFNet/fg1-s32,MoGERNN/mg-s32}/predictions/lifecycle-v3-08dbd1e121/lce_*_results.txt`). The four
misses are Join MAE, Warm MAE, Ready MAE, Ready RMSE — all on the join track, all on nodes that do have
their own history. The paired-bootstrap significance for this pair lives on the server; the local
`reports/paired_bootstrap/` copy is from an older model pair and an older protocol, so do not quote it.

`kt` reached **35/36** (only Ready RMSE missing) by distilling the retired `rf_l3s`/A2 teacher into a frozen
two-stage schedule with 15,317 trainable parameters. That violates the end-to-end constraint, so it is a
**recorded upper bound and not an active path**: the numbers stay in `outputs/ALFNet/kt-s32`, and the code,
configs and teacher checkpoint are gone. ALFNet now *raises* on `lambda_kd`, `teacher_checkpoint` or
`teacher_config_path` rather than ignoring them, so a stale YAML cannot reintroduce a two-stage run under a
mainline name. Do not reintroduce distillation to chase 36/36, and do not reconstruct the teacher.

Unrelated but easy to confuse: `ALFGraph`'s `teacher_forcing_ratio` is DCRNN scheduled sampling and has
nothing to do with distillation. Leave it alone.

Candidates that have been run and did not improve `fg1` on validation S+J+F: `a b c d` (and combinations),
the frozen label-residual family `rt rd rtg`, the training-objective knobs `tq wg tw` (tail weight,
worst-group weight, and both), and `jr` — the end-to-end form of the protected additive residual, which
lost on **all four** validation groups (S+J+F mean 4.282 vs 4.066) with its gate measurably open
(`residual_gate_mean` 0.42), so the branch was active and harmful rather than switched off. Additive
correction over the FiLM anchor is now refuted in both the frozen (`rt/rd/rtg`) and joint (`d`, `jr`)
forms; do not propose a third. For `tq/wg/tw` the 0.1 weight turned out to be far too strong relative to a
base MSE of ~0.157; a finer sweep is deliberately *not* being run. Also decided and closed: no CAGrad or
PCGrad (24 real-batch gradient probes showed all six group-pair cosines positive — no systematic conflict),
no extra gating (FiLM is measurably active), no distribution-shift module (no evidence of shift yet).

**Cross-dataset status: open.** The PEMS-BAY seed-32 pair has been run (`fg1bay-s32`, `mgbay-s32`, both
under hash `e9684b35…`). It is *not* a confirmation yet and the finding is not settled, so no conclusion is
recorded here. Two operating rules do follow from it and are stable:

* **PEMS-BAY test has not been run and must stay unrun** until a candidate wins on PEMS-BAY *validation*.
  Its value is that it has not been inspected; spending it on a validation-stage loss destroys that.
* **Never retune a PEMS-BAY config to close a gap.** The twin rule (`tests/test_pemsbay_configs.py`) is the
  point of the experiment. A real consequence to state in the paper rather than fix: the twins inherit
  hyper-parameters selected on METR-LA, so a transfer gap is partly a hyper-parameter transfer gap.

The diagnostic worth knowing before touching the conditioner: on METR-LA `fg1`'s validation Join MAE falls
~0.99 from `k1` to `k12`, i.e. the conditioner uses accumulating own-history. On PEMS-BAY the same curve is
**flat** (+0.03) while source-native MoGERNN's falls 0.28 — and `fg1bay` early-stopped with its best
checkpoint at epoch 4 of 14, against MoGERNN's epoch 25 of 35. Whether that is under-training plus the
equal-weight `val_lifecycle` rule locking in an early Unseen minimum, or the conditioner genuinely failing
at 325 nodes, is unresolved; `train.py` keeps the per-epoch group curve in `validation_group_history` but
only writes the aggregated volatility, so answering it needs the run's stdout log.

## Housekeeping

Failed candidates are ablation evidence **until the architecture is frozen**. Once it is, delete each dead
candidate's YAML, code branch and tests together — git history and the written reports keep the negative
result, and the repo should not become a candidate archive.

## Keeping this file honest

This file holds stable facts and operating rules, not a per-round experiment log. Update it, in the same
task that causes the change, only when one of these changes:

* the mainline config or the model's declared boundary;
* the training / validation / test protocol;
* the input contract, artefact paths, or the standard commands;
* a candidate is promoted to mainline or permanently retired;
* a conclusion is confirmed across datasets.

Do not write provisional results, guesses, or unconfirmed candidates here.

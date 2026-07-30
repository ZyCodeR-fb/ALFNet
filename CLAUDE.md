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
python scripts/build_router_configs.py                                         # regenerates gs/g1 + twins
python scripts/router_report.py outputs/ALFNet/<run_id>    # per-layer, per-phase, per-k routing
```

## Artefact layout

`--output_root` derives every path (`args.apply_output_root`): `outputs/<model>/<run_id>/{checkpoints,
predictions,manifests,logs,reports,figures,profiles}`. When it is set, `train.py` and the evaluator skip
appending `exp_id` again — the run folder already identifies the run, and `exp_id` is ~176 characters.
Without it the legacy flat layout (`check_points/`, `predictions/`, …) applies.

`utils/run_logger.py` tees stdout **and** stderr into `<run>/logs/{train,test_async}_<timestamp>.log`,
installed by `main.py` before anything prints. This matters beyond convenience: the per-epoch lifecycle
group numbers exist *only* on stdout — `training_summary.json` keeps the aggregated
`validation_group_volatility` and discards the curve it came from — so without the log the single most
useful diagnostic is unrecoverable once the terminal closes. It tees rather than redirects, so a tmux
session still shows progress, and the timestamped name stops a later `test_async` run from truncating the
training log that explains it.

Everything result-shaped is gitignored. Git carries only source, YAML, scripts and tests. Old scattered
result directories (`run_v5/`, `_smoke/`, `outputs/`) stay on disk but must never re-enter the index.

## The three frozen protocols

Defined in `data_provider/lcbt_protocol.py` (training) and `data_provider/ples_protocol.py`
(**LCE v3** main/phase/trajectory tables, **PLES v3** paired event study). They are frozen; do not change
grouping, weighting or aggregation to move a metric.

**Naming.** `lifecycle_training_strategy: lcbt_v3` is the *config identifier* only. The formal name is
**Lifecycle Coverage-Balanced Training v3.2**, and the internal protocol number is
`LCBT_PROTOCOL_VERSION = 5` (`data_provider/lcbt_protocol.py`). All three differ on purpose; do not
"fix" one to match the others.

**The three protocols version independently.** LCBT is at v3.2 / version 5; **LCE and PLES are both still
version 3 and their hashes are unchanged**. Only the part whose semantics actually moved was bumped —
there is no requirement, and no benefit, in keeping the numbers tidy.

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
* Join/Fault evidence lengths come from a **two-dimensional quota** (calendar regime × k). Each regime gets
  `n_r // L` of every k plus a run of extras taken from a cursor carried across regimes, so both margins
  are exact: within a regime and globally, `max - min <= 1`. Measured on the reference schedules —
  METR-LA 356–357 per k, PEMS-BAY 542–543, validation replay exactly 89 and 137. The v3.1 code allocated
  inside each (scenario, regime) cell independently and `_allocate_counts` breaks ties by index, so every
  cell gave its remainder to the smallest k: simulated over twenty realistic cells the counts ran
  monotonically 377 at k=1 down to 357 at k=12.
* A `training_protocol_hash` is written into every manifest and checkpoint summary. Changing anything that
  enters `_manifest_payload` changes it and invalidates cross-run comparability — the loader refuses a
  mismatched manifest unless `--lcbt_overwrite_manifest`. **The hash is per dataset.** Current values:

  | | training (LCBT v3.2) | LCE v3 | PLES v3 |
  |---|---|---|---|
  | METR-LA | `59c9ad67f2dc9c51…` | `659e3583c978a68a…` | `d6c09455e7273d55…` |
  | PEMS-BAY | `caa7dbd3507843c0…` | `b8eeda69790e97ea…` | `68401afbae31d851…` |

  The LCE and PLES hashes are unchanged from v3.1; only the training hash moved.
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

## Strict inductive: no node may own a trainable parameter

A held-out node is one the model never trained on, so it must not carry a parameter indexed by node id.
Every PatchSTG-derived backbone did: `nn.Parameter(node_num, node_dims)`. Measured on the old `fg1` after
five optimiser steps, **all 50 held-out rows took gradient** — with a *larger* mean magnitude than the
known rows — and perturbing only those rows moved the known-node forecast by 1.056. The mechanism is
`_scatter_to_full_nodes`: known nodes are scattered into a full-graph tensor, spatial attention mixes
every node, and only the known ones are gathered back.

`layers/structural_node_encoder.py` replaces the bank with one MLP shared by every node over seven frozen
features — normalised coordinates, weighted in/out/total degree, log total degree, isolated flag. No node
owns anything; the mapping generalises, the identity does not. Coordinates degrade to zero when a dataset
ships none. Lifecycle features are deliberately absent: they are ALFNet's method inputs and must not reach
a baseline.

Adapted: **ALFTemporal, PatchSTG, STID, SLPF, GWNET, STBP**. Audited clean: MoGERNN, FCLSTM, HA,
MCPST_FSL, STFiT, STGCN, STGODE. Two details worth keeping:

* GWNET uses **two independent** encoders for the left and right factors of its adaptive adjacency. One
  encoder reused on both sides gives `softmax(relu(E E^T))`, a symmetric operator — a different model.
* `SLPF.rank_emb` is sized by node count but indexed by a *data-derived rank*
  (`torch.argsort(local_values)`, recomputed per sample and timestep), so no node owns a row. It is the
  only entry in `JUSTIFIED_NODE_SHAPED_PARAMETERS`; adding another requires writing down why.

`StructuralNodeEncoder` carries a `structure_hash` over the raw adjacency, the positions **and** the
derived features. Hashing only the features collided: they are coordinates plus degree summaries, so two
graphs with the same coordinates and degree sequence but different edges produced the same digest.

## MoGERNN's batch adapter

The source detects unknown nodes with `sum(x_enc, dim=(0,1)) == 0` — a node is unknown only if it is zero
across the **whole batch**. That is correct where one fixed sensor set is unobserved for the entire
dataset, which is what MoGERNN was written for. LCBT hides a different cohort per sample, so the batch-wide
intersection is usually empty: measured on METR-LA with disjoint cohorts, B=1 finds 8 unknown nodes and
**B>=4 finds 0**. Training and validation therefore ran with the completion switched off while
`test_async` (batch size 1) ran with it on.

`DataPreparation` now applies the original rule per sample. At B=1 it is bit-identical to the reference,
and a mixed batch equals the concatenation of its samples. This costs ~205 ms per `data_pre` call at B=64
on CPU; a strictly-equivalent vectorisation is possible but must pass the B=1 and mixed-cohort equivalence
tests before it replaces the loop.

## Checkpoint provenance

A checkpoint records `model`, `model_variant`, `router_mode`, `canonical_config_hash`, the three protocol
hashes and `structural_fingerprints`, and the loader refuses a mismatch. This is not paranoia: `fg1`, `gs`
and `g1` are byte-identical in structure and differ only in the router mode, so loading the wrong one was
silent.

`utils/canonical_config.py` freezes the config hash **when parsing finishes**, from an explicit allowlist.
The previous implementation hashed `vars(args)` at checkpoint time, which is a live namespace the trainer
writes `best_val_metric`, per-epoch history and resolved paths onto — so it identified the run, not the
configuration. The hash inlines the nested expert YAMLs, which is what makes `fg1`/`gs`/`g1` distinct, and
excludes `output_root`, `run_id`, `device`, `num_workers` and every runtime field.

A checkpoint predating provenance needs `--allow_legacy_checkpoint`, and results from one are not
comparable.

## ALFNet structure

`model/ALFNet/ALFNet.py` composes two experts built from their own YAML (`family.experts.{graph,patch}`):

* `ALFGraph` — Residual-Delta DCRNN over a Lifecycle-MoGE routed graph conv (`layers/lifecycle_moge.py`),
  `graph_conv=lifecycle_moge`, 5 propagation supports, top-k=1 sparse routing, decoder predicts a delta at
  residual scale 0.3. **This is the prediction backbone** — the final forecast always comes from its
  decoder. It exposes a detached `encoder_state` for downstream conditioning. Its internal GRIN is off, so
  recovery happens once in the shared imputer. **Its router is broken at `moge_top_k: 1` — see below.**
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

### The top-1 router is not trained by the forecast loss

`moge_router_mode` selects `noisy_topk` (default, historical), `soft`, or `st_top1`. The default is kept so
existing runs reproduce, but at `moge_top_k: 1` it is **provably untrainable from the prediction
objective**: `top_k_logits` is a single column, `nn.Softmax(1)` of one element is identically 1.0 with
derivative 0, and the top-k indices carry no gradient. The only path to `w_gate` is the load-balance term —
a uniformity prior that knows nothing about prediction error. Measured, not inferred:

* forecast-only gradient on `w_gate`: **exactly 0.0** at k=1, non-zero at k=2 and k=3, and 0.0 for all 12
  routers through the full ALFNet;
* on the shipped `fg1-s32` checkpoint the injected training noise is **~586×** the learned logits, and
  training-time routing agrees with eval-time routing **20.0%** of the time — chance for five experts;
* `scripts/router_report.py` on the same checkpoint: `entropy_ratio` 1.0000 on every layer and phase, with
  `top1_margin` between 2e-6 and 2.6e-4. The routing table looks specialised per phase and is a coin flip.
* top-1 buys no compute either: all five supports are evaluated before the gates are applied (2.165 ms at
  k=1 against 2.222 ms at k=5), so sparsity here costs differentiability for nothing.

So `fg1`'s headline mechanism currently behaves as expert *dropout*, not as routing. Both candidate modes
are deterministic — an ST-Gumbel variant was measured at 33% train/eval agreement and rejected for
reproducing the defect in milder form. All three modes carry an identical parameter count, so the
candidates are pure ablations. Both were run under the old protocol and neither achieved lifecycle
specialisation: per-state entropy stayed at ~0.99 for `soft` and ~0.9996 for `st_top1`, whose hard argmax
sat on margins of 2e-4 to 1.4e-2, so its 3:1 usage split was an argmax artefact rather than confident
routing. Those runs are superseded; treat the router question as open, not as settled either way.

Diagnostics: per-layer expert usage and entropy are collected at eval only and land in
`training_summary.json` under `routing_diagnostics` as `router/<layer>/…`; the phase-by-k table comes from
`scripts/router_report.py`, which reconstructs it from `w_gate` because the router's only input is the
lifecycle state and LCBT makes that an exact function of (scenario, k).

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

**Every result on disk is invalid and everything must be retrained.** Three corrections landed that change
what the models are and what they were trained on:

* strict inductive — parameter counts and `state_dict` changed for ALFTemporal/ALFNet (1,041,561 →
  1,037,529), PatchSTG, STID, SLPF, GWNET (300,920 → 299,104), STBP (351,352 → 270,072);
* MoGERNN's batch adapter — its completion was off during training and on at test;
* LCBT v3.2 — the training schedule and therefore `training_protocol_hash` moved on both datasets.

So `fg1-s32`, `gs-s32`, `g1-s32`, `gsbay-s32`, `g1bay-s32`, `jr-s32`, `ad-s32`, `kt-s32`, `rg2-s32`,
`mg-s32` and `mgbay-s32` are all superseded. The files stay in `outputs/` as a record of the old protocol;
none of their numbers may be quoted against a new run. **Every architectural conclusion below predates
these fixes and is provisional until re-measured.**

`fg1` (`configs/ALFNet/fg1.yaml`) is the mainline and satisfies the end-to-end constraint: FiLM-conditioned
graph anchor. Under the old protocol it reached **32/36** main+phase cells against MoGERNN on METR-LA, with
the four misses on Join MAE, Warm MAE, Ready MAE and Ready RMSE — all on the join track, all on nodes that
do have their own history. Whether that survives the corrections is unknown.

`kt` reached **35/36** (only Ready RMSE missing) by distilling the retired `rf_l3s`/A2 teacher into a frozen
two-stage schedule with 15,317 trainable parameters. That violates the end-to-end constraint, so it is a
**recorded upper bound and not an active path**: the numbers stay in `outputs/ALFNet/kt-s32`, and the code,
configs and teacher checkpoint are gone. ALFNet now *raises* on `lambda_kd`, `teacher_checkpoint` or
`teacher_config_path` rather than ignoring them, so a stale YAML cannot reintroduce a two-stage run under a
mainline name. Do not reintroduce distillation to chase 36/36, and do not reconstruct the teacher.

Unrelated but easy to confuse: `ALFGraph`'s `teacher_forcing_ratio` is DCRNN scheduled sampling and has
nothing to do with distillation. Leave it alone.

Candidates run **under the old protocol** that did not improve `fg1` on validation S+J+F — all of these
predate the strict-inductive and MoGERNN corrections, so they are prior evidence rather than settled
results: `a b c d` (and combinations),
the frozen label-residual family `rt rd rtg`, the training-objective knobs `tq wg tw` (tail weight,
worst-group weight, and both), and `jr` — the end-to-end form of the protected additive residual, which
lost on **all four** validation groups (S+J+F mean 4.282 vs 4.066) with its gate measurably open
(`residual_gate_mean` 0.42), so the branch was active and harmful rather than switched off. Additive
correction over the FiLM anchor is now refuted in both the frozen (`rt/rd/rtg`) and joint (`d`, `jr`)
forms; do not propose a third. For `tq/wg/tw` the 0.1 weight turned out to be far too strong relative to a
base MSE of ~0.157; a finer sweep is deliberately *not* being run. Also decided and closed: no CAGrad or
PCGrad (24 real-batch gradient probes showed all six group-pair cosines positive — no systematic conflict),
no extra gating (FiLM is measurably active), no distribution-shift module (no evidence of shift yet).

**PEMS-BAY is the second development dataset, not a zero-tuning confirmation set.** It has been used to
compare `gs` against `g1`, so its validation split has informed a design choice and cannot be described as
untouched. A genuinely external confirmation now needs a third dataset that has taken no part in any
selection. Its *test* split has still never been run, and two operating rules follow:

* **PEMS-BAY test has not been run and must stay unrun** until a candidate wins on PEMS-BAY *validation*.
  Spending it on a validation-stage loss destroys the only untouched split left on this dataset.
* **Never retune a PEMS-BAY config to close a gap.** The twin rule (`tests/test_pemsbay_configs.py`) is the
  point of the experiment. A real consequence to state in the paper rather than fix: the twins inherit
  hyper-parameters selected on METR-LA, so a transfer gap is partly a hyper-parameter transfer gap.

**The strongest prior observation, now provisional.** Under the old protocol `fg1`'s validation Join MAE
fell ~0.99 from `k1` to `k12` on METR-LA but was flat or inverted at every epoch on PEMS-BAY, while
source-native MoGERNN held −0.25 to −0.75 throughout — the conditioner was not using accumulating
own-history there. Two explanations were ruled out at the time: it was not under-training (the response
was never present after epoch 2) and not the checkpoint rule (re-selecting by S+J+F still left MoGERNN
ahead, 2.4058 vs 2.4253).

Per-phase, `fg1` beat MoGERNN on **Cold** and **Outage** on both datasets and lost **Warm** and **Ready**
on both at near-identical magnitude — the same four cells that missed on METR-LA — suggesting the
join-track weakness was a property of the architecture rather than of PEMS-BAY.

**All of it was measured with the leaking node embeddings and the disabled MoGERNN completion in place**,
and the MoGERNN side of every comparison was handicapped. It is the best hypothesis to re-test first, not
a result to build on. `scripts/validation_curve.py` prints the k1→k12 slope per epoch, which is the
diagnostic to re-run once the new pair is trained.

## Protocol changes that were considered and declined

A round of protocol suggestions was raised and, on independent review, **not implemented**. They are
recorded here so they are not re-proposed as defects: each would make the protocol more balanced or more
complete, but the existing protocol is sound without them, and changing an evaluation definition costs
every result on disk.

* **Per-cell natural-availability thresholds** (input and target both >= 0.30). Pooled masked loss and
  masked metrics already skip cells that cannot be computed. A threshold is a defensible design choice,
  not a correctness fix, and it would silently drop data from the main tables.
* **Natural-outage stress table**, **origin-specific LCE cohorts**, **node exposure `max-min <= 1`**.
  Coverage improvements. The current cohort draw is valid; wider coverage would reduce variance, not
  remove a bias.
* **Nested bootstrap in place of the crossed design.** The crossed cohort draw matches what the manifest
  declares, so implementation and manifest agree. Changing it without changing the design would be the
  inconsistency.
* **Per-sample cohort loss.** Implemented briefly, then reverted. Pooling every valid cell weights a
  sample by its cohort size, and re-weighting to equal-per-sample is a change to the training objective,
  not a fix — the training loss is not required to match the evaluation weighting. It also makes the
  gradient noisier on the 2.5% cohorts (about 48 cells per sample), an unmeasured risk.
* **LCE/PLES v4 and a wide YAML rename.** Nothing in LCE or PLES was shown to be wrong; renaming for
  version tidiness is churn.

The one suggestion that *was* adopted is the k quota, because it contradicted a manifest field
(`event_evidence_distribution: uniform_k=1..L`) by construction rather than by rounding, and the fix
changes no metric definition.

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

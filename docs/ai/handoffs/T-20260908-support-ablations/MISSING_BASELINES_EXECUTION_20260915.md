# Missing baseline execution — 2026-09-15

## Facts and authority

User explicitly authorized `paper/baseline_adaptation_codex_prompt.md`: six
validation-only seed32 slots, wait for old four, then GPU preflight and concurrent
formal training. Its real protocol-conflict exception applies to GCN-M only.
No test/commit/push, no old run restart/termination, no paper table or final model
selection change. Checked Git: paper/final, 9346dc0fc1dfb6b4aebfffe25224f1337f7a94cb,
upstream0:0; all pre-existing dirty changes retained.

Plan: `725a80ed153317b77834823ee42b696e623f65848f59357cfa5d55427a79c8bd`.
Executable frozen root, relative to the repository:
`workspaces/missing_inductive_baselines_frozen_725a80ed153317b7/`.
Plan file inside it: `configs/paper/missing_inductive_baselines_seed32_v1.json`.
Preparation directory is `workspaces/missing_inductive_baselines_20260915/`;
running jobs do not import the preparation directory or root training source.
Read-only data references are reused by symlink. No new source under outputs_v4.

[Detailed source audit](../../../../workspaces/missing_inductive_baselines_frozen_725a80ed153317b7/source_audit/ADAPTATION_AUDIT.md)
records the user-copy comparison, missing files, source recipes, protocol changes
and original-source inventory. This relative link is resolved from this document's
directory; the frozen root above is also a repository-relative location.

## Completed

- All six YAMLs and slots frozen; four executable, two protocol-blocked. Builder
  `--check` passed in the frozen root. Old root BAY builder `--check` also passed,
  confirming its source/config identity was not overwritten.
- Four actual METRLA/PEMSBAY × GinAR/UIGNN CPU sessions: finite output/loss,
  backward, source optimizer update, future-target/context/hidden-input blindness,
  all-missing handling, checkpoint prediction equality and production last.pt
  resume. Raw hidden-signal poison uses synthetic data with real full-N graph,
  the actual loader and scaler; no held-out test observations are evaluated.
- Source-core parity: GinAR forward max abs2.84e-14, parameter gradient9.09e-13
  (float64). UIGNN aligned head/body gradient tolerance1e-12. Declared structural
  parameterization and independent multistep quantile head are not falsely called
  whole-model source equivalence.
- Nine targeted tests passed: probability gradients/legal masks, source Adam/lr
  schedule, resume scheduler ownership, terminal/active-worker dependency guards,
  restart/orphan protection, no forbidden imports and shared 33-cell reductions.
- Certificates/logs: frozen `verification/cpu_preflight.json`, `cpu_passed.log`,
  `metric_tests.json`, `metric_tests.log`. Failed harness attempts remain archived.
- Restart guard revision preserves byte-identical training/model/loader/loss/
  optimizer/collector code. `verification/revision_check.json` records the two
  changed Python files. The superseded ede4a845… plan, source and waiting state
  remain preserved; it launched **no GPU work** and its own waiter was retired.

## Source adaptations and blockers

| Model | Source → adapter | Outcome |
|---|---|---|
| GinAR | model1/ginar_arch.py + ginar_cell.py → comparison_model/GinAR | Source recurrent/interpolation/learned-graph/decoder core; ordinary shared structural node parameterization; equivalent additive attention factorization. Adam .006, batch16, clip5, source milestones. |
| UIGNN | structure_full.py + EDL_loss.py + quantile_reg.py → comparison_model/UIGNN | Missing files restored; source default quantile + reconstruction alpha1 retained. Independent12×3 head and proper pinball fix; point median. Adam .001, batch4. |
| GCN-M | models/model.py + gnn.py + memoryModule.py + official retrieve_hist | User copies correct, reference data module incomplete. Genuine external recent/day/week x_hist exceeds12-step budget. No legal zero-history config found; both runs blocked, no fake runnable adapter. |

GCN-M synthetic source probe: input5000…5011, memory1×60×12×1, indices962…5010,
654/720 values outside current input. Zero-count variants raise official
stack/concatenate errors. METRLA source is GCNMdynamic; BAY source is GCNM.
Changing the input-history permission needs a separate explicit protocol decision
and new plan, not a silent fix. This does not block GinAR/UIGNN.

## Current old dependency observation

Host verified at 2026-09-15 00:58 +0800. Values expire; scheduler rechecks every30s.

| Old job | Verified PID | Latest completed epoch (zero-based) | Actual state |
|---|---:|---:|---|
| spectral32 | 3300898 | 39 | running train epoch40 |
| linear32 | 3301018 | 39 | running train epoch40 |
| spectral42 | 3301482 | 39 | running train epoch40 |
| linear42 | 3301617 | 39 | running train epoch40 |

All four actual cwd/commands point to root pemsbay_transfer_worker and the
34ecd952… plan. best.pt/last.pt exist; COMPLETE, final report and selected table
are not yet present. GPU0 utilization96%, free29050MiB; no old termination claimed.
Logs remain under `outputs_v4/pemsbay_focal_transfer_v1/34ecd952b01ebbad540a1345260d2793f37e4b71f743c817cc492e6c410b3a0a/logs/`.

## Persistent scheduler and new task state

tmux: `missing-baselines-20260915`. Verified scheduler PID **3509128** at00:58.
Command: `/home/xd/anaconda3/envs/ReaCH/bin/python -u scripts/run_missing_inductive_baselines.py`
from the frozen source root. Use current host identity, not this PID alone.

New output root:
`outputs_v4/missing_inductive_baselines_seed32_v1/725a80ed153317b77834823ee42b696e623f65848f59357cfa5d55427a79c8bd/`.
It contains `plan.json`, `source_snapshot.json`, `scheduler.lock`, `scheduler.log`
and the authoritative `scheduler_state.json`.

| New slot | Seed | Worker PID / epoch / log | Status at00:58 |
|---|---:|---|---|
| METRLA GinAR | 32 | none; formal log not created | waiting_dependency |
| METRLA GCN-M | 32 | none | protocol_blocked |
| METRLA UIGNN | 32 | none; formal log not created | waiting_dependency |
| PEMS-BAY GinAR | 32 | none; formal log not created | waiting_dependency |
| PEMS-BAY GCN-M | 32 | none | protocol_blocked |
| PEMS-BAY UIGNN | 32 | none; formal log not created | waiting_dependency |

After verified old terminal artifacts/exit codes and no actual old worker in two
observations, the scheduler launches sequential **single-batch admission probes**,
then **four concurrent short training/validation probes**, then four concurrent
formal runs at native batch sizes. Target six is not represented as achieved:
the two GCN-M slots remain paused for the explicit protocol conflict. No global
training/validation phase semaphore is added. Probe results live separately in
`outputs_v4/missing_baselines_preflight_725a80ed153317b7/`. Peak GPU memory,
phase intervals and overlap prove actual concurrent training, not only processes.

Single/concurrent/formal logs are created under the new output's `logs/`, named
`DATASET_MODEL_MODE.log`. Formal outputs are dataset/model/configSHA-prefix/seed32,
with best.pt, recoverable last.pt, epoch history, run_report.json,
development_validation_table.json (33 metrics) and COMPLETE. All use strict
validation LifeAvg MAE, earlier ties, max100/patience10, and no test phase.

## Modified but unverified / next action

GPU preflight, resource capacity for concurrent probes, actual new training
throughput and final baseline results are **pending**. The full repository test
suite was not run. UIGNN quantile calibration is not established. This handoff
does not promise a completion time or infer quality from CPU warmup losses.

Continue by reading scheduler_state/log and checking its actual host process.
Do not start another launcher while it owns the lock. A dispatched/terminal
scheduler restart fails closed; inspect jobs and durable checkpoints before any
recovery. A waiting-only restart is permitted only with no active own worker.
Engineering failures preserve records; no result-driven recipe changes or extra
seeds. Do not revive the superseded ede4a845… scheduler.

## Ruled out

No hidden/full validation/test history cache, no repeated/zero fake GCN-M memory,
no ALFNet role/age/context conditioning or auxiliaries in these baselines, no
GWNET substitute, no old final_plan authority, no root-source overwrite, no
external-process action, no test, no commit/push or automatic final-model choice.

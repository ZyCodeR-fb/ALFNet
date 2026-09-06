# Project context

## Purpose and current scope

This repository studies lifecycle-aware, topology-known but signal-inductive traffic
forecasting. The current research question is whether ALFNet improves forecasting for
nodes that are unseen, joining, steady, or faulted under a common data and evaluation
contract. The latest explicitly selected paper shortlist contains nine models:
ALFNet, MoGERNN, HA, FCLSTM, STGCN, STGODE, STID, SLPF, and STBP.

## Accepted protocol contract

- Datasets: PEMS03, PEMS04, PEMS08 (flow, target `q`, payload channel 0), and
  METR-LA, PEMS-BAY (speed, target `v`, payload channel 2).
- Input/horizon 12/12, stride 1, chronological 60/20/20 time split.
- Full-N topology is visible; dynamic signals are scoped to K in training, K∪V in
  validation, and K∪T in test. The scaler is fit on training time and K only.
- Four lifecycle scenarios are sampled equally: zero-history, steady, join, fault.
- Run seeds are 32/42/52; validation design seed is 12026.
- Primary reporting contains 33 values per model/dataset: five main rows plus six
  phase rows, each with MAE/RMSE/MAPE. WorstGroup is not part of the primary table.
- Checkpoint selection uses validation LifeAvg MAE with strict improvement and the
  earlier epoch on ties. Test is not used for method or configuration selection.
- Flow and speed MAE/RMSE are not aggregated together. MAPE on low-flow cells is a
  known limitation that must be reported honestly.

The executable protocol constants are primarily in `lib/lcbt_v4.py`; data identity
and exposure checks are in `lib/runtime_context.py` and
`data_provider/v4_lifecycle_loader.py`; metric rows and arithmetic are in
`data_provider/v4_metrics.py`; model/training dispatch and checkpoint reporting are
in `experiments/v4_runner.py` and `scripts/train_v4.py`.

## Current implementation

At HEAD `cf8f259ebc7fcc4e51d9ae362cf14443ca560191`, ALFNet's accepted baseline is
the `unified_BASE` architecture with the AR_BASE training/decoder recipe used by the
latest seed32 runs. Its active implementation combines a shared GRIN imputer,
ALFGraph, ALFTemporal, FiLM-style fusion, graph refinement, lifecycle context, and
an ALFNet-specific role-balanced forecast objective. Candidate switches still exist
in current code; their presence does not make them accepted protocol.

Comparison-model wrappers are registered in `experiments/v4_runner.py`. Their source
and permitted interface adaptations are documented in
`docs/comparison_model_audit.md`. Common data/exposure/evaluation rules are shared;
ALFNet-specific lifecycle inputs and objectives must remain exclusive to ALFNet.

Dataset layout and patch authority comes from the current manifests and runtime
context. Current resolved maximum partition capacities are PEMS03 12, PEMS04 10,
PEMS08 6 (component-aware graph BFS), METR-LA 7, and PEMS-BAY 11 (KD-tree over
verified WGS84 coordinates). A METR-LA seed32 exploratory experiment used patch
capacity 8; that override is not yet a five-dataset final protocol decision.

## Authority warning

`scripts/build_final_configs.py --check` currently validates 75 generated YAMLs.
However, `configs/paper/final_plan.json` is stale at this HEAD: its builder check
fails because multiple hashed source files changed. It also describes 13 models,
whereas the latest paper shortlist has nine. Therefore it is historical run identity,
not current authorization for new formal runs. A replacement plan must be explicitly
frozen after the active model set, METR-LA patch choice, and evidence scope are agreed.

## Pending verification

- Whether METR-LA and patch capacity 8 will be retained in the final dataset/protocol
  set, and whether seed42/52 plus test evaluation are authorized.
- Whether the nine-model shortlist is final for the paper or only the next reporting
  set.
- The canonical Python environment: `environment.yml` declares environment `moge`,
  while recent verified commands used `/home/xd/anaconda3/envs/ReaCH/bin/python`.
- The current paper source location was not identified during this checkpoint.

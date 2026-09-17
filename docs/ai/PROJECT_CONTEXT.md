# Project context

## Purpose and current scope

2026-09-17 COMPLETED: cleanup verified; all20 seed32 release tests finished under
plan63b50760. Results: `paper/test_seed32/`; provenance: `artifacts/release_completion.json`.
Focal remains locked. It ranks first for LifeAvg MAE/RMSE/MAPE on both datasets,
with45/66 first and13/66 second across four tables; BAY early/ready weaknesses remain.
No active task workers; no further training, repeated test, or Git publication authorized.
Later-in-file pending/running statements are historical. Current task handoff leads.

2026-09-17 current authority: user explicitly authorizes cleanup and then test
for the locked20 seed32 models (METRLA/PEMSBAY × Focal+9 baselines). Historical
no-test restrictions below are superseded for exactly this scope. No retraining,
test-based selection, commit or push. Source/weights/configs frozen in release
plan63b50760; full validation gate precedes test (18 historical-table checks,
2 STFiT archived-source fixed-RNG parity checks). See active handoff.
Weights: `artifacts/models/main/`;293 historical best checkpoints retained in
catalog. Old experiment trees archived, not executable workspace authorities.
SLPF single-stage code merged to root; three-stage code retired. Validation tables
are now `paper/validation_seed32/`; test completion tables will be `paper/test_seed32/`.

Latest result update (2026-09-16 23h): all6 valid reruns are complete, no task
worker/queue remains. Final tables are now `paper/final_seed32_updated_20260916/`:
20 audited rows (6 new+14 reused), Focal+9 baselines, allcanonical33 selection.
Four tables/Excel contain660 verified values and first-bold/second-underline
formatting. This supersedes pending/running and missing-baseline statements below.
Focal remains locked; SLPF is explicitly the single-stage forecast adaptation.
Root three-stage cleanup/single-stage merge remains a separate outstanding code
task; this update changed only saved-result tables/documentation, with no test.

2026-09-16: Focal is locked; no further optimization. User authorized only necessary
seed32 baseline reruns on METRLA/PEMSBAY, max5 concurrent, and then explicitly
simplified SLPF to its historical source forecast backbone with common single-stage
training. Do not restore three-stage ADP/forecast/aggregation or add ALFNet modules.
Describe SLPF as a single-stage forecasting adaptation, not full source-method
reproduction. METRLA SLPF is reusable; BAY SLPF needs canonical33 checkpoint selection.
Current scope:6 reruns (both STID/STGCN, BAY STBP/SLPF),14 reused rows including
Focal. Five non-SLPF jobs run under plan5ebe616f; BAY SLPF plan8eb8b2b9 waits for
a slot in workspaces/slpf_single_stage_20260916; combined cap5. Root code remains
frozen for active jobs; simplify/merge SLPF there only after they finish. Original
two three-stage jobs are superseded, not resumable current work. No test/commit/push.
Current authority: `paper/baseline_retrain_20260916/README.md` and active handoff.
Earlier cleanup/prepared configs and historical queue descriptions are superseded;
historical results themselves are not rewritten.

Latest selection (2026-09-15, after all seven completion runs): use Full+MAE-Focal
on both METR-LA and PEMS-BAY for the requested seed32 tables, retaining original
Full as control/backup. Eight candidate cells are verified; Focal improves LifeAvg
MAE/MAPE on both datasets but not RMSE, and BAY early-join regression plus prior
seed42 counterevidence remain. This supersedes the historical primary-Full choice
only for this bounded selection, not a multi-seed superiority conclusion.
See `paper/final_selection_seed32_20260915/analysis_report.md` and
`final_selection.json` in that directory. Current ranked tables leave STID/SLPF
pending qualified results; BAY STGCN/STBP retain historical selector caveats.
The seven-run queue has finished; no unified all-baseline rerun has started.

This repository studies observation-support-aware, topology-known but signal-inductive
traffic forecasting. The current paper and next optimization scope use METR-LA and
PEMS-BAY; other datasets remain historical/supplementary context. Lifecycle scenarios
instantiate changing observation support under the shared evaluation contract.
The user reopened final method selection on 2026-09-15: complete seed32 results
for Full/Focal/Focal+spectral/Focal+history on both speed datasets. Five existing
cells are reused and three are missing. ST-FiT/PatchSTG are rerun on both datasets,
for seven new runs, maximum five concurrent. The final nine baseline names are
UIGNN, GinAR, MoGERNN, ST-FiT, SLPF, STGCN, STID, PatchSTG, STBP; GWNET is now
historical/audit-only. All baselines are to be rerun after method confirmation,
under a new independent plan. Existing tables remain historical evidence.

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

Latest: plan459749c0 (full hash in active handoff) queues seven validation-only
completion runs from an independent frozen workspace. Root fixes include ST-FiT
batch graph isolation and equivalent dense ChebConv, STID calendar indexing,
GinAR/UIGNN adapter integration, unified source GWNET and initial archive cleanup.
Full-family output/loss/gradient cleanup parity passed on five bounded configs.
SLPF's historical wrapper is forecast-only; formal reruns now fail closed until
a legal three-stage pipeline is verified. STBP remains a disclosed offline adaptation.
The following older final-selection decisions are superseded pending eight cells.

On 2026-09-15 all four PEMS-BAY transfer and four GinAR/UIGNN runs completed.
The user requested final tables and retained two versions: original Full is the
primary, Focal is the backup. Frequency/history variants remain archived.
New tables and checkpoint pointers: `paper/final_two_speed_seed32_20260915/`.
PEMSBAY STGCN/STID/SLPF/STBP retain older micro-MAE checkpoint selection, unlike
the current 33-cell LifeAvg selector; this mixed historical provenance is disclosed.
GWNET inclusion is now explicitly authorized, with cross-dataset adapter differences
disclosed. GCN-M remains blocked by external history beyond the 12-step budget.

Historically, after the METR-LA selection, the user explicitly reopened a bounded PEMS-BAY
transfer screen on 2026-09-14: spectral and history-linear candidates at seeds32/42,
exactly four runs. Original Full was retained pending those results. Existing
PEMS-BAY Full42 runs use earlier recipes, so only Full32 is an equivalent reference;
seed42 supports comparison between the two candidates, not a Full superiority claim.

On 2026-09-14 the final METR-LA optimization round completed and the user requested
final selection. Original Full with its original role-balanced MSE was retained.
Focal gains did not repeat consistently in paired seed42 results; neither the
frequency auxiliary nor the history-linear correction passed preregistered gates.
Those switches remain archived experiments, not default model components. This
selects a recipe, not the best seed; seed32 and seed42 evidence remain separate.
PEMS-BAY stays on its existing Full version. See
`paper/optimization/focal_last_round_20260914/final_selection.json`.

At the 2026-09-12 checkpoint, HEAD is `9346dc0fc1dfb6b4aebfffe25224f1337f7a94cb`
on `paper/final`, with uncommitted ablation/analysis changes recorded in the active
handoff. ALFNet's accepted Full baseline remains
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
verified WGS84 coordinates). METR-LA capacity 8 was subsequently frozen by the user;
PEMS-BAY uses 11. Capacity is tokens per partition, not the number of partitions.
ALFTemporal fills undersized partitions with actual adjacency-ranked node indices,
so capacity changes can alter contextual inputs. No global optimality is established.

## Authority warning

`scripts/build_final_configs.py --check` currently validates 75 generated YAMLs.
However, `configs/paper/final_plan.json` is stale at this HEAD: its builder check
fails because multiple hashed source files changed. It also describes 13 models,
whereas the latest paper shortlist has nine. Therefore it is historical run identity,
not current authorization for new formal runs. Builder statements here are historical
checks, not checks rerun on 2026-09-12. Any new optimization needs its own frozen
plan/code/config identity; METR-LA patch 8 is the retained reference setting.

## Current authorized supplement (2026-09-08)

- The user retained METR-LA and froze its patch capacity at 8 for confirmation.
- Main paper analysis currently uses METR-LA and PEMS-BAY only.
- Six seed32 replacement ablations under `configs/paper/support_ablations_plan.json`
  have completed; Full was reused, not retrained. METR-LA paired trajectory and
  q/layout analyses have also completed; see the experiment index and active handoff.
- No new test access or seeds42/52 authorization is implied by that supplement.
- Read the active handoff for current uncommitted opt-in architecture switches.
- Paper drafts, appendices and figures belong under `paper/`; `docs/ai/` is AI
  context management. On 2026-09-12 the user requested a weaknesses checkpoint
  before further two-dataset optimization. On 2026-09-13 the user explicitly allowed
  GWNet fusion exploration. An independent ALFWave prototype has passed bounded
  performance/behavior checks; it is not an accepted replacement for Full and has
  no completed accuracy training. The two prior optimization rounds completed all
  eight runs without a screening pass.

## Pending verification

- METR-LA seed42 was explicitly authorized on 2026-09-14 for exactly four final
  Full/Focal/spectral/history-linear runs. PEMS-BAY spectral/history-linear at
  seeds32/42 was subsequently authorized for exactly four runs. Seed52 and test
  remain unauthorized. Seed42 also changes validation/test node membership;
  comparisons in that round must use its paired Full42/Focal42 controls.
- Whether the nine-model shortlist is final for the paper or only the next reporting
  set.
- The canonical Python environment: `environment.yml` declares environment `moge`,
  while recent verified commands used `/home/xd/anaconda3/envs/ReaCH/bin/python`.

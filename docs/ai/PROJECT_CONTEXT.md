# Project context

2026-10-17 CLEANUP: advisor-accepted experiments frozen. Current entry
`docs/FINAL_PROJECT_INDEX.md`; research lessons `docs/RESEARCH_LESSONS.md`.
Final ALFNet and eight baselines retained; all final-family ablation/HP weights
preserved. Cleanup inventory `archive/cleanup_20261017/plan.json`: 587 retained
checkpoints, 222 obsolete/excluded checkpoints deleted; old catalog is historical.
Registry now excludes retired baselines. Frozen historical plan hashes are not
new-run authority. Source/evidence archive excludes deleted checkpoint binaries.


2026-10-17 FINAL FIGURES: `paper/final_figures_20261017` has final-checkpoint
validation/test figures2–6 and CSV evidence. PEMS08 mechanism now WD=.001;
other final weights unchanged. Main mechanism figures consistently validation,
test companions retained; not per-figure split cherry-picking. Four displayed
ablations exclude GraphRefinement at user request; complete five-item evidence
retained. v5 manuscript, experiments draft and advisor script updated.14 missing
flow OFAT tests complete with exact validation replay; no training/reselection.

2026-10-12 COMPLETION: all5 new WD=.001 PEMS08 ablations trained and tested,
all validation replays exact. Test4/5 worsen all3LifeAvg; across4datasets3/5
(AdaptiveRouting/SpatialContext/TemporalFiLM) worsen all12. StateAge11/12
(PEMS04RMSE exception); GraphRefinement10/12(PEMS08MAE/RMSE improve).
Current paired tables: paper/pems08_wd_ablation_test_20261012; manuscript
table2 now uses new WD=.001 Full+5ablations. No new training queued.

2026-10-12 UPDATE: PEMS08 final selection is now `13_PEMS08_weight_decay_0.001`.
New authority: `paper/pems08_wd_test_20261012/selected_configuration_manifest.json`;
other three datasets unchanged. Full validation replay exact; test LifeAvg
36.779822/46.566932/42.999878,30/33first. Five paired ablations at WD=.001 are
queued/running under independent plan `2c86d208...`, max5. Old paired ablations
and mechanism figures remain explicitly labeled historical WD=.0001 anchors.
Earlier dated frozen plans/results remain historical evidence, not current selection.

2026-10-10 TEST COMPLETE: all48 ablations tested,48validationreplays exact.
Two types worsen all4datasets×3LifeAvg ontest: SpatialContext andTemporal–FiLM;
validation remains0types. Full1716celltesttables andsplitcomparison audited under
paper/all_ablations_test_20261010. Retain all12types andsingle-seed/selection
limitations. No model changes,newtraining orfurthersearch this analysis turn.

2026-10-10 NEW TEST AUTHORITY: user explicitly requests test all48 completed
ablations (12types×4main datasets). Plan35f30008 launched,max3eval; exact frozen
weights/adapter contract,completevalidationreplaygate beforetest each.4Full tests
reused bycheckpointidentity. All outcomes retained; no model reselection,training,
baseline rerun orPEMS03. Detailed live-state pointers in activehandoff.

2026-10-10 COMPLETED: all48 ablation runs (12types×4datasets) finished and
audited with4 fixedFull references.0types worsen all12LifeAvg metrics; closest
ALFRecon/TemporalFiLM11of12. Full complete1716-cell results/analysis under
paper/ablations_complete_20261010. All counterexamples retained, no model or
test-table changes and no further experiments launched/queued this turn.

2026-10-09 LATEST: mechanism15/16 complete, combined10 ablation types currently
0 with all4datasets×3LifeAvg worse; completedcounterexamples alreadyruleoutlast
pendingtype. All findings published in paper/ablation_progress_20261009, no
outcome-based suppression. User-authorized bounded8 gradient-path controls queued
under29df497b after current16; samefinalanchors,max5,validationonly. No model
reselection or claim that every component is universally beneficial.

2026-10-08 CURRENT: BAY retry training finished but report export failed due to
missing recovery-plan status. Repaired report from saved history + full best-weight
validation replay, no train steps/checkpoint edits/test. All24 original ablations
now packaged924cells. Mechanism16 queue unblocked and first GPU preflight started;
formal training begins after all preflights. Exact details in active handoff.

2026-10-08 LATEST: user authorizes more mechanistic ablations after flow
noRefiner/noAge improvements.16 controls (four per frozen main dataset) queued
under3b52de24 behind single OOM recovery. No production architecture or final
configuration change. Report all prior/new results, never filter solely to make
Full win every metric. New controls separate expert lifecycle input, FiLM gain,
FiLM shift and neighbor mixing in output refinement. No PEMS03/newtest.

2026-10-08 RESULTS: OFAT14/14 complete; ablation23/24 complete, BAY Spatial
Context failed from GPU OOM. Existing results audited and packaged under
paper/final_supplements_20261008. Single identical retry a5ab583c launched in
independent root; automatic offline table refresh scheduled. No newtest or
final-model replacement. Flow WD improvements and non-universal ablation
benefits documented; preserve all outcomes, do not hide better ablated variants.

2026-10-06 LATEST: user authorizes six named component ablations AFTER current
OFAT completion, explicitly corrects scope to FOUR datasets (no PEMS03).
Plan27c243b2 final_ablations_20261006 schedules24 new+4 frozenFull reuses, seed32,
max5. Waiting tmux starts no GPU until predecessor success; preflight then formal
training. No test, new model selection or baseline rerun. Other-userPID2546461
protected. Prior five-dataset draft plan withdrawn before launch.

2026-10-06 CURRENT RUN: user accepts completed speed fixed-anchor OFAT and
authorizes only PEMS04/08 completion, max5. Study flow_ofat_20261006 uses final
flow anchors, reuses4 exact configs and trains14 missing single-factor settings
for capacity/LR/temporal dropout/weight decay. Plan3000c818; queue launched into
preflight, then automatic formal training. No ablations/new test/model reselection
in this immediate scope. Full runtime status in active handoff.

2026-10-06 LATEST ACCEPTED FINAL: user ends scheme search and accepts one
parameterized ALFNet backbone with dataset-specific input handling. Freeze
METRLA capacity7/LR1/b0/s.3/alpha0, PEMSBAY capacity16/LR2/WD1e-5/b0/s.3/alpha0,
PEMS04 capacity10/LR1/b1/s1/alpha0, PEMS08 capacity6/LR1/b1/s1/alpha.5.
All dropout.1, other WD1e-4, seed32. Exact weights/configs and planned paired
ablation/OFAT studies: paper/final_scheme_20261006/. No further method search.
Interpolation is current-visible-window model input handling, not offline complete
data preprocessing. Alpha0 disables centering. Current turn froze identities and
analyzed saved results; new study configurations/preflight/queue not yet launched.
Max5 training persists; no baseline retraining or PEMS03 reopening.

2026-10-04 NEW AUTHORITY: transfer accepted support scheme to METR-LA and
PEMS-BAY with dataset-specific hyperparameters, max5 concurrent. Frozen16-run
speed_support_20261004 study: selected capacities7/16, dropout.1; center0/.5
x LR.5/1/2 plus center1/LR1 each, and two PEMSBAY WD1e-4 controls against
selectedWD1e-5. Preserve old speed configuration unless new scheme meets goals
and shows improvement. No automatictest/baseline training; PEMS03 remains stopped.

2026-10-04 LATEST RESULT/STOP: all six PEMS03 support-family configurations
now have authorized test results. Best coverage remains alpha.5/LR1 at20/33;
none meets27/33. Five additional fullvalidation replays are exact, no evaluation
adapter bug found. User instructed to stop if repair is complex: stop PEMS03
optimization, keep METR-LA/PEMS-BAY/PEMS04/PEMS08 main scope and retain complete
PEMS03 evidence for appendix/limitations. No dataset/weight deletion. New evidence
in paper/pems03_remaining_test_20261004; descriptive node/time shift analysis
must not be presented as isolated causal proof or a prespecified benchmark choice.

2026-10-04 LATEST AUTHORITY: user accepts one parameterized support scheme
(visible interpolation + decoder residual scale + centering strength), with
dataset-specific hyperparameters. PEMS04 alpha0/LR1 and PEMS08 alpha.5/LR1
are accepted as the same scheme; identical fixed numerical values are not
required. Reopen PEMS03 evaluation under this scheme: six validation trainings
alpha{0,.5,1} x LR{1,2}, max5 concurrent. Retain existing PEMS03 evidence;
main-paper reinstatement depends on reviewed results, not yet decided.
Baseline/speed training and new test are outside this immediate run scope.

2026-10-03 NEW AUTHORITY: main-paper dataset scope is METR-LA, PEMS-BAY,
PEMS04 and PEMS08; retain PEMS03 as archived/appendix evidence, no raw/weight
deletion. User reopened ALFNet optimization after two flow repair rounds failed
to reach80% first cells. First evaluate new04/08 candidates, max5 concurrent;
speed ALFNet transfer is authorized if promising, baselines must not retrain.
Current study plan and live status are in the active handoff. Prior locked-model
and last-round-stop statements are superseded for this authorized research only.

## Purpose and current scope

2026-09-22 LATEST: manuscript reorganized by RQ1–4. Main Figures3/4/6 focus
known-node paired validation windows;held-out full views remain appendixA.13.
Mask-only moved intact toA.14;w/o names unified. Source predictions unchanged.
New offline views and mapping:paper/rq_experiments_20260922/. Verified108points,
9216residualcells,12unchanged final-table hashes. No new inference/test/training.

2026-09-22 LATEST: intervention/metric audit complete. Mask-only changes both
role/age inputs and role-dependent refinement scale (0/1/1/.25 to mean.5625);
not isolated role ablation. LocalTemporalContext reorganizes same modules pernode,
not standalone TSA deletion;GRIN ablation uses train-mean fill. Regression Focal
is (e/std)*tanh(.1e),not classification Focal or MAPE. Keep UIGNN and consistent
MAE inFigures3/4;MAPE does not reverse held-out weakness. Figure6 now all64-window
residual atlas,old individualcases appendixA.11. Newaudit/figures:
paper/support_metric_audit_20260922/. No training/inference/newtest/modelchange.

2026-09-22 CURRENT presentation: user accepted cohort-specific design. Paper4.5
uses held-out arrival versus known-node interruption;4.6 shows both cohorts
separately. Final predictions reused offline, no new inference/test. Figures in
paper/support_cohorts_20260922/. Main ablation table has7structural variants;
2objective variants moved to appendixA.12 with findings retained in main prose.
Case centers selected from truth mean/std/change,post-hoc descriptive,not wins.
Original figures/failures remain archived and linked. Model/main tables unchanged.

2026-09-22 LATEST COMPLETE: manuscript revised to academic experiment prose.
Final-checkpoint validation mechanism plan1dcdf704 complete:ALFNet,UIGNN,MoGERNN,
STBP;representatives selected by prior validation LifeAvg MAE in three classes.
128 paired windows,20 arms,10240 predictions audited. Figures3/4/6 and main-text
cases:paper/final_support_20260922/. All9 ablations retained,including unfavorable
results. Test-feedback history moved to appendixA.10,mechanism protocolA.11.
No architecture/training/main-table change or new test.

2026-09-21 LATEST COMPLETE: authorized nine METRLA ablation tests finished under
plan08f0a1b1 after all9 validation gates reproduced exactly. Paper4.4 now test;
prior validation retained in appendixA.9; hyperparameter curves stay validation.
Full reuses identical main-table record. Mask-only and no_role_balance improve
all three test LifeAvg metrics, with early-stage tradeoffs; no blanket claim that
every module helps. Final model/main tables unchanged, no test-based pruning.
Reports: paper/ablations_metrla_test_20260921/. No evaluation workers remain.
Older dated state statements below are historical and superseded by this entry.

2026-09-21 COMPLETED: all9 final-Focal METRLA ablations audited, paper4.4 now
validation results (not test). Removing temporal-FiLM/context/imputer increases
MAE15.10/7.37/5.11%; removing refiner improves all LifeAvg metrics; role balancing
has condition tradeoffs. Final model unchanged. Recommended final-paper layout:
main comparison and fixed ablations test, hyperparameters validation. Nine ablation
test evaluations are NOT yet done/authorized by this analysis request; no retraining
needed if later authorized. No active workers observed. See latest task handoff.

2026-09-20 LATEST: user explicitly authorized two METRLA candidate tests and
test-based comparison, superseding earlier no-test/validation-only selection scope
for lr_double and patch12 only. Both tests completed (planaefefd7c), original
retained:32 first cells vs25/27; both candidates worsen MAE/RMSE. This is exploratory
test reuse, not independent confirmation. Nine final-Focal METRLA ablations now
running under plan196607d5, max5; full9 real preflight passed. Original core model
and train_v4 unchanged; study-only adapter required to rebuild ablation checkpoints.
Paper main LifeAvg table retains both datasets; other displays METRLA, BAY retained
in appendix. No more test or final model replacement automatically authorized.

2026-09-20 COMPLETED: METRLA9/9 hparameter runs audited, including capacity7/10/12;
capacity8 reused. lr_double best LifeAvg, patch12 most improved cells29/33, wd_high
also improves three LifeAvg metrics; all fail frozen early-stage constraint, lr_double
also fails Unseen limit. Keep final METRLA Focal8 and BAY Focal16; paper4.7 now
contains completed validation sensitivity. No new test, no final-table replacement.
Analysis/audit:paper/hparams_metrla_20260920/. No active task workers observed.

2026-09-20 NEW AUTHORITY: final model/results fixed; update paper to final test
and run METRLA sensitivity, maximum5 parallel. Plan f45c7f30: nine new seed32
single-factor runs (capacity, all-group LR, dropout, weight decay), reused Focal
baseline. Train/validation only; no architecture/loss change, automatic final-model
replacement, or new test. See latest task handoff for actual execution state.

2026-09-20 COMPLETED: user authorized candidate test/table update. Only PEMSBAY
patch16 passed the frozen validation screen; its prior-session validation/test
finished. Current session repaired offline table publication, without new inference.
Use Focal capacity16 for PEMSBAY updated tables, capacity8 unchanged for METRLA.
New tables: paper/{validation,test}_seed32_hparam_patch16_20260918; old release
tables/weights preserved. No active task training/evaluation. See latest handoff.

2026-09-17 NEW AUTHORITY: user requests PEMSBAY-only hyperparameter sensitivity
with final Focal architecture unchanged, permits4–5 concurrent training runs.
First independent plan1ec318e9: seed32 patch capacity12/14/16 and effective optimizer
LR multipliers0.5/2 (all three groups). Reuse current Focal32 as baseline; no new test,
METRLA training, architecture/loss change, or automatic final-model replacement.
Plan/report: `paper/hparams_pemsbay_20260917/`; inspect active handoff/queue status.
Current user-committed HEAD783cbf7 supersedes older Git-state descriptions below.

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

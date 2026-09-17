# Experiment evidence index

## E-FINAL-SEED32-TABLES-20260916 — completed, nine-baseline pool

Six valid reruns complete/exit0;14 historical rows reused. Both execution plan
hash checks pass. All20 artifact/checkpoint/selection/exposure/scaler/count audits
pass, all canonical33 selection, skipped0. No live task worker/queue at read-only
host check. Four updated tables in `paper/final_seed32_updated_20260916/` include
Focal+9 baselines;660 Excel numeric cells and bold/underline styles roundtrip pass.
Focal first/second: METRLA main11/1 phase12/5, BAY main5/7 phase7/5; total35/18
out of66 correlated/derived cells. Single-stage SLPF and offline STBP adaptations
disclosed. Collector `scripts/update_final_seed32_tables.py`; saved validation
only, no inference/test/new training. Previous tables retained for provenance.

## E-BASELINE-RERUN-20260916 — five active, single-stage SLPF queued

Current scope:6 necessary seed32 runs,14 reused rows. Full artifact audit20 rows;
18 current module-instantiation isolation checks; prior18 interface/poison tests
retained after code-SHA checks. Plan5ebe616f GPU probes7/7 passed. Formal METRLA
three-stage SLPF failed on entirely missing history; BAY three-stage stopped on
user simplification instruction. No result claimed for either. Other5 formal
STID/STGCN(two datasets)+BAY STBP jobs running. New plan8eb8b2b9 queues only BAY
source-forecast single-stage SLPF, combined cap5; METRLA old single-stage reused.
Single-stage CPU checks: both historical checkpoints strict-load, target/lifecycle
blindness, all-missing finite output and gradient update pass; historical adapter
AST unchanged except label and core bytes identical. Evidence and current plan
pointers: `paper/baseline_retrain_20260916/README.md`, `eligibility_current.json`,
`module_isolation.json`, `single_stage_slpf_checks.json`,
`slpf_single_stage_semantic_parity.json`, `slpf_single_stage_plan_pointer.json`.
No test/commit/push. New results are not yet complete.

## E-FINAL-CODE-AUDIT-20260916 — no formal training

Final Focal locked; cleanup checkpoint parity on two legal train windows/dataset:
max absolute difference METRLA1.862645e-9, BAY0, strict checkpoint loads passed.
Nine baselines × two datasets:18 CPU interface/gradient/isolation cases passed.
Targeted repair tests20 passed; shared runner/entrypoint regression91 passed
(overlapping suites). Initial multiprocessing pytest stalled after78 passes and
was interrupted; single-process rerun completed in455s. SLPF full three-stage
entrypoint CPU one-batch-per-stage smoke passed, not accuracy evidence.
Artifacts: `paper/final_code_audit_20260915/README.md`, `focal_cleanup_parity.json`,
`baseline_interfaces.json`, `repair_tests.log`, `targeted_tests_singleprocess.log`.
Prepared20 configs+4 Focal expert configs in `configs/paper/final_models_v1/` pass
builder --check; not a training plan. No GPU/test/commit/push. Historical result
tables remain unchanged pending independently authorized retraining.

## E-FINAL-FOCAL-SEED32-20260915 — completed and ranked

Plan459749c0 seven runs complete/exit0; candidate coverage8/8. Selected Focal
for both datasets, original Full retained. Frozen builder check passed; 26 runs
audited, zero skipped batches. Excel1320 numeric cells and bold/underline rankings
roundtrip verified. Evidence: `paper/final_selection_seed32_20260915/`:
`analysis_report.md`, `final_selection.json`, `results_and_provenance.json`,
`verification.json`, `final_seed32_tables.xlsx`, `final_tables.html`/`.md`.
Historical tables separately retain excluded STID/SLPF old values. Current tables
have seven ranked baselines plus two pending rows; BAY legacy selectors disclosed.
No new evaluation/test/training. Earlier queue-running statements below are historical.

## E-SEED32-COMPLETION-20260915 — seven new runs, max five concurrent

Plan `459749c0482f937db06f13a42090be8578eb4ffb0b28178360e6cfd3c00e8324`:
METRLA spectral/history32; BAY Focal32; ST-FiT/PatchSTG on both datasets32.
Five existing Full-family cells have verified reuse hashes. Source:
`workspaces/final_completion_frozen_459749c0482f937d/`; outputs:
`outputs_v4/final_completion_seed32_v2/<hash>/`. Seven CPU/GPU preflights and
all seven concurrent probes passed; five formal workers running, two queued.
Full-family cleanup parity: five configurations, exact two-window prediction,
objective and all parameter gradients. Dense ChebConv: six double-precision
operator tests plus both datasets/phases full-model comparisons (<1.3e-7 max).
Root baseline audit:20 CPU interfaces passed, with SLPF semantic blocker retained.
No test access. Tables/coverage and audit:
`paper/final_completion_20260915/`; finite collector updates saved-artifact summaries.
The previous primary-Full decision and final eight-baseline table are historical.

## E-FINAL-TWO-SPEED-20260915 — Full primary / Focal backup, saved-artifact tables

Old four BAY candidates and four GinAR/UIGNN complete/exit0. User confirmed GWNET
inclusion and requested two retained versions. New artifacts:
`paper/final_two_speed_seed32_20260915/` (Excel, Markdown, CSV, source SHA audit,
selection report, primary/backup checkpoint pointers). Four formal seed32 tables
contain Full + eight baselines, 594 values; separate Focal candidate has33 METRLA
values and an explicit missing BAY entry. Scope is validation only, no new inference.
OriginalFull remains primary, Focal backup. Historical BAY baseline micro-selection
and cross-dataset GWNET adaptation differences remain visible; no claim of uniformly
rerun baselines or multi-seed significance. Rebuild: `scripts/build_final_two_speed_tables.py`.

## E-MISSING-BASELINES-20260915 — adapted, persistent dependency wait

Plan `725a80ed153317b77834823ee42b696e623f65848f59357cfa5d55427a79c8bd` freezes
six seed32 slots, METRLA/PEMSBAY × GinAR/GCN-M/UIGNN. Four GinAR/UIGNN CPU
production paths, core parity/isolation/checkpoint-resume probes and nine targeted
tests passed. Two GCN-M slots are explicitly blocked by official external-memory
access beyond the12-step budget. No fake memory or substitute model is trained.
Frozen executable source is in workspaces/, not outputs_v4. tmux
`missing-baselines-20260915` persistently waits for old four verified terminal
states, then GPU admission/concurrency probes and four legal formal runs.
At2026-09-15 01:00, still waiting; no new GPU training or test access.
Details: [execution handoff](handoffs/T-20260908-support-ablations/MISSING_BASELINES_EXECUTION_20260915.md).

## E-PEMSBAY-FOCAL-TRANSFER-20260914 — four-run transfer launched

Plan `34ecd952b01ebbad540a1345260d2793f37e4b71f743c817cc492e6c410b3a0a`:
spectral/history-linear × seed32/42, PEMS-BAY only, exactly4 runs/max4 concurrent.
21 targeted tests, oldFull8-window parity, within-seed pairing/target-blind,
4CPU/4GPU preflights and synchronized4-job16-batch probes passed. Probe train
38.86–39.97s, peak reserved15.73GiB. Root-source execution, frozen snapshot.
Only Full32 equivalent reference exists; Full42 superiority is unconfirmed.
Evidence/output: `outputs_v4/pemsbay_focal_transfer_v1/<hash>/`; execution:
`docs/ai/handoffs/T-20260908-support-ablations/PEMSBAY_TRANSFER_EXECUTION_20260914.md`.
No test/extra runs/automatic promotion.

## E-METRLA-FOCAL-LAST-FINAL-20260914 — complete, original Full retained

All4 plan8e2a8b3b… runs complete/exit0, independently audited from saved reports/logs/checkpoint metadata. Neither new candidate passes frozen gates. Focal gains fail seed42 confirmation; history correction improves27/33 cells versusFull but Unseen RMSE+10.42%. Original Full MSE remains final recipe; stop search. Full results, stage timing, 33 cells, hashes and selection pointers in `paper/optimization/focal_last_round_20260914/{results_report.md,results_analysis.json,final_selection.json}`. No inference rerun/test or production-code change.

## E-METRLA-FOCAL-LAST-20260914 — four seed42 runs started

- Independent plan `configs/paper/metrla_focal_last_v1.json`, hash `8e2a8b3b2009ecb5a0ddf3225699290ba39fee68967de9c58449f5980f1f59ca`; Full/Focal controls plus spectral/history-linear candidates, METR-LA only, max4 concurrent.
- 19 targeted tests, oldFull8-window parity, seed42 common weights/RNG/exposure/scaler and target-blind checks, 4CPU/4GPU single-batch preflights, synchronized four-job16-batch probe passed. No completed accuracy evidence yet.
- Design and literature: `paper/optimization/focal_last_round_20260914/design.md`; execution: `docs/ai/handoffs/T-20260908-support-ablations/FOCAL_LAST_EXECUTION_20260914.md`. Results automatically update under `outputs_v4/metrla_focal_last_v1/<hash>/`.
- Seed42 also changes validation node membership; compare within42. Retain Full, no automatic promotion/next search/BAY/test. Root-source execution with snapshot backup; original dirty changes preserved.

## E-METRLA-MAE-TAIL-20260914 — completed and analyzed

- All4 d9051606… runs COMPLETE/exit0; best epochs60/49/52/59, epochs71/60/63/70.
- Builder current, saved tables/reports/history selections/win counts verified;
  no skipped batches. No new inference, fitting or test access this analysis.
- Focal LifeAvg4.537516/6.911326/13.475396; Full-relative−2.17%/+1.54%/−7.87%.
  All4 fail gates; Focal retains33/33 MoGERNN wins,27 original8 first places,
  15 first places including GWNET. Main/phase counts and full differences saved.
- `paper/optimization/mae_tail_results_20260914/report.md` and `analysis.json`;
  reproducible offline script `scripts/analyze_mae_tail_results.py`.
- Decision: end current optimization search; retain Full and Focal alternative,
  no new candidate/BAY/test launch. Original candidates status updated.

## E-METRLA-MAE-TAIL-20260913 — four trainings launched

- Plan `configs/paper/metrla_mae_tail_v1.json`, identity
  `d90516065cc844010bdb13ee1e2a2552deb6d58535030bee56909a588e0d93cb`.
- Root implementation, Full structure, seed32, four loss-only candidates;
  max4 concurrent, batch64, max100 epochs with original early stopping/selection.
- 15 synthetic tests, configuration audit and plan loader passed; Full/four variants
  strict-checkpoint parity over8 windows maxdiff0; 4CPU/4GPU probes passed.
- Four concurrent16-batch probes passed, train phase17.98–18.15s each,
  peak reserved10.1055GiB each. No full-epoch performance conclusion yet.
- tmux `metrla-mae-tail-20260913`; output root
  `outputs_v4/metrla_mae_tail_v1/<identity>/`, launcher_status/logs/report/candidate_summary.
- Full retained; no BAY queue, test access, external-process manipulation or promotion.
- Details: `handoffs/T-20260908-support-ablations/MAE_TAIL_EXECUTION_20260913.md`.

## E-FULL-MAE-RESIDUAL-20260913 — paired validation diagnosis complete

- Diagnostic identity `6c867b03c41f3ad12765ed0d13b37c87e8f15a0ed9282b996bae8a742fca63ad`;
  output `outputs_v4/full_mae_residual_diagnosis_v2/`, Full + Full-MAE, METRLA seed32.
- 6725 loader / 6696 valid windows, 874197 paired positions; independent 33-cell
  reproduction maximum error1.913e-7; exact canonical RMSE attribution verified.
- v1 retained after Full succeeded and MAE relative-config loading failed;
  v2 hashes/reuses Full and runs MAE only. No fitting, test or old-run overwrite.
- Analysis, win lists and source hashes: `paper/optimization/full_mae_residuals/analysis.json`;
  report and PNG/PDF in the same directory. Eleven-paper evidence screening,
  four loss-only candidates and retention gates are proposed, not launched.

## E-TWO-SPEED-ROUND2-20260913 — preflight complete / queued

- Isolated code: outputs_v4/round2_workspace_20260913; first-round hashed source
  unchanged. Round2 plan configs/paper/two_speed_round2.json under that root,
  identity 159322422fa9ece24aa434832405b0824e6e1d867a411da8388d4bb7831c27b8.
- Four seed32 runs: freshness/refiner_uniform × METRLA/PEMSBAY, original Full
  budgets and protocol. No compound candidate, Full promotion or test access.
- 24 targeted tests passed / 3 deselected; builder and production authority passed;
  8 baseline parity checks max error0; four CPU/four GPU batch64 probes passed.
- Queue waits for first four COMPLETE markers and original collector result;
  second round uses three phase slots and explicit wait/compute timing.
- Monitor/report: outputs_v4/two_speed_overnight_20260913/two_round_report.md and
  two_round_summary.json. tmux two_speed_overnight_20260913_v2. No action on
  external PID1734562. No next-round result claim yet.
- Diagnosis/limits: handoffs/T-20260908-support-ablations/OVERNIGHT_20260913.md.

## E-TWO-SPEED-MECHANISM-20260912 — diagnosis complete / training launched

- Diagnostic plan `configs/paper/two_speed_diagnostic_v1.json`, identity
  `4809eb19144dbce8b310437daea0ed06d1f0fa7d100d088cd9c28542730228c3`;
  outputs under `outputs_v4/two_speed_diagnostic_v1/<identity>/`.
  Both METRLA/PEMSBAY COMPLETE: 256 natural validation windows, six arms each.
  Paired targets/masks/metadata, completion hashes and saved-array MAE checked.
- Training plan `configs/paper/two_speed_mechanism_v1.json`, identity
  `afd9f69d16e5b59429a8eda619b7e9874e1e21cd28de4af4873ad3e86ee5bdf9`;
  outputs under `outputs_v4/two_speed_mechanism_v1/<identity>/`.
  route_shrink/bounded_mae × METRLA/PEMSBAY × seed32, four runs launched.
- 22 targeted tests passed, 3 deselected; builder check, invariant config comparison,
  eight baseline prediction parity checks (max error 0), four CPU and four GPU
  batch64 probes passed. Peak allocated GPU memory 9.4/14.8 GiB by dataset.
- Status authority: preflight.json, launcher_status.json, logs/ and eventual
  candidate_summary.json. No candidate performance/promotion claim yet.
- Scope: user authorized bounded parallel training; four processes, two GPU phase
  slots; no new test or seeds42/52. Full retained. Execution details in
  `handoffs/T-20260908-support-ablations/EXECUTION_20260912.md`.

## E-TWO-SPEED-DIAGNOSIS-20260912 — saved validation reanalysis

- Design: `handoffs/T-20260908-support-ablations/OPTIMIZATION_CANDIDATES_20260912.md`.
- Reproducible CPU-only source: `reanalyze_saved_validation.py` in the same directory;
  output `DIAGNOSTIC_EVIDENCE_20260912.json` records input hashes and full contrasts.
- PEMS-BAY Full retains Join gains over Mask-only/static while Steady/Fault expose
  negative effects. METR-LA held-out low-speed [0,20) cells explain much of selected
  MAPE regressions, but represent only 45 steps, six windows and four nodes.
- All observations use existing seed32 validation artifacts, not new forecasts.
  Paired metadata/targets/masks matched; saved-window MAPE recomputation maximum
  difference 4.07e-5 percentage points. Original snapshot hashes matched 87/88;
  ISSUES_20260912.md drift is preserved and disclosed.
- No candidate implemented or trained; proposed gates and four-run screening cap
  are design suggestions, not execution authority. Natural Unseen/PEMS-BAY
  speed-bin attribution and internal pathway diagnosis remain unperformed.

## E-SUPPORT-ABLATIONS-20260908

- Authority: `configs/paper/support_ablations_plan.json`; six seed32 validation-only
  replacements on METRLA/PEMSBAY, no Full retraining.
- Analysis: `outputs_v4/reports/support_analysis_v1/`; condition_results.csv,
  condition_gains.png/pdf, METRLA_cases.png/pdf, PEMSBAY_cases.png/pdf,
  case_records.json and provenance.json. Existing Full checkpoints only.
- Preflight: `outputs_v4/support_ablations_setup/preflight.json` (six CPU passes),
  `gpu_preflight.json` (batch64 PEMS-BAY local context).
- Runs: `outputs_v4/support_ablations/<plan_sha256>/launcher_status.json`.
- Status: all six runs COMPLETE; run reports, best epochs and source-file hashes
  rechecked on 2026-09-12. Exact result/config/log paths are indexed by
  `paper/experiments_evidence_snapshot.json` (22 source hashes matched).
- Limits: these original cases are descriptive; stage evidence retains original
  seed32 validation mixed-source scope. Controlled intervention is indexed separately
  below; no new test claim.

## E-OBSERVATION-SUPPORT-V1 — completed METR-LA controlled analyses

- Evidence: `outputs_v4/reports/observation_support_v1/`; COMPLETE, manifest.json,
  pairs.json, metrics.csv, paired_predictions.csv, relative_gains.csv,
  verification.json and figure_manifest.json.
- Scope: validation, reused ALFNet/MoGERNN/Mask-only checkpoints; 128 paired windows,
  20 unique arms, 7,680 forecasts. Six trajectories plus 14 distinct q/layout arms.
  No Full retraining. Naturally complete history/future required for selection.
- Recorded verification passed for paired targets/masks, all three metric arithmetic,
  full-history anchors, metadata-blind control invariance and unchanged checkpoints.
  The 2026-09-12 checkpoint reads this record; it does not rerun model inference.
- Latest figures in `paper/figures/`: condition gains use fixed MoGERNN; trajectory
  uses 64 held-out windows/23 nodes, known-node controls in appendix; q/layout retains
  128 windows (known/held-out 64 each), with mature AAS/age=1 metadata fixed.
- Figure manifest: all 11 listed artifact hashes matched on 2026-09-12.
- Limits: descriptive paired forecast responses, not independent seed significance
  or isolated internal routing causality. Pooled metrics differ from LifeAvg.
  The output README's old "main figures average 128" sentence is superseded for
  Figure 3 by the latest figure_manifest, script and paper appendix.
- Manuscript: `paper/experiments_draft_cn.md`, `paper/experiments_appendix_cn.md`.

## E-OPTIMIZATION-ISSUES-20260912 — existing evidence reanalysis only

- Inventory: `docs/ai/handoffs/T-20260908-support-ablations/ISSUES_20260912.md`.
- Findings: current LifeAvg beats MoGERNN on both datasets/all three metrics;
  METR-LA Unseen MAPE and PEMS-BAY Unseen all three metrics do not. PEMS-BAY
  Mask-only/static slightly outperform Full overall. Held-out Full-history/Join-6
  also expose internal-control weaknesses. No new training or evaluation occurred.
- Prior candidates checked: METR-LA LF/SUP summary under
  `outputs_v4/metrla_lf_sup_candidates/df329665605fc2335c94d957dce70421bd0b59494a01a6c0c5d9d9cbb314d793/`;
  speed SUP5/LF_SUP5 summary under
  `outputs_v4/speed_sup5_candidates/16e11d667c3374f594934f52a0a27e0994dcd8b84211546942ea0001ae303f64/`.
  Both index historical recipe-specific tradeoffs, not fresh comparisons to current
  Full; some referenced original config paths require recovery before reproduction.

This file indexes evidence; it does not duplicate full metric tables. Conclusions are
limited to the stated dataset, seeds, phase, code identity, and provenance.

## E-4D-11M-3S — Four-dataset, 11-model validation/test report

- **Evidence:** `outputs_v4/reports/four_dataset_11_models_final/7523ac4c11f14e222dddedc905c9b7d50bc7eece1fc4deebc83a8994c382ac84/summary.json`
- **Scope:** PEMS03, PEMS04, PEMS08, PEMS-BAY; 11 models; seeds 32/42/52;
  validation and test.
- **Status:** complete historical report tied to source plan hash beginning `c8d6b8`.
- **Limit:** it is not the latest nine-model/METR-LA experiment and is not evidence
  that current HEAD satisfies the old plan hash.

## E-4D-SEED32-MIXED — Mixed-source four-dataset diagnostic

- **Evidence:** `outputs_v4/reports/seed32_hybrid_validation/59d38810eaa142934cb603ebb73b7c03fa0d0a29a2f030fdacef503a32bcd083/summary.json`
- **Scope:** validation, seed32, four datasets; corrected ALFNet/GWNET/MoGERNN mixed
  with earlier baseline artifacts.
- **Status:** diagnostic only.
- **Limit:** mixed provenance forbids presenting it as one formal comparison run.

## E-METRLA-9M-SEED32 — METR-LA nine-model candidate

- **Plan:** `configs/paper/metrla_seed32_nine_models_plan.json`, plan hash
  `057073f745ba24710cc4849abe91cc65c00b13ce4136337eb67563f60da35ff1`.
- **Evidence root:** `outputs_v4/metrla_seed32_nine_models/057073f745ba24710cc4849abe91cc65c00b13ce4136337eb67563f60da35ff1/`.
- **Scope:** METR-LA, nine selected models, seed32, validation only, ALF patch 8.
- **Checked state:** all nine run directories contain `COMPLETE` and
  `plan_validation_eval.json`; the plan builder check passed at the checkpoint.
- **Re-evaluation:** eight models agreed exactly with training-time validation. STGCN
  recorded `agrees=false`, maximum absolute difference
  `7.081064183012131e-06` against tolerance `1e-6`; cause remains pending.
- **Limit:** no seed42/52 conclusion and no test conclusion.

## E-5D-9M-SEED32-MERGED — Five-dataset reporting workbook

- **Evidence:** `outputs_v4/reports/five_dataset_seed32_nine_models_validation/05df96f892b8ccb532f081c4ece8a389c035849af3b50a264de5669f220cd6d4/`.
- **Files:** `five_dataset_seed32_nine_models_validation_tables.xlsx`, `tables.json`,
  and `provenance.json`.
- **Scope:** nine selected models, five datasets, seed32, validation only.
- **Status:** exploratory mixed-source report, as declared by its provenance file.
- **Limit:** the four pre-existing datasets and METR-LA were not produced by one newly
  frozen nine-model plan; it is not a formal three-seed or test table.

## 2026-09-13: completed optimization rounds and bounded fusion profiling

- Two rounds / eight seed32 runs are COMPLETE, all failed their screening gates;
  retained Full unchanged. Combined evidence: `outputs_v4/two_speed_overnight_20260913/two_round_summary.json`.
- New bounded performance identity `4423e8ebc5fb58f78bb746ac8f95a3c8f3d23790eaf520296c42f55e21bc5b71`:
  `outputs_v4/speed_fusion_probe_20260913_v2/{manifest,results,validation_reductions,log_timing_summary,supplemental_provenance}.json`.
  Six batch64 forward/backward cases, no optimizer step/checkpoint/test; ALFWave
  ~2.4x faster than Full in this short probe. Three synthetic tests passed; two
  dataset target-perturbation checks gave exact zero. Accuracy remains untested.
- First probe hook failure is preserved under `outputs_v4/speed_fusion_probe_20260913/`.
- Diagnosis, limits and proposed new plan scope:
  `docs/ai/handoffs/T-20260908-support-ablations/SPEED_AND_GWNET_20260913.md`.

## Historical run-state checkpoint

Completion analysis for plan ef6bfd48…: all five METR-LA runs completed by
2026-09-13 18:10 +0800; four method candidates failed frozen gates. Full-MAE
LifeAvg 4.548997/6.947354/13.692081 improves MAE/MAPE but regresses RMSE versus
Full. It beats the retained MoGERNN validation table in all 33 cells (seed32 only).
PEMS-BAY is not launched. Detailed verified tables and timing interpretation:
`handoffs/T-20260908-support-ablations/METRLA_WAVE_RESULTS_20260913.md`.

2026-09-13 14:14 +0800 supplement: METR-LA five-arm plan `ef6bfd48f3c4b69840d06a7a3b2a61e03aa798de361efed26dbccbbd2114ba85`
is running, with preflight/config audit/concurrent probe passed. See
`handoffs/T-20260908-support-ablations/METRLA_WAVE_EXECUTION_20260913.md`.
All five workers survived the conversation interruption; PEMS-BAY is not queued.

At `2026-09-06 15:29:32 +08:00`, process/GPU inspection found no GPU process whose
working command belonged to this repository. Existing tmux sessions were not treated
as evidence of an active job. This observation expires immediately after the stated
time and must be rechecked before operational decisions.
# 2026-09-17 — cleanup release and final seed32 test (complete)

- Independent plan63b50760;20 validation gates then20 authorized test evaluations,
  max4 concurrent, using relocated unchanged best checkpoints. No retraining or
  test-based model selection; no task workers remain.
- Tables: `paper/test_seed32/` (Excel/HTML/Markdown/660-cell CSV/provenance).
  All660 numeric cells and rank formats verified; first bold, second underlined.
- Focal LifeAvg: METRLA3.931467/6.191635/10.426073; BAY2.808320/4.359018/6.646911
  (MAE/RMSE/MAPE), all first among final10. Main/phase first counts: METRLA15/17,
  BAY7/6; total45/66 first,13/66 second. Single seed; BAY Cold/Ready weaknesses remain.
-18 historical validation tables reproduced (max4.22672e-6);2 STFiT archived-source
  fixed-RNG full validation pairs identical. Historical STFiT RNG is unavailable;
  source stochastic sampling remains enabled, historical deviations disclosed.
- Cleanup evidence: `artifacts/cleanup_verification.json`, `release_completion.json`;
  293 best weights and1259 metadata files SHA-verified.156-file evaluated source/config
  snapshot retained alongside run results; original research history separately archived.

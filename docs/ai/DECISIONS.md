# Decision log

## D-20260917-FROZEN-RELEASE-AND-TEST

用户明确授权整理并清理历史产物后，用整理后的最终20权重（双速度数据集seed32）执行test。模型保持Focal，不再选择/优化。保留293份最佳权重及正文附录分析，旧源码/配置/日志/报告压缩按原路径归档并逐SHA核验，清理重复last.pt和旧目录。训练计划退休，test独立身份，不因整理伪造原训练配置哈希。

SLPF按已确认单阶段版本合入root。完整validation复算检查strict权重、scaler、暴露、样本计数及33指标；统一float32 `atol=1e-5,rtol=1e-5`，其确定发生在test打开前，依据同源码/同配置两次GPU重算的约1e-6相对波动，旧失败门槛证据保留。test使用checkpoint scaler、零TF，不允许按test选模型。历史test记录仍保留，不能称本数据集从未被评估。

## D-20260916-SIMPLIFY-SLPF-AND-RERUN-ONLY-NECESSARY

用户明确不追求SLPF复杂三阶段完整复现，优先保留源model并作最低限度协议适配。采用历史源forecast单阶段适配，保留共同masked full-N输入、日历、既有结构节点映射和历史AdamW/归一化MAE；不使用ALFNet创新。必须称“SLPF单阶段预测适配”，不声称完整原论文训练流程。METRLA历史canonical33结果复用；BAY旧micro选点统一后重训。原三阶段计划两项取消；其完整训练全缺失历史报错不通过额外模块补救。

最终范围仅6项：METRLA/PEMSBAY STID、STGCN，PEMSBAY STBP、SLPF。复用14项包含最终Focal两项，不再优化。5个非SLPF正式任务使用5ebe616f，SLPF新计划8eb8b2b9独立workspace等待空位，两个调度器合计max5。根冻结代码在当前5项结束前不改，之后再归档三阶段和合入简单adapter。无test/commit/push授权。

## D-20260916-LOCK-FOCAL-AND-REPAIR-BASELINES

用户明确最终Focal不再改，只精简弃用候选并修复九基线，之后再指定旧结果重训。保留原Full控制和论文消融，不做模型搜索，不沿用旧final_plan授权。清理前后选中checkpoint的CPU输出/损失/梯度在1e-7绝对/1e-6相对容差内，严格加载兼容。

SLPF完整源管线必须作信号预算适配：ADP仅遮住当前本来可见历史的20%并监督这些合法位置；forecast监督合法sensed未来；aggregation监督共同loss_mask∩unsensed。前两阶段分别以固定validation合法任务MAE选点，恢复各自最佳并冻结；最后仍按完整canonical33 LifeAvg选点。无聚合监督的批次不更新。每阶段最多max_epoch、独立优化器/可恢复阶段状态，最多3倍单阶段预算；不是原forecast-only结果的等价重命名，也不是原私有数据管线逐位复现。新运行必须独立冻结计划。

STID新配置恢复官方速度数据集288时刻槽+星期、Adam/lr.002/里程碑[1,50,80]/clip5/原尺度MAE，修复不等时间嵌入维度；STGCN恢复来源Adam/原尺度MAE。其他共同数据、scaler、mask、12/12、指标和最终选点不变。STBP固定图离线边界继续披露，不宣称在线跨年扩展已复现。详见`paper/final_code_audit_20260915/README.md`。

## D-20260915-FINAL-FOCAL-SEED32

八候选格补齐后，按用户要求选双数据集统一Focal，Full保留对照/备选。当前七可用基线池Focal赢35/66，Full28、频域27、历史30；Focal的LifeAvg MAE/MAPE两数据集改善，RMSE代价+1.54%/+0.40%。频域仅METR-LA小幅更好，在BAY不及Focal；历史也无稳定优势。此为seed32范围取舍，不撤销旧seed42门槛失败或宣称多种子最优；BAYCold等退化必须披露。

最终指针和报告位于`paper/final_selection_seed32_20260915/`。对比主表STID/SLPF缺合格结果留空，旧数值另列历史表；BAY STGCN/STBP的旧选点差异保留。第一名加粗、第二名下划线按未舍入值计算。当前表不是九基线统一修复重训后的最终论文表。没有新训练/test/commit/push，所有基线重训仍待独立计划。

## D-20260915-SEED32-COMPLETION

用户重新开放双速度数据集四方案seed32选择，明确九基线清单（不含GWNET），授权缺失项补训、ST-FiT/PatchSTG各两数据集重跑和适配清理。复用5格，补3格，基线重跑4，共7个新run/max5。独立计划459749c0使用冻结工作区；所有探针成功后自动正式队列。旧fb9bbbb3只做探针，因ST-FiT稠密图消息性能问题被等价矩阵实现取代，没有正式训练。Full四方案仍未最终选择；结果齐后统一选一个配方，不按格/种子拼接。

根目录修复STID时刻、ST-FiT批次图与合法合成/原尺度目标；合并已验证GinAR/UIGNN适配；统一GWNET源核心但不加入最终名单。失败原型/模块归档而非删权重。SLPF源三阶段未被旧wrapper调用，禁止把forecast-only作为完整算法正式重跑；三阶段合法训练适配是下一轮全基线计划的前置条件。STBP离线适配边界保留披露。所有基线重跑待最终模型确认后独立计划；本队列不开test/commit/push。

## D-20260914-PEMSBAY-TRANSFER-FOUR

用户在Full最终选择后明确要求再做PEMS-BAY两候选各seed32/42，共4run。选METR-LA最有区别的折中：Focal+历史修正及Focal+频域项，按BAY原Full配置迁移，不重新调参。上条停止/BAY未授权状态被本次范围取代；Full保留，最终跨数据集取舍待结果。无等价Full42，因此只在32对Full判门槛，在42进行两候选配对，不自动添加对照或宣称全面确认。独立计划34ecd952…；不开test、commit/push或额外run。

## D-20260914-FINAL-ORIGINAL-FULL

用户要求分析最后四运行并决定最终版。四组正常完成，频域/历史修正均未达到事前门槛；Focal seed42总体MAE近持平、MAPE略退化、RMSE+4.317%，seed32收益未稳定确认。历史修正虽赢Full27/33格，Unseen RMSE仍+10.422%、总体+3.025%。最终保留原Full架构与原role-balanced MSE，结束搜索；候选保留为归档，无默认开关晋升、无新训练/BAY/test。选择为配方而非挑最好seed。具体身份/指针：`paper/optimization/focal_last_round_20260914/final_selection.json`；证据：同目录`results_report.md`及`results_analysis.json`。

## D-20260914-FOCAL-LAST-PAIRED

用户在停止建议后明确授权最后一轮，并选择 Full42、Focal42、两新候选42，共4个METR-LA训练，最多4并发。该授权取代上一条停止状态，但不开放BAY/test或后续搜索。新候选为Focal+正交复数频域L1(0.1)、Focal+零初始化444参数合法可见历史直接修正，彼此不叠加。seed42会改变验证节点划分，判定只用同轮配对参照。保留原Full。冻结方案见`paper/optimization/focal_last_round_20260914/design.md`及独立计划`configs/paper/metrla_focal_last_v1.json`。

## D-20260914-END-OPTIMIZATION-SEARCH — evidence-based stopping decision

- User requested analysis and a decision whether to end or continue. All4 loss
  candidates completed but failed frozen gates; Focal improves Full-MAE overall
  yet retains Full-relative RMSE regression, particularly Steady.
- End current candidate search. Keep original Full; retain Focal as a tradeoff
  alternative, not an automatically accepted replacement. Do not relax old gates.
- C-S/C-M/C-E have already been trained; C-U learnable variant untested but fixed
  local intervention offered no consistent positive signal. Do not run it merely
  to exhaust the old list. No new BAY/test/seeds42/52 scheduled.
- If adoption is revisited, prioritize an independently scoped paired confirmation
  over additional coefficient search. Report: paper/optimization/mae_tail_results_20260914/report.md.

## D-20260913-MAE-TAIL-SCREEN — accepted execution scope

- User explicitly authorized the four proposed loss candidates after paired residual diagnosis.
- Keep Full architecture; run METRLA seed32 only: tail quadratic strengths0.25/0.5,
  source-verified MAE-Focal, and0.9MAE+0.1MSE. Max4 concurrent trainings, batch64,
  max100 epochs, original teacher forcing and LifeAvg-MAE checkpoint selection.
- MAE-Focal follows author code `2*sigmoid(.2*raw_absolute_error)-1`, gamma1,
  no detach; divide raw loss by train std and retain project mask/role aggregation.
- Independent identity `d90516065cc844010bdb13ee1e2a2552deb6d58535030bee56909a588e0d93cb`.
  Explicit MAE/MAPE retention and Full-RMSE recovery gates frozen in plan.
- No PEMS-BAY queue, new test, Full promotion or Git publication. Root source
  implementation with immutable run snapshot; do not create another outputs code workspace.

## D-20260913-01 — Preserve round1 and queue a distinct second round

- User explicitly authorized next-round code and automatic training after the
  current four runs complete; do not stop them or touch other user's PID1734562.
- Isolate next-round code in outputs_v4/round2_workspace_20260913 to preserve
  round1 source hashes and its original collector. No root model edits this turn.
- Round2: freshness and refiner_uniform × METRLA/PEMSBAY × seed32, four runs;
  no composition with round1 candidates, unchanged Full training budgets/protocol.
  Independent plan identity 159322422fa9ece24aa434832405b0824e6e1d867a411da8388d4bb7831c27b8.
- Queue requires all four round1 COMPLETE markers, successful round1 collection,
  round2 preflight passed and matching builder identity. No automatic restart.
- Three GPU phase slots with free-memory admission replace conservative two-slot
  scheduling for round2 only; log wait and execution separately. Other jobs are
  observation-only. No test, Full promotion, seeds42/52, commit or push.
- Details: handoffs/T-20260908-support-ablations/OVERNIGHT_20260913.md.

## D-20260912-02 — Two-speed diagnosis and bounded candidate execution

- **Authority:** user accepted the recommended next stage and explicitly allowed
  parallel training, at most five concurrent training jobs; both METR-LA and
  PEMS-BAY must be diagnosed. This supersedes the prior diagnosis-only stop.
- **Scope:** natural-validation mechanism diagnosis first, then at most two
  evidence-triggered single-mechanism candidates on both datasets, seed32
  (four candidate runs); concurrency never exceeds five. No new seed42/52,
  test access, Full replacement, commit or push is implied.
- **Diagnostic identity:** configs/paper/two_speed_diagnostic_v1.json,
  plan 4809eb19144dbce8b310437daea0ed06d1f0fa7d100d088cd9c28542730228c3.
  Each dataset uses 32 randomly selected natural schedule windows per
  scenario/phase stratum; selection seed 20260912, no target/error filtering.
  Full, Mask-only, Static and MoGERNN are checkpoint controls; uniform routing
  and a 0.75 observed-sender GRIN decoder mixture are paired diagnostic arms.
- **Research distinction:** these fixed inference interventions locate sensitivity,
  not retrained candidate performance. Expert-specific Theta means duplicate
  support matrices alone do not establish identical expert outputs.
- **Invariants:** Full and old outputs preserved, capacity METR-LA=8/PEMS-BAY=11,
  split/exposure/scaler/metrics/selection unchanged. New training requires its
  own frozen code/config/data identity and preflight within this authorization.
- **After both diagnostics completed:** first screen selects C-S route_shrink and
  C-M bounded_mae; C-U fixed group intervention gave no consistent Unseen benefit
  and is deferred. C-E is not included. This selects experiments, not a new Full.
  Plan `configs/paper/two_speed_mechanism_v1.json` has identity
  `afd9f69d16e5b59429a8eda619b7e9874e1e21cd28de4af4873ad3e86ee5bdf9`.
  Four seed32 runs inherit the dataset-specific Full training recipe, and freeze
  the proposed numerical screening gates. Details and limitations are recorded in
  `handoffs/T-20260908-support-ablations/EXECUTION_20260912.md`.

Only consequential, still-relevant decisions belong here. Evidence paths are links;
implementation status is stated separately from the accepted intent.

## D-20260906-01 — Shared AI context structure

- **Status:** accepted and implemented for repository guidance.
- **Decision:** keep thin `AGENTS.md` and `CLAUDE.md` entries; maintain shared workflow,
  stable context, decisions, experiments, and task handoffs under `docs/ai/`.
- **Reason:** prevent Codex and Claude from carrying divergent copies of project rules
  and from treating old chat as current authority.
- **Scope:** AI collaboration documentation only; no scientific code or result changed.

## D-20260906-02 — Common inductive contract, model-specific innovation

- **Status:** accepted design; implemented in the v4 data/runner paths at the checked
  HEAD, subject to plan re-freezing.
- **Decision:** all methods share dataset identity, exposure masks, split, scaler,
  origins, evaluation arithmetic, and checkpoint-selection rule. A baseline may keep
  architecture/loss behavior required by its source, but must not receive ALFNet-only
  lifecycle inputs or objectives.
- **Reason:** comparable evidence requires equal observable data, not identical model
  internals.
- **Source:** `lib/lcbt_v4.py`, `lib/runtime_context.py`,
  `experiments/v4_runner.py`, and `docs/comparison_model_audit.md`.

## D-20260906-03 — Current ALF choice is unified_BASE / AR_BASE

- **Status:** accepted as the current ALF baseline for the latest seed32 comparisons.
- **Decision:** LF, SUP, direct-head, masking/reconstruction, and other candidates are
  not selected merely because code or old outputs remain. Current follow-up starts
  from unified_BASE with the AR_BASE recipe unless a new validation-only decision
  replaces it.
- **Reason:** prior candidate work did not establish a uniformly superior replacement;
  preserving a reproducible baseline takes precedence over silent retuning.
- **Scope:** does not validate or authorize the stale 13-model final plan.

## D-20260906-04 — Nine-model reporting shortlist

- **Status:** accepted user requirement for the current reporting task; not yet encoded
  in a new formal multi-seed plan.
- **Decision:** report ALFNet, MoGERNN, HA, FCLSTM, STGCN, STGODE, STID, SLPF, and STBP.
- **Alternative replaced:** the older 13-model plan remains historical evidence but is
  not the current paper shortlist.

## D-20260906-05 — METR-LA patch 8 was exploratory (subsequently frozen)

- **Status:** originally accepted for the METR-LA seed32 validation experiment;
  superseded by the user freeze recorded on 2026-09-08 below.
- **Decision:** allow patch capacity 8 for that experiment, one token above the
  maximum original partition size of 7; do not describe it as the final uniform
  five-dataset setting without a new decision and frozen plan.
- **Reason:** the choice was informed by validation history and therefore requires
  explicit provenance and further confirmation rather than retroactive generalization.
- **2026-09-12 correction:** the former wording "one more partition" was inaccurate.
  This is capacity, not partition count; ALFTemporal adds actual node context tokens.
# 2026-09-08 — Authorized support replacement ablations

- User authorized exactly six seed32 runs, METR-LA/PEMS-BAY × Mask-only,
  static graph routing, local-only temporal context. Full training/results remain
  untouched. No new test access, commit, or push is authorized.
- Keep data exposure, target, loss, teacher forcing, batch64, checkpoint criterion
  and existing dataset-specific Full recipes. METR-LA patch8 remains frozen.
- A1 repeats the observation mask into role/age slots before all inference paths
  and uses a constant mean refinement scale; role-balanced training supervision
  is retained to isolate inference input conditioning.
- A2 retains the graph module/operators with a learned static mixture; 660 fewer
  parameters are disclosed rather than hidden with unused padding.
- A3 reuses the temporal branch's parameters for per-node temporal attention and
  shared latent-token conditioning, removing cross-node signal mixing only in
  that branch. Same parameter count does not imply equal FLOPs or expressivity.
- Four GPU phase slots admit six persistent workers without changing model math
  or batch size. New plan and snapshots are independent of stale final_plan.

## D-20260912-01 — Checkpoint before two-dataset model optimization

- **Status:** user intent recorded; issues inventory/checkpoint completed, new
  optimization not started.
- **Decision:** future optimization scope is METR-LA and PEMS-BAY. Preserve current
  Full results and distinguish overall gains from condition/metric weaknesses.
  GWNet is a potential reference, not an accepted fusion or guaranteed improvement.
- **Evidence constraint:** retain PEMS-BAY Mask-only/static counterevidence and
  held-out trajectory weaknesses. Do not improve apparent ranking by hiding controls
  or changing metric/test rules. Existing failed LF/SUP recipes are bounded evidence,
  not proof that every future operator or low-speed-error approach must fail.
- **Scope:** current request ends with handoff and stop; it does not initiate a new
  training/search plan or authorize test access, commit or push.
- **Details:** `handoffs/T-20260908-support-ablations/ISSUES_20260912.md`.

## 2026-09-13: GWNet fusion exploration authorized; Full retained

- User explicitly requested GWNet fusion and investigation of training speed.
  This supersedes the earlier restriction against selecting fusion by default.
- Eight completed small-mechanism candidates failed their screening gates.
  No Full replacement is accepted. ALFWave is an unregistered independent
  prototype replacing the recurrent predictor with a temporal-context-conditioned
  GWNET direct head; bounded speed/gradient/target-isolation checks passed.
- Proposed next independent scope is W-MSE / W-MAE / source GWNET-MAE on both
  speed datasets, seed32, validation only. Formal plan/config registration and
  audits are not yet complete; no full training was launched this turn.
- Never reuse old final_plan as new authority. No test, commit, push or action
  on protected external PID1734562. See the active handoff's SPEED_AND_GWNET report.
# 2026-09-17 — release ST-FiT stochastic replay boundary

Final model selection remains Focal; no test-based reselection. Retain ST-FiT's
source Gumbel sampling during evaluation. Historical best checkpoints do not store
the runtime RNG state, so their stochastic validation realization cannot be replayed
exactly. Do not disable sampling or enlarge numerical tolerance to conceal this.
For both datasets, compare current versus archived training source using the same
checkpoint/config and fixed training_runtime subseed32; verify dependencies bytewise.
Both full 33-cell validation pairs passed with maximum difference0 under release
plan63b50760. Historical maximum differences0.0443968527 (METRLA) and0.0239694599
(PEMSBAY) are recorded separately. The other18 checkpoints retain historical-table
validation gates. Test sampling uses the same fixed declared runtime subseed,
with unchanged evaluation-design seed12026 and checkpoint scaler.

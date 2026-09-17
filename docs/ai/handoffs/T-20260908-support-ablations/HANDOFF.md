# T-20260908-support-ablations

## COMPLETE — 2026-09-17: cleanup and all20 authorized tests finished

- **Facts**：Git `paper/final` / `9346dc0`，保留既有dirty改动，未commit/push。最新计划SHA `63b50760d680148364a493f6d9f70aa582ada871e0358c090c65c8e40c5a3f41`；全部20 validation门槛通过后仅打开一次test，20/20完成，当前无本任务评估进程/tmux。不要再启动已完成队列。
- **Completed / cleanup**：293份最佳权重、1259份配套元数据及旧研究档案SHA复核通过，main20/history273。9367个旧文件路径清理（约4.104GiB旧文件；其中必要权重/元数据已另存，不能当作净节省）。291原始配置可恢复，仅2个旧未完成MoGERNN尝试配置原先缺失，最终20项完整。`artifacts/cleanup_verification.json`及`release_completion.json`为完成证据。
- **Completed / test**：`paper/test_seed32/`包含Excel、HTML、Markdown四表、660格CSV、原始报告与verification；20项共同曝光/计数/scaler、TF0、有限结果通过；660个Excel数值和第一加粗/第二下划线回读通过。原validation表独立保留。
- **Results**：Focal两数据集LifeAvg MAE/RMSE/MAPE均第一。METRLA主表第一15/15、阶段17/18（其余Cold MAE第二）；BAY主表第一7/15、第二6/15，阶段第一6/18、第二6/18。总体45/66第一、13/66第二。BAY Cold/Ready等仍有弱项，不宣称全面领先或多种子显著优越，不依据test重新选模型。
- **Validation boundary**：18项历史33格复算最大绝对偏差4.22672e-6；两项ST-FiT固定RNG归档/当前全量配对差异0，历史采样偏差单列。ST-FiT源Gumbel保留，best无历史RNG；SLPF单阶段与STBP离线适配已披露。156文件本次源码/配置快照回读通过。定向18项与补充14项测试通过（有重叠），完整suite仅收集281项，未执行全套。
- **Modified but unverified**：没有待完成的本轮代码/表格验证；不声称完整suite通过。正文草稿未自动改写为test数值。
- **Not started / not authorized**：新训练、test调参/重复评估、三种子扩展、Git提交/推送；本轮无需启动。
- **Ruled out**：不把历史开发test称作从未打开，不把保留历史候选称作当前最终模型，不将适配版SLPF称作完整三阶段源方法复现。当前用户任务已完成，无需持续监控。

## Live checkpoint — all20 validation passed; test opened

Plan63b50760全部20项validation门槛通过，`VALIDATION_COMPLETE.json`与`TEST_OPENED.json`已生成，当前自动执行授权20项test、max4；尚不能声称test完成。两项ST-FiT完整配对差异0。评估源码/配置快照156文件逐SHA回读通过，位于当前评估目录`source_and_configs.tar.gz`，外部权重/数据/历史参考档案身份仍由plan固定。下一步等待TEST_COMPLETE及paper/test_seed32表格验证，再做最终交接；不重复test、不重训、不按test重新选模型。

## Current execution — plan63b50760

前一计划ST-FiT参考实例遗漏`from_runtime_context`绑定，报错发生于validation，未打开test。修正后两数据集各一完整validation批次归档/当前前向差异均0，证据`artifacts/stfit_pair_preflight.json`。新冻结计划`63b50760d680148364a493f6d9f70aa582ada871e0358c090c65c8e40c5a3f41`，tmux `release-test-seed32-verified`，日志`artifacts/evaluation/queue_verified.log`；ST-FiT优先完整复算。旧f04afaaf已结束并保留validation-only计划/日志。当前仍须等待全部validation门槛及20项test完成；数值模型/权重未修改。

## Latest — 2026-09-17: stochastic validation gate; release queue f04afaaf

- Facts：清理及293份best SHA复核完成。最新冻结计划`configs/paper/release_test_seed32.json` SHA `f04afaaf309f68b53a24b3e82f2736c49244b162c9485435fcff6550a9fc986c`，tmux `release-test-seed32-paired`，日志`artifacts/evaluation/queue_f04afaaf.log`。必须读取实时queue_status/TEST_COMPLETE，不沿用PID。
- Completed：18项定向测试及新增14项collector/seed/release协议测试通过；281项仅收集，未跑完整suite（部分测试涉及当前范围外PEMS08 test）。293最佳权重再次逐SHA核验一致。
- Diagnosis：STGCN float32复算误差约1.6e-5，使用统一abs1e-5+rel1e-5门槛。ST-FiT源模型eval保留Gumbel随机采样，历史best不含RNG，历史validation偏差约0.044并非浮点；不扩大容差。两项ST-FiT用固定training_runtime子种子、同权重/配置对照归档训练源码与当前源码做全量validation，依赖逐字节核验；18项仍与历史33格比较。
- Modified but unverified：配对ST-FiT验证与20项test正在队列中，完成以逐项报告和标记为准。此前三个失败计划仅运行validation，均无TEST_OPENED，原计划保留在各自目录`validation_only_superseded_plan.json`。
- Not started at entry：本轮test尚未打开；全部20项门槛通过后自动打开，最多4并发。
- Ruled out：不关闭Gumbel、不更改模型/选点/权重、不用test选择模型、不重训、不动其他用户进程、不commit/push。

## Latest — 2026-09-17: release cleanup complete; full validation and test explicitly authorized

- **Authority**：用户锁定最终模型，授权清理outputs/logs/paper/configs/paper/workspaces/scripts/tests，保留正文附录后续所需权重/配置，再用整理后的权重执行METRLA/PEMSBAY seed32各10模型test。覆盖此前no-test限制；不授权新训练/重新选模型/commit/push。
- **Facts / cleanup**：Git paper/final /9346dc0，既有dirty保留。293 best.pt逐SHA复制到`artifacts/models/{main,history}/<dataset>/<model>/seed<seed>/[run_id]/`，main20项；保留报告/原配置/来源。291配置按原SHA取得，其中46从Git历史恢复；两次旧未完成MoGERNN尝试原配置早已缺失，单列unresolved，不影响main20。最后快照293份及旧零散树清理；9367文件/约4.104GiB移除，保留权重约1.98GiB、压缩档案84MiB，因此不能称净节约4GiB。
- **Archive**：`archive/research_history_20260917.tar.gz`含8436个原路径成员，逐成员解压回读SHA核验；原输出元数据/日志/配置/源码/workspaces/旧tests均有追溯。索引`artifacts/archive_manifest.json`，原权重路径映射`catalog.json`。25个观测支持/案例分析文件额外提取到`artifacts/analyses/`方便正文附录使用。data/data_v4/references未动，其他人GPU进程未操作。
- **Code**：SLPF已合回完成训练时的单阶段adapter，删除根三阶段adapter/pretraining及过时tests；Focal/其他核心未改。root保留10个通用脚本，旧训练计划fail-closed退休。主配置为`configs/paper/final_seed32/`；ALFNet只重定位专家配置路径，原配置与哈希并存。validation表迁至`paper/validation_seed32/`。
- **Verification**：18项最终损失/基线修复/denseCheb/release-gate单测通过，保留测试281项收集成功（未声称全跑）。293权重与归档SHA通过。正式一致性验证是整理后20模型全validation重算，身份/scaler/计数严格相同。
- **Numerical gate**：两次validation-only计划1782b278、8d52ce1f因METRLA STGCN浮点差异触发过严门槛，均未打开test、保留日志和旧plan。复核config及模型/loader/collector/metrics源码字节相同，最大绝对差1.59e-5、相对1.13e-6。最终统一float32容差`atol=1e-5, rtol=1e-5`，不修改权重或指标公式，不依据test调门槛。
- **Running**：当前冻结plan `configs/paper/release_test_seed32.json` SHA `8f5b1faefd9fe8c6165d08c3e8238c1fec9b35e354c637682491f640cac987de`。tmux `release-test-seed32-final`，日志`artifacts/evaluation/queue_8f5b1fae.log`；状态/逐项结果`artifacts/evaluation/seed32/<hash>/`。max4且保留24GiB启动余量，先全20validation passed才写VALIDATION_COMPLETE/TEST_OPENED并评估test。失败停后续，不自动重试。不要直接沿用PID。
- **Test contract**：checkpoint scaler、TF=0、相同33格collector与冻结划分、严格load_state_dict，test完整预测要求有限。仅最终20权重，不按test选模型。旧更早计划的test报告已保留，此次不宣称数据集从未被评估。完成后自动`scripts/collect_release.py`生成`paper/test_seed32/`四表/Excel/格式核验。
- **Next**：持续核验validation与test队列；完成后检查TEST_COMPLETE20项及660格表。当前尚不能宣称test完成。代码与主配置已冻结，评估运行中不修改。

## Latest — 2026-09-16 23: tables updated after six completed baseline reruns

- **Facts / authority**：用户要求训练结束后更新表格；不重新选模型，最终Focal锁定。Git paper/final /9346dc0，已有dirty保留。fresh宿主检查无本轮worker/queue。5ebe616f的5个有效任务和8eb8b2b9的BAY单阶段SLPF均complete/exit0；原两个三阶段SLPF仍failed/已取消，不纳入。
- **Completed**：两个计划check通过；6新+14复用共20个checkpoint/报告/表SHA、最佳epoch、seed、共同暴露/scaler/计数核验；全20项canonical33选点，训练skipped0。新脚本`scripts/update_final_seed32_tables.py`只读已有validation工件，不推理。
- **Artifacts**：`paper/final_seed32_updated_20260916/`包含`final_seed32_tables.xlsx`、`final_tables.html`/`.md`、`all_cells_and_ranks.csv`、`results_and_provenance.json`、`verification.json`、`README.md`。四张表均Focal+九基线，无空行待补；660个Excel数值与第一加粗/第二下划线回读通过；按未舍入值排名、精确并列共享名次。
- **Results**：Focal METRLA主表第一11/15、第二1/15；阶段第一12/18、第二5/18。PEMSBAY主表第一5/15、第二7/15；阶段第一7/18、第二5/18。合计第一35/66、第二18/66。池为九基线，不能直接与旧七基线池格数比较；不是独立统计检验。
- **Limits / unmodified**：SLPF单阶段预测适配、STBP固定图离线及既有结构映射边界在表注披露；没有test/训练/新推理/commit/push，不更改旧表或checkpoint。根三阶段代码后续归档和单阶段合回仍未执行，本轮仅完成表格更新；不得恢复已取消运行。
- **Next**：本轮表格请求已完成；当前更新版以上述目录为准。代码收尾是独立待办，不自动重训/调参。

## Latest — 2026-09-16: user simplifies SLPF; five formal jobs active, one replacement queued

- **Authority / changed scope**：用户明确“SLPF不要分别训练三个阶段，保留源model，其他尽量适配，不必为完整流程过度复杂化”。采用历史源forecast主干单阶段适配，标注适配边界，不再三阶段。最终必要重训6、复用14：METRLA SLPF旧canonical33直接复用；BAY SLPF旧micro选点仍需重训。Focal不变。
- **Observed failure / supersession**：初始7GPU探针全部complete/exit0；正式METRLA三阶段SLPF遇到全历史无可见节点的窗口抛错，未产生完成结果。BAY三阶段依新用户指示，fresh读取队列与/proc核验后仅向本任务该worker发SIGINT。原两SLPF日志/status保留failed，不得重试。临时empty-sum修复workspace未启动，已ABANDONED。
- **Running**：计划5ebe616f当前5个正式worker为两DS STID/STGCN和BAY STBP，均进入真实训练，METRLA两项已完成epoch0训练+验证。初轮训练耗时METRLA STID48.6s、STGCN69.1s；验证178.8/185.0s；这是当时并发首轮，不外推固定速度。其他进程未动。
- **SLPF replacement**：`workspaces/slpf_single_stage_20260916/`保留历史SLPF预测核心，adapter AST除method_variant标签外相同，核心字节相同；train入口跳过三阶段pretrain。METRLA/BAY历史checkpoint严格加载、真实窗口反向/更新、目标/生命周期污染不变性、全历史缺失前向有限通过；证据`single_stage_slpf_checks.json`及`slpf_single_stage_semantic_parity.json`。保留历史AdamW/归一化MAE；没有新增ALFNet模块，不宣称完整原三阶段复现。
- **Queued**：新计划`workspaces/slpf_single_stage_20260916/configs/paper/baseline_retrain_seed32_20260916.json`，SHA `8eb8b2b9d1f5ec9f6db9e9158c7ac335a0512f231fc4377a6619a70741b27f6c`，仅BAY SLPF seed32；tmux `slpf-single-20260916`，日志`paper/baseline_retrain_20260916/slpf_single_queue.log`，输出`outputs_v4/slpf_single_stage_seed32_20260916/<hash>/`。持久等待原5项空位，再GPU探针→单项正式；合计max5，失败不自动重试。读取实时状态，不沿用PID。
- **Evidence / current docs**：`paper/baseline_retrain_20260916/README.md`顶部为当前权威，`eligibility_current.json`为14复用/6重训清单。`module_isolation.json`新做18个实际模块实例检查通过，均无ALF*、lifecycle_moge、source_grin或Focal目标。此前初始7项计划仅剩5项有效，SLPF由本节覆盖。
- **Deferred cleanup**：根代码正被5项使用，不能为SLPF改写其冻结身份；根SLPF三阶段源码/旧准备配置暂作原计划历史，不是最新SLPF运行权威。等当前5项结束再将单阶段适配合回根目录并归档三阶段及未执行修复草稿。不可自动重训这些已完成/进行中的5项。
- **Next / constraints**：核验五项进展及SLPF等待器存活；训练结束再汇总结果。不开test、不提交/推送，不改Focal、不复活旧队列。

## Latest — 2026-09-16: necessary seed32 baseline reruns explicitly authorized

- **Authority**：用户要求确保九基线不引入ALFNet独有模块/设计，复用合格旧结果，重训必要的METRLA/PEMSBAY seed32，最多4–5并发。替代前节等待授权；Focal锁定，不优化、不重训，不开test/commit/push。
- **Eligibility**：全部20项checkpoint/报告/33格表SHA、选点、共同暴露/scaler重新核验。复用13：Focal两项、UIGNN/GinAR/MoGERNN/STFiT/PatchSTG各两项、METRLA STBP。重训7：两数据集STID（源日历/优化器/原尺度MAE修复）、SLPF（合法三阶段替代forecast-only）、STGCN（Adam/原尺度MAE），以及PEMSBAY STBP（旧micro改canonical33选点）。证据`paper/baseline_retrain_20260916/eligibility.json`。
- **Plan**：`configs/paper/baseline_retrain_seed32_20260916.json`，SHA `5ebe616fbb6b16b1fa4aff200b75c3f3f0ba73369e92daa07f63f13585ec7888`。root脚本`scripts/baseline_retrain_20260916.py`注册新plan；源码/配置/payload/unseen/manifests/证据按SHA冻结。运行根仓库代码，outputs只存工件。训练期间不得改冻结代码；旧final_plan和初始未执行草稿不具权威。
- **Validation**：先核验旧18接口证据对应维护源码未变，源冻结实现差异检查支持复用；新增计划/源码篡改拒绝检查通过，2项test拒绝/仅ALFNet角色混合回归通过。worker验证实际模块不含model.ALF*/lifecycle_moge/source_grin，禁止Focal目标。各run GPU预检每阶段最多1轮/8训练批/2验证批，SLPF覆盖三阶段；全部成功才正式训练。预检不是论文新结果。
- **Running**：宿主tmux `baseline-retrain-20260916`，队列日志`paper/baseline_retrain_20260916/queue.log`。状态`outputs_v4/baseline_retrain_seed32_20260916/<hash>/{concurrency_probe,launcher_status}.json`；查看实时状态，不直接沿用PID。文件锁、已有队列拒绝、max5、12秒错峰、显存余量；成功后自动补位，失败不重试。num_workers0、CPU线程4、batch16。SLPF每阶段max100/patience10，故可能明显长于其他基线。
- **Boundaries**：无ALFNet生命周期模块/角色平衡Focal目标；共同协议掩码和源方法可见性保留。SLPF可见历史监督、结构节点映射、STBP离线主干而非跨年在线等适配差异继续披露，不声称完整源论文等价。详细说明`paper/baseline_retrain_20260916/README.md`。旧结果/模型/别人的进程均不改。
- **Next**：核验正式队列启动及进展；完成后按新工件身份汇总13复用+7新结果统一主表/阶段表，不自动增跑或改Focal。

## Latest — 2026-09-16 12:20: final Focal locked, cleanup and baseline repairs verified

- **Facts**：用户明确不再修改最终Focal，只清理放弃候选、检查修复九基线，之后用户再指定重训。Git paper/final /9346dc0 /upstream0:0，已有dirty保留；本轮未正式训练/GPU/test/commit/push。历史结果/权重/references/冻结工作区保留，不替换现有论文数值。
- **Completed**：活动路径移除频域/历史头、tail/MSE混合、LF/SUP/有界MAE、soft-shrink、多尺度support、DIRECT_HEAD及弃用实验启动器/构建器，52个源码/测试/脚本文件保存在`workspaces/archived_candidates_final_focal_20260915/`并有SHA清单。Full控制与论文消融保留。Focal独立`experiments/focal_loss.py`；根训练入口拒绝旧候选计划。
- **Repairs**：SLPF现为三个原源核心的ADP→forecast→agg，合法可见历史遮挡监督、阶段最佳恢复/冻结、独立Adam和可恢复状态；agg仅共同loss_mask∩unsensed，无监督批次跳过。最终仍完整canonical33选点。STID分钟/1440保留，修复tid/diw维度计算；新配置恢复官方288槽+星期、Adam/.002/里程碑[1,50,80]/clip5/原尺度MAE。STGCN恢复Adam/原尺度MAE。STBP固定图离线边界继续披露。
- **Verification**：`scripts/check_final_focal_cleanup.py compare`两选中checkpoint严格加载，两个合法窗口预测/损失/全梯度最大差METRLA1.862645e-9、BAY0；`scripts/audit_final_models.py`九基线×两数据集18/18通过，含训练/验证目标和生命周期隔离、同暴露/scaler、各阶段梯度；SLPF真实入口CPU各阶段单批预检通过。修复测试20通过，共享训练/选点/恢复回归91通过（有重叠），配置builder --check及diff --check通过。证据`paper/final_code_audit_20260915/`，详见README。
- **Operational note**：首轮扩展pytest在DataLoader fork后挂起，78通过后只对当时新核验的本任务pytest PID4192352发SIGINT，原日志保留；未操作训练/他人进程。测试辅助`_session`默认num_workers=0后91项于455s通过。旧PID不再作为操作权威。
- **Modified but unverified**：没有GPU/正式收敛或新准确率验证，没有完整仓库测试（含历史过时计划及真实test检查）。接口通过不证明原论文全流程等价；SLPF属于明确的可见历史预算适配、STBP不是在线跨年复现。根旧生成configs仍作历史，正式要用新准备配置。
- **Hypotheses / limits**：源配方修复可能改变基线表现，尚无新结果。Focal选择仍保留seed42和BAY Cold反例；清理不消除这些缺陷。SLPF最多3×max_epoch，聚合skipped须按有无unsensed监督解释，不套用全零要求。
- **Not started / next**：`configs/paper/final_models_v1/`20外层+4专家配置已准备且check通过，`prepared_manifest.json`明确不是运行计划。等待用户指定重训算法/数据集/种子/并发后，建立新的独立冻结plan/code/config/output身份，再做获授权GPU预检启动；不可用旧final_plan或维护快照启动。STID/SLPF等正式表仍待新结果。
- **Ruled out**：不再优化或替换Focal，不复活旧实验队列，不自动扩成九基线全跑、不打开test、提交或推送；未修改之前的结果表/选择工件。

## Latest — 2026-09-15 23:35: completed seed32 selection and ranked tables

- **Facts**：计划`459749c0482f937db06f13a42090be8578eb4ffb0b28178360e6cfd3c00e8324`七项complete/exit0，末项22:53:28结束，宿主无该队列worker/launcher/watcher。八候选结果齐全。Git paper/final /9346dc0，原dirty保留。
- **Decision / Completed**：双数据集统一选Focal，原Full保留；频域/历史不选。当前七个可用基线池赢格Focal35/66、Full28/66、频域27/66、历史30/66。MAE/MAPE两数据集改善，LifeAvg RMSE仍+1.54%/+0.40%。不得称多种子最优；BAY Cold RMSE+10.48%、旧METRLA42 RMSE+4.32%保留。
- **Evidence / verification**：冻结builder --check通过，26个运行工件SHA/CPU checkpoint元数据/选点/暴露scaler/计数审计通过，skipped0；Excel1320个数值及加粗/下划线逐格回读通过。只读取已有validation结果，没有训练/重新推理/test。新脚本`scripts/finalize_completion_results.py`；产物`paper/final_selection_seed32_20260915/`含analysis_report.md、final_selection.json、Excel、HTML/Markdown四表、历史四表、provenance与verification。
- **Modified but unverified**：只新增离线汇总和文档；未修改根默认训练配方或重跑完整仓库测试。保留先前适配语义限制，不能将来源审计等同重新复现。
- **Not started / blocker**：STID修复后无结果，SLPF完整三阶段仍缺；当前表两者留空排除排名，历史表另存旧值。BAY STGCN/STBP旧micro选点和STBP离线适配披露。九基线统一重训尚未开始，需要独立计划；当前不是最终统一重训论文表。
- **Ruled out**：不复活七run/旧队列，不按数据集挑不同候选或按格拼接，不自动开test/commit/push，不触碰他人进程。此前主Full决定为历史，本节及新final_selection.json为当前seed32选择权威。

## Latest — 2026-09-15: seven seed32 completion runs and baseline repair audit

- **Facts**：用户最新清单为UIGNN/GinAR/MoGERNN/ST-FiT/SLPF/STGCN/STID/PatchSTG/STBP；GWNET不入表但检查统一适配。最终模型重新在两数据集Full/Focal/频域/历史修正的8个seed32格中选择。缺METRLA频域/历史、BAY Focal，共3；STFiT/PatchSTG两数据集重跑4。只7新run，最多5，旧5格核验SHA复用。Git仍paper/final /9346dc0，dirty保留，无test/commit/push。
- **Completed**：7CPU/7GPU前后向；STFiT合法donor四代表epoch全origins检查，其他epoch运行时fail-closed；5配置Full清理前后预测/损失/梯度差0；6dense Cheb算子测试＋两数据集两阶段真实梯度对照，最大预测8.57e-8、梯度1.27e-7；36入口/修复测试、11dense/损失测试、根3新增基线测试通过。旧v1全部并发探针完成，但性能差，**没有正式训练**。
- **Running**：v2计划`459749c0482f937db06f13a42090be8578eb4ffb0b28178360e6cfd3c00e8324`，源码`workspaces/final_completion_frozen_459749c0482f937d/`。tmux `final-completion-20260915`自动先run_final_completion.py --probe，成功再正式七run队列。状态在`outputs_v4/final_completion_seed32_v2/<hash>/{concurrency_probe,launcher_status}.json`；核对这些文件及宿主进程，不用历史PID重启。独立源码不会随根目录后续适配变化。根CPU十算法×两数据集接口审计进行中，`paper/final_completion_20260915/interface_audit.json`逐项写入。
- **Modified but unverified**：根合入GinAR/UIGNN适配器、损失/optimizer/scheduler；STID分钟/1440修复；GWNET统一官方核心与最小adapter，原实现归档；STFiT修复及等价矩阵实现；拒绝SLPF forecast-only正式训练。尚需完成根20接口审计及最终启动核验。不能把接口通过当源方法全等。完整全仓测试未完成。
- **Hypotheses / limitation**：矩阵实现大幅降低STFiT显存/短批耗时，不改变数学运算，但未验证训练轨迹逐位相同。全训练收益未知。8格最终选择必须一套双数据集配方；旧seed42反例保留。旧基线赢格暂不作公平最终结论。
- **Not started / blocker**：SLPF完整ADP→forecast→aggregation尚未适配，原wrapper只forecast；新增formal guard避免再次错误重跑。需要合法伪未感知历史监督、分阶段optimizer/恢复/选点方案，并独立预检。STBP是离线主干适配，必须披露。全9基线正式重跑只在本轮模型确定后另立计划；当前不自动扩队列。
- **Ruled out**：不动其他人进程，不复活旧四BAY/四GinARUIGNN、旧消融/trajectory，不启动v1正式训练，不开test，不提交推送，不按格拼接模型/种子。失败原型/校准模块及对应历史测试归档于`workspaces/archived_candidates_20260915/`；权重/历史配置/冻结源码保留。
- **Verified launch update (13:44–13:47 +0800)**：v2全部7探针complete/exit0；正式5running2queued，宿主核验五个worker真实存活，GPU96%、约42.3GB；METRLA频域/历史、BAY Focal、METRLA STFiT/PatchSTG在跑，BAY两基线排队。根20接口全部通过，并逐项对Full核验train/validation exposure与scaler相同。新增根dense/基线9测试通过，官方GWNET source_model与references逐字节相同。SLPF语义阻塞不因接口测试通过而解除。
- **Collector / next**：tmux `final-completion-collector-20260915`只读已完成报告，自动写`paper/final_completion_20260915/completion_summary.json`与候选主表/阶段表CSV；当前5/8候选格，全部终态退出。后续读取状态与终态结果再确定最终模型。详细范围/审查见`paper/final_completion_20260915/README.md`；不要把当前候选CSV当最终公平九基线表。完整epoch耗时尚未统计，短探针含固定日程准备，不能线性外推全轮耗时。

## Latest — 2026-09-15: final selection and two-speed seed32 tables

- **Facts**：旧四BAY候选及四GinAR/UIGNN均complete/exit0，宿主无相关worker。用户要求最终选择，随后明确保留两个版本（一主一候选），并确认GWNET纳入新表。
- **Completed**：主版原Full，候选Focal；频域/历史修正不作为最终候选。新文件位于 `paper/final_two_speed_seed32_20260915/`：Excel（四张正式表＋说明＋Focal候选）、Markdown、CSV、完整结果与SHA来源、最终checkpoint指针及选择说明。正式表为原Full+MoGERNN/STGCN/STID/SLPF/STBP/GWNET/GinAR/UIGNN，仅METRLA/PEMSBAY seed32 validation。594格正式值和候选33格导出回读核对；BAY Focal留空。
- **Evidence / limitations**：METRLA32 Focal相对Full MAE−2.17%、MAPE−7.87%、RMSE+1.54%；对八基线最优值赢16/33，Full9/33。已有seed42 Focal RMSE+4.32%不能因为不做三种子而忽略。BAY无单独Focal；METRLA无频域/历史修正seed32，不能跨seed代填。BAY两扩展均未过原定门槛。
- **Hypotheses**：Focal适合优先MAE/MAPE的METRLA32描述性展示，但尚无双数据集/多种子总体最优证据。原Full并非所有候选赢格最多。
- **Modified but unverified**：只新增离线汇总脚本/表格并更新交接，没有重新推理或训练。保存工件身份、历史最优checkpoint、数据暴露/scaler/统计计数核对通过；不宣称重新复现实验。BAY STGCN/STID/SLPF/STBP历史micro选点与当前33格选点不同，表下注明；GWNET两数据集适配配方不同也注明。
- **Not started / ruled out**：不跑第三种子/补缺run，不开test，不commit/push，不改原报告/权重。GCN-M主输入本已12，外部x_hist越预算，仍无合格结果。所有训练已结束，不要复活历史launcher。最终指针 `paper/final_two_speed_seed32_20260915/final_selection.json`。

## Latest — 2026-09-15: missing baseline adaptation and persistent wait

- **Facts**：当前权威为用户指定的 `paper/baseline_adaptation_codex_prompt.md`，替代旧优化范围；六个 seed32 槽位为 METRLA/PEMSBAY × GinAR/GCN-M/UIGNN，validation-only。Git paper/final / 9346dc0 / upstream0:0，已有 dirty 保留，旧四项依赖根目录的训练代码未改。
- **Completed**：GinAR model1 核心保留，普通结构编码适配节点参数，注意力等价分解；补全 UIGNN structure_full/EDL/quantile 源文件，保留全图主体和重建，修复独立分位数及12步头。四个 CPU 生产路径前后向、合法 mask、信息隔离、checkpoint/续训通过；9项针对性测试通过。新冻结plan `725a80ed153317b77834823ee42b696e623f65848f59357cfa5d55427a79c8bd`，builder --check；源码快照位于 `workspaces/missing_inductive_baselines_frozen_725a80ed153317b7/`。
- **Hypotheses / blocker**：GCN-M 官方 x_hist 实际越过12步预算，合成索引5000…5011读取到962；零 nh/nd/nw 配置在官方函数报错，重复当前窗口被明确禁止。两项协议阻塞，不伪造适配；其他四项不受影响。UIGNN概率头属于明确修复，未验证概率校准；GinAR共享节点参数并非整模型原样数值等价。
- **Modified but unverified**：GPU单批、四个合法任务并发探测、正式结果均未开始；须等旧四项真实终态。完整仓库测试未跑，未访问test。CPU热身数字不作为模型性能结论。
- **Running / next**：宿主 tmux `missing-baselines-20260915` 中唯一持久调度器按30秒轮询旧终态，保存状态并上锁。当前旧四项仍在跑，已完成epoch39，进入40（0起算）；不重启/停止它们或他人进程。等待结束后自动单批GPU→并发探测→四项正式训练。两项GCN-M保持阻塞，不能称六并发。详情、状态/日志路径见 `MISSING_BASELINES_EXECUTION_20260915.md`。
- **Ruled out / not started**：不开test、不commit/push、不重训Full/旧消融、不改论文主表或最终模型选择、不用旧final_plan授权。首个准备快照ede4a845…在GPU启动前被新版取代，仅停止了本任务自己的等待器以加强重启防重复保护；保留旧状态证据。不要复活旧等待器。

## Latest — 2026-09-14 17:40: PEMS-BAY four transfer runs active

- **Facts**：用户明确要求PEMS-BAY两最优候选×seed32/42，共4run，再确定最终模型。选择Focal+频域损失、Focal+444参数历史修正，复用已核验设计，保留BAY原Full底座/patch容量11。最多4并发，不开test、不加对照训练，不操作他人进程。
- **Completed**：根目录新增独立builder/preflight/worker/launcher/collector、21针对性测试通过；计划`34ecd952b01ebbad540a1345260d2793f37e4b71f743c817cc492e6c410b3a0a`冻结并check通过。入口仅添加新plan注册，模型/损失不变。Git paper/final HEAD9346dc0/upstream0:0，dirty保留。
- **Hypotheses / limits**：METR-LA折中是否迁移到BAY未知。仅有等价Full32；找到的Full42旧结果是更早配方，不能当当前Full42。因此32对Full32判门槛，42只两候选配对/排序稳定性，不宣称两种子都超Full。
- **Modified but unverified**：4CPU/4GPU单批及配对审计全部通过，旧Full8窗口预测差0，每seed初始化/RNG/暴露/scaler/target-blind通过。四并发16批全部exit0，训练段38.86–39.97s、峰值预留15.73GiB；正式训练收益未知，完整仓库测试未跑。
- **Running / next**：tmux `pemsbay-focal-transfer-20260914`，launcher3300858；spectral32 PID3300898、linear32 PID3301018、spectral42 PID3301482、linear42 PID3301617均进入正式train_one_epoch，17:40存活检查通过。GPU利用率96%、每组约16.8GB（16776–16794MiB）。新输出`outputs_v4/pemsbay_focal_transfer_v1/<hash>/`自动更新报告/summary，snapshot已保存。不要重复启动。完整执行记录见`PEMSBAY_TRANSFER_EXECUTION_20260914.md`。保留旧Full及METR-LA失败证据；BAY明确授权取代之前的停止范围。
- **Ruled out**：不调新强度/不叠两个候选，不重跑METR-LA/旧消融，不开test/commit/push，不把4run偷偷扩成更多。设计：`paper/optimization/pemsbay_focal_transfer_20260914/design.md`。

## Latest — 2026-09-14 16:20: four results audited, original Full selected as final

- **Facts**：四组COMPLETE/exit0，15:56:24全部结束，原launcher/四worker均退出。Full/Focal/频域/历史修正分别67/65/65/73epoch，选56/54/54/62（0起算）。LifeAvg分别4.467556/6.641539/12.710354，4.465825/6.928255/12.716585，4.400747/6.884835/12.645045，4.414710/6.842457/12.269469。三改动均未过预注册门槛。
- **Completed**：builder --check、checkpoint CPU元数据、选点/表/collector/33格计数/暴露scaler配对、全部skipped0、阶段日志核验通过。离线分析脚本`analyze_focal_last_results.py`，完整报告`paper/optimization/focal_last_round_20260914/results_report.md`，最终指针`final_selection.json`。完整epoch中位约4.49分钟。
- **Decision / Hypotheses**：用户要求决定最终版，保留原Full+原role-balanced MSE；Focal seed32收益未在42配对稳定重现。历史修正赢Full27/33格但Unseen RMSE+10.42%，不是可忽略边界失败；场景平均归因显示Unseen主导退化，未作单样本因果诊断。
- **Modified but unverified**：仅新增离线分析/选择记录及更新交接文档；没有改生产模型或默认配方，无重新推理、GPU计算、完整仓库测试。原dirty及全部候选权重保留。Git paper/final HEAD9346dc0/upstream0:0。
- **Not started / Ruled out**：搜索结束，不开新训练/BAY/test，不commit/push，不将Full42替换seed32表或挑最好种子。Focal/频域/历史头只作归档，不成为最终主版本。PEMS-BAY保持原Full，不能据本轮宣称跨数据集收益。下一步仅依据最终冻结记录进行论文结果/缺陷表述；无待启动任务。

## Latest — 2026-09-14 10:50: final bounded seed42 round, all four running

- **Facts**：用户明确接受2对照+2新候选，四个METR-LA seed42从头训练、最大四并发。原Full保留，最后一轮；不扩BAY/test/commit/push。seed42还改变V/T划分，须按本轮Full42/Focal42配对，不能沿用seed32绝对门槛。
- **Completed**：8篇新增原始文献定向筛选（含方法/源码与摘要筛选范围区分），选Focal+0.1复数频域L1、Focal+444参数可见历史直接修正；19合成测试及builder/check、git diff --check通过。计划`8e2a8b3b2009ecb5a0ddf3225699290ba39fee68967de9c58449f5980f1f59ca`。Git paper/final HEAD9346dc0/upstream0:0，原dirty保留。
- **Hypotheses**：联合12步误差约束、可见历史直接修正可能追回Steady/接入RMSE；Full-MAE残差诊断不能冒充Focal直接机制证据；不保证最优或收益。
- **Modified but unverified**：根目录两个opt-in模块、runner/入口、新builder/preflight/worker/launcher/collector已实现。旧Full8验证窗口预测差0，四组seed42共同初始化/RNG/暴露/scaler一致、target-blind通过；4CPU/4GPU单批全部通过。四并发16批全部exit0，训练段17.39–18.35秒、峰值预留10.11GiB。完整训练收益未知，完整仓库测试未跑。
- **Running / Not started**：tmux `metrla-focal-last-20260914`，launcher3065848；Full3065860/Focal3065986/频域3066103/历史修正3066228均进入正式train_one_epoch，10:50检查存活，GPU利用率91%、各约11028MiB。新输出`outputs_v4/metrla_focal_last_v1/<hash>/`自动更新candidate_summary.json/report.md；禁止重复启动。BAY/test/额外种子未开始。详见[执行记录](FOCAL_LAST_EXECUTION_20260914.md)。
- **Ruled out / next**：无需再次确认授权；不再堆失败LF/SUP/tail网格，不误读patch容量8；旧plan源码哈希可能因本次代码变更失效，完成运行使用旧snapshot，不刷新旧身份。10:42只读GPU约94GiB空闲，无本任务旧训练；没有操作他人进程。设计：[最后一轮方案](../../../../paper/optimization/focal_last_round_20260914/design.md)。

## Latest — 2026-09-14: four loss trials analyzed; optimization search ended

- **Facts**：d9051606…四训练全部COMPLETE/exit0，最后01:48结束。tail025/tail050/Focal/mix分别71/60/63/70epoch，所选60/49/52/59，均正常早停。四候选均未通过冻结门槛。
- **Completed**：builder --check、run_report/33格/summary一致性、历史最小LifeAvg-MAE选点、无skipped、赢格与阶段计时核验通过。Focal总体4.537516/6.911326/13.475396，对Full −2.17%/+1.54%/−7.87%；对Full-MAE总体三项均改善，原8第一27/33、含GWNET第一15/33，对MoGERNN33/33。
- **Hypotheses**：Focal是有价值的MAE/MAPE折中备选；其Steady/接入RMSE代价尚存，多种子稳定性未知。两个尾部强度导致Unseen明显退化，不能据loss形式推断尾部一定改善。
- **Modified but unverified**：本轮仅新增离线结果分析脚本/报告与文档；没有重新推理、读取权重、运行GPU或完整仓库测试。全部生产代码、旧checkpoint保持。
- **Not started**：新配方、PEMS-BAY扩展、新种子确认、test均未开始。本次用户要求决定继续或结束，决定结束当前优化搜索；保留原Full，Focal不自动晋升。
- **Ruled out / next**：C-S/C-M第一轮已做、C-E第二轮已做，不再当未完成清单；C-U只做固定局部干预、未测可学习完整版，现无前置正响应支持训练。停止密集loss网格/失败模块堆叠；未来如考虑Focal应先独立确认而非继续搜配方。详见[结果与停止决定](../../../../paper/optimization/mae_tail_results_20260914/report.md)。无test/commit/push/他人进程操作，dirty worktree保留，HEAD9346dc0。

## Latest — four METR-LA loss candidates running (2026-09-13 19:49 +0800)

- **Facts**：用户授权四候选训练。根仓库实现tail025、tail050、源MAE-Focal、0.9MAE+0.1MSE；独立计划`d90516065cc844010bdb13ee1e2a2552deb6d58535030bee56909a588e0d93cb`，仅METRLA seed32，最大4并发。
- **Completed**：15项合成测试、配置/plan loader审计、4CPU/4GPU单批预检全部通过。Full及四候选加载Full权重后8窗口原尺度预测最大差0；各候选1037529参数，batch64峰值预留显存约10.08GiB。Focal作者源码已取得，使用2sigmoid−1、原尺度beta0.2、gamma1、无detach，证据纳入哈希。
- **Hypotheses**：尾部约束可能回收RMSE并保留MAE/MAPE，尚无完整训练精度结论。
- **Modified but unverified**：生产runner仅增加opt-in损失分派，训练入口增加独立计划注册；完整仓库测试未跑，四并发16批次探针已全部exit0，约18秒/候选，各峰值预留10.1055GiB；完整训练收益未知。
- **Running / Not started**：tmux `metrla-mae-tail-20260913`，launcher2596935；tail025 PID2596948、tail050 PID2597092、focal PID2597232、mix PID2597686全部进入train_one_epoch，GPU各约11026MiB。PEMS-BAY/test/新种子/自动晋升不执行。
- **Ruled out / next**：上次额度耗尽阻止探针，用户要求继续后只读核对无已有进程，完成探针再启动正式训练；不要重复启动。查看新输出根launcher_status.json、logs/、report.md及candidate_summary.json。保留原Full/Full-MAE、全部dirty改动；不操作他人PID1734562。执行详情见[四候选执行](MAE_TAIL_EXECUTION_20260913.md)。

## Latest — Full-MAE paired residual diagnosis and loss candidates complete

- **Facts**：Full-MAE保留Full全部结构，训练base/role损失MSE→MAE。原8基线第一格数主表10→12、阶段12→13；加入本轮GWNET后主表5→7、阶段4→4；对MoGERNN32→33/33。
- **Completed**：两个checkpoint配对验证推理，6725 loader窗口/6696有效窗口/874197位置；33格独立复现最大偏差1.913e-7，目标/mask元数据一致、正式加权RMSE精确归因通过。诊断身份`6c867b03c41f3ad12765ed0d13b37c87e8f15a0ed9282b996bae8a742fca63ad`，输出`outputs_v4/full_mae_residual_diagnosis_v2/`。v1在Full完成后嵌套路径失败，旧证据保留并哈希复用，仅补跑MAE。
- **Hypotheses**：新增/持续大残差、40–60mph和较大速度变化、后半段预测更值得针对。原Full尾部整体改善，不能只讲旧离群点恶化；低速收益应保护。不是因果或多种子结论。
- **Modified but unverified**：仅新增两个诊断/分析脚本、研究报告与图表，没有改生产模型/训练。推理配对和数值验证通过，未做完整仓库测试。Focal作者util.py网页抓取失败，源码因子/尺度仍待核对。
- **Not started**：温和berHu两强度、MAE-Focal、MAE/MSE混合四候选仍为草案；未实现/冻结训练计划、未训练或排队，PEMS-BAY/test/新种子未开始。
- **Ruled out / next**：不以新门槛追认旧失败；首选保持Full结构的损失修正，目标保留80%MAE/MAPE收益并追回RMSE。核对源实现、根目录实现与测试、独立计划及资源范围确定后才能训练。完整报告：[诊断与11篇文献筛选](../../../../paper/optimization/full_mae_residuals/report.md)。无test、commit/push、他人进程操作；原dirty worktree保留，HEAD9346dc0。

## Latest — METR-LA wave screen complete and analyzed

- **Facts**：ef6bfd48…五运行均COMPLETE/exit0，最后完成2026-09-13 18:10；四方法候选均未过冻结门槛。Full-MAE总体MAE/MAPE改善1.92%/6.39%，RMSE退化2.06%；对原MoGERNN33/33更好。大替换MAE对GWNET总体MAE仅改善0.55%，未达到大幅优势。
- **Completed**：本次builder --check通过；五run_report/所选table/collector一致，history最小值及早epoch平手规则核对通过，skipped均0；完整33单元与实际阶段耗时分析见 [结果分析](METRLA_WAVE_RESULTS_20260913.md)。
- **Hypotheses**：损失目标而非额外结构是更值得继续的方向；RMSE退化是否集中在残差尾部尚未配对诊断。
- **Modified but unverified**：本次仅分析/交接文档，无模型/训练改动；未重新推理、未验证多种子或test。
- **Not started**：PEMS-BAY不启动；MAE/MSE混合候选、残差诊断、统计批量化仅为后续建议，未形成新执行计划。
- **Ruled out / next**：保留Full和Full-MAE证据，不合并失败TCN/大替换，不重复五训练。独立源码目录及snapshot继续保留，不当普通输出清理。无test、commit/push、他人进程操作。

## Latest — 2026-09-13 14:14 +0800: five METR-LA trainings survived interruption

- **Facts**：用户收窄为先5个METR-LA，PEMS-BAY不排队。独立计划ef6bfd48f3c4b69840d06a7a3b2a61e03aa798de361efed26dbccbbd2114ba85已启动；tmux `metrla-wave-screen-20260913`、launcher2381162及五worker均活跃，额度中断没有停止训练。最新完成epoch索引约3–5，不能下精度结论。
- **Completed**：13合成测试、5CPU/5GPU预检、Full/零残差8窗口预测差0、源文件及前后向对照、配置审计、五进程同步16批次预检passed；正式源码snapshot已保存，当前builder --check仍通过。
- **Hypotheses**：Full-MAE、轻量TCN、源GWNet替换MSE/MAE与原生GWNet参考可区分损失和结构收益；完整结果未知。
- **Modified but unverified**：新训练行为仅在独立目录，未合并根代码。根与独立目录对比：154个Python文件相同，4个既存文件改变，9个Python文件新增；复制不代表全部修改。完整仓库测试未跑。
- **Running / Not started**：五训练PID2381183/2381332/2381622/2381786/2382045；PEMS-BAY、新test及自动晋升未开始。不要重复启动。
- **Directory authority**：实际cwd及Python导入根为 `outputs_v4/wave_fusion_workspace_20260913`；data/data_v4/references/paper/outputs_v4为指向根目录的符号链接，未复制数据集。这个目录不是可随意删除的普通输出。运行中不移动/修改源码；结束后再将选中改动整理成根目录可审阅补丁，保留实验snapshot。源码放在outputs下容易混淆，应避免后续沿用该目录组织。
- **Next / Ruled out**：继续已有五训练，查看结果report.md/candidate_summary.json；无重启、无他人PID1734562操作、无test/commit/push。执行详情见 [METR-LA五对照](METRLA_WAVE_EXECUTION_20260913.md)。

## Latest — 2026-09-13: eight runs complete; GWNet prototype profiled

本节取代下文“运行/排队”状态。详见 [两轮结果、速度与GWNet融合](SPEED_AND_GWNET_20260913.md)。

- **Facts**：两轮8训练全部COMPLETE，均未通过筛选；Full保留。用户已明确允许GWNet融合。第二轮实际parallel3、无实质phase等待；METR-LA完整epoch中位4.6分钟、PEMS-BAY7.48–7.78分钟。
- **Hypotheses**：用GWNET直接预测器替换图循环可兼顾速度/精度；MSE→MAE需单独对照。速度收益已测，精度仍未知。
- **Completed**：新ALFWave独立原型，3合成测试passed；两数据集Full/GWNET/原型共6项GPU短批次，原型约2.4倍前后向加速，target-blind差0。probe身份4423e8eb…；旧统计CPU批量拷贝的局部计时/归约核对完成。
- **Modified but unverified**：仅新增model/ALFWave、两性能脚本及tests/test_alfwave.py；未登记正式训练路径。完整33单元/选点统计优化等价性、融合精度、完整仓库测试未做。
- **Not started**：下一轮完整训练、正式two_speed_wave_fusion_v1冻结/配置builder与审计。建议三臂×两数据集seed32，详见新文档；该建议不是可执行旧plan。
- **Ruled out**：不重启两轮、六消融或trajectory/q分析；不重复失败小修补配方；不开test、不commit/push、不操作PID1734562。
- **Git / next**：paper/final，HEAD9346dc0，本地upstream0/0，原改动均保留。下一步在独立正式身份中接入W-MSE/W-MAE/GWNET参考并完成审计；先测并发吞吐，再启训练。现无本任务完整训练运行。

## Latest — 2026-09-13: round1 continues, round2 queued after completion

用户要求保留运行中四训练，完成后自动跑下一轮；明确不操作他人PID1734562。
详见 [耗时分析与夜间串联](OVERNIGHT_20260913.md)。以下09-12状态为历史。

- **Facts**：第一轮仍运行，源码/计划afd9f69d…完整保留。四进程只有2个GPU整阶段槽，epoch耗时包含排队；METR-LA中位约8分钟、PEMS-BAY约10分钟。PID1734562只读观察，不作任何操作。
- **Hypotheses**：第二轮测试freshness与refiner_uniform分别定位时间支持表示和输出细化缩放；不是继续叠加失败配方。
- **Completed**：独立代码目录`outputs_v4/round2_workspace_20260913/`和计划15932242…已冻结，4个seed32候选，builder检查及合成测试通过。原代码无变化，避免第一轮收尾哈希失败。
- **Modified but unverified**：第二轮4CPU/4GPU batch64预检全部通过，八组旧Full/控制预测最大误差0；24相关测试passed、3 deselected。新候选实际收益未知，完整仓库测试未执行。
- **Running / Not started**：一次性串联tmux `two_speed_overnight_20260913_v2`已运行，阶段waiting_for_round1；第二轮正式训练尚未启动。第一轮四COMPLETE和原collector结果+第二轮预检passed是自动启动门槛。
- **Next**：查看`outputs_v4/two_speed_overnight_20260913/two_round_report.md`、同目录JSON及CHAIN_ERROR.json（若存在）。第二轮GPU阶段槽3、显存空闲门槛22GiB，不干预其他任务，不保证明早全部训练完成。
- **Ruled out**：不停止/重启第一轮，不触碰1734562，不开test，不commit/push，不自动替换Full。不要重复启动串联任务。

## Current execution — 2026-09-12, both diagnostics complete; four runs launched

用户已明确授权双数据集诊断及并行训练（最多5个同时运行），覆盖下文旧“未授权训练”状态。
本轮执行说明见 [EXECUTION_20260912.md](EXECUTION_20260912.md)。

- **Facts**：两数据集自然验证机制诊断 COMPLETE；诊断计划 `4809eb19…`。四训练独立计划 `afd9f69d…` 已冻结，选择 C-S route_shrink / C-M bounded_mae × METRLA/PEMSBAY × seed32。Full 保留，无 test/seed42/52/commit/push。
- **Hypotheses**：有限收缩可减少部分状态负迁移、有界 MAE 可缓解低速度偏差；尚无新候选训练收益。
- **Completed**：每数据集256自然窗口 × 六 arm；输入/目标配对、完成时来源哈希核验通过。新机制相关测试22 passed/3 deselected；builder --check 和训练/协议/嵌套配置一致性检查通过。两数据集四控制各8窗口的新旧代码预测原尺度最大差异均0。
- **Modified but unverified**：新增 opt-in 路由/损失、训练计划/启动/收集脚本；四CPU batch1及四GPU batch64单批预检全部通过。完整仓库套件未跑（含真实test测例）；新候选最终效果未验证。
- **Running / Not started**：四候选正式训练已启动；candidate_summary、多种子确认、test尚未完成/执行。不要把单批预检当正式训练结果。
- **Ruled out / deferred**：C-U 固定增强GRIN观测消息无一致Unseen收益，本轮不训练；C-E暂缓；不整体去路由、不原样复用LF/SUP。
- **Next**：`tmux` 会话 `two_speed_mechanism_20260912` 已启动 `scripts/run_two_speed_mechanism.py --parallel 4`（GPU阶段槽2）；检查新输出根 launcher_status.json 与 logs/，完成后自动收集所选 checkpoint 的33单元。不要重复启动；旧实验禁止重启。实际状态以文件与当前进程核验为准。
- **Identity warning**：新代码改变使旧诊断/消融计划在当前源码下check过期，完成时哈希已通过且诊断源码快照已保存；不要重写旧计划。当前训练权威只有 two_speed_mechanism_v1.json。当前HEAD仍9346dc0，所有既存改动保留。

## Latest checkpoint — 2026-09-12 resumed: diagnosis and candidate design complete

本次用户已接续原停止点，授权两数据集的诊断与候选设计。最新交付是
[候选与验证门槛](OPTIMIZATION_CANDIDATES_20260912.md)、
[离线重算证据](DIAGNOSTIC_EVIDENCE_20260912.json) 和
[重算脚本](reanalyze_saved_validation.py)。以下旧 checkpoint 保留为历史，不再表示本轮尚未开始诊断。

- **Facts**：branch/HEAD 未变，相对本地 upstream 0/0、未 fetch、暂存为空；原 88 文件哈希 87 匹配，只有 ISSUES_20260912.md 漂移，原文与快照均保留。没有覆盖既存改动。
- **Hypotheses**：有限路由收缩、观测/重建消息分组、观测新鲜度编码、有界 MAE 辅助项均仅为候选；未确认机制因果关系。
- **Completed**：重算 Full 对内部控制的全部主/阶段差异；PEMS-BAY Join 优势与 Steady/Fault 负收益已分开。METR-LA 已保存 held-out 配对样本中，低于 20 的速度箱解释多条 MAPE 退化的大部分贡献，但只有 4 节点/6 窗口，不能外推自然 Unseen 或 PEMS-BAY。
- **Modified but unverified**：新增离线脚本及设计/证据文档；脚本执行及配对/指标断言通过，无模型验证。候选阈值是拟议工程门槛，未冻结为运行规则；环境声明差异仍待解决。
- **Not started**：自然验证 Unseen 分箱、内部路径干预、有效路由测量、候选实现、任何新训练/重评/test。无可执行新 plan；seed42/52 未授权。
- **Ruled out**：整体去状态作为默认修复；仅凭低速度少量样本决定两数据集损失修改；旧 final_plan 或候选文档直接授权运行。
- **Commands / outputs**：git status/rev-parse/rev-list/diff，rg/cat/sed；ReaCH Python 3.11.13 标准库 SHA256/JSON/CSV 汇总，`reanalyze_saved_validation.py` 输出到 DIAGNOSTIC_EVIDENCE_20260912.json；apply_patch 文档编辑。无训练模块导入、GPU、模型 forward、原始 test 读取、commit/push。
- **Checks**：git diff --check、文档链接/行尾空白、JSON 解析、四个重算输入/脚本哈希通过；600 个分箱记录构成 120 组，样本计数与 MAPE 分箱贡献守恒通过。保留 66 行完整内部控制比较；未运行模型或完整仓库测试。
- **Next**：审议 C-S/C-U 的前置诊断范围；若需要新增 validation-only 推理，先明确样本、路径干预臂及独立身份。只有通过机制诊断的最多两个候选进入拟议 seed32 四运行上界；这不是本轮训练授权。

## Previous checkpoint — 2026-09-12, optimization issues inventoried; stopped then

用户要求基于现有实验梳理模型缺陷与遗留问题，完成 checkpoint 后停止。未来优化仅涉及 METR-LA、PEMS-BAY。本轮没有训练、模型修改、推理重评、新 test 访问、commit 或 push；尚未选择或启动任何新优化方案。

先读 [问题清单](ISSUES_20260912.md)，再读 [代码与证据快照](CHECKPOINT_20260912.json)。不要从旧对话重新启动已完成的六个消融或两个分析实验。

## Facts / 已验证事实

- 当前分支 `paper/final`，HEAD `9346dc0fc1dfb6b4aebfffe25224f1337f7a94cb`；相对本地 upstream 引用 ahead/behind=0/0（未 fetch），暂存区为空。HEAD 不能完整表示当前实验代码：工作区有既存改动及未跟踪文件，详见 JSON 快照。
- 既存 tracked 代码改动：`model/ALFNet/ALFNet.py`、`model/ALFTemporal/ALFTemporal.py`、`model/ALFTemporal/patchstg.py`、`scripts/train_v4.py`。六消融的 opt-in 开关及训练入口登记尚未提交，当前 Full 默认路径保留。
- `configs/paper/support_ablations_plan.json` 独立于旧 final_plan；run identity 为 `d55490c48695f36829d3978de1190bf68b583b7f6d76aeea4c1f1663a12fc67a`。计划语义哈希不是 JSON 文件 SHA256，勿混用。
- 旧 `configs/paper/final_plan.json` 有历史哈希漂移，不能作为本轮或后续新运行权威；本轮未执行 builder/check/preflight。
- 两数据集目标是速度 v（payload channel 2），输入/预测 12/12，chronological 60/20/20；训练只暴露 K 的信号，validation 为 K∪V。当前结果 seed32、validation-only、mixed-source；没有新的 test 或多种子结论。
- 用户已保留 METR-LA，冻结其 patch **容量** 8；最大原分区大小 7。PEMS-BAY 容量 11 等于最大分区大小。容量不是分区数；不足位置补入实际邻接排序节点 token，不是零填充。尚无最优容量证明。
- 两数据集 LifeAvg 三指标均超过 MoGERNN；METR-LA 唯一落后 MoGERNN 的主表单元为 Unseen MAPE，PEMS-BAY 为 Unseen 三指标。具体数字、强基线弱项和内部控制反证见问题清单。
- Full 的状态专家路由在 ALFGraph，但状态信息还参与重建、时间表示、细化；dense soft routing 与单预测+上下文条件特征并非已证缺陷。state_encoder 输出未用于最终预测，不能算有效条件化贡献。
- six COMPLETE 文件与 reports/epoch 记录本轮复查存在；论文 snapshot 所列 22 个源文件 SHA256 全部匹配；figure_manifest 所列 11 个图/CSV 文件 SHA256 全部匹配。
- 本轮未检查运行中进程/GPU；历史 PID 和空闲观察均已过期，不能据此启动、停止或重启工作。

## Hypotheses / 假设，未作为结果

- 无历史预测的空间补偿、重建偏差或上下文调制可能解释 Unseen 短板；尚未隔离原因。
- 显式状态可能在完整历史/早接入或 PEMS-BAY 上带来条件依赖的负迁移；单种子不能确定其稳定性。
- 低速度分母、损失与 checkpoint 指标不对齐可能解释 MAPE，但尚无分箱归因，不能沿用“低流量导致”的结论。
- 有效 support 重复可能降低路由可辨识性；两数据集算子数值距离/路由塌缩尚未新测。
- GWNet 的时间卷积等可作候选灵感；没有选定融合或证明其结构是涨分原因。
- 早接入已有弱项证据；故障后恢复全过程尚未单独测量。两者不可等同。

## Completed / 已完成

### 六个 replacement ablations

| 数据集 | 实现变体 / 论文名称 | best epoch | epochs run | 状态 |
|---|---|---:|---:|---|
| METR-LA | mask_only / Mask-only | 49 | 60 | COMPLETE |
| METR-LA | static_graph / Static Propagation | 59 | 70 | COMPLETE |
| METR-LA | local_temporal / Local Temporal Context | 59 | 70 | COMPLETE |
| PEMS-BAY | mask_only / Mask-only | 59 | 70 | COMPLETE |
| PEMS-BAY | static_graph / Static Propagation | 59 | 70 | COMPLETE |
| PEMS-BAY | local_temporal / Local Temporal Context | 16 | 50 | COMPLETE |

- Mask-only 以完整 mask 重复到 role/age 输入槽，细化 role scale 使用固定均值；原 role-balanced loss、mask、cohort、teacher forcing 不变。
- static_graph 保留全算子，以学习的全局静态混合替代状态路由，少 660 参数（1,036,869 vs 1,037,529）。
- local_temporal 保留时间分支名义参数，按节点独立进行该分支注意力；全模型其他空间路径仍在，计算几何并不相同。
- 既往验证：19 targeted tests passed、15 deselected；六 CPU preflight、四 batch64 GPU train/validation probes。此为此前记录，本轮未重跑，不是完整仓库测试。
- Full 未重新训练、已有结果未覆盖。所有报告、checkpoint、日志、配置路径见 `paper/experiments_evidence_snapshot.json` 的 runs/sources。

### 已完成分析与论文组织

- 原 condition/case：`outputs_v4/reports/support_analysis_v1/`。
- METR-LA trajectory intervention 与 q/layout 均已完成：`outputs_v4/reports/observation_support_v1/`，128 paired windows × 20 unique arms × 3 models = 7,680 forecasts，各预测 12 步。
- 既存 `verification.json` passed：target/mask 一致，三指标重算、Full anchor 复现、metadata-blind controls 不变、旧 checkpoint 不变。此前分析定向测试 3 passed；本轮只读核验报告与哈希，没有重新执行验证器或模型。
- 正文 `paper/experiments_draft_cn.md`；附录 `paper/experiments_appendix_cn.md`；图表计划 `paper/experiment_figure_plan_cn.md`；图在 `paper/figures/`。旧 `docs/paper/` 路径不是当前论文位置。
- 表 1 两数据集 LifeAvg；图 2 METR-LA 全条件/三指标相对固定 MoGERNN；表 2 METR-LA 消融；图 3 held-out 64 窗口（23 节点），MoGERNN 外部对照与 Mask-only 内部控制分面板；图 4 q/layout 原 pooled 128 窗口、known/held-out 各 64。Method 占图 1。
- 附录保留逐条件最强基线、known-node 干预（64 窗口/53 节点）、全三指标、q 的精确 mask、案例、PEMS-BAY 消融。
- q=0/3/6/9/12，12 步实际可用数；四布局 Recent/Early/Dispersed/Middle gap。自然完整窗口中合法删除，AAS/age=1 固定，不是 K-study，也不是故障后恢复测试。
- 当前论文是可继续修改的实验稿；用户现在转向模型优化，不能把旧稿所有方法解释视为不可修正事实。
- 本轮新增问题清单、版本/证据快照，修复 handoff 的“分析未完成”等过期状态，并同步发生变化的共享事实/决策/实验索引。

## Modified but unverified / 已修改未验证及验证边界

- 本轮只有 AI 文档和 checkpoint 文件修改；handoff 链接、JSON 解析、88 个快照文件哈希、未跟踪文档空白及 git diff --check 均通过，共享文档 diff 已审阅。没有代码行为改动可声称已验证。
- 既存四个 tracked 代码文件及未跟踪分析/测试/config 已完成上述有限验证，但**未做完整仓库测试、未在新多种子/正式新 test 上确认**；“未提交”不等于“从未验证”。
- 输出 README 的“main figures average 128”旧句尚未修订，与当前图 3 的 64 held-out 不一致；最新 figure_manifest、脚本及论文附录为图分组依据。本轮不改旧输出文件。
- 老 SUP5 等 summary 部分配置路径已不存在；具体配方复现能力尚未恢复，不能仅凭旧 selected 字段复用。
- 环境仍用已验证的 `/home/xd/anaconda3/envs/ReaCH/bin/python`；environment.yml 声明与安装环境差异未解决。

## Not started / 未开始

- 新一轮优化候选选择、代码修改、训练或重新评估；GWNet 融合；patch 容量/分区搜索。
- 新种子确认、正式显著性结论、已冻结新方案的 test 评估；本轮均未授权执行。
- 低速度分箱归因、operator 数值距离/有效路由诊断、故障后恢复全过程的独立分析。
- 完整仓库测试、STGCN 历史约 7.08e-6 重评偏差原因、旧计划重新冻结。
- 下一会话应先依据问题清单区分具体缺口和候选机制，再确定独立运行身份与用户授权范围；不能直接续跑旧任务或覆盖现有 Full。

## Ruled out / 有证据排除的结论或具体配方

- 排除“当前所有条件三指标均胜 MoGERNN/所有基线”；排除“状态/动态路由在两数据集所有指标必需”。
- 排除“当前 dense soft 本身是 bug”“只有硬稀疏路由才有贡献”“patch 8 是 8 个分区”“接入等同故障后恢复”等推断。
- 历史 speed SUP5 / LF_SUP5 未通过两数据集统一替换门槛；METR-LA LF_SUP5 对其历史 BASE 33 单元全败。不原样再试作为已知可靠升级。
- METR-LA LF 改善 MAPE 但牺牲绝对误差；LF_SUP 的 selected 标签不代表胜过历史 BASE。这些排除只覆盖已测试配方，不排除全部算子优化或低值误差研究。
- PEMS-BAY GWNET 在已有混合来源记录中 MAE/MAPE 更强、RMSE 更弱；排除“GWNet 所有指标已证最佳”与“整网融合已经被证实有效”。
- 不以隐藏 PEMS-BAY 消融反证、降低控制容量、改变 MAPE 口径或 test 调参追求领先。

## Git / verification / stop

- 不自动 commit/push；保持当前 reviewable worktree，不清理、还原或暂存用户既存改动。
- 本轮命令范围：git status/rev-parse/rev-list/diff，rg/sed/cat，只读 Python JSON/CSV 与 SHA256 核验，apply_patch 文档编辑，文档检查；无模型 forward、无训练、无 GPU 工作、无新 test 读取。
- 精确代码/配置/论文/证据哈希、既存改动列表、检查范围保存在 `CHECKPOINT_20260912.json`。handoff 及问题清单与该快照共同构成恢复点。
- **交接完成后停止。** 用户下一阶段意图是两数据集优化；本 checkpoint 不等于已批准某种融合、搜索计划、test 打开或 Git 发布。

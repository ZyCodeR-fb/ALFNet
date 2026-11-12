# T-20261103-extension-experiments

任务说明：[SERVER_EXPERIMENT_PROMPT_CN.md](../T-20261020-paper-writing/SERVER_EXPERIMENT_PROMPT_CN.md)"提示词正文"一节（用户 2026-11-03 在服务器会话中要求执行）。

范围：四个补充对比方法（Merlin、HD-TTS 属 2.2；STONE、STOP 属 2.3）在 v4 协议下的适配、冻结计划、训练、验证选点与一次测试；两项整体消融（w/o SAHC、w/o OS）。测试访问由用户 2026-11-03 授权，仅限本任务的最终检查点、每个一次。不改论文正文、不 commit/push、不改 `references/`、`data/*/original/`、`data_v4/` 与已有 checkpoint。服务器共享：不动他人进程（服务器所有进程同属用户 `xd`，只能按本任务记录的 PID 停止），本任务同时运行的训练/评估进程最多 3 个。

## 检查点 0：上下文恢复 — 2026-11-03 18:10

- **Checked state**：主机 `xd`；分支 `paper/final`，HEAD `8322d9c4505b96b17785fdeb121d16c402e031f7`，与 `origin/paper/final` 0/0（未 fetch）；开工时工作区干净、无未跟踪文件。
- **Facts**：
  - 提示词要求的文件全部存在：`survey/{ARGUMENT_MAP,BASELINE_CRITERIA,LITERATURE_MATRIX}_CN.md`、`paper/BASELINE_FAIRNESS_20260928.md`、`paper/SLPF_STBP_INTEGRITY_20260926.md`、`docs/comparison_model_audit.md`、`experiments/alfnet_ablation_adapter.py`、`model/ALFNet/ALFNet.py`、`experiments/v4_runner.py`、`scripts/train_v4.py`、`drafts/manuscript_v6_cn.md`、`paper/final_scheme_20261006/`、`paper/pems08_wd_test_20261012/selected_configuration_manifest.json`、`paper/all_ablations_test_20261010/`。
  - `references/` 下已有 `Merlin/`、`hdtts/`、`STONE-KDD-2024/`、`STOP/`。
  - GPU：1 张 RTX PRO 6000（97,887 MiB），观察时他人进程 PID 2443908（`pytorch2.9` 环境 UrbanPG 预训练）占 1.5 GiB、利用率 54%。磁盘：工作盘 1.4 TB 可用，系统盘 281 GB 可用；内存 125 GB。
  - 解释器 `/home/xd/anaconda3/envs/ReaCH/bin/python`：Python 3.11.13，torch 2.8.0+cu128，CUDA 可用。
  - `scripts/train_v4.py::load_plan_authority` 的 `registered` 为空，所有旧计划均已退役；历史研究通过各自的 study launcher（`scripts/run_*_2026*.py`，以 `unittest.mock.patch` 注入 adapter）调用 `train()`。
  - 最终 ALFNet：METR-LA `patch7`、PEMS-BAY `05_PEMSBAY_combo2`（二者无插值、无中心化，decoder scale .3）；PEMS04 `06_PEMS04_bridge_delta1`（`support_transfer_variant=bridge_delta1`：插值开、中心化关、decoder scale 1）；PEMS08 `13_PEMS08_weight_decay_0.001`（`unified_variant=triple_halfcenter`：插值开、中心化 .5、decoder scale 1）。
- **Completed**：按 CLAUDE.md 顺序阅读 WORKFLOW、PROJECT_CONTEXT、HANDOFF、论文任务最近记录与本任务依据文件。
- **Not started**：任务 A、任务 B 的全部实现、计划、运行。
- **Next**：检查四个参考仓库与旧 Merlin 适配（`archive/cleanup_20261017/sources_and_evidence.tar.gz`），实现 wrapper 与冒烟测试。

## 检查点 1：适配与 CPU 冒烟测试 — 2026-11-03 18:50

- **Facts**：
  - `references/Merlin` 与 GitHub 上游 main 的 stid_arch/mlp/Loss/mask_metric/train_teacher/train_student_merlin 逐字节一致；hdtts、STONE、STOP 无 git 元数据与 LICENSE 文件，按文件 SHA256 记录。
  - 旧 Merlin 适配（归档于 `archive/cleanup_20261017/sources_and_evidence.tar.gz` 的 `model/comparison_model/Merlin/adapter.py`、`scripts/train_merlin_v4.py`）已重新审计；与源实现的差异与本次改动见 `docs/comparison_model_audit.md` 2026-11-03 节。
  - 源实现行为如实保留：Merlin 星期嵌入索引恒为 0；STONE 图编辑器掩码跨迭代累积；STOP 发布代码无消息扰动/DRO。
  - HD-TTS 依赖 tsl 0.9.5、torch_scatter 2.1.2+pt28cu128、torch_sparse 0.6.18+pt28cu128，以 `pip --no-deps --target` 装在 `outputs_v4/extension_20261103/pydeps`（未改 ReaCH 环境），仅 HD-TTS 进程加入 PYTHONPATH。PyG 2.7 的 `propagate` 模板生成与 HD-TTS 子类不兼容，改用通用 `MessagePassing.propagate`。
  - w/o SAHC 复用：METR-LA `12_METRLA_no_imputer`、PEMS-BAY `13_PEMSBAY_no_imputer`（2026-10-06 计划 27c243b2）配置语义 = 当前最终配置 + no_imputer，权重哈希 = 2026-10-10 测试记录，验证重放差 0.0；该计划冻结后的代码变化仅为 `v4_runner.py` 删除退役模型注册项与 GWNET 裁剪分支（归档副本与冻结哈希一致）。
- **Completed**（均未提交）：
  - `model/comparison_model/{Merlin,HDTTS,STONE,STOP}/`（source/ 字节副本、`adapter.py`、`__init__.py`、`SOURCE_MANIFEST.json`）；`model/comparison_model/source_loader.py`。
  - `experiments/extension_baselines_20261103.py`（注册、源训练配方、STONE 目标、Merlin 两阶段）；`experiments/alfnet_extension_ablation_adapter.py`（no_sahc、no_os）。
  - `scripts/run_extension_20261103.py`（freeze/check/preflight/queue/worker/collect）；`scripts/test_extension_20261103.py`（训练完成后再冻结的测试计划与报表）。
  - `tests/test_extension_20261103.py`：带私有依赖 10 passed；不带时 7 passed、3 skipped（HD-TTS）。
  - `docs/comparison_model_audit.md` 新增 2026-11-03 节（四行）。
- **Modified but unverified**：GPU 预检正在用草稿计划（scratchpad）试跑，尚未正式冻结 `configs/paper/extension_20261103/plan.json`。
- **Not started**：正式冻结、正式预检、训练队列、测试、报表。

## 检查点 2：计划冻结与预检 — 2026-11-03 19:00

- **Draft dry run**（scratchpad 草稿计划 b4691fdb，输出不在 artifacts/）：22/22 通过。发现并修正：预检报告中优化器字段读的是补丁前的函数（训练本身经模块属性调用、用的是源配方；已在补丁上下文中单独核实 STOP=ClippedAdam lr2e-3/wd1e-5/eps1e-8+MultiStepLR，STONE=两组 ClippedAdam+StepLR(10)）；预检多进程共用一个汇总文件的竞争改为按键名分文件。
- **Formal plan**：`configs/paper/extension_20261103/plan.json`，plan_sha256 `5867446b73b1d45526b0dde8c027d4b9fc0370d4164b96945edf84d129041195`；22 个训练（16 基线 + 4 w/o OS + 2 w/o SAHC），2 个复用（METRLA/PEMSBAY w/o SAHC）；381 个文件哈希（代码、配置、数据、锚点与复用权重、references 源文件）+ 私有依赖 RECORD 哈希；git HEAD 8322d9c；torch 2.8.0+cu128。`check` 通过。
- **Formal preflight**：输出目录 `artifacts/experiments/extension_20261103/5867446b…/`；本任务进程 PID 2550625（HD-TTS，PYTHONPATH=pydeps）、2550656（其余 18 个），日志 `preflight_hdtts.log`、`preflight_main.log`。22/22 通过，`PREFLIGHT_COMPLETE.json` 已写（两进程均已退出）。泄漏审计（16 基线）、身份与配对初始化（6 消融）、w/o OS 输入置零与对角色/进度不变性、优化器/调度器类型全部通过；峰值显存基线 0.2–6.6 GiB，ALFNet 消融 6–15.5 GiB。
- **Training queue**：2026-11-03 19:16 启动，tmux 会话 `alfnet_ext_queue`，queue PID 2567130，日志 `<输出目录>/queue.log`；worker PID 与命令实时记录在 `<输出目录>/queue_status.json`（每个 worker 日志 `<输出目录>/<key>.log`）。顺序：6 个 ALFNet 消融优先，然后 16 个基线；最多 3 个并发，启动前要求空闲显存 > max(8, 1.5×预检峰值+4) GiB，间隔 ≥60 s。失败不自动重试，写 `ATTENTION_REQUIRED.json`。第一个 worker：`16_METRLA_no_os` PID 2567143。
- **停止方法**（只停本任务进程）：先 `tmux kill-session -t alfnet_ext_queue` 停队列，再按 `queue_status.json` 中 status=running 的 pid 逐个 `kill`；不要按名字批量 kill（服务器所有进程同属用户 xd）。恢复：每个运行目录保留 `last.pt`（train_v4 的续跑状态），但本计划的 worker 不带续跑入口；中断或失败的运行需按 recovery_policy 另立恢复计划后才能续跑或重跑。
- **Next**：等待训练完成 → `collect` → 冻结测试计划 `scripts/test_extension_20261103.py freeze` → 测试队列（≤3）→ `report`。

## 检查点 3：训练、测试与结果整理 — 2026-11-05

- **Facts**：
  - 训练队列 22/22 完成（`COMPLETE.json`，`collect` 自动执行，`validation_results.json`），无失败、无重试；`run_extension_20261103.py check` 仍通过。
  - 测试计划 `configs/paper/extension_test_20261103/plan.json`，plan_sha256 `4742c34967eab16169d01ee81c98b9d90528a1e9282fdcd7c6e26112fe507044`；tmux `alfnet_ext_test`（queue PID 3933836，已自行退出）；22/22 完成，测试前验证重放全部 0.0 差。输出 `artifacts/evaluation/extension_test_20261103/4742c349…/`。
  - 测试第一格数（ALFNet 严格最低，每数据集 33 格）：现有 8 个 33/29/29/30（=论文现状）；+Merlin、+STONE 不变；+HD-TTS 33/29/26/30；+STOP 31/15/13/11（70/132）；+全部 4 个同为 31/15/13/11。
  - STOP 测试 LifeAvg 在 PEMS-BAY、PEMS04、PEMS08 优于 ALFNet（PEMS08 MAE 22.70 对 36.78），差距几乎全部来自 Unseen。用 STOP PEMS08/METR-LA 最终权重在真实验证样本上做反事实：替换全部未来目标与全部不可见输入，输出逐元素不变。
  - 消融（测试 LifeAvg）：w/o SAHC 与 w/o OS 在四个数据集三项上几乎全部变差，唯一例外 PEMS04 w/o SAHC RMSE −0.85%；变差格数 w/o SAHC 33/30/29/31、w/o OS 33/30/30/24。
- **Completed**（未提交）：`scripts/test_extension_20261103.py`（冻结前移除报表部分）、`scripts/report_extension_20261103.py`；`paper/extension_baselines_20261103/`、`paper/extension_ablations_20261103/`（README、results_and_provenance.json、长表、LifeAvg 汇总）；`paper/latex_neurips_20261020/results_extension_20261103/`（comparison/ablation xlsx、first_place_counts.json、README_CN.md）。
- **Not touched**：论文正文、`tables/*.tex`、`selected_tables.json`、已有权重与结果。
- **Needs user decision**：是否在论文中报告 STOP（报告则"≥80% 第一"不成立，需改写 2.3 节与实验结论）；2.2 选 Merlin 还是 HD-TTS；是否追加可选 `mask_only`。
- **Next**：作者决定报告集合后，再改写正文与表格（需另行授权）。

## 检查点 4：第二批 2026 对比方法（MCPST-FSL、MetaDG）— 2026-11-05

- **用户指令**：导师要求再加 2026 年对比模型；适配 `references/MCPST`、`references/MetaDG`，同样要求，四个数据集，最多同时 4 个。用户回答（AskUserQuestion）：MetaDG `node_emb` 换成共享结构编码；训练后每个最终检查点各测试一次。
- **Facts**：
  - MCPST = MCPST-FSL（arXiv 2602.01936；文件日期 2026-08-26）。2026-09 曾有一版适配（已退役，归档），其运行在逐副本 `eigh` 中被手动中断（KeyboardInterrupt），且用了缩减的结构超参数、未用物理特征。
  - MCPST 无逐节点参数；只运行源域训练阶段（FOMAML 与目标域微调属于目标数据适应，不在本协议内）；源数据为 Git LFS 指针，按真实大小推断原始 2 通道（第二通道按 DCRNN 惯例为日内时间）。
  - MCPST 光谱模块逐副本特征分解：GPU 上副本差 ≤1.2e-7、特征值差 ≤3.4e-5，改为首副本分解复用（容差 1e-5），PEMS-BAY 速度由约 3 h/epoch 降至约 16 min/epoch。
  - MetaDG 源码 `init_hidden_state` 在 CPU 上建状态，GPU 上直接运行会失败；改为在模型设备上创建同样的零张量。
- **Completed**（未提交）：`model/comparison_model/{MCPST,MetaDG}/`、`experiments/extension2_baselines_20261105.py`、`scripts/run_extension2_20261105.py`、`scripts/test_extension2_20261105.py`、`tests/test_extension2_20261105.py`（8 passed；连同第一批与旧消融测试共 23 passed）；`docs/comparison_model_audit.md` 2026-11-05 节。
- **Plan**：`configs/paper/extension2_20261105/plan.json`，plan_sha256 `a8d285aae02cc3f8425488ec411482a4aa30128510a1b6bf46b0ad7563d0cdf4`，8 个运行，最多 4 并发；输出 `artifacts/experiments/extension2_20261105/a8d285aa…/`。预检进程 PID 4058949（本任务）。
- **Next**：预检通过后启动 tmux `alfnet_ext2_queue`；训练后 collect → 冻结 `scripts/test_extension2_20261105.py` 测试计划 → 测试 → 更新报表（第一格数纳入 MCPST、MetaDG）。
- **2026-11-05 16:27 失败与恢复**：`00_METRLA_MCPST` 第 0 轮验证时退出（`validation skipped an invalid batch`，日志 `artifacts/experiments/extension2_20261105/a8d285aa…/00_METRLA_MCPST.log`）。诊断（不经模型）：批大小 8 时 METR-LA 验证设计中第 745、832 批的 8 个样本全无监督位置（验证集 6725 个样本中 29 个无监督且时间上相邻），公共 runner 本就跳过这类批；其余三个数据集 0 个；训练中 0.85% 的样本无监督，打乱后整批为空的概率约 3e-17。恢复计划 `configs/paper/extension2_recovery_20261105/plan.json`（`034802bd…`，在看到任何恢复结果前冻结）：只放宽检查——跳过的验证批必须恰为预先算出的空批、跳过的训练批必须无监督、有监督的非有限损失仍失败；配置文件与哈希、模型、数据、优化与选点均不变；对 MCPST 其余数据集是空操作，故不重跑。tmux `alfnet_ext2_recovery`（PID 4081542），等待基础队列无排队且运行数 <4 后启动，保证本任务不超过 4 个进程。
- **2026-11-05 19:50 第二个失败与恢复**：`07_PEMS08_MetaDG` 第 0 轮训练中出现非有限损失（公共 runner 跳过该批，严格包装器要求 0 跳过而退出；预检 20 批通过；METR-LA（同一源配置块）、PEMS-BAY、PEMS04 MetaDG 在严格规则下无跳过）。恢复计划 `configs/paper/extension2_recovery2_20261105/plan.json`（`4f1cea2f…`，在看到恢复结果前冻结）：允许公共 runner 对非有限/无监督训练批的跳过（不更新），逐个记录批号与首个非有限子模块（诊断重跑前后恢复全部 RNG 与 curriculum 计数），任一轮跳过 >1% 即失败，验证不得跳过；对 MetaDG 其余数据集为空操作。tmux `alfnet_ext2_recovery2`，与 MCPST 恢复一起计入 4 个进程上限。
- **2026-11-06 00:20 恢复 2 发现真实发散并停止**：恢复 `4f1cea2f` 运行中从第 0 轮第 138 批起每批预测全为非有限，首个非有限子模块 `core.tod_embedding`（参数已为 NaN）——某个损失有限的批产生了非有限梯度，经裁剪与 Adam 更新写入全部参数。按其 1% 规则必然失败，本任务 PID 4168534 由代理停止（核对命令行后 kill），记录 `STOPPED.json`、`skip_events_at_stop.json`（166 次）。
- **恢复 3**：`configs/paper/extension2_recovery3_20261106/plan.json`（`b136d9b6…`；首次冻结 `d74f0c3f` 因守卫函数未绑定为方法、在构造学习率调度器时崩溃，未训练，保留为 `plan.superseded_d74f0c3f_crashed_at_scheduler_construction.json`）：每次优化步前检查梯度，非有限则跳过该步（不改参数与优化器状态）并记录参数名；损失跳过同样记录；每轮事件 >1% 即失败；验证严格。对 METR-LA、PEMS04（已完成）与 PEMS-BAY（严格规则下运行中）为空操作。tmux `alfnet_ext2_recovery3`，PID 88426。源实现无此守卫，属披露的适配。
- **2026-11-06 恢复结果**：8/8 训练完成（00 来自恢复 `034802bd`，07 来自恢复 3 `b136d9b6`：仅第 0 轮第 137 批 1 次非有限梯度步被跳过，最佳第 20 轮，验证 LifeAvg MAE 39.522），其余 6 个来自 `a8d285aa`；全部 train_skipped=0。

## 检查点 5：第二批测试 + 对比算法三种子（自动链）— 2026-11-06

- **用户指令**：本轮训练、测试结束后自动开始 3 种子；先不跑 ALFNet；对比算法 UIGNN、GinAR、MoGERNN、SLPF、STGCN、STID、PatchSTG、STBP、Merlin、STONE × 四个数据集；最多同时 4 个；以前在最终协议下跑过的种子可复用。用户回答（AskUserQuestion）：3 种子训练后自动各测试一次。
- **第二批测试计划**：`configs/paper/extension2_test_20261105/plan.json`，plan_sha256 `60b911ddf956928181edc62e4f3c3268e3faa9c287a5b51fa4b3a054e07b8c9b`（8 个检查点，训练来源 a8d285aa×6、034802bd、b136d9b6），输出 `artifacts/evaluation/extension2_test_20261105/60b911dd…/`。
- **三种子训练计划**：`configs/paper/seeds_20261106/plan.json`，plan_sha256 `839232c95df80dbddc0bb6bcd4021512fcb1fa41162a358fbe49769ead890ef4`；种子 32/42/52（种子同时决定节点划分）。种子 32 复用最终运行（40 个，检查点、配置字节与验证表逐一核对；METR-LA/PEMS-BAY 用 `artifacts/models/main/*/seed32/config.original.yaml`，即训练时的字节，`final_seed32` 版仅多/少注释行）；种子 42/52 新训练 80 个，配置文件与种子 32 完全相同。历史种子 42/52 运行 46 个均不可复用（旧 micro LifeAvg 选点，其中 42 个配置也不同），审计写入计划。预检门槛：当前代码重放种子 32 检查点验证表须一致（abs 1e-5 + rel 1e-5），不一致则该对加训种子 32（规则已冻结）。8 个原基线在裸 runner 下训练（与种子 32 一致），Merlin/STONE 在 2026-11-03 研究上下文中训练；训练跳过批按 runner 原行为计数，任一运行跳过 >1% 标记为发散并在测试前停止。手动冒烟（未写结果）：METR-LA STID/STONE/Merlin 重放差 0.0，种子 42 冒烟通过。
- **测试/汇总**：`scripts/test_seeds_20261106.py`（训练全部干净后冻结；先重放验证再测试，各一次；复用的种子 32 不重测）；`scripts/report_seeds_20261106.py`（mean ± 样本 std，写 `paper/seeds_20261106/` 与 `paper/latex_neurips_20261020/results_seeds_20261106/`）。`tests/test_seeds_20261106.py` 3 passed。
- **运行**：tmux `alfnet_ext_seeds_chain`（chain PID 407906，`scripts/chain_seeds_20261106.sh`，日志 `artifacts/experiments/seeds_20261106/chain.log`），2026-11-06 10:57 启动。顺序：第二批测试队列（PID 407923）→ 三种子预检（4 个分片子进程，PID 记在 `<输出目录>/preflight_status.json`）→ 训练队列（≤4，worker PID 在 `<输出目录>/queue_status.json`）→ collect → 若 COMPLETE.passed 则冻结测试计划并测试（≤4）→ 汇总。
- **停止方法**：`tmux kill-session -t alfnet_ext_seeds_chain`，再按各 `queue_status.json`/`preflight_status.json` 中记录的 PID 逐个 kill（先核对命令行）。
- **Next**：第二批测试完成后整理 MCPST/MetaDG 对比表与第一格数；三种子完成后核对汇总并报告。
- **2026-11-06 11:13 第二批测试完成**：8/8，验证重放差 0.0；表 `paper/latex_neurips_20261020/results_extension2_20261105/`（`scripts/report_extension2_20261105.py`）。测试第一格：现有 8 + MCPST + MetaDG = 33/28/27/30（118/132，每数据集 ≥27）；MetaDG 只赢 3 个 MAPE 格，MCPST 不赢任何格；两者 LifeAvg 均劣于 ALFNet。
- **2026-11-06 11:47 三种子计划两次作废（均未开始训练）**：
  - `839232c9`（v1）预检中 METR-LA GinAR 种子 42 冒烟失败（验证无有限预测）。诊断：lr .006 下第 13 次优化步 58 个参数梯度非有限 → 全部参数 NaN；种子 32/52 与 PEMS-BAY 三个种子前 20 步无事件。本任务停止链（PID 407906、队列 419257、预检分片 419268/419271，核对命令行后 kill），已完成的 32 个预检文件保留在 `artifacts/experiments/seeds_20261106/839232c9…/preflight/`。
  - `5e70abfa`（v2）= v1 + 全方法非有限梯度步守卫。冻结后诊断：守卫下 METR-LA GinAR 种子 42 前 20 批仍有 4 次守卫事件、验证 MAE 1.16e15 → 守卫不足；v2 从未运行。
  - 证据：`artifacts/experiments/seeds_20261106/diagnostics/{ginar_lr0.006_divergence_20261106.json,diag_ginar_20261106.py}`；作废计划存为 `configs/paper/seeds_20261106/plan.superseded_{839232c9…,5e70abfa…}.json`。
- **三种子计划 v3**：`configs/paper/seeds_20261106/plan.json`，plan_sha256 `58eecfb5cf25829b4d963a07c91c1eba8a04b2ecb297d8dd5743ba1c675cf448`，82 个新训练。规则（看到任何恢复结果前写入）：(1) 全部运行带非有限梯度步守卫（跳过并记录；任一轮事件 >1% 整轮批数即失败；无事件时为空操作）；(2) GinAR 在 METR-LA、PEMS-BAY 按 2026-09-27 流量数据集先例 lr .006→.001（仅改该行与 variant 标签，配置 `configs/paper/seeds_20261106/GinAR_lr0.001/`），使四个数据集一致；这两对的种子 32/42/52 全部新训练，预检跑完整首轮 + 完整验证；论文现有 lr .006 种子 32 结果不改动、不计入三种子均值。其余 38 对种子 32 复用（重放门槛不过则加训）。
- **重启**：tmux `alfnet_ext_seeds_chain`（chain PID 437987），2026-11-06 11:58；链脚本对已完成的第二批测试不重跑。
- **2026-11-06 13:00 预检 v3 通过并开始训练**：40/40 通过；38 对种子 32 重放全部在容差内（最大差 2.2e-5，STGCN），无需加训；冒烟无守卫事件；GinAR lr .001 完整首轮验证 MAE：METR-LA 6.57、PEMS-BAY 4.81。训练队列 PID 438017（82 个，≤4 并发），首批 worker 505769/506167/506556/506953（Merlin）；实时 PID/日志见 `artifacts/experiments/seeds_20261106/58eecfb5…/queue_status.json`。

## 检查点 6：三种子完成与整理 — 2026-11-10

- **Facts**：自动链 2026-11-10 08:12 完成（`chain.log`：chain COMPLETE）。训练 82/82（`COMPLETE.json` passed，无 missing/flags），无非有限梯度事件、无训练跳批；METR-LA UIGNN（批 4）每轮跳过 6 个无监督验证批，与种子 32 相同。测试计划 `configs/paper/seeds_test_20261106/plan.json`（`92f8956b…`），82 个测试，测试前验证重放全部一致（最大差 1.4e-5）；38 个种子 32 表复用。
- **Observed**：METR-LA GinAR 种子 42（lr .001）验证 LifeAvg MAE 7.02，测试 Steady MAE 50.3 → LifeAvg 18.30；按协议不剔除不重跑。GinAR lr .001 种子 32 与论文 lr .006 种子 32 测试相近（METR-LA 7.043/7.092，PEMS-BAY 4.992/5.059）。种子 32 划分在 METR-LA/PEMS-BAY/PEMS08 偏易、PEMS04 偏难（基线中位偏差 −3.5/−2.3/+2.7/−3.1%）。
- **Completed**（未提交）：`scripts/report_seeds_overview_20261110.py`；`paper/latex_neurips_20261020/results_seeds_20261106/{seeds_overview_20261110.xlsx,seeds_overview_20261110.json,README_CN.md}`（16 行方法，缺失种子留空）。暂定测试第一格（10 个三种子方法）：同种子 32 121/132；对均值 120/132（33/30/27/30）；对最好种子 95/132。
- **Not started**：ALFNet、HD-TTS、STOP、MCPST、MetaDG 的种子 42/52（用户指示后续补充）。
- **Next**：补跑后在 `collect()` 中加入新来源并重跑整理脚本；ALFNet 三种子补齐后以均值对均值重算第一格数。
- **2026-11-10 后台残留**：发现 4 个本任务的后台轮询循环（PID 420328/420443/423832/427110），在等待已作废计划 `839232c9…` 的 `PREFLIGHT_COMPLETE.json`（永不出现），已运行约 4 天，仅 sleep，无 GPU；核对命令行后已 kill。tmux 服务器进程 437986 的命令行显示本任务的 new-session，但它同时承载他人会话（formal、streamlit*），不得 kill。教训：等待循环必须带超时并指向当前计划目录；任务结束前清理自己的后台等待。
- **2026-11-10 用户下一步指令**：新会话中跑 ALFNet、MCPST、MetaDG 四个数据集三种子（已有结果复用），最多同时 5 个；STOP 不进入报告；ALFNet 先审计四个数据集当前配置是否为已跑超参数组合中的最优（胜格最多），再在最优配置下跑其他种子；后台监控只在会话内确认运行正常前使用，之后不留后台任务。新会话提示词：`docs/ai/handoffs/T-20261103-extension-experiments/NEXT_SESSION_PROMPT_CN.md`。

## 检查点 7：ALFNet 配置审计（三种子前）— 2026-11-10 12:20

- **用户指令（2026-11-10）**：跑 ALFNet、MCPST、MetaDG 四数据集三种子（种子 32 复用须重放一致），最多 5 个进程；训练前先审计 ALFNet 配置并汇报，若当前最终配置不是验证最优或验证与测试结论不一致则停下交由作者决定；不确定的候选可新开测试（对测试使用规则的明确例外）；HD-TTS、STOP 不进报告。
- **Checked state**：HEAD `c1d1b2c`，分支 `paper/final`；开工时已有未提交改动（`docs/ai/*`、seeds_20261106 产物等），均保留未动。
- **Facts**：
  - 审计 `paper/alfnet_config_audit_20261110/`（`README_CN.md`、`candidates.json`、`candidates_summary.md`、`replay/`、`new_tests/`；脚本 `scripts/audit_alfnet_candidates_20261110.py`）。122 个种子 32 设置候选（METR-LA 20、PEMS-BAY 23、PEMS04 39、PEMS08 40），排除 76 个组件消融与 1 个种子 42 运行。选点、数据/划分身份、预算全部一致；当前代码在原研究上下文中重放 122/122 一致（最大差 0.0；`main/METRLA`、`main/PEMSBAY` 用内容相等的 `configs/paper/final_seed32/` 配置，因原嵌套配置已移动）。
  - 验证第一格（对手集 8+Merlin+STONE+MCPST+MetaDG，GinAR 速度数据集 lr .001；改用 lr .006 计数不变）：当前最终 METR-LA 29（patch12 31、lr_double 30 更高）；PEMS-BAY 24（四方并列，按验证 LifeAvg MAE 排第 4）；PEMS04 26（三方并列，验证 LifeAvg 三项最差，但测试最好）；PEMS08 32（唯一最高）。
  - 新开测试 1 次（授权）：PEMS-BAY `speed_support_20261004/PEMSBAY_center_half_lr2_wd1e-05`，验证重放 0.0，测试 LifeAvg 2.6790/4.1352/6.2317，测试第一 31/33（当前最终 2.7232/4.2006/6.2761，28/33）。
  - 已测候选的验证-测试 Spearman：METR-LA 0.10（第一格）/−0.13（MAE）；PEMS-BAY 0.58/0.55；PEMS04 0.89/−0.13；PEMS08 0.87/−0.51。
- **Completed（未提交）**：上述审计；三种子框架 `scripts/run_seeds2_20261110.py`（select/freeze/check/preflight/queue/worker/collect；ALFNet 按种子 32 研究上下文训练，预检含 1 个完整种子 32 epoch 与原第 0 轮记录比对，容差 rel 1e-3）、`scripts/test_seeds2_20261110.py`、`scripts/chain_seeds2_20261110.sh`、`scripts/report_seeds_overview2_20261110.py`（输出 `paper/latex_neurips_20261020/results_seeds2_20261110/`，试跑仅写 scratchpad）、`tests/test_seeds2_20261110.py`（6 passed、1 skipped：计划未冻结）。
- **Not started**：选择文件、计划冻结、预检、训练、测试。
- **Blocked on author decision**：四个数据集各用哪一个种子 32 配置（见审计 README“待作者决定”）。决定后：`run_seeds2_20261110.py select --choice ... --authority "<原话>"` → `freeze` → tmux `alfnet_seeds2_chain` 运行 `scripts/chain_seeds2_20261110.sh`。

## 检查点 8：三种子计划冻结与启动（ALFNet、MCPST、MetaDG）— 2026-11-10 13:10

- **作者决定（AskUserQuestion，2026-11-10）**：METR-LA “保留 patch7 (Recommended)”、PEMS-BAY “保留 combo2 (Recommended)”、PEMS04 “保留 bridge_delta1 (Recommended)”；PEMS08 保留 13_PEMS08_weight_decay_0.001（验证唯一最优）。记录于 `configs/paper/seeds2_20261110/alfnet_selection.json`。
- **计划 v1 作废（未训练）**：`643ed7a4`（`configs/paper/seeds2_20261110/plan.superseded_643ed7a4_epoch0_validation_tolerance.json`）。预检中 ALFNet 第 0 轮复现门槛把 1 个 epoch 后的验证指标限在 rel 1e-3，PEMS-BAY（2.1e-2）、PEMS08（2.5e-2）未过，队列按设计未开训（`artifacts/experiments/seeds2_20261110/643ed7a4…/ATTENTION_REQUIRED.json`）。诊断 `artifacts/experiments/seeds2_20261110/diagnostics/`：当前代码独立复现彼此前 5–10 批逐位相同、之后漂移，一个 epoch 后验证差 1–4%，原运行落在新运行分布内（≤1.4 sd）；epoch 平均训练损失与原运行差 ≤1.2e-4（新运行之间 ≤3e-4）；负对照（PEMS08 漏掉 flow_unified adapter）训练损失 +24%、验证 +36%。PEMS08/PEMS-BAY/METR-LA/PEMS04 的 ALFNet 代码与配置哈希与原训练计划一致（`v4_runner.py` 等改动只涉及退役模型）。
- **计划 v2**：`configs/paper/seeds2_20261110/plan.json`，plan_sha256 `d6f66d2073190a15ed4b8e9c6dd7a07dca5c2050c381c18257de4630fd67d06b`；36 条（12 对 × 种子 32/42/52），24 个新训练；种子 32 复用（重放门槛 abs 1e-5 + rel 1e-5，不过则加训）；ALFNet 训练上下文 METR-LA/PEMS-BAY 裸 runner、PEMS04 support、PEMS08 support+unified；第 0 轮复现门槛（对四个数据集一致）：计数/方法契约/参数量精确，epoch 平均训练损失 rel 1e-3 + abs 1e-6，验证指标 rel 0.1 粗检；全部运行带非有限梯度步守卫（每轮事件 >1% 失败）；runner 跳批计数，训练跳批 >1% 标记发散并在测试前停止；最多 5 并发。
- **预检 v2**：12/12 通过（种子 32 重放全部 0.0；ALFNet 复现训练损失最大偏差 1.4e-4，aux 项按绝对容差；冒烟无守卫事件）。
- **运行**：tmux `alfnet_seeds2_chain`（chain PID 2934198，`scripts/chain_seeds2_20261110.sh`，日志 `artifacts/experiments/seeds2_20261110/chain.log`），2026-11-10 12:55 启动。顺序：预检（5 分片）→ 训练队列（queue PID 2934211，≤5 并发，最长任务先跑；worker PID/日志/命令在 `artifacts/experiments/seeds2_20261110/d6f66d20…/queue_status.json`，日志 `<该目录>/<key>.log`）→ collect → 若 COMPLETE.passed 则冻结 `scripts/test_seeds2_20261110.py` 测试计划并测试（≤5）→ `scripts/report_seeds_overview2_20261110.py` 汇总到 `paper/latex_neurips_20261020/results_seeds2_20261110/`。首批 worker：PEMSBAY_MCPST_seed42 2939107、PEMSBAY_MCPST_seed52 2939469、PEMSBAY_ALFNet_seed42 2939848、PEMSBAY_ALFNet_seed52 2940226、PEMSBAY_MetaDG_seed42 2940595。
- **停止方法**：`tmux kill-session -t alfnet_seeds2_chain`（只杀该会话，勿 kill tmux 服务器），再按 `queue_status.json` 中 status=running 的 pid 逐个核对命令行后 kill。中断的 worker 无续跑入口；重启队列会把 running 且无结果的标为 interrupted，不自动重跑。
- **测试授权**：本轮新检查点各测试一次（用户 2026-11-10）；测试结果不得用于选择、调参、换方法或重跑。

## 检查点 9：MetaDG PEMS-BAY 种子 42 失败与冻结恢复 — 2026-11-10 14:15

- **Facts**：
  - 13:39 `PEMSBAY_MetaDG_seed42`（worker PID 2940595）在第 0 轮结束时退出：非有限梯度步守卫记录 56 次（约为该轮约 1,950 批的 2.9%），超过计划的 1% 限额（`AssertionError: ('nonfinite-gradient steps', 56)`，日志 `artifacts/experiments/seeds2_20261110/d6f66d20…/PEMSBAY_MetaDG_seed42.log`）；守卫使参数保持有限；逐批事件未落盘。同数据集 `PEMSBAY_MetaDG_seed52` 第 0 轮在 1% 下正常（验证 MAE 3.455）。推测（未验证）：MetaDG DGQ 中 `exp(InstanceNorm(pos_mask)×2)` 在稀疏掩码时溢出，与种子 32 PEMS08 单次事件同类。
  - 用户决定（AskUserQuestion）：“冻结 10% 恢复方案 (Recommended)”。恢复计划 `configs/paper/seeds2_recovery_metadg_20261110/plan.json`，plan_sha256 `694a8c243cc411b361774d354d8a060b8acee9d166d790d988890f2d3f149290`（`scripts/recover_seeds2_metadg_20261110.py`），在任何恢复结果前冻结：对 MetaDG 全部数据集/种子一致——主队列结束后，凡在 1% 限额下失败的 MetaDG 运行用同一配置/种子/上下文重跑，每轮限额改为 10%，逐事件记录（epoch、优化步序号、参数名），验证须有限；1% 下已完成的不重跑；其他类型失败不覆盖。10% 是在看到 2.9% 之后提出的，须披露。
  - `scripts/test_seeds2_20261110.py` freeze 改为：主计划只缺恢复覆盖的运行时合并恢复行（并核对同 (数据集, 种子) 计数一致），否则拒绝冻结。
- **运行中**：
  - 主链 tmux `alfnet_seeds2_chain`（chain PID 2934198，queue PID 2934211）：14:13 时 5 个 worker 运行（PEMSBAY ALFNet s42/s52 各 5 轮；MCPST s42/s52、MetaDG s52 各 1 轮），18 个排队，1 个失败；显存约 55.8 GiB。主链在主队列结束后因 COMPLETE 缺 1 个运行而在测试前停止（预期）。
  - 恢复链 tmux `alfnet_seeds2_recovery`（PID 3009623，`scripts/chain_seeds2_recovery_20261110.sh`，日志 `artifacts/experiments/seeds2_recovery_metadg_20261110/chain.log`）：每 5 分钟检查主队列（若主队列进程消失而仍有任务则停止）→ 恢复重跑 → 合并冻结测试计划 → 测试（≤5）→ 汇总 `paper/latex_neurips_20261020/results_seeds2_20261110/`。
- **停止方法**：`tmux kill-session -t alfnet_seeds2_recovery`、`tmux kill-session -t alfnet_seeds2_chain`（只杀这两个会话），再按两个 `queue_status.json` 中 running 的 pid 核对命令行后 kill。
- **Next**：训练与测试结束后核对 `COMPLETE.json`、恢复的事件比例、测试重放差，分析 `seeds_overview_20261110b.xlsx`（均值对均值与逐种子第一格）。

## 检查点 10：三种子完成、核验与分析 — 2026-11-12

- **Facts**：主链 2026-11-12 02:57 结束（23/23 完成，缺 `PEMSBAY_MetaDG_seed42`，按设计在测试前停止）；恢复链 05:18 `recovery chain COMPLETE`：恢复 1/1（第 0 轮 56 次守卫事件 = 2.87%，之后 0 次）、测试计划 `configs/paper/seeds2_test_20261110/plan.json`（`d7dda3a885fd046d009e3a71565c3ca5fe4ac02c7e52e31ecbad455dca069a09`，24 个新检查点各测一次，测试前重放差 0，12 个种子 32 表复用）、汇总。2026-11-12 复核：三个计划哈希检查通过；24 个运行验证有限、训练跳批 0；守卫事件另有 PEMS08 MetaDG 种子 52 1 次；测试计数与 2026-11-06 研究逐格一致；两个 tmux 会话已自行退出，无残留进程。
- **结果**（测试，比较集 8+Merlin+STONE+MCPST+MetaDG）：三种子均值对均值第一格 33/28/30/30 = 121/132，四个数据集均 ≥27；同划分种子 32/42/52 为 118/104/102；只用种子 42/52 均值 31/22/25/28（PEMS-BAY、PEMS04 不足）；LifeAvg 三项均值在四个数据集全部第一，逐种子 35/36。丢失格集中于 Unseen（PEMS08 三项输 UIGNN 15–18%，PEMS04 MAE/RMSE 输 UIGNN）。验证主口径 29/25/30/31。
- **Completed（未提交）**：`scripts/analyze_seeds2_20261112.py`；`paper/latex_neurips_20261020/results_seeds2_20261110/{seeds_overview_20261110b.xlsx,.json,seeds_analysis_20261112.xlsx,.json,README_CN.md}`。
- **注意（论文写作）**：“≥80% 第一”依赖均值口径并含种子 32（其配置保留曾参考测试）；不宜写成各划分均成立。ALFNet 早停起始第 40 轮、MoGERNN 第 20 轮、其余第 0 轮，需披露。MetaDG PEMS-BAY 种子 42 的 10% 限额为事后提出，需披露。
- **Not started**：论文正文与表格更新（需另行授权）；HD-TTS、STOP 的种子 42/52（已决定不进报告）。
- **2026-11-12 导师版简表**：按作者示例 `___.xlsx` 版式生成 `paper/latex_neurips_20261020/results_seeds2_20261110/{main_phase_3seed_20261112.xlsx,ablation_seed32_20261112.xlsx}`（`scripts/make_advisor_tables_20261112.py`，只读冻结记录）。脚本内校验：示例 8 方法的种子 32 数值与论文现表逐格一致（速度数据集 GinAR 因 lr .001 不同）；两批消融的 Full 均为最终检查点、与论文 ALFNet 测试表一致。消融变差格数：w/o State Progress 25/30/29/24、w/o Adaptive Routing 30/33/29/26、w/o Spatial Context 33/32/28/33、w/o ALFTemporal–FiLM 33/28/31/33、w/o SAHC 33/30/29/31、w/o OS 33/30/30/24。

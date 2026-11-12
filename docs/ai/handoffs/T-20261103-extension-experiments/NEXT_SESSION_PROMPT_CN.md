# 新会话提示词：ALFNet、MCPST、MetaDG 三种子（2026-11-10）

请用简体中文与我交流。在 Ubuntu 服务器上的 ALFNet 仓库中继续任务 `T-20261103-extension-experiments`，完成 ALFNet、MCPST、MetaDG 在四个数据集（METR-LA、PEMS-BAY、PEMS04、PEMS08）上的三种子实验。

## 1. 恢复上下文（先读，不要凭记忆）

按 `CLAUDE.md` 的顺序阅读：`docs/ai/WORKFLOW.md` → `docs/ai/PROJECT_CONTEXT.md` → `docs/ai/HANDOFF.md` → `docs/ai/handoffs/T-20261103-extension-experiments/HANDOFF.md`（重点读检查点 4–6 及其后的 2026-11-10 记录）。另外阅读：

- `docs/ai/EXPERIMENTS.md` 中 2026-11-05、2026-11-06、2026-11-10 三条；`docs/comparison_model_audit.md` 中 MCPST、MetaDG 两节。
- 上一轮三种子框架（要复用其结构与规则）：`scripts/run_seeds_20261106.py`（冻结计划、种子 32 重放门槛、预检分片、非有限梯度步守卫、队列、collect）、`scripts/test_seeds_20261106.py`、`scripts/report_seeds_overview_20261110.py`、`scripts/chain_seeds_20261106.sh`、`configs/paper/seeds_20261106/plan.json`（`58eecfb5…`）。
- MCPST/MetaDG 的适配与恢复：`experiments/extension2_baselines_20261105.py`、`scripts/run_extension2_20261105.py`（`run_context()` 内含 MCPST `set_scaler`）、`scripts/recover_extension2_20261105.py`（MCPST METR-LA 空验证批）、`scripts/recover3_extension2_20261105.py`（MetaDG 非有限梯度步守卫，规则适用于 MetaDG 全部数据集）、`configs/paper/extension2_test_20261105/plan.json`（种子 32 的 8 个检查点与表）。
- ALFNet 当前最终配置：`paper/pems08_wd_test_20261012/selected_configuration_manifest.json`、`decision.json`、`selected_tables.json`；超参数研究记录在 `configs/paper/` 与 `scripts/` 下的 `metrla_hparams_20260920`、`pemsbay_hparams_20260917`、`hparams_completion_20260928`、`hparams_combinations_20260930`、`flow_*_2026100x`、`flow_ofat_20261006`、`pems08_wd_test_20261012` 等（以文件与记录为准，自行核对清单是否完整）。
- 汇总结果：`paper/latex_neurips_20261020/results_seeds_20261106/README_CN.md`、`seeds_overview_20261110.xlsx`。

读完后用中文简要复述：当前已完成什么、本轮要做什么、哪些结果可复用、有哪些约束，再开始动手。

## 2. 本轮任务

1. **ALFNet 配置审计（训练前，必须先完成并向我汇报）**
   - 盘点四个数据集上所有已训练的 ALFNet 超参数/设置候选（种子 32），核对它们是否在当前协议下可比：33 格验证 LifeAvg MAE 选点、同一数据与划分、当前代码可重放；不可比的列出原因。
   - 主排序依据**验证集**：以最终报告的对比集合（8+4）为对手，统计每个候选在验证集 33 格中严格第一的格数，并列出验证 LifeAvg MAE/RMSE/MAPE。已有的测试记录只作为已发生事实列出，若是有不确定的，可以给候选新开测试，看测试结果怎么样，因为最终的论文表格结果是测试结果（这里适当打破4的约束，其他未说明的地方要符合约束）。
   - 若当前最终配置不是验证集上的最优、或验证与测试结论不一致，先停下，把对比表和差异交给我决定，不要自行更换配置。注意：PEMS08 当前配置是"test-informed retention"（见 `decision.json`），需在汇报中注明。
   - 确定配置后，再冻结三种子计划。
2. **三种子训练**
   - 方法：ALFNet、MCPST、MetaDG；数据集：四个；种子 32/42/52（种子同时决定 K/V/T 节点划分）。
   - GinAR 在 METR-LA/PEMS-BAY 的种子 32（lr .001）已在计划 58eecfb5 中重训，三种子表已使用，无需再跑；论文现有 lr .006 种子 32 仅单列备查。
   - 复用已有结果：MCPST、MetaDG 的种子 32 来自 `extension2_test_20261105`（训练计划 a8d285aa 及恢复 034802bd、b136d9b6）；ALFNet 的种子 32 来自最终配置检查点（若审计后换配置，则使用该配置已有的种子 32 运行）。复用前用当前代码重放验证表，须在 abs 1e-5 + rel 1e-5 内一致，不一致则该对加训种子 32。另外检查是否已有可复用的种子 42/52 运行（选点规则、配置字节、代码均须一致），不符合的写明原因。
   - 新训练的配置文件与种子 32 完全相同（同 SHA256），只改种子。
   - ALFNet 的训练路径（各最终配置所需的研究上下文/适配器，如 flow support transfer、flow unified 等）必须与其种子 32 运行完全一致：先查清当初训练它的脚本与上下文，再用 1 个 epoch 的种子 32 复现与原运行第 0 轮记录比对，确认训练路径等价后再开训。
   - MCPST、MetaDG 用其研究上下文（`run_extension2_20261105.run_context()`）；沿用已冻结的恢复规则：runner 对无监督批的跳过照常计数；全部运行带非有限梯度步守卫（无事件时为空操作，任一轮事件 >1% 即失败）。
   - 预检：每对做种子 32 重放 + 种子 42 短跑冒烟，预检分片并行；全部通过后入队。
   - 并发：本任务最多同时 5 个训练进程（预检、测试也计入），每次启动前按显存余量判断。
3. **训练后测试**：训练全部干净完成（无缺失、无发散标记）后，冻结测试计划，对每个新检查点测试一次（先重放验证再测试）。授权：我授权本轮新检查点各测试一次，测试结果不得用于任何选择、调参、换方法或重跑。
4. **汇总**：扩展 `scripts/report_seeds_overview_20261110.py`（或新建同类脚本），把 ALFNet、MCPST、MetaDG 的新种子并入 xlsx；**HD-TTS、STOP 不进入报告**；第一格数的比较集 = 现有 8 个（UIGNN、GinAR、MoGERNN、SLPF、STGCN、STID、PatchSTG、STBP）+ Merlin + STONE + MCPST + MetaDG，ALFNet 与各方法均以三种子均值对均值比较，同时保留逐种子同划分比较。GinAR 在 METR-LA/PEMS-BAY 的三种子均为 lr .001（种子 32 已在计划 58eecfb5 中重训，无需再跑）；论文现有 lr .006 种子 32 仅单列备查，报告中注明。

## 3. 后台任务与会话结束规则

- 长任务用带前缀的 tmux 会话（如 `alfnet_seeds2_*`）运行一条自动链：预检 → 训练队列 → collect → 冻结测试 → 测试 → 汇总；记录 PID、命令、日志路径、计划哈希到任务 handoff。
- 本会话内可以用后台等待/监控来检查预检和训练是否正常，但每个等待必须带超时、且只指向当前计划的输出目录。
- 确认"彻底跑起来"即可结束：预检全部通过，首批 5 个 worker 都至少完成 1–2 个 epoch、日志与验证指标正常、显存稳定、无报错。之后停止并清理本会话启动的全部后台等待/监控（只保留 tmux 中的训练链），用中文向我报告状态、预计耗时、查看进度与停止的方法，然后结束。训练结束后我会再来找你分析结果。

## 4. 约束（继续有效）

- 服务器有同门在使用：不要终止、暂停或修改任何不是本任务启动的进程（包括 tmux 服务器进程和他人的 tmux 会话），不清理他人的文件、显存或缓存，不改系统配置、驱动或公共环境；每次启动前用 `nvidia-smi`、`ps` 查看占用；运行前检查磁盘空间。
- 只停自己记录过的 PID，kill 前核对命令行。
- 输出写到本轮专属目录，不覆盖已有结果、权重与日志；不修改 `references/`、`data/*/original/`、`data_v4/` 与已有 checkpoint。
- 不 commit、不 push；不修改论文正文。
- 出现显存不足或数值发散：先停下记录现象；恢复方案在看到恢复后的结果之前写入并冻结，对同一方法的全部数据集一致执行；失败与负面结果完整保留并报告。
- 测试结果不得用于选择检查点、调整超参数、更换方法或决定是否重跑。
- 解释器：`/home/xd/anaconda3/envs/ReaCH/bin/python`。

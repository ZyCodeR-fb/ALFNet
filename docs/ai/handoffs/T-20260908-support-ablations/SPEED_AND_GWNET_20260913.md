# 两轮结果、训练耗时与 GWNet 融合 — 2026-09-13

用户本轮明确允许探索 GWNet 融合。Full 仍保留；此前“不要默认融合”不再限制本轮原型开发。
以下均为 seed32 / validation 证据或有限 train 批次测速，不是新模型精度结论。

## 已核验的结果

两轮 8 个候选全部 COMPLETE，均未过各自预先设置的筛选门槛。来源：
`outputs_v4/two_speed_overnight_20260913/two_round_summary.json`，
同目录 `two_round_report.md`。原报告与 checkpoint 不修改。

- METR-LA：Full LifeAvg MAE 4.638117，四候选 4.770494–4.815248；部分 MAPE 改善不能抵消绝对误差损失。
- PEMS-BAY：Full LifeAvg MAE 3.039729，四候选 3.006424–3.027518；总体有改善但 Join/早期阶段退化，不能自动替换 Full。
- freshness 的 PEMS-BAY Unseen MAE/MAPE 4.855330/13.071821，优于 Full 5.009514/14.131389；Join MAE/MAPE 却从 2.432881/6.370869 变成 2.538395/6.995400。这是条件间折衷，继续增强同一干预缺乏依据。
- 历史 PEMS-BAY GWNET LifeAvg 2.983848/4.748691/7.489858；Full 3.039729/4.637845/7.996064。GWNET 的总体 MAE/MAPE 强，RMSE 弱；Unseen MAE/RMSE 也弱。来源为 corrected_baseline_seed32 下 GWNET 的 development_validation_table.json。当前输出检索未找到同等权威的 METR-LA GWNET validation 表。
- 该 GWNET 配方使用 MAE，而 Full 使用 MSE 加角色混合；不能把所有差异归因于结构。

## 耗时：可确认的原因与仍无法追溯的部分

第二轮实际 parallel=3 / phase slots=3，前三项 02:01 启动，第四项 06:40 才启动。整轮 12:45 完成。四项累计 phase wait 各不足 0.1 秒，因此本轮进入训练后的长 epoch 不再是阶段锁等待。

| 运行 | 完整 epoch 间隔中位数 | train 阶段中位数 | validation 阶段中位数 |
|---|---:|---:|---:|
| METR-LA freshness | 276 s | 171.03 s | 98.73 s |
| METR-LA refiner_uniform | 276 s | 171.23 s | 98.77 s |
| PEMS-BAY freshness | 467 s | 360.50 s | 96.10 s |
| PEMS-BAY refiner_uniform | 448.5 s | 362.45 s | 81.93 s |

各列独立取中位数，不要求相加等于 epoch 中位数；还存在保存及其他调度开销。
第一轮四进程共享两个整阶段槽，epoch 间隔中位数 METR-LA 484/501 s，PEMS-BAY 594/630.5 s。这个保守配置由助手设置，确实引入等待；日志未记录等待时间，无法精确倒推出占比。

用户的历史速度有依据：2026-08-20 development_base 的 PEMS-BAY seed32 日志，中位数 212 s/epoch。该 run_report 同样记录 1,037,529 参数、489 train batch、163 validation batch；不能用“现在样本更多或模型参数翻倍”解释。旧 report 的选点是旧 validation life_avg_mae，与当前严格 33 单元表 LifeAvg 选点不同，不能把旧精度直接用于现在选型。旧并发负载、逐阶段时间及 GPU 遥测不足，尚不能定量区分代码演化和共享资源竞争。

当前 Full 默认路径未新增大结构；route_shrink 只加 144 参数，freshness 是小型输入卷积。已有旧 Full/控制预测 parity 为 0，说明默认预测行为保留，但预测 parity 本身不能证明速度完全不变。

### 同批次 GPU 测量

独立有限诊断 `outputs_v4/speed_fusion_probe_20260913_v2/`，身份
`4423e8ebc5fb58f78bb746ac8f95a3c8f3d23790eaf520296c42f55e21bc5b71`。
PyTorch 2.8.0+cu128，ReaCH Python，RTX PRO 6000 Blackwell，4 CPU threads。
每项 batch64，train 前64窗口、2 warmup + 5 次同步计时 + 1 profiler，epoch50 free-run；无 optimizer.step、无 checkpoint。三模型均从新初始化测量，不是训练后精度比较。

| 数据集 | Full 前向+反向 | 原生 GWNET | ALFWave 原型 | Full / 原型峰值显存 |
|---|---:|---:|---:|---:|
| METR-LA | 0.27848 s | 0.03653 s | 0.11606 s | 9.34 / 3.93 GiB |
| PEMS-BAY | 0.40528 s | 0.06051 s | 0.16978 s | 14.65 / 6.09 GiB |

原型两数据集短批次分别快 2.40/2.39 倍。数据整理、完整验证、optimizer、保存、并发不在此倍率内。各项顺序执行时外部任务仍在，因此不是独占 GPU 的严格性能竞赛。

CUDA event 对 profiler 批次的前向分解：Full 图循环 92.6/87.9 ms，GRIN 44.6/47.1 ms，时间分支 3.8/2.9 ms。事件只覆盖相应模块且 profiler 会扰动计时，不能当整 epoch 占比。Full 该批有 1,390 次 aten::bmm、499 次 aten::mm；图递归/GRIN 顺序计算与大量小算子是真实开销。不能把主要负担归给 patch 容量8，也不能据此删掉有消融支持的跨节点时间分支。

GPU 不是没使用：profiler 有实际 CUDA kernel。训练之外的一次 nvidia-smi 快照为30%利用率，不能代表昨夜训练利用率，更不能据此计算历史闲置比例。他人 PID1734562 全程只读观察。

### 验证统计存在可优化同步

`ValidationCellCollector.update/_error_sums` 及 runner 的逐样本 scenario 统计，对 GPU tensor 重复 `bool/int/float`，造成 CPU 等待 GPU。用已有 Full validation 预测的各64窗口单独计时，同一 `_error_sums` 运算：

- METR-LA：GPU逐标量读取 69.35 ms；先整批复制 CPU 再归约 4.04 ms。
- PEMS-BAY：68.44 ms → 4.36 ms；CPU拷贝已计入。
- 计数相同，归约和最大相对偏差 2.11e-7；这不是完整33单元或选点等价验证。

这里只隔离了 collector 的一部分，不能宣称整次 validation 快16倍。若上线 CPU统计或批量GPU归约，必须新身份、旧 checkpoint 全部33单元验证及选点容差检查；本轮未改变生产 metric 实现。该开销也不足以独自解释所有分钟级差异。

## 融合原型与有区分力的下一轮

已实现 `model/ALFWave/__init__.py`：共享 GRIN → ALFTemporal 跨节点特征 → FiLM 调制 GWNET skip feature → 一次输出12步 → 原 graph_refiner。仅替换循环预测器，去掉其自回归与动态专家路由；保留原角色目标/输入及 refiner。GWNET 自适应图继续由共享结构编码器产生，不新增未见节点专属可训练 embedding。该结构改变不只是换激活函数，不能把后续收益单独归因于时间卷积。

原型 971,624 参数，少于 Full 1,037,529。独立目录、不改 Full，未登记正式 training dispatcher，不能拿旧 plan 直接启动。构造目前复用 ALFNet 配置接口，正式接入前须登记独立模型身份和配置 builder。

建议下一阶段最多6个 seed32 run（三臂×两数据集），独立名称 `two_speed_wave_fusion_v1`，尚未冻结为执行计划：

1. **W-MSE（优先）**：ALFWave，保留 Full 的 MSE/角色混合目标，其余配置不叠加失败配方。判断替换循环预测器是否同时改善精度和速度。
2. **W-MAE（对照）**：相同 ALFWave、相同初始化和角色混合系数，仅基础及角色逐元素误差改 MAE。判断 GWNET 历史优势有多少来自误差目标。与旧 0.05 bounded-MAE 辅助项不同；预先约束 RMSE 退化。
3. **GWNET-MAE（基线）**：两数据集同当前曝光/数据/指标，独立 source-native 输入，无 ALFNet 生命周期特征。PEMS-BAY 旧 checkpoint 只有语义等价审计通过才可复用；METR-LA 需要补齐相同协议参考。它与 W-MAE 对照融合增益。

保留 batch64、METR-LA容量8/PEMS-BAY11、seed32、train K/validation K∪V、12/12、100epoch上限、patience10、strict LifeAvg MAE/earlier tie；为比较固定 early-stop-start40。直接预测声明无 teacher forcing，不能伪报 TF40 仍在使用。先2并发测总体 samples/s，再决定3；不是直接把进程数加到5。并发上界5不变。

拟议筛选门槛须在正式计划冻结时固定：LifeAvg MAE 至少改善1%、MAPE至少改善2%、RMSE不劣于 Full；Unseen MAPE改善至少2%、MAE/RMSE不退化；Join及Cold/Warm/Ready三指标最多退化0.5%，其余单元最多2%。通过只是 seed32 筛选，不能声称多种子稳定或自动替换 Full。若 W-MAE 改 MAPE 但损 RMSE，则保留为条件折衷，不继续事后调门槛。

不建议立即再叠加 freshness/shrink/refiner_uniform；暂不默认两整网预测平均，缺少同窗口误差互补证据，而且保留了 Full 的计算开销。可后续用同源验证预测先做固定权重扫描判断互补性，再决定是否值得额外训练。

## 验证与边界

- 新合成测试3 passed：恒等FiLM与原生GWNET完全一致、decoder target隔离/拒绝teacher、条件参数与卷积梯度存在。
- 两数据集 batch64 原型反向通过，目标扰动最大预测差0，FiLM梯度有限。尚未完整训练/精度验证、完整仓库测试、完整正式计划与配置审计。
- 首次 profiler 因 ALFTemporal.forward_v4 绕过外层hook未记录事件而失败；保留 `speed_fusion_probe_20260913/`，v2改挂backbone后六项通过，未丢弃正式训练结果。
- Full及已完成两轮源码未修改；无 test / commit / push / 新一轮完整训练，无他人进程操作。

# 两数据集机制诊断与候选执行

## 授权与运行范围

用户已接受建议并明确允许两个数据集同时诊断、多个训练并行，最多五个训练同时运行。本轮采用两候选 × METR-LA/PEMS-BAY × seed32，共四个训练；最多四个进程并行，GPU 活跃训练/验证阶段保守限制为两个。不是要求用满五个名额。保留原 Full 与全部旧证据，不执行 seed42/52、test、commit 或 push。

Git 基点仍为 paper/final / 9346dc0fc1dfb6b4aebfffe25224f1337f7a94cb；本轮在既存未提交改动之上新增 opt-in 功能。旧配置默认路径不分配新参数、不更换损失。代码修改使旧 plan 的源码哈希过期，这是预期的历史身份边界，不能重写旧 plan 修正它。

## 已完成诊断

诊断计划 `configs/paper/two_speed_diagnostic_v1.json`：
`4809eb19144dbce8b310437daea0ed06d1f0fa7d100d088cd9c28542730228c3`。
结果位于 `outputs_v4/two_speed_diagnostic_v1/<上述哈希>/{METRLA,PEMSBAY}/`。
两个 COMPLETE 均存在，原始预测、目标、有效 mask、q、邻居支持密度保存在各 arm 的 npz；summary.json、selection.json、effective_experts.json 保留汇总和身份。运行时逐 arm 检查目标/mask/origin/支持一致，结束时全部源文件与旧 checkpoint 哈希检查通过。源码快照与环境信息保存在根目录 snapshot/、environment.json。

每数据集按 Unseen/Steady/六阶段各随机取 32 个自然 schedule 窗口，共 256 窗口；seed=20260912，无误差或自然完整性筛选。四个 checkpoint 控制为 Full、Mask-only、Static、MoGERNN；两组固定推理干预为均匀路由、GRIN decoder 观测发送者占 0.75 的分组聚合。各 arm 都预测全图，但评分仅用原合法 cohort/loss mask。

这是自然验证集的**分层子样本 pooled 诊断**，不替代完整 33 单元平衡主表。各数据集 Unseen 仅 32 窗口，低速度单元数量小；不宣称统计显著性、整个验证集总体归因或训练后的因果效果。

| 诊断 | METR-LA | PEMS-BAY | 决策 |
|---|---|---|---|
| 均匀路由对 Full | Steady MAE +16.06%，Unseen MAE +6.44% | Steady MAE +24.92%，Warm MAE +19.35% | 不能整体去动态路由；保留向动态端初始化的有限收缩候选 |
| 专家实际输出离散程度 | 相对 RMS 均值 0.604 | 0.574 | 相同支持不等于相同专家：每个专家有独立 Theta，不能由矩阵冗余推出预测冗余 |
| 固定 GRIN 分组对 Full 的 Unseen | MAE −0.08%，RMSE +0.39%，MAPE +0.24% | MAE +0.21%，RMSE +1.71%，MAPE +0.44% | 未建立一致正响应，C-U 不进首轮训练；不等于排除所有 GRIN 方案 |
| Unseen [0,20) | 77 步，Full 偏差 +29.806，MoGERNN +27.042 | 17 步，Full 偏差 +26.135，MoGERNN +16.697 | 低速度高估在两数据集都有，但不能作为唯一原因 |

MAPE 的全样本差异与分箱贡献必须区分：METR-LA Unseen Full/MoGERNN=15.302/15.317，低于20箱贡献差 +0.258 个百分点、≥60箱 +0.403，中间箱改善抵消它。PEMS-BAY=6.604/6.250，低于20箱贡献差 +0.164、≥60箱 +0.263。不能把新子样本结果写成主表排名改变，也不能把“低速度高估”写成所有 MAPE 差异的解释。

## 首轮候选与冻结身份

训练计划 `configs/paper/two_speed_mechanism_v1.json`：
`afd9f69d16e5b59429a8eda619b7e9874e1e21cd28de4af4873ad3e86ee5bdf9`。
输出根 `outputs_v4/two_speed_mechanism_v1/<上述哈希>/`。

1. **route_shrink（C-S）**：每个 ALFGraph router 使用 soft 动态混合与学习静态先验的凸组合。alpha 由原六维合法状态线性投影加 sigmoid 得到，初始化 0.95；新参数零/常数初始化，不额外消耗随机数。BASE 算子及各自 Theta、GRIN、时间分支、FiLM、细化和原损失不变。
2. **bounded_mae（C-M）**：仅在 ALFNet 显式声明时，在原总训练损失上增加 0.05 × role-balanced weighted MAE（标准化残差）。权重为 clip(tau/max(abs(raw target),1e-6),1,3)，tau 仅由训练时间 K 有效速度第20百分位确定。METR-LA tau=53.5，PEMS-BAY tau=61.20000076293945；有效训练值分别 2,984,367 / 7,629,636。权重不用于验证指标，非旧 signed_log_huber，不加校准头。

选择 C-M 的依据是两数据集自然 Unseen 都出现了目标值分层偏差，且旧 LF 的误差交换要求保留强 MAE/RMSE 守门；这只是值得测试的机制，不能保证比旧 LF 更好。C-S 的干预证明路由有效，不证明静态收缩已有效，需由训练筛选门槛判断。C-U 与 C-E 本轮不训练，不叠加候选。

四运行均继承各数据集 Full 的 batch64、优化器、学习率、teacher forcing、最大100 epoch、patience10、early-stopping-start40，以及 patch 容量 METR-LA8/PEMS-BAY11。选 checkpoint 仍为严格改善的 validation LifeAvg MAE，tie 保留更早 epoch。

筛选规则冻结在 plan.gates：LifeAvg 任一指标退化不得超过0.5%，任一33单元退化不得超过2%；C-S 要求 PEMS-BAY Steady/Fault MAE/MAPE 各改善至少2%，Join和METR-LA Unseen 不退化超过0.5%；C-M 要求两数据集 Unseen MAPE 改善至少2%、LifeAvg MAPE 至少1%，LifeAvg/Unseen MAE/RMSE 不退化。跨数据集分别判断，筛选通过不自动替换 Full。

## 验证与运行入口

- 合成机制测试及既有控制/暴露检查：22 passed、3 deselected；排除真实 test split 及不在本轮验证范围的测例。没有运行完整仓库套件。
- builder `--check` 已通过；训练 quantile 重算及源/配置/数据/诊断证据哈希匹配。
- 预检入口 `scripts/preflight_two_speed_mechanism.py`：每数据集四控制、八窗口预测与诊断前代码输出比对（原尺度容差1e-5）；四候选 CPU batch1 + GPU batch64 单批训练/验证，显式拒绝 test context。
- launcher `scripts/run_two_speed_mechanism.py --parallel 4`：要求匹配身份且 passed 的 preflight.json，保存源码/配置快照、逐任务日志与 launcher_status.json；每次分配任务重新检查身份，最多五进程参数上限。
- 完成后自动调用 `scripts/collect_two_speed_mechanism.py`，写 candidate_summary.json，公开33单元差异与失败门槛；不自动晋升 Full。
- 资源观察：沙箱外 GPU 为 NVIDIA RTX PRO 6000 Blackwell，97,887 MiB；诊断前约981 MiB占用、无训练进程。沙箱内不能访问驱动，GPU 命令在已授权范围使用沙箱外执行。
- 运行状态、实际峰值与预检结果以新输出根内 preflight.json / launcher_status.json 为准；不能以本文计划文字代替完成证据。

## 启动核验

全部八项预检通过，八个控制预测一致性检查最大误差均0；METR-LA route_shrink/bounded_mae 的 GPU 峰值9.381/9.341 GiB，PEMS-BAY为14.746/14.668 GiB。
2026-09-12 14:31:21 +0800 四任务已由持久 tmux 会话
`two_speed_mechanism_20260912` 启动。launcher PID1370200；四 worker
PID1370245/1370246/1370247/1370248。约14:33核验均存活；两METR-LA任务已取得GPU槽训练，PEMS-BAY等待阶段槽，GPU约24,215 MiB、利用率86%。这是带时间戳的观察，后续须重新查进程。

尚处于首个epoch，尚未核验正式last.pt/best.pt生成；不能把预检成功当作训练完成。恢复点由原训练入口在epoch边界保存。完整训练后launcher自动收集candidate_summary.json；没有自动重启、追加候选或打开test的逻辑。

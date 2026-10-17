# 最终 ALFNet 的 flow 扩展：2026-09-26

## 范围和研究问题

用户授权 PEMS03/04/08、ALFNet/MoGERNN、容量与学习率搜索，最多同时5项训练。
本轮 seed32，训练和 validation；不打开 test、不替换已有正式表、不改架构和目标函数。
检验冻结的条件化模型能否从 speed 推广到 flow，以及容量、学习率的敏感性。

## 数据和结构

保留三个原有 flow 数据集：PEMS03/04/08 的 graph_native_component_bfs
分别以12/10/6为最小合法容量。固定32个分区及其成员；容量是每个分区的节点槽位，
不是分区数。更大的容量会改变补齐节点上下文。无需伪造 WGS84 坐标或因旧结果较弱更换数据集。
speed 使用 KD-tree，flow 使用图分块，需披露此静态布局适配，不能将两者差异单独归因于坐标。
PatchSTG 官方实现使用 KD-tree（https://github.com/lmissher/patchstg）；
ASTGCN 官方提供 PEMS04/08 flow 数据（https://github.com/wanhuaiyu/ASTGCN）。
保留 speed+flow 有助于检验跨测量量的泛化；不跨数据集合并有不同单位的 MAE/RMSE。

SUP 统一使用 identity/mean/weighted_mean/forward_diffusion/backward_diffusion。
二值邻接可能使 mean、weighted_mean 和前向扩散重合；预检记录实际矩阵距离。
不为了使算子不同而改图、不复活失败 SUP5 配方。Focal 的 raw-error 权重也保持不变；
flow 量纲可能使 tanh 权重更早饱和，此为需分析的迁移因素，不预设容量能解决所有问题。

## 固定候选

每个数据集5项 ALFNet，共15项。架构和 Focal 配方来自正式 METRLA ALFNet 配置，
只换数据字段、合法容量和学习率。所有训练从头初始化，不迁移 speed 权重。

| 数据集 | 默认容量 | 容量候选（LR×1） | LR候选（默认容量） |
|---|---:|---|---|
| PEMS03 |12|16、20|×0.5、×2|
| PEMS04 |10|14、18|×0.5、×2|
| PEMS08 |6|8、10|×0.5、×2|

ALFNet 三优化器组 LR=5e-4/1e-4/5e-5，整体同比缩放；batch64，100epochs上限，
early stopping 从40开始、patience10。Focal、role-balanced beta0.1、dropout0.1不变。
MoGERNN 使用修正训练器历史权重，须核验数据、scaler、seed、exposure、模型契约、
严格权重加载及完整33格validation复现；不满足则停止，不默默复用。
MoGERNN保留自身MSE与源早停（start20/patience3），不引入 ALFNet 模块或损失。

## 选择和报告

每个 checkpoint 由 validation LifeAvg MAE 选取，相同取较早epoch。
候选配置也按完整 validation LifeAvg MAE最小选取，精确相同则优先默认、其后按计划顺序。
同时完整报告15主表格+18阶段格，三个指标和所有候选，不按test或隐藏弱项选型。
替换默认参数的门槛：LifeAvg MAE至少降低0.5%，RMSE/MAPE各不退化超过0.5%，
Unseen MAE不退化超过1%，Cold/Warm/Ready平均MAE不退化。
若最小MAE候选不通过门槛，保留默认并报告，不另以获胜格数挑选。
80%是对MoGERNN的评估目标：全33格至少27胜，同时主表至少12/15、阶段至少15/18；
按未四舍五入的值及1e-6绝对容差，平局不算胜。这不能证明胜过尚未运行的全部基线。
这是单种子开发结果，最终test须另行冻结和授权。

## 运行控制

独立 plan.json 固定代码、配置、数据和复用证据SHA256；启动前check。
预检全部候选的同数据身份、参数初始化、实际容量/LR及GPU两批训练/验证。
最多5训练进程，显存不足等待；失败保留日志，停止派发后续任务，不自动降batch。
运行目录为 artifacts/experiments/flow_extension_20260926/<plan_sha256>/。
历史修正MoGERNN需完整validation再评估通过；先复用，避免重复训练。

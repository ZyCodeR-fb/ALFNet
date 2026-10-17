# Flow消融的最后一轮复验：诊断与待确认范围

状态：诊断完成；新增Full训练范围等待用户确认，未启动训练。不是冻结运行计划，不能用本文替代带配置/代码/数据摘要的正式plan。

## 1. 已核验的结果

以下是消融相对同数据集、同seed32最终Full的test LifeAvg误差变化。负数表示消融更优。

| 数据集 | 消融 | MAE变化% | RMSE变化% | MAPE变化% | 三指标是否均变差 |
|---|---|---:|---:|---:|---|
| PEMS04 | w/o State Age | -0.64 | -1.71 | +3.00 | 否 |
| PEMS04 | w/o Adaptive Routing | +0.66 | +0.10 | +0.68 | 是 |
| PEMS08 | w/o State Age | +2.63 | +2.82 | +26.83 | 是 |
| PEMS08 | w/o Adaptive Routing | -2.27 | -1.67 | +5.28 | 否 |

来源：`all_ablations_test_20261010/results_and_audit.json`。验证结果：`ablations_complete_20261010/results_and_audit.json`。

PEMS04 no_age在validation上MAE/RMSE也更好，因而当前证据不是仅在test随机反转。PEMS08 static_router在validation三指标均更差、test MAE/RMSE更好，可能反映时间/节点分布及指标权衡；单次运行不足以确定原因。两项在test的MAPE均变差，说明并非完全无贡献，但不支持所有误差尺度全面改善。

## 2. 实现诊断

- `experiments/alfnet_ablation_adapter.py:remove_age` 在完整模型前向入口将age通道替换为mask，保留角色、真实监督标签和评价掩码。图/时间分支收到的显式age已经被替换；目前未发现需要“再关闭别的模块”才能实现该定义的证据。保留的角色和mask本来就可能承载部分阶段信息，这是独立age特征的互补性检验。
- static_router由嵌套图配置设定`moge_router_mode=static`，保留传播算子，使用全局可学习混合。它是合理且较强的非自适应对照，不是应当故意削弱的实现错误。
- 已做过更宽的`no_expert_lifecycle`（同时削去专家输入的角色/age信息），其test只有4/12项变差；“关更多就一定更差”并不受已有结果支持。
- 当前`lib/v4_seeds.py`将run_seed直接用于node_split，并派生模型初始化、训练日程、shuffle、教师强制等子种子。同seed重跑不等于独立随机重复；新run_seed也不是纯优化噪声实验。

## 3. 推荐的有界复验（待范围确认）

不修改消融定义、不搜索新超参数、不按结果追加运行。

| 数据集 | 模型 | 新run_seed | 新训练数 |
|---|---|---|---:|
| PEMS04 | Full、w/o State Age | 42、52 | 4 |
| PEMS08 | Full、w/o Adaptive Routing | 42、52 | 4 |

共8项，最多5项并行，按显存门槛调度。已有seed32的同配置同干预结果作为第三组完整记录保留；已满足test标准的两个数据集/干预组合不额外训练。新种子只允许同种子Full与消融配对，不能拿seed42/52消融对seed32 Full。

这检验的是既有种子契约下的跨划分与训练随机性稳健性。时间划分比例和协议保持不变，节点成员及训练scaler可随run_seed变化。若目标必须是“固定seed32节点划分、只改变初始化”，需要另立明确分离split_seed与optimization_seed的新协议，不能偷偷改现有seed契约；本轮不默认做这一扩展。

训练前核验：新种子数据身份可构建；每对共享数据/掩码/监督、保留张量初始化一致；Full和干预继承当前已冻结的flow输入处理与超参数；使用独立输出根和计划身份。预算保持max_epoch100、patience10、start40、按validation LifeAvg MAE存best。先汇总完整validation，不自动打开新test。

完成后固定报告每种子结果、配对差值、平均差及样本标准差；n=3只作描述性证据，不夸大显著性。不以“三指标是否变差”筛掉任何运行，不用较差重跑替换原seed32消融。原正式主表配置与数值不变。

## 4. 修改消融定义的可选研究问题

- **Age→常数0**：可额外检验当前用mask替代age这一替代编码的影响。但必须命名为独立对照并保留原no_age，不在观察结果后以较差者替代原项；也不保证会变差。
- **固定均匀算子混合**：检验完全不学习传播权重的情况，比当前全局可学习静态混合多去掉了权重学习能力，不能把两者差距全部归因于输入自适应。
- **Age与Routing联合消融**：可以回答二者联合依赖问题，但不能用于证明Age或Routing单独必要。若做交互结论，需要Full、去Age、静态Routing、两者同时干预的完整2×2对照及同种子配对。

当前推荐先做上面的配对复验，不额外增加这些定义。若差异继续保留，应将贡献写成误差尺度/条件间的互补，而非所有组件在所有数据集都降低全部误差。

## 方法依据

- Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021：区分数据抽样、初始化等变异来源，不能将所有重跑差异当作同一种随机噪声。https://arxiv.org/abs/2103.03098
- Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 2010：在有噪声的评估结果中挑选有利结果会产生选择偏差。https://www.jmlr.org/beta/papers/v11/cawley10a.html

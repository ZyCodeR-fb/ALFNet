# 最后一轮flow定向修复 — 2026-10-02

授权：用户明确要求文献检索、定位异常误差并最后尝试多种修复，最多5训练并发。本计划限定PEMS03/04/08各4候选，共12项seed32训练；复用3个当前原版对照，不重训任何比较算法或speed模型。没有自动test、删除/替换数据集、提交或推送。

## 证据与可检验问题

原版完整validation重放3/3最大差0。无历史目标低于10时，PEMS08真实均值5.41、预测96.93、填补末值136.59；PEMS03对应5.30/21.03/34.25，PEMS04为6.21/38.11/52.66。高于300区间则普遍低估。这些是微平均分箱诊断，不是论文canonical33重放加权数值。填补锚点与最终误差方向一致，但不能用相关性认定全部由填补造成。

上轮标准化Focal在03/04变差，08的validation收益也未迁移到test。因此恢复原raw Focal对照，不重复标准化Focal、角色平衡删除或图细化删除，不复用旧LF/SUP具体配方。

## 固定矩阵

| 候选 | 单独改动 | 检验目标 |
|---|---|---|
| reconstruction | 原Focal＋0.25×隐藏历史重建MAE（标准化单位） | 直接监督GRIN输出，减轻无历史输入锚点的中间值收缩 |
| degree_supports | 两个重复mean槽位改为有界度数放大/衰减；其余3槽及全部Theta保留 | 测试均值传播丢失度数尺度信息的影响，非扩大传播跳数 |
| reconstruction_degree | 前两项组合 | 只测已明确设计的两机制交互 |
| relative20 | 0.8×原Focal＋0.2×受限相对MAE | 对低流量高估直接施加梯度，同时保留原绝对误差目标 |

固定原容量/LR/dropout/WD：03=C16/LR1/0.1/1e-4，04=C10/LR1/0.1/1e-4，08=C6/LR2/0.1/1e-4。全部max_epoch100/patience10/早停起点40/TF40/batch64/32分区；初始化参数逐项一致，无新增可学习参数。0.25、0.2及下限0.1是本轮冻结的工程候选，不宣称理论最优，不在观察结果后改值。

## 实现与信息边界

- `experiments/flow_last_round.py`以进程内作用域适配器实现；生产ALFNet、loader、评价器不改，不复制工作区。新方法身份进入checkpoint contract，后续恢复和评估须安装同一适配器。
- 重建损失仅使用**训练时间K节点**人工隐藏但自然有效的历史标签，由batch.origin确定过去12步；读取前先截取K和时间，禁止V/T和未来。标签只在forward结束后计算loss，不注入模型输入、decoder teacher或validation。训练时额外使用隐藏历史监督是本候选的方法差异，需披露；不宣称各方法优化的标签集合完全一样。源GRIN骨干与参数数量不变。
- degree_supports保持identity、forward/backward diffusion；mean与weighted_mean槽改为rho*P和P/rho。rho=clip(log(1+d)/mean_positive(log(1+d)),0.5,2)，只依赖当前合法静态邻接。不是PNA完整复现，不增加max/min/variance聚合，不是旧多跳/PPR SUP配方；规则图上rho可能相同，不夸称任何图都去冗余。
- relative20相对项为abs(raw_pred-raw_target)/max(abs(raw_target),0.1*train_std)。系数只用于训练loss；预测、零值掩码、MAPE阈值与33格统计完全不改。相对项独立于原Focal，保留80%原目标；不是已失败的signed_log_huber＋校准头或旧MSE＋0.05 bounded-MAE配方。仍可能牺牲绝对误差，所以单独检验。
- 不采用普通逐节点RevIN：Unseen没有自身统计量；不使用测试节点均值、额外speed、OSM/车道或仿真信息。没有证据时不伪造坐标、不认定BFS是bug。

## 预检与运行

针对性测试包含隐藏历史监督的K/时间/可用性隔离与梯度、图置换等变/孤立点/保留算子、相对loss零值和负预测有限性。每项GPU20train＋4validation预检；初始参数、曝光/scaler/layout/LR/WD/dropout与对照一致；零skipped batch。预检通过后独立plan队列最多5并发，显存门控和30秒启动间隔。失败保留、不无限重试。

checkpoint仍仅按完整validation LifeAvg MAE选择。完成后导出全部15项control+candidate的495格validation主表/阶段表、对8个固定baseline的胜格数与三项LifeAvg。80%为27/33目标，不是保证；无额外硬性退化门槛，保留逐条件取舍。新test需候选冻结与授权，不自动扩大。

## 停止规则与数据集

这是最后一个已授权训练矩阵，不自动根据本轮结果追加超参数组合。若没有可复核的改善，保留原权重并结束修复。可以将论文主张收窄到speed，把已完成flow作为泛化边界完整报告；不因胜率低隐藏原实验。更换数据集意味着所有baseline需要新训练，与当前约束冲突，因此本计划不启动替换。既有test反馈必须如实说明，单seed不作显著性宣称。

# 三轮消融完整汇总（2026-10-10）

全部48项训练已完成；12种干预×4数据集，复用4项Full，共1716指标格。满足“四数据集三项LifeAvg都严格变差”的消融共 **0种**。
按原始未舍入数值与各自固定Full比较；正值=消融更差，负值=消融更好。参数/数据身份、最佳epoch选择、完整计数、有限权重及零跳批审计通过；无新test/训练。

| 消融 | 变差指标/12 | 三指标均变差的数据集 |
|---|---:|---|
| w/o State Age | 7 | METRLA, PEMSBAY |
| w/o Adaptive Routing | 10 | METRLA, PEMSBAY, PEMS08 |
| w/o Spatial Context | 10 | METRLA, PEMSBAY, PEMS04 |
| w/o ALFRecon | 11 | METRLA, PEMSBAY, PEMS08 |
| w/o ALFTemporal–FiLM | 11 | METRLA, PEMSBAY, PEMS04 |
| w/o Graph Refinement | 6 | METRLA, PEMSBAY |
| w/o Expert Lifecycle Context | 1 | 无 |
| w/o FiLM Multiplicative Modulation | 4 | METRLA |
| w/o FiLM Additive Modulation | 5 | METRLA |
| Pointwise Refinement | 8 | METRLA, PEMSBAY |
| Stop Graph-to-Reconstruction Gradient | 7 | METRLA, PEMS08 |
| Stop Temporal-to-Reconstruction Gradient | 6 | 无 |

## 本轮两个梯度对照

| 数据集 | 阻断图→重建 ΔMAE/RMSE/MAPE% | 阻断时间→重建 ΔMAE/RMSE/MAPE% |
|---|---|---|
| METRLA | +7.77/+6.20/+10.88 | +1.03/+0.29/-0.12 |
| PEMSBAY | -2.22/-2.05/-2.51 | -0.58/+0.25/+0.55 |
| PEMS04 | -4.93/-4.93/+3.80 | -4.15/-4.89/+5.29 |
| PEMS08 | +9.15/+7.86/+2.77 | -0.59/+0.48/-7.60 |

图分支梯度对METR-LA、PEMS08有一致正向作用，但PEMS-BAY阻断后更好，PEMS04为MAE/RMSE与MAPE取舍。时间分支梯度没有在任何数据集同时改善三项LifeAvg的证据（以本轮方向比较为准）。
新两项均未满足全四数据集条件。原先缺失的BAY Pointwise结果也已纳入，未改变此前结论。
## 全部LifeAvg相对变化

| 消融 | METRLA | PEMSBAY | PEMS04 | PEMS08 |
|---|---|---|---|---|
| w/o State Age | +1.73/+0.78/+1.30 | +0.27/+1.62/+2.96 | -2.76/-3.14/+2.34 | -1.95/-0.74/-0.77 |
| w/o Adaptive Routing | +4.02/+2.68/+2.52 | +1.51/+1.87/+4.68 | -0.13/-0.69/+1.41 | +1.96/+1.43/+8.12 |
| w/o Spatial Context | +7.15/+4.05/+11.65 | +5.74/+5.29/+5.89 | +3.40/+1.23/+12.31 | -0.31/-1.33/+0.48 |
| w/o ALFRecon | +5.30/+3.91/+3.78 | +0.29/+0.78/+1.98 | +1.77/-0.12/+17.46 | +9.27/+6.21/+9.11 |
| w/o ALFTemporal–FiLM | +15.48/+15.28/+11.83 | +2.05/+4.30/+4.70 | +5.39/+5.18/+21.58 | +11.62/+10.61/-5.04 |
| w/o Graph Refinement | +1.66/+1.33/+0.56 | +0.44/+1.78/+3.25 | -0.93/-1.02/-1.19 | -4.26/-3.59/-0.64 |
| w/o Expert Lifecycle Context | -0.18/-0.59/-0.77 | -1.38/-0.56/-1.40 | -2.37/-3.56/+2.12 | -5.01/-5.77/-4.37 |
| w/o FiLM Multiplicative Modulation | +1.20/+1.41/+2.31 | -0.87/-0.49/-0.77 | -0.07/-1.00/+3.20 | -8.32/-7.70/-15.88 |
| w/o FiLM Additive Modulation | +1.75/+1.08/+1.87 | -1.79/-1.72/-2.24 | -3.39/-3.42/-1.13 | +2.59/+2.32/-6.72 |
| Pointwise Refinement | +1.68/+0.57/+2.20 | +0.92/+1.84/+4.11 | +0.04/+0.30/-1.14 | -1.39/-1.15/-2.76 |
| Stop Graph-to-Reconstruction Gradient | +7.77/+6.20/+10.88 | -2.22/-2.05/-2.51 | -4.93/-4.93/+3.80 | +9.15/+7.86/+2.77 |
| Stop Temporal-to-Reconstruction Gradient | +1.03/+0.29/-0.12 | -0.58/+0.25/+0.55 | -4.15/-4.89/+5.29 | -0.59/+0.48/-7.60 |

## 解释范围

“没有任何项在全部12指标变差”不等于完整模型没有优势，也不等于所有组件无效；它否定的是当前单seed配置下所有组件具有跨数据集、跨指标一致增益的强说法。
应保留数据集依赖和指标取舍，核心结构删除、条件信息控制、梯度路径控制分别解释。不要把它们当作同等层级的独立创新。
建议暂停以达到全12格变差为目标的搜索。若继续研究，优先验证稳定性或解释已发现的冲突；这不是本次自动启动的新任务。
正式比较表、最终模型配置和已有test表均未修改。后续论文引用应使用本完整版本，先前中期汇总仍保留为历史快照。

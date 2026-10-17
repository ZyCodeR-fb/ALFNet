# 48项消融test结果分析

48/48项成功；48项完整validation复核通过，最大绝对差0。四个Full按相同checkpoint复用，1716格及第一加粗/第二下划线回读验证通过。

符合四数据集三项LifeAvg均严格变差的消融：**2种**，分别为w/o Spatial Context、w/o ALFTemporal–FiLM。

| 消融 | validation变差/12 | test变差/12 |
|---|---:|---:|
| w/o State Age | 7 | 10 |
| w/o Adaptive Routing | 10 | 10 |
| w/o Spatial Context | 10 | 12 |
| w/o ALFRecon | 11 | 8 |
| w/o ALFTemporal–FiLM | 11 | 12 |
| w/o Graph Refinement | 6 | 10 |
| w/o Expert Lifecycle Context | 1 | 4 |
| w/o FiLM Multiplicative Modulation | 4 | 5 |
| w/o FiLM Additive Modulation | 5 | 5 |
| Pointwise Refinement | 8 | 10 |
| Stop Graph-to-Reconstruction Gradient | 7 | 5 |
| Stop Temporal-to-Reconstruction Gradient | 6 | 10 |

## 两项一致test收益

| 数据集 | w/o Spatial Context：ΔMAE/RMSE/MAPE% | w/o ALFTemporal–FiLM |
|---|---|---|
| METRLA | +14.52/+9.20/+21.22 | +10.98/+7.82/+13.77 |
| PEMSBAY | +8.62/+6.95/+11.72 | +6.13/+6.82/+11.40 |
| PEMS04 | +1.84/+2.53/+6.97 | +6.50/+5.52/+13.35 |
| PEMS08 | +4.17/+4.20/+16.73 | +6.12/+7.57/+22.72 |

跨节点时间上下文及整条时间条件分支是本次test上支持最一致的两项机制。两种干预互有关联：前者保留时间分支/FiLM，仅移除跨节点上下文；后者删除整条分支。因此不是两个完全独立模块贡献的证明。
## validation与test的差别

Spatial Context的PEMS08 MAE/RMSE在validation改善、test变差；ALFTemporal–FiLM的PEMS08 MAPE在validation改善5.04%、test变差22.72%，使它们从10/12与11/12转为12/12。
State Age、Adaptive Routing、Graph Refinement、Pointwise Refinement和时间→重建梯度对照在test均为10/12，仍有一个flow数据集MAE/RMSE例外。ALFRecon则从validation11/12降为test8/12，其flow误差尺度存在取舍。
同checkpoint的validation复核完全一致，未发现因适配器错误或权重混用导致的分歧。validation/test时间段及V/T未见节点不同，分布改变可以解释方向变化的可能性；仅凭聚合指标不能区分时间/节点效应或声称因果证明。
## 论文使用建议

主文可强调跨节点上下文与时间条件分支在test上的跨数据集一致收益，同时保留原六项完整结构消融；细粒度FiLM、生命周期输入与梯度路径实验可另列机制分析/附录。不能只保留两项而将其余结果抹去。
validation表与test表明确标注；12/12是描述性方向一致，不是统计显著性。单seed与既有test反馈参与研究设计的限制仍需保留。本次未改模型、未新训练，也未启动下一轮。

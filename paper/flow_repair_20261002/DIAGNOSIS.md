# Flow差距诊断与旧配置重选 — 2026-10-02

## 已有结果重新选型

范围为ALFNet与8个基线，排除ST-FiT、Merlin。全部67配置的两个split排名见`../final_four_tables_20261001/eight_baseline_candidate_ranking.json`。不拼接不同checkpoint的单个最优格。旧test已经参与选型，这不是新的独立test确认。

| 数据集 | 选定容量/LR倍率/dropout/WD | validation第一/33 | test第一/33 | test LifeAvg MAE / RMSE / MAPE |
|---|---|---:|---:|---|
| METR-LA | 7 / 1 / 0.1 / 1e-4 | 29 | 33 | 3.8421 / 6.0846 / 10.3365 |
| PEMS-BAY | 16 / 2 / 0.1 / 1e-5 | 24 | 29 | 2.7232 / 4.2006 / 6.2761 |
| PEMS03 | 16 / 1 / 0.1 / 1e-4 | 16 | 8 | 35.2678 / 47.3101 / 50.4365 |
| PEMS04 | 10 / 1 / 0.1 / 1e-4 | 19 | 22 | 38.1597 / 51.6398 / 25.0477 |
| PEMS08 | 6 / 2 / 0.1 / 1e-4 | 21 | 21 | 38.4054 / 49.2460 / 52.2712 |

仅PEMS04替换：旧combo1为21格，LifeAvg39.9994/54.0097/25.2619；default多一格且三项平均误差一致改善。validation/test同步这一权重，不能分别展示最优配置。

PEMS03的WD-low有9格，但LifeAvg36.5545/48.8053/52.3279，三项比当前高3.65%/3.16%/3.75%。综合平均质量保留8格，明确不称其覆盖最多。没有任何已跑配置同时实现“每个单项最小且胜格最多”。其余三个数据集选定配置仍位于test最高胜格集合。

目前test合计113/165；LifeAvg14/15第一，缺的是PEMS03 MAPE。两个speed超过80%，三个flow均未达到至少27/33。新修复的成功不能预先承诺。

## 新增数据侧证据（无新test读取）

命令：`/home/xd/anaconda3/envs/ReaCH/bin/python scripts/diagnose_flow_generalization.py`。只统计train时间K和validation时间V，不建test context、不做模型forward。原值由同一train-K scaler反标准化；仅使用共同自然有效掩码。完整精度、样本数见`distribution_diagnostics.json`。

| 数据集 | train-K节点时间均值的CV | validation V无直接K邻居比例 | validation V低值占比 |
|---|---:|---:|---:|
| METR-LA | 0.114 | 0.00% | 0.13% |
| PEMS-BAY | 0.040 | 0.00% | 0.01% |
| PEMS03 | 0.509 | 2.22% | 7.23% |
| PEMS04 | 0.426 | 13.16% | 5.85% |
| PEMS08 | 0.410 | 9.52% | 3.50% |

CV=各K节点训练时间均值的标准差/均值，衡量节点间幅值差异；邻居按已给定图任一方向的一跳连接、不计自环；低值定义为正有效目标低于该数据集train-K均值的10%，仅作跨量纲描述，不是修改MAPE口径。表中统计单位是时间点/节点，不是论文33格的重放加权指标。

这提供了比“flow没有KD-tree”更直接的解释线索：flow节点的典型流量差异显著更大，且部分V节点缺少一跳已观测邻居。无自身历史时，模型必须从邻域及上下文推断节点流量水平；speed的节点均值相对集中，这一步通常更容易。PEMS08的validation-V总体均值仅为train-K的0.877倍，亦有幅值迁移，但时间和节点集合同时改变，不能单独归因于其中一个。

这些是数据差异和机制假设，不是证明BFS或Focal造成全部差距。PEMS04的test Unseen已优于剩余baseline，而PEMS08的validation/test出现反转；说明泛化难度与样本划分也需考虑。当前材料不能证明泛化差来自某一处代码错误。

## 对用户所列原因的判定

1. **Focal尺度：确认存在，优先做控制。** 原损失`z*tanh(0.1*sigma*z)`。z=0.1时，flow权重约0.89–0.91，BAY约0.09；flow更接近MAE。新候选改`z*tanh(z)`。这更强调较大标准化残差，是否有利Unseen/RMSE、是否损害低流量MAPE，必须看训练结果。
2. **算子重复：确认矩阵重复，不等于专家相同。** flow旧支持诊断三矩阵距离为0，但每个专家具有独立Theta，重复不能直接推出预测无效或删专家必然有利。保留原算子，不复用失败SUP。
3. **MAPE低值：有统计支持，但不能解释全部。** flow低值比重大，Unseen MAE/RMSE也有缺口，所以不能靠改阈值解决。
4. **BFS/KD-tree：相关性不是因果。** 邻接一跳覆盖较弱可能增加无历史难度，KD-tree换分区不会凭空增加真实观测或图边；不为分块伪造GPS坐标。
5. **dense routing：属实，但非flow独有问题。** soft模式全专家计算，改top-k参数不提供稀疏加速。当前修复关注预测质量，不额外引入训练/推理路由差异。

## 执行及保留条件

按PLAN.md启动三个flow各三种修复：unit Focal；unit Focal+beta0；unit Focal+refiner scale0。旧模型作复用对照。输入、指标、数据划分、scaler、checkpoint准则、种子、超参数与baseline均不变；没有训练speed或baseline。前两种是损失配方变化，第三种额外关闭原有输出残差，不替换主干。

新方案是候选，不预先替换最终模型。先按validation33格/三项LifeAvg及各条件取舍；若出现明确改善再冻结后续评估身份。单seed无法排除随机波动；原test已被观察，这一限制继续保留。

## 数据集最终建议

本轮保留五数据集，不启动新数据集和baseline重训。两套speed在Graph WaveNet（IJCAI2019）、DCRNN（ICLR2018）有先例，但论文是否充分还取决于研究问题、协议、机制验证与结论范围。TrendGCN（CIKM2023）使用四flow+两speed，说明混合类型同样合理。链接见PLAN.md。

更新的例子：[Trafformer（AAAI2023）](https://ojs.aaai.org/index.php/AAAI/article/download/25980/25752)亦以METR-LA与PEMS-BAY两套speed开展实验；[MoGERNN（TRC2025）出版社摘要](https://www.sciencedirect.com/science/article/abs/pii/S0968090X25000841)明确报告两套真实数据集。因此“两套一定不够”不成立，但这些先例不能保证本研究只用两套时已充分验证所有泛化主张。

如果本轮flow仍不理想，建议将结论明确限定为speed上的主要优势，并把已有flow作为正文简洁说明/附录完整展示的泛化边界；不隐去已有反证，不只因结果差换数据集。真实传感器经纬度和车辆GPS轨迹不是同一类数据，当前没有必要把任务改成GPS预测。

## 验证记录

- 4个新增测试通过：单位变化不改变新目标及有限梯度；真实compute_losses只监督合法目标、适配上下文恢复原函数；scale0在非零refiner参数下仍恒等且梯度正确；无效scaler拒绝。
- 9项YAML对原锚点逐字段比较：仅运行身份、loss，以及B的beta或C的refiner scale变化。
- 01/02各1485格重新读取并验证排名；各1320个baseline数值保持原精度（旧Excel导出与JSON序列化差异最大5.7e-14）；只更新PEMS04的33格。03/04的CSV字节完全相同，单变量布局和所有数值不变，仅Excel最终配置标记同步。
- 原始结果、权重、共享模型/训练源码未改；旧test计划哈希检查通过。新9项GPU预检/实际训练状态以队列证据为准，不把20-batch预检当训练完成。

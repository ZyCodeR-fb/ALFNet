# 原因诊断与文献决策

## 可核验事实

`cohort_diagnostics.json`分离时间漂移（train K→validation K）和同时间节点组差异（validation K→V），仅汇总train K及validation K/V；test只检查公开节点索引互斥，不读取其目标做统计。

| 数据集 | K/V/T节点数 | 时间均值比 | V/K同期均值比 |
|---|---:|---:|---:|
| PEMS03 | 268/45/45 | .9919 | .9948 |
| PEMS04 | 230/38/39 | 1.0656 | .9597 |
| PEMS08 | 128/21/21 | 1.0195 | .8605 |

08的验证Unseen组均值低13.95%，明显大于04的4.03%。这支持节点组尺度差异假设，不构成对test节点的分布结论。03均值相近仍困难，说明仅整体均值不足以解释长尾误差。三组索引互斥，未发现据此可认定的划分bug，也不为改善结果更换划分。

既有`flow_unified_20261003/residual_bins.json`中08原版Unseen真实0–10区间均值5.406、预测96.930；03约5.30/21.03，04约6.21/38.11。08低值≤30仅占4.64%位置却贡献49.14%微平均APE。这是低流量系统性高估，而不只是MAPE分母小。微平均诊断不等同于协议33格统计。08验证无K邻居比例9.52%，04反而13.16%；不支持简单“没有KD-tree导致08更差”。train节点均值CV03/.509、04/.426、08/.410，08也不是CV最大者。

08半中心化已改善Join和多数阶段，但test Unseen三项仍输UIGNN；先前纯空间均值中心化未解决，邻居均值不能保证目标节点尺度正确。原单位Focal和重复SUP是合理机制疑点，但unit-Focal、degree-SUP、relative20已有失败证据，不重复原配方。

两数据集LR分别1/2、dropout .1/weight decay1e-4相同。此前结果支持数据集内机制比较，却不能排除跨数据集最优机制差异来自优化器设置。先补齐因子矩阵比同时新增几个模块更有判别力。现阶段不宣称学习率能够根治Unseen问题。

## 检索与适用边界

使用Academic Research Suite本地研究设计流程、Sciverse语义检索和论文原始页面；下列为针对性叙述检索，非穷尽系统综述。Sciverse命中FUNS等节点异质性工作，也返回若干不相关交通论文，不据此认定最新或最优。

| 原始来源 | 可借鉴信息 | 对本轮决策的作用与边界 |
|---|---|---|
| [DIR, ICML2021](https://proceedings.mlr.press/v139/yang21m.html) | 连续标签不均衡及分布平滑 | 支持按流量分箱诊断；不等于低值越重越好，不直接改变本轮损失。摘要/会议原文页面核验。 |
| [Balanced MSE, CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html) | 从标签分布校正回归目标 | 来源任务是视觉回归；不能推定对零历史交通泛化有效。保留备选，先消除LR混淆。 |
| [Group DRO](https://arxiv.org/abs/1911.08731) | 平均好不代表少数组泛化好；正则化重要 | 支持分组审查与匹配优化条件。其分类结果不能直接当作本任务收益保证。摘要核验。 |
| [RevIN, ICLR2022](https://openreview.net/pdf?id=cGDAkQo1C0p) | 局部归一化与还原减轻分布漂移 | q=0没有自身统计量，不能使用Unseen隐藏真值求均值；已有中心化只取可见输入。 |
| [LTSF-Linear/NLinear](https://arxiv.org/abs/2205.13504) | 简单时间建模及归一化对照的重要性 | 不据此替换ALFNet；比较已有中心化强度与LR。摘要与既有作者实现核验。 |
| [GraphNorm, ICML2021](https://proceedings.mlr.press/v139/cai21e.html) | 图级归一化与优化行为相关 | 说明归一化和训练条件可能交互；其图分类结果不能保证节点预测收益。 |
| [GRIN, ICLR2022作者实现](https://github.com/Graph-Machine-Learning-Group/grin) | 通过关系归纳偏置重建缺失值 | 重建与全无历史预测不是同一任务；不能把低值预测偏差全归因填补器。 |
| [GinAR, KDD2024论文](https://arxiv.org/abs/2405.11333) | 插值注意力与自适应图处理变量缺失 | 支持保留不完整观测基线及现有插值对照，不引入其完整架构。 |
| [Robust Traffic Forecasting against Spatial Shift over Years](https://arxiv.org/abs/2410.00373) | 显式研究空间关系漂移 | 摘要级启发，跨年设定不同；不借其结果推断本项目原因。 |
| [On the Role of RevIN, 2026预印本](https://arxiv.org/abs/2603.11869) | 重新审视归一化各组件的作用 | 仅作为最新反证线索，不把所有中心化默认为有益，不作为已核验顶会结论。 |

结论：当前最有依据的是“跨节点尺度差异＋少量低值的大偏差＋机制/LR交互待分离”。本轮实施4个新的匹配LR配置，未实施新论文模块。不能保证80%；若失败保留记录，不通过删强基线或改MAPE掩盖。

## 基线选择

`baseline_pressure.json`按当前04bridge_delta1、08halfcenter和已有test逐一列出每个模型胜过ALFNet的格，允许重叠。该统计只定位困难条件，不能作为删除基线的理由。保留8基线、相同指标与数据，不重训。后续只有任务适配、来源实现或方法代表性理由才能支持替换。

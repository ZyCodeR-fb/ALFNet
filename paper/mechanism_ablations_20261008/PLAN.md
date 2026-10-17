# 条件机制补充消融与flow诊断

2026-10-08：四数据集，每个4项，共16项；固定既有Full配置/seed32/协议，最大5并发。当前唯一OOM补跑成功结束后再开始GPU预检和训练，避免叠加争抢显存。仅validation，不打开test，不更换模型。此前24项结果全部保留，本轮全部16项也保留，不将“四数据集三项LifeAvg全部变差”作为结果收录门槛。本轮是观察已有结果后的探索性机制实验，不是事前独立确认。

## 问题与证据

已核验：移除Graph Refinement后，04三项LifeAvg变化约-0.93/-1.02/-1.19%，08约-4.26/-3.59/-0.64%；08移除Age约-1.95/-0.74/-0.77%。均为配对seed32，不构成统计显著性结论。

代码实际为 prediction + 0.05 × role_scale × tanh(MLP([prediction,forward_neighbors,backward_neighbors]))，不是直接用均值替换预测。输入来自图解码及中心化还原后的预测。流量训练节点平均值CV04/08约.426/.410；较强空间水平异质性可能使邻居预测不适合直接用于本地输出修正。归一化空间的最大修正0.05乘训练std后，在flow原单位约7.4–7.7（role_scale=1时），在speed约0.45–0.65；这是幅度上限，不是实测修正幅度，也不自动证明有害。与插值/中心化、Focal目标和有限样本优化的交互均是待验证假设，不能把所有原因归结为KD-tree或数据类型。

Age在已有角色、观测掩码、近期历史/中心化中可能存在冗余；模型有能力利用它，不代表有限训练一定获益。旧noAge将年龄改为mask，并没有删除角色或观测状态，不足以否定整个条件化设计。最终不同epoch的重训差异也不能直接解释为推理时单模块偏差。

## 本轮预先固定对照

| 对照 | 改动及保持项 | 回答的问题 |
|---|---|---|
| w/o Expert Lifecycle Context | 共享重建器输入的四角色及年龄通道替换为mask，保留raw/mask；原始角色仍用于refiner和监督 | Age单独冗余是否掩盖联合生命周期信息的作用？不是旧Mask-only：不更改refiner角色尺度 |
| w/o FiLM Multiplicative Modulation | gamma置0，保留beta及时间分支 | 乘性条件调制有无独立收益？ |
| w/o FiLM Additive Modulation | beta置0，保留gamma及时间分支 | 加性条件调制有无独立收益？ |
| Pointwise Refinement | 仅refiner接收单位邻接；保留全部线性层、参数初值、角色尺度，其他图运算不变 | 有害来源更接近邻居混合还是输出修正整体？ |

四项均保留参数维度/初始化（FiLM被屏蔽输出的权重不再影响预测），不刻意构造不可用输入或破坏真值。对应源代码仅在独立study adapter挂钩，旧模型源码不修改。收录全部结果，报告33格与三项LifeAvg，支持/不支持假设分别记录。

## 文献检索与使用范围

以下为本轮定向检索的主要原始来源，借鉴实验拆分原则而非照搬其结论：

1. GRIN，ICLR2022：[论文全文](https://arxiv.org/html/2108.00298)，区分历史重建架构和图结构作用；不把重建收益等同预测收益。
2. FiLM，AAAI2018：[论文全文](https://arxiv.org/html/1709.07871)，条件仿射由乘性与加性两部分组成，据此拆分本轮gamma/beta干预；原任务并非交通预测。
3. Graph WaveNet，IJCAI2019：[官方全文](https://www.ijcai.org/proceedings/2019/0264.pdf)，讨论预定义连接不等同实际依赖；支持将邻居信息与变换模块分开对照，不证明本项目图细化失败的原因。
4. AGCRN，NeurIPS2020：[官方全文](https://proceedings.nips.cc/paper_files/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf)，节点自适应参数与自适应图结构是可分离因素。
5. RevIN，ICLR2022：[作者实现和论文入口](https://github.com/ts-kim/RevIN)，强调统计水平与归一化/还原；本模型只有部分中心化，不等同完整RevIN。OpenReview全文本轮遇验证页，未声称阅读全文。
6. GinAR，KDD2024：[作者全文](https://finleywang.github.io/files/2024_KDD_GinAR.pdf)，插值注意力与自适应图卷积分别负责缺失与依赖建模；不把其插值注意力等同本项目线性可见历史插值。
7. DLinear/NLinear，AAAI2023：[官方论文](https://ojs.aaai.org/index.php/AAAI/article/view/26317)，提供简单结构与复杂结构比较的背景；不能由此推导删模块必然有益。

检索中发现另一篇2026同名PatchSTG结果，未将其误认成本项目2025原方法；本轮不据此添加重复空间上下文消融。既有局部上下文和整条Temporal–FiLM删除已经覆盖较粗层次，本轮细化条件作用和图细化来源即可。

## 执行与复用

四个Full直接复用；四个新干预与既有六种干预均不相同。独立配置哈希与当前source/data身份，CPU保留参数初值检查，16项GPU20train/4val预检后正式运行。失败保留记录，不静默更改batch/配置。全部完成输出20配置660validation格。图细化假设若不被支持，也保留Pointwise结果。

# 异常flow误差的证据与文献筛选

## 实测诊断

2026-10-02使用当前原版选定权重，只重放validation；3个数据集canonical33与原报告最大差均为0。脚本`diagnose_flow_residual_bins.py`记录真实目标、最终预测和填补末值三个量。填补末值是解码初始输入；与未来目标的偏差同时包含预测难度，**不是对同一历史时点真值的纯填补误差**。不能据此断言填补器单独造成全部问题。

| validation无历史，真实目标0–10 | 数量（预测位置） | 真实均值 | 预测均值 | 填补末值均值 | 分箱MAPE |
|---|---:|---:|---:|---:|---:|
|PEMS03|13569|5.30|21.03|34.25|426.89%|
|PEMS04|4412|6.21|38.11|52.66|1038.98%|
|PEMS08|1336|5.41|96.93|136.59|2577.91%|

反方向同样存在：目标>300时，预测平均偏差03为−51.71、04为−80.01、08为−60.50。这符合向中间流量收缩的现象，与既有train-K节点均值CV .41–.51、无历史节点需要推断幅值的困难相容。低值误差不是单纯“MAPE数字不好看”：PEMS08低值区间MAE也达91.52。部分03/04负预测存在，但08这一低值区间无负预测，单纯截断负数无法解决。

统计为不分calendar/k权重的micro诊断，不替代论文canonical33；不将分箱MAPE拿去与主表MAPE直接对比。数据、零值规则、cohort、horizon未修改。原始汇总见residual_bins.json。

## 检索方法与证据层级

使用Academic Research Suite本地research-design流程，Sciverse两组语义检索（无历史flow的幅值异质性；图均值聚合/度数与过平滑），再以论文原站、作者仓库核对机制。以下为有针对性的9项筛选，不声称系统综述或查尽全部论文。正式发表与预印本分开，不因标题相近就采用；也不把文献在其他任务上的提升当作ALFNet收益保证。

| 文献/来源 | 核验内容和本问题关系 | 本轮决定 |
|---|---|---|
|[GRIN，ICLR2022作者仓库](https://github.com/Graph-Machine-Learning-Group/grin)，[论文摘要](https://arxiv.org/abs/2108.00298)|已有组件来自这一图重建方法。当前ALFNet主要通过forecast梯度训练填补输出，代码没有直接隐藏历史重建loss。论文支持以重建任务学习图时序表示；OpenReview全文访问遇到验证页，未冒称已读该入口全文。|候选A只新增训练K隐藏历史MAE辅助目标，不替换GRIN，不宣称完全复现其所有训练目标。|
|[PNA，NeurIPS2020原文](https://proceedings.neurips.cc/paper_files/paper/2020/file/99cad265a1768cc2dd013f0e740300ae-Paper.pdf)|全文机制：多种聚合配合度数缩放，单一mean/max有区分能力局限。支持给重复均值槽补入degree信息，但未证明交通flow一定获益。|候选B采用有界度数放大/衰减，保留原5个权重槽；是PNA启发的受控变体，不是完整PNA。|
|[GinAR，KDD2024论文](https://arxiv.org/abs/2405.11333)，[作者代码](https://github.com/GestaltCogTeam/GinAR)|摘要/官方代码介绍插值注意力和自适应图卷积，强调缺失变量恢复与预测联合建模。|支持先检查重建路径，而非再大换主干；不借用其baseline权重或重训baseline。|
|[RevIN，ICLR2022作者代码](https://github.com/ts-kim/RevIN)|实例归一化/反归一化利用输入序列均值与标准差缓解分布迁移。|Unseen没有自身历史，直接按节点RevIN无合法统计；不使用验证/test节点长期均值。暂不采用。|
|[ST-Norm，KDD2021](https://doi.org/10.1145/3447548.3467330)，[作者代码](https://github.com/JLDeng/ST-Norm)|出版社摘要/代码介绍空间与时间归一化，用于分离局部/高频成分。|不可据此推断全局loss标准化就是正确方向；本轮不进一步改变观测掩码上的归一化。|
|[Non-stationary Transformers，NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html)|摘要明确讨论过度平稳化可能损失序列差异，恢复非平稳信息的重要性。|作为“归一化不必然更好”的反证线索，符合上一轮flow失败；不直接套用其Transformer模块。|
|[Balanced MSE，CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html)|从标签分布不平衡角度改造回归目标；视觉回归设置与本任务不等价。|不用均匀标签分布假设或跨样本密度估计。候选D仅检验明确低值高估的受限相对项，不能称Balanced MSE复现。|
|[DenseWeight/DenseLoss，Machine Learning2021](https://link.springer.com/article/10.1007/s10994-021-06023-5)|目标密度用于回归损失加权，说明改变训练重视程度是一条可检验路线。|不引入额外密度拟合/验证调权；保留原Focal并预先固定相对项系数。|
|[HINT，2025预印本](https://arxiv.org/abs/2512.17984)|Sciverse段落及作者摘要明确指出相邻主路/匝道flow尺度异质，方法同时利用全网speed、OSM和仿真等信息。不是本次已核验的顶会正式发表。|与本诊断方向相关，但输入条件更强；拒绝直接套其节点校准或额外输入，以免破坏当前比较信息边界。|

## 为什么选这四项而不是继续无目标搜索

A针对填补路径，B针对重复均值传播的尺度表达，C测试A+B交互，D针对低流量相对梯度。A/B/C保持原Focal；D不删结构。全部从同一原版开始，不沿用上一轮已经失败的unit-Focal结果。重建监督的标签范围、degree算子变化和loss差异均进入方法身份。

本轮**没有**找到一个已证实能修复所有flow的现成方案。可复核的是误差模式与候选机制的对应关系；是否保留由完整validation33格及LifeAvg决定。不能因低值区间MAPE改善就忽略高流量RMSE、Fault或早期阶段。

若最后一轮无效，建议将模型主要优势限定为speed、完整保留flow的泛化边界。仅因结果不利删除3个数据集会使证据选择依赖结果。换新speed数据集也不能免除重训全部baseline的成本，本轮不启动替换。

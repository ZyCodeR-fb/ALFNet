# 统一机制的证据与文献

## 为什么当前优选不同

优选来自两个不同单种子数据分布与覆盖/平均误差权衡，不是两个数据集理论上必须使用不同模型。04原版LR1，08LR2，节点数、训练分布和held-out节点不同，尚无控制实验可把差异唯一归因于某一项。04bridge_delta1 test29格，08只有20格；08bridge_center25格，但LifeAvg MAPE落后UIGNN。统一算法、数据集级validation超参数，是首选论文表述；若最后分speed/flow配方，也应同结构并透明披露，不改名掩盖模块差异。

## MAPE并非所有场景都变差

根据保存test表（无新test访问），08原版Unseen/Steady/Join/Fault MAPE=145.382/12.107/35.710/15.886；bridge_delta1=171.473/13.015/30.975/14.904；bridge_center=161.753/14.422/26.599/18.331。桥接降低Join MAPE，但Unseen恶化抵消了收益。bridge_center的Unseen增加16.371点，对等权LifeAvg贡献约+4.093点；Join改善约-2.278点。不能因此否认接入收益，也不能只展示Join。

04bridge_delta1四场景MAPE均下降，说明同公式在不同节点/时段分布下产生不同误差取舍。结果不能证明数据或KD-tree有bug。历史方案unitFocal/relative20/SUP度缩放/重建训练并未稳定解决test泛化，故本轮不重复损失加权，而检验多节点残差参考与增量幅度。

## 补充validation分箱

scripts/diagnose_unified_flow_bins.py重放原版04/08以及04bridge_delta1、08bridge_delta1、08bridge_center，共五权重；完整canonical33必须复现才保存分箱。只访问validation；micro目标区间分箱不是正式33格加权指标。GRIN最后填补值与未来truth对比也不是严格同时间重建误差。

首次脚本未正确进入旧候选adapter，原版两项delta0，但第一候选delta69.271699触发断言，未保存该候选结果、没有访问test。修复入口后复用已验证原版，只重放三候选。该失败属于诊断脚本重建错误，不是训练权重或正式表的问题；最终有效记录仅以每项delta0为准。

最终五项全部canonical33复现delta0。08validation无历史真实0–10区间均值5.406，原预测均值96.930，bridge_delta1为90.348，bridge_center为75.279；三个都严重高估，但该小区间在validation反而改善，不能把旧test MAPE退化直接说成已验证由该区间造成。bridge_delta1在10–30/30–100/100–300三个区间微观MAPE分别由460.9/135.6/36.6恶化为516.4/153.4/43.4；bridge_center高流量>300平均低估由-60.5扩大至-94.1。故存在更广的尺度和泛化取舍，不只是0附近分母问题。分箱未重新打开test，test具体值区间归因仍未证实。

## 定向文献检索

Academic Research Suite本地设计流程；Sciverse检索GraphNorm相关原文片段，再用会议/作者一手来源核对。延续上一轮十篇相关工作，本轮重点补查下列机制，非穷尽式综述，也无最优保证。

|来源|核验范围|启发与边界|
|---|---|---|
|[GraphNorm, ICML2021](https://proceedings.mlr.press/v139/cai21e.html)|会议摘要、Sciverse原文片段|过度移除均值可能损伤表达，作者用可学习平移；本轮.5平移仅做无新参数控制，不是完整GraphNorm。|
|[Non-stationary Transformers, NeurIPS2022](https://papers.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html)|会议摘要及作者摘要|归一化可能丢失有用差异；本轮不移植attention或宣称同一结论已在ALFNet验证。|
|[GRU-D, Scientific Reports2018](https://doi.org/10.1038/S41598-018-24271-9)|出版社摘要和机制段落|掩码/间隔可条件化动态响应；本轮qdelta是确定性观测比例规则，不是GRU-D可学习衰减，也不用全数据均值。|
|[GRIN, ICLR2022](https://github.com/Graph-Machine-Learning-Group/grin)|作者仓库说明/前轮已核验|保留图重建处理无历史节点，不能把有观测节点的插值推广为无历史的真实数据。|
|[NLinear, AAAI2023](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)|前轮作者源码逐行核验|减last再加回；本轮检查多节点不同支持下的参考一致性，保留原ALFNet而非换成线性预测器。|
|[RevIN, ICLR2022](https://github.com/ts-kim/RevIN)|前轮作者说明|可逆变换的范式；q0不能假造其自身均值和方差。|
|[DCRNN, ICLR2018](https://arxiv.org/abs/1707.01926)|作者摘要|递归预测背景；倍率1或qdelta来自本项目假设，不是作者推荐常数。|

不引入节点专属test统计、GPS/OSM或隐藏标签；不同候选是统一模型族的独立对照，不是按条件选用已训练模型的测试集后处理。最终优先选择共同机制，再验证speed，不把四个数据集各自的test最优模块拼成一个未实现模型。

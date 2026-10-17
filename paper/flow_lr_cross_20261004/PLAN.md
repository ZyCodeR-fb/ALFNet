# PEMS04/08 机制与学习率交叉确认

日期：2026-10-04。当前用户授权选择优化方向、尝试多个方案，最多5个训练。选择方向2，不根据竞争强弱删除基线。

只新增4项，复用4项；机制为插值＋倍率（bridge_delta1）与插值＋倍率＋半中心化（triple_halfcenter）。每个数据集、每种机制均比较学习率×1和×2。

| 数据集 | 新训练 | 复用 |
|---|---|---|
| PEMS04 | bridge_delta1 LR2；triple_halfcenter LR2 | 同机制LR1 |
| PEMS08 | bridge_delta1 LR1；triple_halfcenter LR1 | 同机制LR2 |

容量分别10/6（不是分区数）；两者32分区，dropout .1、weight decay 1e-4、seed32、batch64、max_epoch100、patience10、early start40、teacher forcing40完全不变。统一机制不要求不同图使用相同padding容量。三组优化器学习率同时成比例修改，参数初始化与对应数据集锚点逐项校验。

不新增模型模块、损失、监督、数据划分、mask或指标；沿用已有隔离适配器，源代码与旧权重保持原样。模型仍以完整validation LifeAvg MAE选择checkpoint。最终分析同时报告33格、三项LifeAvg和Unseen，不引入硬性退化门槛，不以删基线提升胜格。

验收：先4项GPU预检，每项20训练batch与4验证batch；校验权重身份、学习率、dropout、分区、协议、无跳过batch。随后4项正式训练并发，不超过5。完成自动生成8项264格validation表，保留所有对照。无新test、无speed/基线重训，无自动追加搜索。

主要判别：相同LR时04/08是否仍偏好不同中心化强度；降低08LR能否改善Unseen而保留Join收益。当前研究已受历史test反馈启发，后续结果不能称为全新独立测试确认。单seed不能证明统计显著性。本轮先隔离优化器混淆，若仍无改善，再根据逐节点validation残差判断是否值得修改跨节点尺度估计。

# 五数据集全量test（2026-10-01授权）

范围：seed32，67个ALFNet已完成超参数配置＋五数据集×十个现有基线=117个checkpoint。只评估best.pt，不重训、不改变模型/损失/协议。允许按用户要求探索test上的最优配置，但将validation选型与test探索选型分开保存，后者不能作为未参与选择的独立test证据。

训练完成：最后12项exit0；新候选未普遍叠加收益。Test打开前选定validation配置：METRLA_lr_double；05_PEMSBAY_combo2；16_PEMS03_lr_half；PEMS04_lr_half；03_PEMS08_combo1。

- METRLA维持容量8/LR2：25格第一，新容量12/LR2仅24格，虽然RMSE略低。
- BAY选择容量16/LR2/WD1e-5：仍18格第一，MAE相对原选定增加约0.14%，RMSE/MAPE略有改善。
- 03、04新组合未改善当前折中，维持原选定。
- 08选择容量6/LR.5/drop.05/WD1e-5：13格第一（原9）；相对原选定MAE/RMSE略升、MAPE略降。容量6/LR.5/drop.05/WD1e-3为平均误差更好备选（8格第一）。两者均完整保留，不拼接单元格。

检查当前源代码与配置/权重hash；既有test仅在checkpoint及源模型/数据代码一致时复用。其余先验证checkpoint/scaler/曝光/seed，完整validation重现（1e-5绝对＋相对容差）后test。ST-FiT原Gumbel评估有随机性，单独固定种子重放并披露与历史validation差异；不把重放相同称为历史逐值重现。

test使用原数据划分、checkpoint训练scaler、无teacher forcing、共同33格收集器。所有预测必须有限，每数据集跨模型/配置校验test曝光、scaler与各格计数。最多5任务，显存门槛；保存逐项日志与结果。失败隔离，已完成结果不覆盖；无自动新增训练或追加搜索。

输出两套55记录表：validation_selected_test和exploratory_test_selected；另输出所有67项超参数test主表/阶段表，四轴固定参考点及组合分组。第一名加粗、第二名下划线。现有旧表保留。

探索性test选型规则（打开新test前记录）：若存在三项LifeAvg均优于各指标最强基线的候选，先在其间按33格第一数最大，平局按三项相对最强基线比值均值最小，再MAE最小；若没有，先取LifeAvg Pareto集合再相同排序。完整公布全部候选，规则只给出覆盖优先的可复现探索选择，不声称全局最优或统计显著性。模型优缺点、MoGERNN patience3和SLPF/STBP适配等既有差异继续披露。

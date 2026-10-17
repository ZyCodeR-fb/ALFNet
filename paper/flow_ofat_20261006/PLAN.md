# PEMS04 / PEMS08 单因素超参数补实验

用户于2026-10-06授权：只补两个flow数据集，固定锚点可不为最优配置，优先复用严格等价历史结果，并发最多5。METR-LA/PEMS-BAY已有固定锚点实验视为完成，不重跑。不包含消融、新test、基线训练或最终模型再次选型。

本轮选定最终方案作为锚点，避免旧输入处理的敏感性被误写成新方案效果。PEMS04：容量10，插值开启，解码残差1，中心化0；PEMS08：容量6，插值开启，解码残差1，中心化0.5。两者LR倍率1、dropout0.1、weight decay1e-4。其余协议不变：seed32，batch64，max_epoch100，patience10，early stopping start40，teacher forcing40；checkpoint选完整validation LifeAvg MAE最小值。

| 单因素 | PEMS04 | PEMS08 |
|---|---|---|
| 容量 | 10/14/18 | 6/8/10 |
| LR倍率 | 0.5/1/2 | 0.5/1/2 |
| dropout | 0.05/0.1/0.2 | 0.05/0.1/0.2 |
| weight decay | 1e-5/1e-4/1e-3 | 1e-5/1e-4/1e-3 |

每个数据集共9个唯一配置，四条曲线共享锚点。复用2个锚点和2个LR×2；补14项。容量为每分区token容量，分区数量32不变；LR同步缩放三个优化器组；dropout沿用既有实验定义，仅改变ALFTemporal的crossformer_dropout，非所有模块的全局dropout；weight decay不是角色损失权重。

历史旧方案容量/LR/dropout/WD结果单独保留，不与本轮混成单因素曲线。新计划对展开后的全部嵌套配置作语义比较，仅忽略运行名称与文件路径；复用核验checkpoint哈希、报告、最佳epoch、零跳过批次。旧flow LR计划代码哈希检查作为当前实现等价依据。所有14项还需CPU身份与参数初值核验、GPU20训练批次+4验证批次预检，全通过后自动入正式队列。显存门槛与错峰启动可使实际并发低于5，避免影响其他用户任务。

结果自动保存完整validation主表/阶段表，18配置×33格=594格；新结果不自动覆盖论文最终表，不以test筛选参数。预检、失败记录和各run权重保留。独立计划路径configs/paper/flow_ofat_20261006/plan.json；启动器scripts/run_flow_ofat_20261006.py。

# 四数据集最终配置配对消融

2026-10-06用户明确更正为四数据集、六项消融，排除PEMS03，等待当前flow超参数完成后开始，最多5个训练；不触碰其他用户PID2546461。本轮validation-only，无自动test，无超参数重选。此前五数据集计划未启动，已保留为plan_superseded_five_datasets.json，不能用于运行。

METR-LA/PEMS-BAY/PEMS04/PEMS08使用paper/final_scheme_20261006/selected_configuration_manifest.json的冻结配置。四项Full权重及结果复用，不重训。PEMS03不运行。

| 名称 | 实际干预 |
|---|---|
| w/o State Age | 年龄通道改为可见掩码；保留角色、监督和教师信号 |
| w/o Adaptive Routing | 保留传播算子，改用全局可学习静态混合 |
| w/o Spatial Context | ALFTemporal逐节点处理，取消跨节点上下文，保留时间表示和FiLM |
| w/o ALFRecon | 删除共享学习式历史重建，缺失填训练均值（标准化0）；flow的可见历史插值及中心化仍保留 |
| w/o ALFTemporal–FiLM | 删除整条时间表示及FiLM调制分支 |
| w/o Graph Refinement | 删除输出图细化 |

每个数据集固定其Full容量/LR/dropout/WD/损失/协议，seed32、max100、patience10、start40、TF40不变，checkpoint按完整validation LifeAvg MAE最小值选。保留不利结果，不为消融单独调参。

复用检查扫描历史study_ablation配置并展开嵌套专家配置比较；旧METRLA容量8与当前7不等价。24项新训练+4项Full复用，主表和阶段表共924指标格。训练前逐项检查相同曝光/scaler/种子及保留参数初值、优化器LR，执行20训练批次和4验证批次预检；全部通过后正式运行。flow适配器与消融适配器按顺序组合，不改生产模型源码。

队列先检查当前flow_ofat_20261006同一plan全部14项成功及汇总COMPLETE；失败则停止接续而不误启动。等待期间不占GPU。通过后先预检再最多5并发，显存门槛自动保护其他用户任务。完成导出validation_results.xlsx、validation_cells.csv、results.json。新独立计划保存config/code/data/checkpoint哈希，旧冻结运行文件不修改。

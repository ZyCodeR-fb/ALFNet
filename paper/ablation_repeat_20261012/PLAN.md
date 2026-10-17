# 两项flow消融原配置重跑

用户于2026-10-12明确授权：PEMS04 w/o State Age、PEMS08 w/o Adaptive Routing各重训一次。保留seed32、原数据划分、完整超参数、消融定义、100轮上限/patience10/start40和validation LifeAvg MAE权重选择准则。从头训练，不加载旧best权重续训；独立计划及输出目录保留全部旧结果。

仅2项训练，最多2项并行。原始配置直接引用final_ablations_20261006中的02_PEMS04_no_age和07_PEMS08_static_router。对实际模型/训练代码和嵌套配置逐项核验与原运行摘要一致；新计划保存当前依赖摘要。每项先完成继承的配对初始参数/数据身份检查以及20训练batch、4验证batch预检，再开始正式训练。

本轮是同seed重复性检查，不保证结果变化，更不保证消融误差升高。停止规则为两项各完成一次，不依据指标继续重跑。所有新旧结果并列，不能按误差升降筛选或替换原论文结果。Full复用对应seed32现有记录，不重训其他模型或数据集。没有新的test授权，不自动读取test。

脚本：scripts/repeat_flow_ablations_20261012.py。完成后自动输出RESULTS.md、results.json和validation_cells.csv，包含Full、原消融、新消融的主表及阶段表。错误保留并显式报告，不静默追加训练。

# PEMS04/08观测历史迁移实验（2026-10-03）

用户重新授权：正文四数据集METR-LA/PEMS-BAY/PEMS04/PEMS08；继续分析及修复ALFNet，先跑04/08，有效再迁移speed，最多5并发。此前“最后一轮结束”已被本次请求取代。基线不重训。此计划仅10项flow训练，无新test、speed训练或数据集删除。

## 假设与矩阵

原版04test丢11格集中Join/Cold/Warm/Ready；08丢12格集中Unseen/Join/Cold/Warm。两轮针对量纲、角色权重、精修、重建、SUP、相对误差的候选没有在test稳定改善覆盖。新方向优先尝试改善少量观测的利用，而非再次改变指标权重。

每个数据集五种：bridge、center、delta1、bridge_delta1、bridge_center，共10项；原版两个直接复用。bridge在当前12步输入内按同一节点的可见值线性插值、两端最近值延拓，仅替换有至少一个观测的缺失位置，无历史节点保持GRIN。center在GRIN之后仅将数值通道减去最后可见值，在图预测后、refiner前加回；解码teacher同样中心化但不改变teacher mask。无历史节点中心为0，不估计不存在的均值/方差。delta1仅将解码增量系数从.3改1；系数并非硬性能力上限，因为投影可学习，本实验检验优化偏置。

中心化/插值均发生在专家之前，不是旧history_linear_correction预测校正头；不是此前reconstruction额外标签，也不替换SUP，不重复unitFocal/relative20。所有学习参数与初始化相同。桥接可能损失局部趋势、中心化可能丢失有用绝对尺度、delta1可能放大累积误差；通过独立控制及两组组合区分。

## 固定协议

Seed32；04C10/LR1，08C6/LR2；dropout.1/WD1e-4；batch64；max_epoch100/patience10/early_start40；teacherforcing原40epoch策略；32分区不变。Loss原rawFocal与角色混合不变，无辅助损失。容量为每分区token容量，不是分区数。输入输出12/12、掩码、自然缺失、scaler、分割、K/V/T暴露、指标均不变。只使用当前可见历史，无全数据节点统计、未来目标、隐藏历史标签或GPS新增信息。

## 验证及决策

每候选20训练+4验证batch预检；同参数哈希、信号暴露、scaler、优化器、布局审计；正式运行完整33validation，零跳过批次，checkpoint仍以LifeAvg MAE选择。最多5并发，显存门禁，失败保存且不自动无限重试。

目标是每个数据集至少27/33第一且三项LifeAvg优于基线；这是成功目标而非擅加退化硬门槛，未达时仍完整报告Pareto取舍。若某共同方案在04/08有稳定覆盖及平均改善，再为已授权speed迁移生成独立身份，不直接沿用flow配置、不自动对每个候选跑speed。统计是单种子探索，不能保证最终达到目标。既有test曾用于诊断，保留披露；新test需要单独固定评估范围，训练队列不自动打开。正式表与权重不覆盖。

执行代码：experiments/flow_support_transfer.py、scripts/run_flow_support_transfer.py。候选checkpoint的support_transfer方法合同是未来评估重建的必要条件。没有该adapter不能作为同一模型评估。

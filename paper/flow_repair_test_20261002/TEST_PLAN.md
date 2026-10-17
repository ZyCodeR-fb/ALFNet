# PEMS08修复候选的限定test — 2026-10-02

授权：用户本轮“训练已结束，请分析并整理结果，是否有方案满足要求，可以的话跑一下test”。仅新评估下列两个PEMS08 checkpoint，其他flow候选不打开test，不重训模型。

9项均已完整结束；没有修复达到27/33第一。PEMS03/04的覆盖下降，平均误差大部分变差，保留原版。PEMS08两项相对当前原版有一致的LifeAvg改善，因覆盖与平均误差取舍不同，一同固定进入test：

| run | validation第一/33 | LifeAvg MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| 原版PEMS08_lr_double（对照，复用旧test） |21|35.49617|46.92140|34.14484|
|02_PEMS08_unit_focal|21|34.81996|44.70282|31.20052|
|08_PEMS08_unit_no_refiner|22|34.85672|45.01656|31.62994|

同为C6/LR2/dropout0.1/WD1e-4、seed32。A只改标准化Focal，C另将refiner scale置0；模型身份/源文件/训练plan/权重SHA写入新plan。checkpoint仍按训练时validation LifeAvg MAE选择。

执行顺序：验证训练报告/权重一致性→独立完整validation33格重放（沿用原1e-5误差门槛及计数核验）→通过后写test打开记录→原指标、共同mask/scaler、不用teacher forcing完成test33格。训练适配器只在该进程启用，checkpoint method contract必须相等。两项并行，最多2个评估任务，无训练。

评估只报告这两个预先固定候选与原版、8个固定baseline的比较；不根据测试结果再打开其余7项。首先判断相对原版的覆盖与三项LifeAvg是否共同改善；若有冲突，完整列出取舍，不设置新的任意硬门槛，也不拼接不同权重的最优单格。80%仍指至少27/33，无方案达到则如实报告。

原test曾参与修复动机及锚点选择，本次不能称为完全未受test反馈影响的独立方法确认。没有多种子显著性声明。输出独立版本的validation/test完整主表阶段表，保留现有正式四表待结果判断。

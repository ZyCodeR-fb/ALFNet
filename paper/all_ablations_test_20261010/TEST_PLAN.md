# 四数据集全部消融test计划

2026-10-10用户明确授权“都再跑一下test，看test结果是否满足”。范围：前述三轮12种消融×METR-LA/PEMS-BAY/PEMS04/PEMS08，共48个已训练checkpoint；四个固定Full已核验test按checkpoint哈希复用。无PEMS03、无新训练、无新参数组合、无对比算法重跑。

每项保持训练时完整适配器（输入插值/中心化、结构干预、FiLM/梯度路径控制），严格载入best.pt并对照method contract、数据/scaler/曝光身份。先完整复核validation33格，与训练报告一致后才写test打开记录并执行test。测试不使用教师强制，不更改分割或指标公式。

独立计划保存48项权重、配置、source/data哈希，checkpoint固定后不按test反馈换epoch。评估保守最多3并发，显存门槛20GiB，错峰20秒；不操作其他用户进程。每项失败保留日志，结果不能用validation或其他checkpoint填充。

自动导出52模型配置×33格=1716test指标值，主表/阶段表、三项LifeAvg相对Full变化、四数据集全12指标变差计数。全部48项保留，validation为0的事实不改写，不以test方向筛除不利消融或再次调参。历史选型已看test，本轮属于固定模型的授权补充评估，不声称从未观察的独立选型集。

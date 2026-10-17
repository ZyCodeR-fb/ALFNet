# 最后一轮 flow 候选测试计划（2026-10-03）

用户授权：分析已完成训练，对值得保留的候选运行 test。本计划在任何新 test 访问前固定。

## 固定范围

五项 seed32 checkpoint：PEMS03 reconstruction；PEMS04 reconstruction；PEMS08 reconstruction、degree_supports、reconstruction_degree。最多三个并发评估，无新训练。

03/04 reconstruction 同时提高相对八个基线的第一格数及三项 LifeAvg，优于同轮其他方案的这四个汇总维度。08 的三个候选分别代表最大覆盖（23/33）、最低 LifeAvg MAE/RMSE（三指标均优于原版，但20/33）、最低 LifeAvg MAPE（22/33）；在覆盖和三指标上互不支配。原版08为21/33。故保留这三个明确的取舍，不把更多胜数当成平均误差必然改善。

没有候选在 validation 达到至少80%（27/33），测试用于判断局部收益是否值得保留，不宣称已达到目标。旧 test 已参与前期研究设计，新的比较属于探索性验证，不能描述为完全独立的确认。

## 执行约束

使用 experiments.flow_last_round.adapter 重建准确的训练语义；严格核对 checkpoint 方法合同、配置、数据、scaler、信号暴露和权重。完整33格 validation 重放通过后，每项只打开一次 test。评价掩码、阈值、分组、算术保持原协议。

旧模型及八个基线 test 只复用，不重训、不重测。reconstruction 多了训练节点过去值的辅助监督，评估不使用隐藏标签。不得将其描述为所有方法采用相同损失标签集合。

输出新的 validation/test 主表与阶段表、权重审计和第一格数。正式四张论文工作簿暂不覆盖；依据覆盖、LifeAvg及条件级取舍给出保留建议。无自动追加候选、数据集删除、提交或推送。

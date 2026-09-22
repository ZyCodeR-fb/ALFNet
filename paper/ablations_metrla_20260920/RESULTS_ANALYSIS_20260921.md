# 最终Focal消融结果分析（2026-09-21）

Origin Skill: academic-research-suite / experiment-agent validate (inline)  
Verification Status: ANALYZED；离线工件交叉核验通过，未重跑训练或模型评估。

计划196607d53cc319ead629e8d8fc748c368718b8ab834d2e23896f0f890be5e45a。9/9完整结束，退出码0，全部训练/validation无skip；best epoch为完整历史validation LifeAvg MAE首次最小点。checkpoint配置与计划SHA、报告与独立33格表一致。完整audit含权重SHA，见completion_audit_20260921.json。没有活动worker，新test未运行。

## 全部变体的变化

相对同一validation Full，正值表示误差增大；改善格数为变体误差小于Full，不是外部基线获胜格数，也不是独立统计重复。

| 变体 | MAE变化% | RMSE变化% | MAPE变化% | 优于Full格数 | best epoch(0起) | 训练轮数 |
|---|---:|---:|---:|---:|---:|---:|
| mask_only | +3.765 | +1.884 | +4.981 | 18/33 | 61 | 72 |
| no_age | +1.112 | +0.536 | +4.242 | 1/33 | 52 | 63 |
| static_router | +3.581 | +3.429 | +1.580 | 14/33 | 72 | 83 |
| local_context | +7.374 | +4.234 | +14.033 | 0/33 | 57 | 68 |
| no_imputer | +5.111 | +3.562 | +5.430 | 0/33 | 52 | 63 |
| no_temporal_film | +15.098 | +14.450 | +13.553 | 6/33 | 62 | 73 |
| no_refiner | -0.444 | -0.126 | -1.251 | 19/33 | 72 | 83 |
| plain_mae | +0.703 | +1.030 | +3.181 | 7/33 | 65 | 76 |
| no_role_balance | +0.161 | -0.406 | +0.420 | 16/33 | 95 | 100 |

## 可支持和不可支持的结论

- 时间条件分支整体：删去后平均退化最大，但包含分支容量变化，不能单独归因FiLM。
- 局部上下文和无重建：全部33格变差；前者参数量相同，是较直接的跨节点上下文证据。后者减少容量，结论限于与标准化零填充对比。
- 状态/静态路由：平均变差，但18/14格反而更好；必须展示条件与幅度，不只报告有利均值。
- Age：去掉后32/33格更差；保持角色/掩码，可分离其补充作用。
- Plain MAE：三项平均变差，7格更好，不能说Focal全条件优。
- Role balance：对Unseen/Cold有益，对Steady/Fault存在代价，平均收益小且RMSE反向。
- Refiner：删除后全部LifeAvg指标更好、19格改善；Full只有部分条件收益。不能写模块全面有效，不按该单seed结果自动更改已固定模型。

## 论文采用哪个划分

建议最终版：主比较test、固定消融test、超参数曲线validation。超参数负责选参/敏感性，消融在配置固定后负责评价移除设计的效果。先用validation选择每个变体best，再对全部预定变体执行test；Full复用同一checkpoint的现有test，因而其行应与主表一致，不需重训。不能根据test继续选择有利消融或删除不利行。

当前只有validation消融，所以正文4.4显式标记为开发分析并给出同split Full，未伪装成test。Full validation4.537516/6.911326/13.475396与test3.931467/6.191635/10.426073不同，是时间段与保留节点不同，不是配置不一致。跨split绝对值不能直接比较；超参baseline只需与相同validation基准一致，不要求等于test主表。

仅从其他论文数值或图形不能断言其全部使用同一split；应看实验设置和表注。有的采用test消融与validation调参，有的标注不足。我们需要明确标注，不用统一数字掩盖不同数据来源。[官方选参/评价流程](https://scikit-learn.org/stable/modules/cross_validation.html)。

本轮用户请求结果分析与稿件更新，未明确授权本次9项新test。已完成训练、权重可用于后续完整test；不自动开test，保留当前最终模型和全部原始结果。前次已使用test比较候选的研究历史仍须披露，不称全研究首次独立测试。

## 解释限制

单seed、完整33格相关、没有p值/CI；不宣称显著或可加性。9/9全部报告，无幸存者筛选；聚合与条件结果相反已列明，不外推每节点或自然发生率效果。不将移除分支的容量影响归于单一算子，也不将有限配置结果外推为普遍因果结论。早期Full/MSE机制图属于历史实验，不能假定等价于新版Focal。

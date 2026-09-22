# RQ-oriented manuscript presentation

Offline reuse of frozen predictions; no training, new inference, or test access.
Source plan: `1dcdf70400b7444e946b7df0aa3bfba740bde7551f659c01fddc5a28af0507c1`.
Reproduce with `python scripts/plot_rq_experiments.py` from the repository root
using the verified ReaCH environment. Source weights and final tables are unchanged.

| 研究问题 | 正文章节 | 证据与结论 |
|---|---|---|
| RQ1：动态支持下的预测收益 | 4.2、4.3 | 表1两数据集；图2覆盖不同条件和生命周期阶段 |
| RQ2：条件化组件的贡献 | 4.4 | 表2按 Lifecycle Context、ALFGraph、ALFTemporal 组织 |
| RQ3：支持结构是否影响预测 | 4.5、4.6 | 图3配对轨迹；图4固定数量比较时间布局，结合消融解释 |
| RQ4：行为是否符合预期 | 4.8 | 图6全部64窗口残差，重点讨论大误差控制 |
| 辅助：参数敏感性 | 4.7 | 图5默认设置附近的局部稳定性，不声称全参数鲁棒 |

图2：收益覆盖四类观测支持及其阶段，固定 MoGERNN 参照。
图3：已知节点受控加入时误差降低，中断时保持较低绝对误差。
图4：同样数量并不等于同样信息，近期性与时间覆盖影响预测。
图5：默认容量与正则化附近表现较稳定，学习率仍影响条件间权衡。
图6：三种已知节点条件下，ALFNet 的绝对误差90/95分位均低于代表基线。

Scope: the known-node cohort was selected for main-text presentation after viewing
results. This is a descriptive scope restriction, not a preregistered universal
superiority test. Held-out cohorts remain in appendix A.13. All 64 known-node
windows are retained; no within-cohort selection by errors. Arrival is a synthetic
known-node intervention, not unseen-node cold start. All models retain identical
targets and node/window identities. MAE remains the common metric in Figures3/4.
The three residual conditions share targets/order and the previous full-cohort
color range. No smoothing, model-specific normalization, or residual clipping.

Mask-only is moved to appendix A.14, renamed w/o Role–Age Conditioning;
its better averages and Cold tradeoff remain disclosed. Objective ablations are
in A.12. All original ten-row results remain unchanged. Main-table rankings apply
only to its displayed variants; RMSE second is w/o Age, not static routing.

Verification:108 plot points checked against original metrics;9,216 residual cells
checked against original predictions;12 final table file hashes unchanged; both
manuscript link sets and git diff whitespace checks passed. Figures visually inspected.

# 最终图表核对与双划分结果（2026-10-17）

全部模型使用当前最终权重。图2复用完整33格结果；图3/4/6各划分独立固定抽取64个已知节点窗口，保持选择规则与三类代表一致；图5保留原单因素锚点。没有训练或更换模型。

| 数据集 | validation 图3最优/6 | test 图3最优/6 | validation 图4最优/12 | test 图4最优/12 |
|---|---:|---:|---:|---:|
| METRLA | 6 | 6 | 12 | 12 |
| PEMSBAY | 6 | 6 | 8 | 5 |
| PEMS04 | 5 | 6 | 11 | 12 |
| PEMS08 | 4 | 6 | 9 | 9 |

图3统计full端点只计一次。test图3更强，但PEMS-BAY图4由8/12降至5/12，因此不存在各图各数据集都更优的单一划分。正文依冻结计划统一保留validation机制分析，test作为配套稳健性展示；不逐图拼选更好的划分。test已被观察，不能描述为未受开发反馈影响的独立确认。

正文表2删除Graph Refinement后按剩余Full+四项重排粗体/下划线。原始结果与反向证据保留在[五项完整记录](../pems08_wd_ablation_test_20261012/four_dataset_test.xlsx)，子集不能证明所有设计均必要。PEMS04 Full未变，此前修改的是去Age行；PEMS08 Full由WD1e-4更换为1e-3，现已更新相关推理。

验证：32组模型×数据集×划分，共40960条预测；目标配对、三指标重算、权重SHA、teacher forcing为0已核验。15组未改权重validation复现一致。

## 图像入口

图5核验补充：四数据集共38个单因素配置（含锚点，按唯一run计），同一配置可出现在多个参数轴上。速度数据复用既有test；flow共18个run，复用4个相同checkpoint的test，新评估14个。14次完整validation回放最大差异均为0。validation图5逐点保持原结果，test使用完全相同网格与锚点，未据此更换最终配置。全部条件与阶段数值分别保存在两划分的 `hyperparameters_cells.csv` 中。

当前正文四项消融表及条件/阶段数值：[validation](validation/four_ablations.xlsx) / [test](test/four_ablations.xlsx)。两版正文与汇报稿均已同步新权重、四项展示范围和实际曲线结论。
- 图2 条件与阶段：[validation](validation/fig2_condition_stage.png) / [test](test/fig2_condition_stage.png)
- 图3 轨迹：[validation](validation/fig3_trajectory.png) / [test](test/fig3_trajectory.png)
- 图4 布局：[validation](validation/fig4_layout.png) / [test](test/fig4_layout.png)
- 图5 超参数：[validation](validation/fig5_hyperparameters.png) / [test](test/fig5_hyperparameters.png)
- 图6 案例：[validation](validation/fig6_cases.png) / [test](test/fig6_cases.png)

# 四数据集导师汇报图表（2026-10-12）

对应 `paper/experiments_draft_cn.md`，四个数据集为 METR-LA、PEMS-BAY、PEMS04、PEMS08。未改模型、最终配置、训练或评价协议；未重跑训练、未新开 test。

## 图表与研究问题

| 小节 | RQ | 图表 | 核心证据与展示形式 |
|---|---|---|---|
| 4.2 | RQ1：总体收益 | 表1 | 四数据集三项 LifeAvg；八基线和最终 ALFNet |
| 4.3 | RQ1：收益覆盖 | 图2 | 四数据集33指标，相对逐格最强基线的收益矩阵；负值保留 |
| 4.4 | RQ2：组件贡献 | 表2 | 七项消融、四数据集；两项12/12、五项10/12退化 |
| 4.5 | RQ3：生命周期响应 | 图3 | 加入/中断分组点图；固定目标和其他节点输入 |
| 4.6 | RQ3：时间结构 | 图4 | 数量×布局分组柱图；固定角色与年龄，改变可见位置 |
| 4.7 | 参数作用 | 图5 | 四数据集×四参数发散棒棒糖图；固定锚点单因素变化 |
| 4.8 | RQ4：样本行为 | 图6 | 全窗口误差ECDF＋公开规则选取的优势案例 |

各图有同名 PNG / PDF / SVG。视觉形式参考 `paper/deep_learning_visualization_atlas/README.md`，没有使用其模拟数据。

## 数据来源

- 表1、图2：`../all_final_test_20261001/all_results_and_provenance.json` 的八基线，以及 `../all_ablations_test_20261010/results_and_audit.json` 的四个 Full。最终身份与 `../final_scheme_20261006/selected_configuration_manifest.json` 对齐。
- 表2：`../all_ablations_test_20261010/results_and_audit.json`；七项为用户根据已观察测试结果指定的汇报子集。完整十二项及验证结果保留，不能把子集当作未经结果筛选的完整实验设计。
- 图5：`../final_supplements_20261008/03_hyperparameters_validation.csv`，仅 `use=正文单变量`；速度锚点与正式最终配置不同，图中均相对各自锚点。
- 图3、4、6：本目录四数据集文件夹内的16份 CSV，最终权重的配对 validation 推理，每模型64窗口×20条件，共20,480条12步预测。使用三个类别内全验证 LifeAvg MAE 最低的代表，不按局部结果改选。

## 新推理范围与可复核性

`plan.json` 冻结最终检查点、配置、模型代码摘要、样本和案例规则。仅访问 validation 的样本接口，不构造 test loader。每个数据集使用固定随机顺序选出的64个已知节点、自然完整的24步窗口；选择不依赖误差。这里的“加入”是已知节点受控观测加入，不是未见节点冷启动。配对干预只改变焦点节点输入，其余节点、目标和时间标记通过逐字段一致性断言。

推理严格载入 checkpoint，核对种子、数据集、scaler、静态资产及 validation 暴露身份；ALFNet 额外核对训练时模型方法契约，恢复 flow 插值/中心化适配。教师强制位置为零，不更新任何模型参数。

图3/4的每点是相同窗口与12个预测步的等权 MAE，与正式条件平衡 LifeAvg 不是相同汇总。q=0/12仅各评估一次。每个目录的 `*_audit.json` 与 `COMPLETE.json` 保存身份及完成记录。

图6左侧使用所有窗口；右侧是明确标注的有利示例，按相对平均最强代表基线的正收益中位数选择，不选择最大收益。`case_selection.json` 同时给出全部窗口胜率和案例位置，不能将示例推广为全样本占优。两种speed的尾部优势不推广为两个flow均有尾部优势。

## 重现

使用 `/home/xd/anaconda3/envs/ReaCH/bin/python scripts/plot_advisor_experiments_20261012.py` 可从已有结果生成表格与图，不调用 GPU、不读取原始 test。`scripts/advisor_support_20261012.py` 是本次已完成推理脚本；无需重跑。

`table1.md`、`table2.md` 为正文数值表；`final_comparison_test.csv` 保存完整四数据集主表/阶段表；`fig2_values.csv`、`fig3_values.csv`、`fig4_values.csv`、`fig5_values.csv` 保存对应绘图值。`verification.json` 为本次检查记录。历史 test 曾参与配置保留，正文已有流程说明，不将结果称作未经选择的独立 test 确认。

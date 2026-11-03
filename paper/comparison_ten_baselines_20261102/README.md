# 加入 ST-FiT 与 Merlin 的四数据集对比表（2026-11-02）

离线合并已保存的结果，未训练、未推理、未访问新的测试数据。由 `build_tables.py` 生成（Anaconda Python 3.12 + openpyxl）。

## 文件

| 文件 | 内容 |
|---|---|
| `test_comparison_10methods.xlsx` | 测试集，论文表 1 与图 2 的依据。工作表：说明、第一名统计、四个数据集各一张主表与阶段表 |
| `validation_comparison_10methods.xlsx` | 同样结构的验证集版本，供参考 |
| `*_comparison_10methods.csv` | 长格式数值（未舍入） |
| `summary.json` | 第一名格数与 LifeAvg 降幅 |

## 数据来源

- ALFNet 与原表 1 的 8 个基线：`paper/pems08_wd_test_20261012/{test,validation}_main_phase.csv`（最终配置）。
- ST-FiT 与 Merlin：`paper/final_four_tables_20261001/before_model_filter_20261001/{02_comparison_test,01_comparison_validation}.csv`，来自 2026-10-01 统一评估（计划 `configs/paper/all_final_test_20261001/plan.json`，SHA `e696f342…`；117 个权重先做完整验证复现再开测试，`paper/all_final_test_20261001/verification.json` 为 `passed: true`）。
- 合并前，两个来源共有的 8 个基线在测试集与验证集上各 1056 个数值逐一相等。

注意：`paper/final_four_tables_20261001/01_comparison_validation.xlsx` 是 10 月 2 日旧 ALFNet 配置的验证集表（第一名 29/24/19/21），不是 33/29/29/30 的来源。

## 结果摘要（测试集）

| 数据集 | ALFNet 第一/33（8 个基线） | 加入 ST-FiT、Merlin 后 | 被超过的格 |
|---|---:|---:|---|
| METR-LA | 33 | 30 | Merlin 3 格：Steady MAE/MAPE、Outage MAE |
| PEMS-BAY | 29 | 22 | ST-FiT 7 格：Join MAE/MAPE、Cold RMSE、Warm MAE/MAPE、Ready MAE/MAPE |
| PEMS04 | 29 | 15 | Merlin 14 格：Fault 与 Onset/Decay/Outage 全部 12 格、Steady RMSE/MAPE |
| PEMS08 | 30 | 19 | Merlin 11 格：Fault、Decay、Outage 全部 9 格及 Steady MAPE、Onset MAPE（后者 ST-FiT 也优于 ALFNet） |

LifeAvg 三项在四个数据集上仍全部第一。唯一变化是 PEMS08 的 MAE 与 RMSE 最强比较方法由 STID 变为 ST-FiT，降幅由 7.86% 与 8.87% 变为 7.27% 与 7.72%；因此正文"MAE 降低 7.86%–15.63%"会变为 7.27%–15.63%，RMSE 区间 7.51%–10.56% 不变。

验证集：第一名格数由 29/24/27/32 变为 21/18/14/20；LifeAvg 三项仍全部第一，最强比较方法与降幅不变。

## 限制

- 单一种子 32。
- Merlin 的教师模型在训练期可访问已知节点被遮蔽的历史，并保留节点身份参数；ST-FiT 评估含 Gumbel 随机采样（固定种子重放）。见 `paper/BASELINE_FAIRNESS_20260928.md`。
- 二者的权重已在 2026-10-17 清理中删除（`archive/cleanup_20261017/plan.json`）。本表只用已保存结果；若要在二者上补做图 3、图 4、图 6 一类分析，需要重新训练。

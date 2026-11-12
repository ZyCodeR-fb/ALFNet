# 对比算法三种子结果（2026-11-10 整理）

## 文件
- `seeds_overview_20261110.xlsx`（主文件）：全部 16 行方法 × 四个数据集；种子 32/42/52 逐列给出，未跑的留空；三种子齐全时给 mean、std（样本标准差，ddof=1）与 "mean ± std"。
  - `test_LifeAvg` / `validation_LifeAvg`：LifeAvg 三指标的逐种子值、均值、标准差。
  - `*_main_mean_std` / `*_phase_mean_std`：主场景 5 格、阶段 6 格（×MAE/RMSE/MAPE）的 mean ± std。
  - `*_long`：全部 33 格 × 三种子长表（后续补种子时用于核对）。
  - `test_stability`：各方法变异系数、最好/最差种子、种子 32 排名与均值排名。
  - `first_place_provisional`、`test_ALFNet_vs_mean`：暂定比较（ALFNet 只有种子 32）。
- `seeds_overview_20261110.json`：同上数据与暂定第一格数。
- `seeds_test_mean_std.xlsx`、`seeds_validation_mean_std.xlsx`：自动链生成的 10 个三种子方法 mean ± std（与主文件一致）。
- 生成脚本：`scripts/report_seeds_overview_20261110.py`（只读冻结记录，可在补充种子后重跑）。

## 已完成 / 未完成
- 三种子完成（训练计划 58eecfb5，测试计划 92f8956b）：UIGNN、GinAR、MoGERNN、SLPF、STGCN、STID、PatchSTG、STBP、Merlin、STONE。82 个新训练全部完成，无非有限梯度事件、无训练跳批；82 个测试前验证重放全部一致（最大差 1.4e-5）；38 个种子 32 结果复用（当前代码重放一致）。
- 仅种子 32（42/52 留空，待补）：ALFNet、HD-TTS、STOP、MCPST、MetaDG。

## 需要注意
- GinAR 在 METR-LA、PEMS-BAY 三种子均为 lr .001（lr .006 在种子 42 发散）；论文现有 lr .006 种子 32 单列为 "GinAR (lr .006, paper)"。种子 32 两者测试 LifeAvg MAE 接近（METR-LA 7.043 对 7.092，PEMS-BAY 4.992 对 5.059）。
- METR-LA GinAR 种子 42：验证正常（LifeAvg MAE 7.02），测试 Steady 格 MAE 50.3、LifeAvg MAE 18.30，导致该行 std 很大（10.99 ± 6.34）。按协议不因测试结果剔除或重跑，照实报告。
- 种子改变节点划分，且划分难度有系统差异：10 个基线的种子 32 相对其三种子均值的中位偏差为 METR-LA −3.5%、PEMS-BAY −2.3%、PEMS04 +2.7%、PEMS08 −3.1%。因此"ALFNet 种子 32 对基线三种子均值"在 METR-LA、PEMS-BAY、PEMS08 偏向 ALFNet，在 PEMS04 偏向基线；目前唯一同划分的公平比较是种子 32 对种子 32。

## 暂定第一格数（测试，比较集 = 10 个三种子方法；METR-LA / PEMS-BAY / PEMS04 / PEMS08，合计 /132）
| 规则 | 每数据集 | 合计 |
|---|---|---|
| 种子 32 对种子 32（同划分） | 33 / 29 / 29 / 30 | 121 |
| ALFNet 种子 32 对基线三种子均值 | 33 / 30 / 27 / 30 | 120 |
| ALFNet 种子 32 对基线三种子中最好值（最保守） | 33 / 22 / 16 / 24 | 95 |

ALFNet 三种子补齐后，应以均值对均值重算。

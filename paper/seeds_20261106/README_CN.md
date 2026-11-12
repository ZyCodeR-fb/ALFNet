# 对比算法三种子结果（2026-11-06）

- 方法：UIGNN, GinAR, MoGERNN, SLPF, STGCN, STID, PatchSTG, STBP, Merlin, STONE；数据集：METR-LA, PEMS-BAY, PEMS04, PEMS08；种子 32/42/52。
- 训练计划 `58eecfb5`，测试计划 `92f8956b`；新测试 82 个检查点（各一次），复用种子32表 38 个。
- 种子同时决定 K/V/T 节点划分；每个种子用与种子32完全相同的配置文件（同 SHA256）。
- 例外：GinAR 在 METR-LA、PEMS-BAY 上 lr .006 在种子42发散（诊断见 artifacts/experiments/seeds_20261106/diagnostics/），按 2026-09-27 流量数据集先例统一改为 lr .001（四个数据集一致），三个种子均新训练；论文现有 lr .006 种子32结果未改动、不计入本均值。
- 全部运行带非有限梯度步守卫（无事件时为空操作；有事件的运行在 seeds_mean_std.json 来源中可查）。
- 数值为 mean ± std，std 为三种子样本标准差（ddof=1）。
- `seeds_mean_std.json` 含每格三种子原值与来源；`seeds_long.csv` 为长表。

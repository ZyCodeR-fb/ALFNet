# 最终四数据集validation→test排名变化

使用最终选定checkpoint已有结果；无新推理或test数据读取。每格第一定义为严格优于八基线，33格包含主表和阶段表，彼此并非独立样本。

| 数据集 | validation第一 | test第一 | LifeAvg MAE变化% | RMSE变化% | MAPE变化% |
|---|---:|---:|---:|---:|---:|
| METRLA | 29 | 33 | -15.13 | -11.63 | -24.23 |
| PEMSBAY | 24 | 29 | -7.88 | -8.77 | -17.45 |
| PEMS04 | 27 | 29 | -1.14 | -1.83 | -27.13 |
| PEMS08 | 30 | 30 | +8.63 | +5.35 | +36.59 |

## METRLA

| 条件/指标 | 变化 | ALFNet val→test | 最强基线 val→test |
|---|---|---|---|
| Join/MAPE | gain | 14.29715→9.09667 | 13.76558(UIGNN)→10.21895(UIGNN) |
| Cold/MAE | gain | 5.23420→3.67302 | 5.22611(UIGNN)→3.98773(UIGNN) |
| Cold/MAPE | gain | 18.29871→9.53767 | 18.13998(UIGNN)→11.38046(UIGNN) |
| Warm/MAPE | gain | 14.06249→9.12508 | 13.41797(UIGNN)→10.13333(UIGNN) |

## PEMSBAY

| 条件/指标 | 变化 | ALFNet val→test | 最强基线 val→test |
|---|---|---|---|
| Unseen/MAE | gain | 5.15955→4.53955 | 4.71549(MoGERNN)→4.59203(UIGNN) |
| Unseen/MAPE | gain | 13.84182→10.16531 | 12.86960(MoGERNN)→10.93420(MoGERNN) |
| Join/MAPE | gain | 6.65393→5.41657 | 6.30491(UIGNN)→5.52627(UIGNN) |
| Cold/MAE | loss | 2.80354→2.46177 | 2.80744(UIGNN)→2.41195(UIGNN) |
| Warm/MAPE | gain | 6.62655→5.37417 | 6.27153(UIGNN)→5.52520(UIGNN) |
| Ready/MAPE | gain | 6.15796→5.19693 | 5.97218(UIGNN)→5.39441(PatchSTG) |
| Outage/RMSE | gain | 4.37253→4.50258 | 4.37045(PatchSTG)→4.61017(PatchSTG) |

## PEMS04

| 条件/指标 | 变化 | ALFNet val→test | 最强基线 val→test |
|---|---|---|---|
| Steady/RMSE | gain | 34.69532→33.41692 | 34.25798(STID)→33.44773(STID) |
| Ready/RMSE | gain | 40.41955→36.02406 | 40.39804(STID)→36.14146(STID) |

## PEMS08

| 条件/指标 | 变化 | ALFNet val→test | 最强基线 val→test |
|---|---|---|---|

解释：speed并非多了大量新信息，METRLA只跨越4个比较阈值，PEMSBAY净增5（6增1失）。实际误差与基线也随时间/节点群体变化，排名取决于两者相对幅度。flow已在validation有27/30格，改善空间较小；PEMS08尽管LifeAvg变差，仍可保持相同总胜格。格数相同不保证同一组格子不变，更不代表分布相同。

validation与test按时间分段，并使用不同的未见节点V/T，因而时间/节点分布变化是合理来源；但这份已有指标分析没有对两个因素做因果隔离，不能把speed所有变化归结为某一种分布偏移。当前speed原配置的保留曾参考test，存在选择乐观性，不能把test优于validation都解释为天然泛化增强。需如实披露历史选择流程，不把test当作继续调参的独立验证。

# METR-LA 五对照结果分析

## 核验范围

计划 `ef6bfd48f3c4b69840d06a7a3b2a61e03aa798de361efed26dbccbbd2114ba85`。
五运行全部COMPLETE、退出码0；完成时间2026-09-13 18:10 +0800。
在独立源码目录执行builder --check通过。逐项核对run_report与development_validation_table、candidate_summary内容一致，五项history中的最小checkpoint_life_avg_mae及最早平手epoch均与选中报告一致；训练skipped batch均0。没有重新推理或打开test，也未重新加载checkpoint tensor核验预测。

来源：`outputs_v4/metrla_wave_screen_v1/ef6bfd48f3c4b69840d06a7a3b2a61e03aa798de361efed26dbccbbd2114ba85/` 的plan、launcher_status、各run_report/development_validation_table、candidate_summary、logs；Full/MoGERNN参考来自 `paper/experiments_evidence_snapshot.json`。五个新运行的主表有效目标数均874197、窗口6696，各条件/阶段计数相同。这里只分析seed32 validation，不是显著性或test结论。

## 总体结果

| 模型 | best epoch / 已跑epoch数 | LifeAvg MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| 保留Full | 历史参考 | 4.638117 | 6.806835 | 14.626242 |
| Full-MAE | 53 / 64 | **4.548997** | 6.947354 | 13.692081 |
| 轻量TCN-MSE | 57 / 68 | 4.982139 | 7.156756 | 14.960466 |
| 替换版-MSE | 35 / 50 | 4.937411 | 7.060575 | 13.904753 |
| 替换版-MAE | 53 / 64 | 4.770451 | 7.182375 | **13.371278** |
| 同轮GWNET-MAE | 38 / 50 | 4.796742 | 7.181160 | 14.046639 |

四个方法候选均未过冻结门槛。GWNET是参考，不作为待晋升候选。没有任何一项达到100epoch上限；均由早停结束。尾部LifeAvg围绕已选最优上下波动，没有“单纯加长到100就会通过”的证据。

## 最有价值的结果：Full-MAE

相对Full，LifeAvg MAE降低1.92%、MAPE降低6.39%，RMSE上升2.06%。11行的MAE和MAPE全部改善；RMSE只有Onset改善，其余10行退化，合计23/33单元改善。九条失败门槛全部是RMSE。这是有系统性的损失取舍，不应与结构候选退化混为一谈。

| 条件 | Full MAE / RMSE / MAPE | Full-MAE MAE / RMSE / MAPE |
|---|---|---|
| Unseen | 6.609063 / 9.020675 / 20.273579 | 6.546776 / 9.120071 / 19.194421 |
| Join | 4.425512 / 6.630428 / 15.538770 | 4.268788 / 6.770992 / 13.901018 |
| Cold | 5.341122 / 7.462025 / 19.424238 | 5.150291 / 7.555293 / 17.396510 |

Unseen MAPE改善5.32%，Join MAPE改善10.54%，Cold MAPE改善10.44%；对应RMSE仍略升。与“绝对误差目标弱化大残差惩罚”的解释一致，但尚无逐样本残差分布证据，不能宣称已定位低速/极端值因果。

对原MoGERNN参考，Full-MAE在33/33单元都更好，包括Unseen MAPE 19.194421 < 19.689703；这补上了原Full在METR-LA对MoGERNN的唯一指标短板。但这是同种子验证表逐单元比较，不能声称多种子显著优越。

对同轮GWNET，Full-MAE的LifeAvg MAE/RMSE/MAPE分别降低5.16%/3.26%/2.52%，总体三指标均更好；条件上不是全面胜出：Unseen/Join更好，Steady/Fault及部分故障阶段仍落后。结论是保留原架构已经能获得较强总体结果，无须把GWNET替换作为前提。

## 轻量TCN与大替换

轻量TCN相对Full总体MAE/RMSE/MAPE分别退化7.42%/5.14%/2.29%；Unseen MAE/RMSE退化14.28%/11.74%，Cold MAE退化7.50%。虽然零初始化预测parity通过，训练后并没有维持优势。只排除本次输入侧16通道/八层/当前学习率方案，不据此排除所有时间卷积；不建议原样扩展到PEMS-BAY。

替换版-MAE相对同轮GWNET总体MAE仅降低0.55%、MAPE降低4.81%，RMSE上升0.017%（实际近乎持平，不夸大这个极小差值）。它没有达到“大幅超过GWNET”的3% MAE改善门槛。

更关键的是相对Full，替换版-MAE Unseen MAE/RMSE上升11.20%/14.88%，Join MAE/RMSE上升15.13%/13.01%；Ready MAE上升20.24%。Steady和Fault三指标全部改善，却牺牲本任务重点的无历史和接入条件。总体MAPE最低不能证明主干替换成功。

替换版-MSE同样损害Unseen和Join；相对GWNET总体MAE反而差2.93%。它与替换版-MAE的对照支持损失会改变误差取舍，但不同checkpoint选择也属于该配方效果的一部分。

原生GWNET在本协议下相对Full Steady/Fault三指标更好，Unseen三指标更差、Join MAE/RMSE更差；优势呈明显条件依赖。当前证据不支持“直接借用其主干就能修复ALFNet无历史预测”。替换还同时改变循环、teacher forcing、传播/路由与条件化位置，不能把差异全部解释为时间卷积本身。

## 实际耗时

以下为每阶段跨全部epoch的独立中位数，各列不能直接相加当严格epoch中位数；五进程共享GPU且各自不同时间结束，不是独占硬件竞赛。

| 模型 | train / epoch | validation / epoch | 运行总时长约 |
|---|---:|---:|---:|
| Full-MAE | 154.34 s | 78.65 s | 4h15m |
| 轻量TCN | 145.10 s | 75.32 s | 4h21m |
| 替换版-MSE | 77.33 s | 140.89 s | 3h08m |
| 替换版-MAE | 75.81 s | 109.85 s | 3h35m |
| GWNET | 35.52 s | 169.76 s | 2h50m |

图主干替换使训练阶段约减半，但没有等比例缩短整个运行。GWNET累计validation 8319.82 s，train 1847.42 s，约82%的这两阶段时间用于validation。之前定位的逐样本GPU标量同步，在多进程共享时仍是值得优化的路径；这些日志只能证明验证耗时高，不能独立精确分解同步、GPU竞争和CPU调度占比。完整验证的工程瓶颈比继续盲目增加并发更值得处理。生产指标代码本轮没有修改。

## 决策建议

1. 保留原Full作为正式参考，同时保留Full-MAE checkpoint及结果为最有希望的折衷候选；不自动晋升，不因结果不错而事后放宽RMSE门槛。
2. 本轮不继续PEMS-BAY：没有候选满足预先固定的完整门槛，用户已要求METR-LA无合格收益则不扩展。
3. 暂停当前TCN与大替换配方，不合并失败架构到根模型。
4. 若继续METR-LA，优先保留结构，检验MAE优势与MSE大残差约束能否兼得。先在原尺度分箱/残差尾部核对Full和Full-MAE误差来源，再以独立身份选择一个固定MAE/MSE混合目标候选；不原样重跑旧bounded-MAE，不事后挑模型/权重拼接漂亮结果。这个后续只是建议，本轮未启动。
5. 验证统计批量化应独立做33单元和选点一致性检验后再上线。与模型准确率试验分开，避免把metric变化混入收益。

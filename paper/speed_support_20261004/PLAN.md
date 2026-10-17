# METR-LA / PEMS-BAY 支持方案迁移

2026-10-04用户明确授权将插值＋倍率＋中心化迁移到speed，允许各数据集多种超参数，最多5并发。新方案若满足要求且优于当前选定配置才考虑统一；否则保留speed原方案，flow仍用已选新方案。PEMS03停止状态不变。

## 已核验锚点与选择依据

| 数据集 | 当前选定配置 | 容量 | LR倍率 | dropout | weight decay |
|---|---|---:|---:|---:|---:|
| METRLA | METRLA_patch7 | 7 | 1 | .1 | 1e-4 |
| PEMSBAY | 05_PEMSBAY_combo2 | 16 | 2 | .1 | 1e-5 |

依据final_four_tables_20261001/selected_configuration_manifest.json，并与hparams_combinations_20260930的config/report/checkpoint对应。两者dropout已相同，不新增dropout轴。过去drop.05/.2均未稳定改善三项LifeAvg；PEMS-BAY容量16下LR1/WD1e-4 validation为2.91236/4.57859/7.53630，当前LR2/WD1e-5为2.95608/4.60451/7.60303，说明旧test选择与validation排序并不一致，保留当前正式原配置作为对照，不偷换成更弱锚点。METR-LA旧LR.5在容量8下不佳，但新方案改变残差尺度，仍将.5纳入，不能假定旧LR结论直接迁移。

## 固定候选矩阵

每个数据集：alpha{0,.5}×LR{.5,1,2}六项，另加alpha1/LR1一项。METR-LA WD固定1e-4，PEMS-BAY WD固定原选1e-5。PEMS-BAY再加alpha{0,.5}/LR2/WD1e-4两项，与对应WD1e-5构成匹配对照。总METR-LA7＋PEMS-BAY9=16项，加2旧权重复用控制，共18项594格validation。完整中心化在flow偏弱，但speed尺度更稳定，因此保留一个直接对照而不全网格扩张。

同方案：过去12步可见点插值（端点延拓，q0用GRIN），decoder scale1，中心alpha×最后可见值；q0中心0，teacher减中心，输出加回后进入原refiner。直接复用已检验support_transfer/unified适配器，不引入新模块、参数、数据集分支、监督或损失。alpha0/.5/1分别对应bridge_delta1/triple_halfcenter/triple。

固定各自原容量/32分区、dropout.1、batch64、seed32、100epochs/patience10/earlystart40/TF40、原Focal与角色权重。LR倍数相对基础图/时间/gate=5e-4/1e-4/5e-5，非相对当前PEMS-BAY×2再乘一次。WD是优化器weight decay，不是角色损失权重。

已有support/unified/LRcross/PEMS03新方案实验未包含speed，这16项均未训练，不重复旧普通ALFNet超参数实验。旧配置和已公布表不覆盖。

## 验证与判定

每项GPU预检20训练＋4验证batch：参数/初始化身份、优化器LR/WD/dropout、分区/容量、teacher/完整协议、零跳过batch。全部通过后正式队列，最多5并发，显存门控与错峰自动调度。checkpoint仍按完整validation LifeAvg MAE最小选，不能根据test挑epoch。

完成后报告全部33格相对八基线的第一格、三项LifeAvg、相对当前选定原ALFNet逐格胜负及变化比例。至少27格和三项LifeAvg领先基线是既有目标；统一方案还需优于原配置，而不只是达到80%。明确优于原版的证据是胜格不减少且三项LifeAvg不差、至少一项改善；存在取舍者单列讨论，不擅设百分比硬阈值、也不自动宣称统一成功。METR-LA既有test33格第一意味着不可能在test胜格超过33，平胜格下需比较实际误差。训练阶段仅validation筛选，不自动test或覆盖正式表。

研究已受历史test反馈影响，不能称未经观察的独立确认。若speed不适用，允许speed原方案/flow支持增强方案并明确披露，不为统一叙事隐藏失败候选。当前只授权这16项，无自动追加搜索、PEMS03/基线重训、commit/push。

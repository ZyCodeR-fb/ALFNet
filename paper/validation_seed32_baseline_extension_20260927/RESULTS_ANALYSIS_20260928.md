# 五数据集 seed32 补充对比结果分析（2026-09-28）

本轮29项新增训练已全部完成：Merlin五项，flow其他基线24项。恢复队列24/24 exit0；原PEMS03/GinAR失败记录保留，采用学习率0.001的新身份结果。
统一表包含五数据集×11模型（ALFNet+10基线），共55条记录、1815个指标。29项新结果逐一核验best epoch、report/checkpoint SHA、checkpoint参数有限性及无跳过batch；Excel数值和第一/第二名格式独立回读通过。
**全部为validation结果；本轮没有打开test，不能替代已有speed test表，也不是五数据集独立test结论。**

## 整体表现与逐格覆盖

ALFNet的LifeAvg MAE/RMSE/MAPE在五数据集上均排名第一，即15/15个整体指标第一。相对于每个指标分别最强的基线，MAE降低9.15%–14.11%，RMSE降低5.90%–12.29%，MAPE降低4.65%–28.80%。这支持整体生命周期平均误差收益，但不能等价为每个条件均最优。

| 数据集 | ALFNet LifeAvg MAE | RMSE | MAPE (%) | 主表第一/15 | 阶段第一/18 | 全部第一/33 | 第一比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| METRLA | 4.5375 | 6.9113 | 13.4754 | 9/15 | 9/18 | 18/33 | 54.55% |
| PEMSBAY | 2.9124 | 4.5786 | 7.5363 | 6/15 | 5/18 | 11/33 | 33.33% |
| PEMS03 | 31.5382 | 41.3182 | 33.4512 | 4/15 | 2/18 | 6/33 | 18.18% |
| PEMS04 | 37.8733 | 51.4145 | 34.8345 | 8/15 | 3/18 | 11/33 | 33.33% |
| PEMS08 | 34.2022 | 45.5565 | 26.0394 | 7/15 | 0/18 | 7/33 | 21.21% |

五个数据集均未达到“33格在全部模型中至少80%第一”的目标。此前三个flow对MoGERNN的33/33胜是两模型比较，不能推广为对所有基线第一。LifeAvg按既定条件平均；阶段表重复细分Join/Fault，数第一格还忽略误差幅度，因此这两个目标并不等价，也不能把33格当独立统计重复。

## 新基线改变了哪些结论

- Merlin并未超过ALFNet的LifeAvg，但明显强化了已有节点的缺失/中断预测。在PEMS03/04的Steady、Fault及Onset/Decay/Outage共15格全部第一；PEMS08在Fault及对应三个中断阶段共12格第一。与此同时其Unseen/Join误差较高。例如PEMS03 Merlin的Unseen/Join MAE为90.857/61.824，ALFNet为62.726/20.532；其Fault MAE则以17.982优于ALFNet的23.440。这解释了“平均最好”和“很多阶段不是第一”同时出现。
- METR-LA：ALFNet在Unseen、LifeAvg和Onset三指标均第一；ST-FiT在Cold三指标第一，Merlin在Steady三指标第一。
- PEMS-BAY：LifeAvg三指标第一，但Unseen三指标由MoGERNN领先；ST-FiT在Join、Cold、Warm领先。
- PEMS03：LifeAvg三指标第一，Unseen MAE/RMSE由UIGNN领先；接入早期存在多个竞争基线。
- PEMS04：ALFNet在Join、Warm及LifeAvg三指标均第一，Unseen MAE/RMSE第一；中断阶段主要由Merlin领先。
- PEMS08：ALFNet在Unseen和LifeAvg三指标均第一；18个阶段格没有第一，虽多数接近前两名，不能表述为全阶段优势。
- GinAR/SLPF在flow上的整体误差较高，不能仅凭低排名断定运行错误或删去结果。已核验新运行无跳过batch、权重有限；完整源方法适配程度仍按既有审计披露。

## 结论与保留范围

保留已经确定的ALFNet模型和配置，以及全部十个比较算法。本轮支持“五数据集整体LifeAvg稳定领先”，不支持“所有阶段均领先/至少80%格最优”。不根据排名删除Merlin或其他基线，不把validation混入test表。
离线复核此前15个flow超参数结果：PEMS03容量20、PEMS04学习率减半同时具有候选中最多第一格（6、11）和最低LifeAvg MAE。PEMS08存在取舍：当前学习率减半为7格，原默认为9格，学习率翻倍为11格；但翻倍相对当前LifeAvg MAE/RMSE/MAPE分别变差约3.78%/3.00%/31.13%。它仍能保持三个LifeAvg指标领先其他算法，但不再是已有候选中的最低MAE。建议保留当前配置，不为增加4个第一格自动接受这一误差代价；本次没有切换模型或重新运行超参数。所有候选对完整基线的比较另存JSON，不隐去覆盖率更高的候选。
公平性说明继续保留：Merlin是两阶段，每阶段max100/patience10且保留源node-ID参数；部分既有基线采用结构编码适配。SLPF为单阶段forecast适配，STBP为离线结构编码适配。flow GinAR采用数值恢复学习率0.001，speed仍为0.006。单seed结果不作显著性结论，flow/speed绝对误差不跨数据集平均。

## 文件

- final_tables.md / final_tables.xlsx：五数据集完整主表与阶段表，第一名加粗，第二名下划线。
- cells.csv：1815格原精度数值及排名。
- results_and_provenance.json：55条完整结果和来源权重/配置。
- completion_audit_20260928.json：本次独立完成审计、LifeAvg收益、各条件赢家及ALFNet排名。

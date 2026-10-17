# ALFNet 五数据集超参数表

下表对应本次validation选型的实际配置。容量是每个空间分区的token容量，分区数固定32；学习率倍率同时作用于图、时间、门控三个参数组。Dropout指ALFTemporal的crossformer_dropout；权重指AdamW weight_decay，不是损失项权重。

| 数据集 | Patch容量 | LR倍率 | 图分支LR | 时间分支LR | 门控LR | Dropout | Weight decay | 最佳epoch（从1计） | 总epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| METRLA | 8 | ×2 | 0.001 | 0.0002 | 0.0001 | 0.1 | 0.0001 | 60 | 70 |
| PEMSBAY | 16 | ×2 | 0.001 | 0.0002 | 0.0001 | 0.1 | 0.0001 | 59 | 69 |
| PEMS03 | 20 | ×0.5 | 0.00025 | 5e-05 | 2.5e-05 | 0.1 | 0.0001 | 96 | 100 |
| PEMS04 | 10 | ×0.5 | 0.00025 | 5e-05 | 2.5e-05 | 0.1 | 0.0001 | 66 | 76 |
| PEMS08 | 6 | ×0.5 | 0.00025 | 5e-05 | 2.5e-05 | 0.05 | 0.0001 | 63 | 73 |

共同配置：seed32、batch64、AdamW、max_epoch100、patience10、early_stopping_start_epoch40、Focal(mae_focal_raw02)、12步输入/12步预测、teacher forcing线性衰减40轮；参数量1,037,529。

每个数据集复用历史搜索并在其已选锚点补点，并非全部参数组合的网格搜索。比较单变量效应时须固定其他参数；完整55项实值及结果见all_candidates.csv和工作簿“全部超参数实验”。

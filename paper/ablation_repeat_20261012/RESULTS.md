# 两项原配置重跑（validation）

全部新旧结果保留；同seed重复不是独立种子证据，不按误差方向筛选。

| 数据集 / 消融 | 记录 | MAE | RMSE | MAPE |
|---|---|---:|---:|---:|
| 02_PEMS04_no_age | Full | 40.021472 | 54.338938 | 31.914950 |
| 02_PEMS04_no_age | original_ablation | 38.918313 | 52.632650 | 32.661912 |
| 02_PEMS04_no_age | repeat_ablation | 39.150938 | 52.907945 | 34.404012 |
| 07_PEMS08_static_router | Full | 35.671322 | 46.488155 | 31.712988 |
| 07_PEMS08_static_router | original_ablation | 36.369446 | 47.153245 | 34.287939 |
| 07_PEMS08_static_router | repeat_ablation | 35.413311 | 46.159428 | 34.059279 |

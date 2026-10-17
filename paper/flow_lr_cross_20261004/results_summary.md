# 两机制×两学习率匹配对照（validation）

复用4项＋新4项；两数据集各包含两机制×两学习率，全部seed32。

| Run | LifeAvg MAE | RMSE | MAPE (%) | 第一格/33 |
|---|---:|---:|---:|---:|
| 06_PEMS04_bridge_delta1 | 40.02147 | 54.33894 | 31.91495 | 27 |
| 07_PEMS08_bridge_delta1 | 37.86332 | 48.57622 | 34.70261 | 26 |
| 04_PEMS04_triple_halfcenter | 40.00144 | 54.01267 | 36.33760 | 20 |
| 05_PEMS08_triple_halfcenter | 36.13402 | 46.83272 | 32.71708 | 29 |
| 06_PEMS04_bridge_delta1_lr2 | 39.57168 | 53.53290 | 33.02158 | 23 |
| 07_PEMS08_bridge_delta1_lr1 | 36.79748 | 48.18704 | 35.27045 | 25 |
| 04_PEMS04_triple_halfcenter_lr2 | 39.13446 | 52.62805 | 33.83503 | 26 |
| 05_PEMS08_triple_halfcenter_lr1 | 35.67132 | 46.48816 | 31.71299 | 30 |

# 最后一轮三数据集 flow 修复结果（validation）

复用3项原版＋新12项修复，全部seed32；共15组、495格。第一格数相对八个保留基线单独计算，不把ALFNet候选互相加入基线池。本表不含新test结果。

| Run | LifeAvg MAE | RMSE | MAPE (%) | 第一格/33 |
|---|---:|---:|---:|---:|
| PEMS03_capacity_mid | 32.61197 | 42.76921 | 32.93310 | 16 |
| PEMS04_default | 40.29572 | 53.32744 | 35.89610 | 19 |
| PEMS08_lr_double | 35.49617 | 46.92140 | 34.14484 | 21 |
| 00_PEMS03_reconstruction | 31.64373 | 41.72813 | 31.26563 | 18 |
| 03_PEMS03_degree_supports | 33.34651 | 44.01137 | 34.78803 | 11 |
| 06_PEMS03_reconstruction_degree | 32.95046 | 43.52273 | 31.27096 | 18 |
| 09_PEMS03_relative20 | 32.89963 | 42.62923 | 33.86123 | 13 |
| 01_PEMS04_reconstruction | 38.33069 | 51.48366 | 34.81515 | 22 |
| 04_PEMS04_degree_supports | 40.26227 | 53.36170 | 36.92556 | 15 |
| 07_PEMS04_reconstruction_degree | 38.71461 | 52.00050 | 35.91458 | 18 |
| 10_PEMS04_relative20 | 39.52978 | 52.99810 | 34.86571 | 19 |
| 02_PEMS08_reconstruction | 37.01866 | 48.47724 | 33.32031 | 23 |
| 05_PEMS08_degree_supports | 34.23869 | 45.20888 | 33.22611 | 20 |
| 08_PEMS08_reconstruction_degree | 36.64652 | 48.36428 | 31.83860 | 22 |
| 11_PEMS08_relative20 | 37.06522 | 48.04262 | 34.07648 | 17 |

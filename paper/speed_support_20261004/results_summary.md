# Speed支持方案迁移（validation）

复用2项当前选定配置＋新16项，METR-LA7项、PEMS-BAY9项，全部seed32。

| Run | LifeAvg MAE | RMSE | MAPE (%) | 第一格/33 |
|---|---:|---:|---:|---:|
| METRLA_patch7 | 4.52693 | 6.88496 | 13.64209 | 29 |
| 05_PEMSBAY_combo2 | 2.95608 | 4.60451 | 7.60303 | 24 |
| METRLA_center_zero_lr0.5_wd0.0001 | 4.91257 | 7.32464 | 14.72770 | 15 |
| METRLA_center_zero_lr1_wd0.0001 | 4.72363 | 7.17167 | 14.22092 | 23 |
| METRLA_center_zero_lr2_wd0.0001 | 4.55124 | 6.97981 | 13.71026 | 26 |
| METRLA_center_half_lr0.5_wd0.0001 | 4.81986 | 7.16909 | 13.89396 | 24 |
| METRLA_center_half_lr1_wd0.0001 | 4.74021 | 7.08207 | 14.01750 | 22 |
| METRLA_center_half_lr2_wd0.0001 | 4.69815 | 7.18434 | 14.00480 | 26 |
| METRLA_center_full_lr1_wd0.0001 | 4.88780 | 7.40107 | 14.11434 | 16 |
| PEMSBAY_center_zero_lr0.5_wd1e-05 | 2.99460 | 4.73320 | 7.85633 | 12 |
| PEMSBAY_center_zero_lr1_wd1e-05 | 3.00746 | 4.73692 | 7.92137 | 14 |
| PEMSBAY_center_zero_lr2_wd1e-05 | 2.94806 | 4.61993 | 7.69710 | 17 |
| PEMSBAY_center_half_lr0.5_wd1e-05 | 3.02795 | 4.76971 | 8.05885 | 11 |
| PEMSBAY_center_half_lr1_wd1e-05 | 2.97848 | 4.70780 | 7.72882 | 15 |
| PEMSBAY_center_half_lr2_wd1e-05 | 2.94683 | 4.60469 | 7.67318 | 24 |
| PEMSBAY_center_full_lr1_wd1e-05 | 3.01770 | 4.79738 | 7.74646 | 17 |
| PEMSBAY_center_zero_lr2_wd0.0001 | 2.93545 | 4.59203 | 7.61382 | 24 |
| PEMSBAY_center_half_lr2_wd0.0001 | 2.95651 | 4.64594 | 7.66584 | 23 |

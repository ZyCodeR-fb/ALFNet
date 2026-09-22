# 保留脚本范围

`run_pemsbay_hparams.py`是2026-09-17新增的明确授权超参数入口，固定5项PEMS-BAY seed32、独立计划、预检后最多5并发。只调用既有train/validation训练函数，不修改最终模型、不开test；历史训练计划注册限制保持不变。

`eval_release.py`和`collect_release.py`用于当前冻结20项seed32权重评估及汇总；以`configs/paper/release_test_seed32.json`和完成标记为准，不能重复打开已有test队列。

`train_v4.py`、`v4_utils.py`及数据构建/预检脚本保留通用入口；旧训练计划已归档，不自动授权新训练。

`analyze_observation_support.py`、`plot_observation_support.py`和`verify_observation_support.py`保留正文/附录历史分析能力。它们的默认路径属于原实验快照，不能在清理后的根目录直接重跑。已有完整结果位于`artifacts/analyses/`；需要复算时，在独立目录恢复`archive/research_history_20260917.tar.gz`的原源码/配置/报告，再按`artifacts/catalog.json`映射原权重路径。历史消融/Full不能替换为当前Focal权重。该恢复不代表新运行授权。

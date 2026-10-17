# 实验工件索引

2026-10-17清理后请以 `../docs/FINAL_PROJECT_INDEX.md` 和
`../archive/cleanup_20261017/plan.json` 为准。旧 catalog 包含已删除的早期权重，
下文的“20个最终模型／293份权重”属于历史快照，不代表当前保留范围。

2026-09-20新增结果：PEMS-BAY Focal容量16更新版权重仍位于`experiments/pemsbay_hparams_20260917/1ec318e96fdff89dc664ac66868a0d5a34d2d0e0c5ef75d158381a3f4d8b1bc4/runs/patch16/PEMSBAY/ALFNet/bddd477b2c565e53/seed32/best.pt`，原main目录容量11权重不覆盖。精确配置与权重SHA见`configs/paper/pemsbay_patch16_test_20260918.json`和新表provenance。评估结果在`evaluation/pemsbay_patch16/eb8096d22272680525d340befde297d10ab87c6d91186bdc798dd5371f3f400d/`；validation/test/TABLES_COMPLETE均完成。

- `models/main/<dataset>/<model>/seed32/`：最终20个模型；ALFNet就是锁定的Focal。包含原样best.pt、运行配置config.yaml、原始配置config.original.yaml、validation表及provenance.json。
- `models/history/<dataset>/<model>/seed<seed>/<run_id>/`：其余273份历史最佳权重，包含正文控制、消融、附录基线、其他数据集/种子及历史候选；保留不等于建议纳入论文。用catalog.json中的原始路径、完成状态和plan/config SHA区分配方，不按结果混用。
- `catalog.json`：293份权重总目录与旧→新路径映射；所有复制逐文件SHA核验。
- `final_seed32_manifest.json`：锁定的20项test范围、权重和配置身份；不按test选模型。
- `archive_manifest.json`：旧源码/配置/日志/报告的压缩档案索引。归档文件为`../archive/research_history_20260917.tar.gz`，成员保留原相对路径，归档逐成员回读验证。
- `historical_config_recovery.json`：46份配置从Git历史按原SHA恢复；仅两个旧未完成MoGERNN尝试的配置原先已缺失，列为unresolved，权重仍保留。最终20项无此缺口。
- `cleanup_report.json`：清理的具体路径及字节数。没有删除data、data_v4、references或历史最佳权重；删除的是已核验迁移/归档的旧树、重复last.pt及缓存。
- `evaluation/seed32/<plan_sha>/`：独立test计划的validation复算、test访问标记、逐模型结果、日志与队列状态。

`config.yaml`只整理ALFNet专家配置路径；原文件与SHA同时保留，不伪装成原训练配置哈希。模型数值一致性通过完整validation复算后才开放test。旧报告中的原始路径是历史身份，使用catalog映射寻找当前文件。

没有为节约目录空间删除附录可能需要的最佳权重。293份中是否纳入正文/附录，以各自实验协议和论文证据为准；未完成尝试不得当作有效结果。

ST-FiT验证例外：源模型在eval仍使用Gumbel随机采样，best.pt未记录历史采样RNG，因此历史validation不能逐值重现。当前计划用固定training_runtime子种子，对归档训练源码与当前源码进行配对全量validation，且验证依赖文件逐字节相同；不修改模型或权重。其他18项仍以历史33格数值一致性为门槛。计划和逐项报告分别记录历史偏差与配对偏差。

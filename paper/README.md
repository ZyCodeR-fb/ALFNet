# 论文文件

2026-09-20当前更新版：`validation_seed32_hparam_patch16_20260918/`与`test_seed32_hparam_patch16_20260918/`。仅PEMS-BAY Focal改为容量16；METR-LA和九基线复用。原表保留。结论及适配/后验探索限制见`hparams_pemsbay_20260917/RESULTS_ANALYSIS_20260920.md`。

- `validation_seed32/`：最终Focal＋九基线、两速度数据集seed32的已核验validation表。
- `test_seed32/`：本次20项冻结模型test完成后生成的主表/阶段表，第一加粗、第二下划线；以验证报告和TEST_COMPLETE为完成依据。
- `experiments_draft_cn.md`、`experiments_appendix_cn.md`及`figures/`：已有正文/附录草稿与图。草稿的历史实验数值不因生成新test表而自动替换。
- `related_work*`、`baseline_selection_cn.md`：文献与基线说明。
- `../artifacts/README.md`：权重、配置及正文/附录历史证据的统一索引。

旧优化报告、运行计划、适配草稿和中间表已压缩到`../archive/research_history_20260917.tar.gz`。历史文档内原路径可通过`artifacts/catalog.json`或`archive_manifest.json`定位，不应再用旧运行提示启动训练。

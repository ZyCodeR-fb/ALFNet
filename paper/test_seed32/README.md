# 冻结模型的 seed32 test

范围为METR-LA、PEMS-BAY，模型为ALFNet（Focal）、UIGNN、GinAR、MoGERNN、ST-FiT、SLPF、STGCN、STID、PatchSTG、STBP。最终模型在test前锁定，不根据本表重新选择。

独立计划：`configs/paper/release_test_seed32.json`，SHA `63b50760d680148364a493f6d9f70aa582ada871e0358c090c65c8e40c5a3f41`。全部20项validation门槛已通过；本目录完整性以`verification.json`与计划输出目录的`TEST_COMPLETE.json`共同为准。

- `test_seed32_tables.xlsx`：两数据集主表、阶段表与说明。
- `test_tables.html`、`test_tables.md`：同一组四张表。
- `test_cells.csv`：660个未舍入数值及名次。
- `results_and_provenance.json`：逐模型完整结果、权重身份与排名统计。
- `verification.json`：数量、共同曝光/计数/scaler与Excel数值/格式回读检查。

每列数值越低越好，第一名加粗，第二个不同数值加下划线；按未舍入值排名，精确并列共享名次。不是多种子统计结论。保持原速度单位及原MAPE定义，不据test调参。

18项模型复现历史33格validation（统一float32容差），2项ST-FiT使用固定随机种子的归档源码/当前源码全量配对验证，差异均0。ST-FiT保留源Gumbel评估采样；历史best没有当时RNG，不能声称逐值重播历史validation。SLPF为用户确认的单阶段源forecast主干适配，STBP为固定图离线适配。

权重在`artifacts/models/main/`；源码/配置快照、validation/test日志、访问标记和逐项报告在`artifacts/evaluation/seed32/<plan_sha>/`。历史validation表单独保留于`paper/validation_seed32/`，不得与test数值混成一张比较表。更早开发阶段test记录仍在研究档案中；本轮不代表这些数据集历史上从未被评估。

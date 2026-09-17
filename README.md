# MoGERNN / ALFNet 实验仓库

当前最终模型是 **ALFNet（Focal）**：原Full架构＋MAE-Focal。模型已锁定，不依据test调参或重新选择。

## 工件与论文

- [权重与配置索引](artifacts/README.md)：最终20份权重，以及正文/附录等293份历史最佳checkpoint的统一目录。
- [论文文件](paper/README.md)：validation/test表、正文/附录草稿、图和文献。
- [当前配置](configs/paper/README.md)：两个数据集seed32的20项配置、专家配置和数据清单。
- [历史归档索引](artifacts/archive_manifest.json)：旧运行代码、配置、日志和报告。原路径可在压缩包`archive/research_history_20260917.tar.gz`中恢复。

数据、原始数据和参考源码分别保留在`data/`、`data_v4/`、`references/`。`artifacts/analyses/`保留正文和附录所用观测支持/轨迹分析原始结果；其原始权重位置可用catalog映射查找。

## 当前test入口

环境：`/home/xd/anaconda3/envs/ReaCH/bin/python`。

```bash
python scripts/eval_release.py check
```

运行身份为`configs/paper/release_test_seed32.json`。队列先对20项已选checkpoint完整复算validation，全部通过才写`TEST_OPENED.json`并执行test。结果在`artifacts/evaluation/seed32/<plan_sha>/`；完成后由`scripts/collect_release.py`生成`paper/test_seed32/`的四张表。已有队列与结果不能重复启动或覆盖。

模型权重原样迁移、逐文件SHA核验。validation的float32重算统一容差为`abs(error) <= 1e-5 + 1e-5 * abs(reference)`，同时要求样本计数、scaler、种子、配置语义和模型状态一致。test使用checkpoint的scaler、teacher forcing=0，指标定义不变。

ST-FiT在eval保留源Gumbel随机采样；历史best未保存当时RNG，故两项ST-FiT改用同一固定随机种子的归档源码/当前源码全量validation配对门槛。其余18项比较历史validation。此处不声称ST-FiT历史随机结果逐值重现，不因test更换模型或权重。

SLPF为源forecast主干的单阶段适配；STBP为固定图离线主干；基线不使用ALFNet独有模块。旧test记录也随历史报告保留，本次是指定冻结模型的test执行，不宣称这些数据集此前从未被评估。

历史训练计划与启动器已归档。若以后需要新训练，需建立新的明确范围和独立运行身份；test计划不授权训练。
# 当前完成状态（2026-09-17）

项目清理与两数据集seed32共20项test已完成。四张主表/阶段表见[最终test结果](paper/test_seed32/README.md)，权重与历史证据见[工件索引](artifacts/README.md)，完成核验见[release_completion.json](artifacts/release_completion.json)。最终模型仍为Focal；不需重复运行已完成队列。

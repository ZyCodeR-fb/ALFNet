# 当前论文配置

2026-10-17：当前最终配置见 `../../paper/pems08_wd_test_20261012/selected_configuration_manifest.json`。
研究计划与各轮超参数／消融配置保留原路径以供复核；以下20项配置描述是历史状态。

`final_seed32/`只保留最终20项运行配置及ALFNet四个专家配置。对应权重/原训练配置/迁移后的配置SHA见`artifacts/final_seed32_manifest.json`。

`v4_data_manifests/`和`v4_dataset_registry.json`保留固定数据、图结构、布局和节点划分身份（其他数据集仍供附录复现）。

`release_test_seed32.json`由`scripts/eval_release.py freeze`生成，为本次明确授权的20项test独立身份，不授权训练。历史训练计划均已归档，不能作为当前训练或test的运行权威。

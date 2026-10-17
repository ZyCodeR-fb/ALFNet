# 定稿后的项目入口

## 写作与绘图

- 正文：`paper/manuscript_draft_cn-2-v5.md`；实验稿：`paper/experiments_draft_cn.md`。
- 汇报：`paper/advisor_report_cn_v5.md`。
- 最终主表／阶段表：`paper/pems08_wd_test_20261012/selected_tables.json` 与同目录 validation/test 工作簿。
- 图2–6、底层 CSV、两种划分：`paper/final_figures_20261017/`。
- 重绘入口：`scripts/plot_final_figures_20261017.py`、`scripts/plot_figure_hparams_20261017.py`；核对入口：`scripts/audit_final_figures_20261017.py`。这些入口使用已保存结果，重绘不需要训练。
- 全量消融保留在 `paper/all_ablations_test_20261010/`、`paper/pems08_wd_ablation_test_20261012/` 等研究目录；正文删除的行仍可分析。

## 模型与权重

- 最终 ALFNet：`model/ALFNet/`、`model/ALFGraph/`、`model/ALFTemporal/`。
- 八个比较模型：UIGNN、GinAR、MoGERNN、SLPF、STGCN、STID、PatchSTG、STBP；`stgcn_n` 是 STGCN 依赖，不能按多余模型删除。
- 最终四份 ALFNet 权重／配置／SHA：`paper/pems08_wd_test_20261012/selected_configuration_manifest.json`。
- 清理时全部保留权重：`archive/cleanup_20261017/plan.json` 中 `weights` 的 `action=retain`；包含 best/last、最终方案的所有超参数与消融，不限于正文结果。
- 最终 flow 仍依赖 `experiments/flow_support_transfer.py` 与 `flow_unified.py`；消融还依赖 `alfnet_ablation_adapter.py`、`mechanism_ablation_20261008.py` 和 `reconstruction_gradient_20261009.py`。这些不是可删除的废弃插件。
- `artifacts/models/main`、旧 catalog 和旧 manifest 是历史身份，不是当前选型权威；部分旧权重已按本次授权删除，查清理清单。

## 补充实验

环境：`/home/xd/anaconda3/envs/ReaCH/bin/python`。通用入口为 `scripts/train_v4.py`；特定支持处理／消融必须沿用对应 study context，不能仅加载 YAML 就宣称复现了相同模型。

历史运行计划固定了源码哈希。模型注册表精简后，旧计划检查可能拒绝运行，这是预期的身份保护；不得绕过检查或篡改旧计划。若需补跑，应保留权重和原配置，创建新的运行身份并重新预检。此次整理未开启训练或 test。

## 归档与清理边界

`archive/cleanup_20261017/sources_and_evidence.tar.gz` 包含整理时的源码、配置、论文资料和小型日志／报告，共6453个文件。成员保持仓库相对路径；建议在独立目录解压查阅，避免覆盖当前代码。该包**不包含已删除权重**；这些权重不可从本次归档恢复。删除名单、原权重哈希及字节数均在 `plan.json`。

删除了非最终的旧 ALFNet 候选权重，以及退出比较集的模型权重和实现；只删除依赖检查未使用的一次性辅助脚本。训练／绘图依赖的历史脚本、配置和结果保留原路径，以免破坏既有证据索引。保留依据不足的旧权重不强行删除。数据、参考源码、论文原稿及完整消融证据均保留。

逐轮探索经验见 [RESEARCH_LESSONS.md](RESEARCH_LESSONS.md)。进一步发布精简仓库可单独制作导出包，不应在本研究工作区删掉复核所需证据。

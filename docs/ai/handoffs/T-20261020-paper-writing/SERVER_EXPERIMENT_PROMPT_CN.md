# 服务器端 Claude 启动提示词：补充对比实验与消融实验

写于 2026-11-03。把下面"提示词正文"整段发给 Ubuntu 服务器上的 Claude Code。发送前请确认：

1. 服务器上的仓库已同步到包含 `paper/latex_neurips_20261020/survey/` 与本文件的版本（这些文件目前只在 Windows 工作区，尚未提交）。
2. 已把以下代码下载到服务器仓库的 `references/`（目录名按下表，保持原样，不要修改其中文件）：

| 类别 | 方法 | 官方仓库 | 建议目录 |
|---|---|---|---|
| 2.2 不完整观测下的时空预测 | Merlin（KDD 2025） | https://github.com/ChengqingYu/Merlin | `references/Merlin/`（仓库中已有，确认是否为最新版即可） |
| 2.2 | HD-TTS（ICML 2024） | https://github.com/marshka/hdtts | `references/hdtts/` |
| 2.3 传感网络演化下的泛化与适应 | STONE（KDD 2024） | https://github.com/PoorOtterBob/STONE-KDD-2024 | `references/STONE-KDD-2024/` |
| 2.3 | STOP（ICML 2025） | https://github.com/PoorOtterBob/STOP | `references/STOP/` |

备用（只在上表某个方法确实无法忠实适配时才用，并需先告诉用户）：2.2 的 BiTGraph（ICLR 2024），https://github.com/chenxiaodanhit/BiTGraph。

---

## 提示词正文

请恢复 ALFNet 论文项目的上下文，然后在本服务器上完成补充对比实验与补充消融实验的代码适配、计划冻结与运行。

### 一、恢复上下文（先读，不要动手）

1. 按 `CLAUDE.md` 的顺序阅读 `docs/ai/WORKFLOW.md`、`docs/ai/PROJECT_CONTEXT.md`、`docs/ai/HANDOFF.md`，以及 `docs/ai/handoffs/T-20261020-paper-writing/HANDOFF.md` 最近四条记录。
2. 阅读本次任务依据：
   - `docs/ai/handoffs/T-20261020-paper-writing/SERVER_EXPERIMENT_PROMPT_CN.md`（本文件）；
   - `paper/latex_neurips_20261020/survey/ARGUMENT_MAP_CN.md`（论文主张与证据缺口，本次实验对应其中 2.2、2.3 没有或只有一个比较方法、SAHC 与 OS 整体作用没有消融两处）；
   - `paper/latex_neurips_20261020/survey/BASELINE_CRITERIA_CN.md` 的 S1–S12 与信息账本；
   - `paper/latex_neurips_20261020/survey/LITERATURE_MATRIX_CN.md` 中 Merlin（C11）、HD-TTS（C12）、STOP（C21）、STONE（E40）的卡片；
   - `paper/BASELINE_FAIRNESS_20260928.md`、`paper/SLPF_STBP_INTEGRITY_20260926.md`、`docs/comparison_model_audit.md`；
   - `experiments/alfnet_ablation_adapter.py`、`model/ALFNet/ALFNet.py`、`experiments/v4_runner.py`、`scripts/train_v4.py`。
3. 当前论文正文以 `paper/latex_neurips_20261020/drafts/manuscript_v6_cn.md` 为准；本任务不修改正文。
4. 检查分支、HEAD、与 origin 的关系、未提交改动与未跟踪文件。若缺少上面任何一个文件，停下来告诉我，不要凭记忆继续。
5. 在 `docs/ai/handoffs/` 下新建任务 `T-20261103-extension-experiments/HANDOFF.md`，并在 `docs/ai/HANDOFF.md` 的任务表登记；之后每个检查点都更新它（按 WORKFLOW 的 Facts / Completed / Modified but unverified / Not started / Ruled out 结构）。

### 二、服务器使用规则（必须遵守）

- 这台服务器有同门在使用。**不要终止、暂停或修改任何不是本任务启动的进程**，不要清理他人的文件、显存或缓存，不要改系统配置、驱动或公共环境。
- 每次启动前用 `nvidia-smi`、`ps` 查看占用，只在显存充足的 GPU 上运行；本任务同时运行的训练或评估进程**最多 3 个**。
- 本任务的进程统一用带前缀的 tmux 会话或 nohup 日志（如 `alfnet_ext_*`），记录 PID、GPU、命令与日志路径；需要停止时只停自己记录过的 PID。
- 运行前检查磁盘空间；输出写到本任务专属目录，不覆盖已有结果、权重与日志。
- 使用项目既有环境（`docs/ai/PROJECT_CONTEXT.md` 记录的解释器为 `/home/xd/anaconda3/envs/ReaCH/bin/python`，以实际可用为准并记录）。
- 不 commit、不 push，除非我明确要求。不修改 `references/`、`data/*/original/`、`data_v4/` 与已有 checkpoint。

### 三、任务 A：四个对比方法的适配

对象：Merlin、HD-TTS（2.2）；STONE、STOP（2.3）。每类两个都要完成，最终在论文中用哪个由我根据完整结果决定。

原则：

1. **优先原样复制，最小改动适配。** 把源模型的核心文件原样复制到 `model/comparison_model/<方法名>/`，在 `SOURCE_MANIFEST.json` 中记录每个文件的源路径与 SHA256（与 `references/` 字节一致）；适配代码单独写在 wrapper 中，只做张量形状、数据接口与协议所需的改动。
2. **只使用公共协议，不得使用 ALFNet 专有的模块或机制**：观测角色、状态进度、生命周期上下文 S、SAHC、ALFTemporal、ALFGraph、FiLM 条件、角色均衡损失、Focal 类损失都不能进入基线。基线与现有八个基线一样，只接收共同的掩蔽输入（不可用值先均值填补再加掩码）、共同图、日历与静态结构信息（见 `docs/comparison_model_audit.md`"Deliberate fairness adaptations"）。
3. **共同协议**：v4 数据加载与划分；train-K 标准化；12 步预测 12 步，步长 1；训练只读已知节点 K 的动态信号与标签，验证与测试分别加入各自的保留节点；评价掩码与指标沿用 `data_provider/v4_metrics.py`；检查点按验证集 33 格中的 LifeAvg MAE 选择，严格改进、平局取较早轮次。
4. **节点身份参数**：与现有五个基线一致，把逐节点可训练的身份参数替换为共享结构编码（`StructuralNodeEncoder`）；若该参数是方法核心、替换会实质改变方法，先记录并告诉我，不要自行决定。
5. **逐方法的注意事项**：
   - **Merlin**：保留教师—学生蒸馏与多视图对比学习；教师只能使用训练节点 K 内、被合成遮蔽的自然观测，不得读取验证或测试保留节点的信号（参照 `paper/BASELINE_FAIRNESS_20260928.md` 中 2026-09-28 的审计）。2026-10-01 曾有一版适配，权重已在 2026-10-17 清理时删除；可在 git 历史与 `archive/` 中查找旧 wrapper 与 `scripts/train_merlin_v4.py` 作参考，但必须重新审计，不能直接沿用。作者曾以"当时训练结果无效"把它移出正文，请在适配时特别记录与源实现的差异。
   - **HD-TTS**：官方 `experiments/run_realworld.py` 在切窗前对整条序列做 `ffill().bfill()`，会读到窗口之外甚至预测时刻之后的值；必须改为只在当前 12 步输入窗口内处理缺失。官方交通配置为 24→12，改为 12→12。检查模型是否含逐节点参数（模型构造时接收 `n_nodes`），按第 4 条处理。
   - **STONE、STOP**：源设置是按年份划分的分布外实验，改用本项目的划分与协议；推理时不做任何参数更新。若源方法需要额外的年份数据或语义图预计算，只能在训练节点 K 的训练时段内计算，并记录。
6. **超参数与预算**：模型结构超参数按源代码中最接近的数据集配置；训练预算与现有基线一致（最多 100 轮、耐心值 10，按验证选点），源方法必需的多阶段训练按阶段保留并记录。不在测试集上调参。
7. **验证**：每个 wrapper 用小规模合成数据做 CPU 前向—反向冒烟测试；做信号泄漏检查（训练批次中不出现验证或测试保留节点的动态信号与标签，标准化统计量只来自训练 K）；在 `docs/comparison_model_audit.md` 增加四行，写明保留内容与全部适配差异。

### 四、任务 B：两项整体消融

基准为当前最终 ALFNet：METR-LA、PEMS-BAY、PEMS04 用 `paper/final_scheme_20261006/` 的最终配置；PEMS08 用 `paper/pems08_wd_test_20261012/selected_configuration_manifest.json`（权重衰减 1e-3）。消融只改被检验的部分，其余设置、随机种子 32、训练预算与配对初始化都与基准一致。实现方式沿用 `experiments/alfnet_ablation_adapter.py`：在研究专用 adapter 中新增干预键，生产模型代码不改。

1. **w/o SAHC（整体剔除共享补全）**：去除图循环补全、窗口内插值与参考中心化三步，后续模块直接读取掩蔽后的输入（不可用位置为 0，即训练 K 的均值），生命周期上下文 S 照常提供，训练目标不变。
   - 已有的 `no_imputer` 只把补全器换成零填补（`ObservedOnly`），保留了插值与中心化。METR-LA、PEMS-BAY 的插值与中心化本来就是关闭的（ι=0、α=0），若核实其配置、代码与基准身份一致，可以复用 2026-10-10 的已有结果（`paper/all_ablations_test_20261010/`），并记录复用依据；否则重跑。
   - PEMS04（插值开启）与 PEMS08（权重衰减已更新）必须按新定义重跑。
2. **w/o OS（整体剔除 OS 条件化）**：模型的所有可学习计算都不再接收观测角色、状态进度与掩码，包括 SAHC 递推与融合的输入、ALFTemporal 的输入、ALFGraph 编码器的输入与路由器（路由改为与输入无关的全局可学习权重，与已有 `static_router` 一致）、输出修正的角色倍率（改为与角色无关的常数，沿用已有 `mask_only` 的均值倍率约定，并记录）。掩码只保留两种非学习用途：补全时保留观测值，以及损失与评价中的有效目标掩码。训练目标（含角色均衡项）保持不变，以便只检验前向计算中的条件化；这一点要在结果说明中写明。
3. 可选（先问我，确认后再跑）：`mask_only`（去掉角色与进度、保留掩码作为输入），用于检验观测角色与状态进度在掩码之外的作用。

### 五、计划冻结、运行与评价

1. 为本任务建立独立的冻结计划（`configs/paper/extension_<日期>/plan.json`），记录全部配置、代码与数据的 SHA256、种子、输出目录与复用项；先做 preflight（每个运行一个批次的前向—反向、暴露范围与身份检查），全部通过后再进入正式训练队列。
2. 运行数量：对比方法 4 个 × 4 个数据集 = 16 个；消融 w/o OS 4 个，w/o SAHC 2–4 个（视复用结果而定）。同时最多 3 个，按队列依次运行，失败的运行保留日志，不静默替换。
3. 若出现显存不足或数值发散，先停下记录现象；恢复方案（如减小批大小、降低学习率）必须在看到恢复后的结果之前写入计划，并对同一方法的全部数据集一致执行。
4. 全部训练完成、按验证集选定检查点并冻结后，对每个最终检查点做**一次**测试评价（本任务的测试访问由用户于 2026-11-03 授权，仅限这些运行）。测试结果不得用于选择检查点、调整超参数、更换方法或决定是否重跑。
5. 每类两个候选的结果都完整保留并报告；不要因为某个方法优于或劣于 ALFNet 而删除、重跑或改变其配置。最终在论文中报告哪些方法，由我决定。

### 六、产出与汇报

- 结果目录：`paper/extension_baselines_<日期>/` 与 `paper/extension_ablations_<日期>/`，各含 README、`results_and_provenance.json`、验证与测试的 33 格长表与 LifeAvg 汇总表。另给出与现有八个基线和 ALFNet 合并后的对照表（现有数值取自 `paper/pems08_wd_test_20261012/selected_tables.json`，只读不改）。
- 适配差异写入 `docs/comparison_model_audit.md`；任务状态写入新任务的 HANDOFF。
- 中途检查点汇报：适配与冒烟测试完成后、preflight 完成后各汇报一次（用中文，简短），然后继续运行，不必等我确认，除非遇到上面要求先问我的情况。
- 全部完成后用中文汇报：每个方法的适配差异与有效性风险；每个运行的状态；验证与测试的 LifeAvg 三项指标与 33 格第一名统计；两项消融相对完整模型的变化；任何与预期不符或需要我决定的事项。不修改论文正文。

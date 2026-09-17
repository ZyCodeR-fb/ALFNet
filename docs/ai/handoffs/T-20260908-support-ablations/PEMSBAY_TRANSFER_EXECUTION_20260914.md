# PEMS-BAY两候选×seed32/42执行

用户明确授权恰好4run：spectral32、linear32、spectral42、linear42。计划
`34ecd952b01ebbad540a1345260d2793f37e4b71f743c817cc492e6c410b3a0a`，
`configs/paper/pemsbay_focal_transfer_v1.json`。
输出：`outputs_v4/pemsbay_focal_transfer_v1/34ecd952b01ebbad540a1345260d2793f37e4b71f743c817cc492e6c410b3a0a/`。

## 范围

PEMS-BAY原Full AR_BASE底座，patch容量11、batch64/workers0，原训练预算100epoch/TF40/早停从40/patience10。Focal+0.1频域复数L1与Focal+444参数历史修正分别迁移，不叠加、不调强度。四个从头训练，最大4并发，显存准入。原Full保留，不开test，不额外加Full42训练。

Full32对照已核验CPU checkpoint元数据、所选表与run_report一致（epoch37，LifeAvg3.039729/4.637845/7.996064）。旧development_base/final_comparisons seed42配方不等价，不能当当前Full42。seed32对Full32判冻结门槛，seed42只做两候选的同seed比较；结果不能宣称两种子均超过Full。

## 已完成 / 执行中

- `python -m pytest tests/test_focal_last.py tests/test_mae_tail_losses.py tests/test_pemsbay_transfer.py -q`：21 passed。未跑全仓库套件。
- builder freeze/--check、入口编译、四配置/seed/patch11/预算及git diff --check通过。模型/损失代码无变化，train_v4仅新增独立计划注册；原dirty保留。
- `python -u scripts/preflight_pemsbay_focal_transfer.py`运行中。旧Full8验证窗口预测差0；两个seed各自两候选共同参数/RNG/初始预测/exposure/scaler与target-blind检查通过。std9.048990仅训练K拟合；seed32首批136完整向量/1632位置，seed42为120/1440（不是全训练覆盖率）。
- 当前已测GPU batch64预留约15.72GiB。4CPU/4GPU单批全部完成后才准许并发探针；本节不提前宣称全部通过。预检日志`/tmp/pemsbay_focal_transfer_preflight_20260914.log`。

Python固定 `/home/xd/anaconda3/envs/ReaCH/bin/python`，实际cwd/源码均为根仓库。输出内snapshot只作冻结备份。旧METR-LA已完成计划因入口注册变更可出现source drift，不刷新旧plan，其历史完成证据与snapshot保留。

## 待完成顺序

17:40最终核验：launcher3300858，spectral32/linear32/spectral42/linear42的PID分别
3300898/3301018/3301482/3301617，均存活且日志有正式train_one_epoch开始事件。
GPU利用率96%，各16794/16786/16794/16776MiB，剩余29083MiB。
report.md/candidate_summary.json及源码snapshot均已生成。四正式训练已启动，禁止重复启动。

更新：全部4CPU/4GPU预检通过，四并发16批探针全部exit0。
训练段spectral32 39.973s、linear32 39.174s、spectral42 39.909s、linear42 38.863s；
各峰值预留15.7305GiB。实际训练启动tmux `pemsbay-focal-transfer-20260914`，
命令 `python -u scripts/run_pemsbay_focal_transfer.py`，启动器日志
`/tmp/pemsbay_focal_transfer_launcher_20260914.log`。下面顺序保留作历史操作说明，
不应重复执行；最新运行状态以launcher_status.json为准。

1. preflight.json passed，运行 `scripts/run_pemsbay_focal_transfer.py --probe`（4同步进程各16训练批+2验证批）。
2. concurrency_probe.json passed，tmux启动 `scripts/run_pemsbay_focal_transfer.py`。
3. 核对4worker进入train_one_epoch，记录PID。launcher_status.json实时状态，candidate_summary.json/report.md自动汇总；不自动晋升或增加训练。

Git paper/final HEAD9346dc0/upstream0:0。初始GPU只读检查约94GiB空闲，没有操作他人PID1734562或任何其他进程；未打开test、提交或推送。

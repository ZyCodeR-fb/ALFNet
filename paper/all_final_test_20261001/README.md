# 全量test运行入口

冻结计划：`configs/paper/all_final_test_20261001/plan.json`。
SHA：`e696f34276f99d1dba825cac9a90f430959fcb85e381328c0c93a9d2d685e3c1`。
117个seed32权重：67个ALFNet配置＋50个比较算法；23复用、94补测。

启动命令：ReaCH Python `scripts/test_all_final_20261001.py queue`。
持久会话：tmux `all-final-test-20261001`。
输出：`artifacts/evaluation/all_final_test_20261001/<上述SHA>/`。

- `queue_status.json`：117项实时状态；优先登记复用结果，再运行基线和ALFNet补测。
- `<key>.log`：单项validation/test日志。
- `validation/<key>.json`：完整validation核验。
- `opened/<key>.json`：通过核验后该权重的test开启记录。
- `<key>.json`：成功test结果或经核验的历史复用结果。
- `ATTENTION_REQUIRED.json`：失败清单，未失败项继续处理。
- `COMPLETE.json`：全部117项及最终表已汇总成功。

最多5个任务，空闲显存低于20GB时暂缓派发，不动其他用户进程。复用项不执行GPU推理。已有权重不重新训练。当前源文件与旧表被冻结，运行期间勿修改已入hash清单的文件。

离线收集脚本`scripts/collect_all_final_test.py`已用临时模拟117条记录进行端到端测试：两套主/阶段表3630指标及排名回读通过。模拟数据从未写入实际结果目录。真实validation表1815格已单独生成。测试阶段完成前不将计划输出称为已完成test结果。

选择依据和输出区分见[VALIDATION_ANALYSIS.md](VALIDATION_ANALYSIS.md)；授权范围及预先写定的探索规则见[PLAN.md](PLAN.md)。

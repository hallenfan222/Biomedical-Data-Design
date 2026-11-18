# 调参过程记录（实际决策时间线）

2026-11-17，开跑前：
目标是在固定 NK/CD4/CD8/Treg sparse-cell 四折上降低等权平均 MRRMSE。最多 15 个不同参数配置（含 baseline），同一时刻只用一张 GPU。不用 Kaggle 分数。
原仓库 HEAD a1164ba 含 Notebook 合并冲突；提取有效历史 f864556，split_data.py 与当前 HEAD 字节一致。原始数据经 SHA-256 校验。既有 XGBoost 3.2.0/CUDA 环境保留，不安装升级。

阶段 0：运行检查
提交 Job 55819475：T00 baseline，仅 NK 折。只检查数据、GPU、输出 shape、时间和采样峰值显存，不以这一个折做选参。GPU 排队期间准备报告脚本。尚未依据任何分数做选择。

预定阶段 1：结构/正则单变量探索（T00–T07）
T00：120轮、lr=.06、depth=3、child=1、lambda=1、subsample=.9、colsample=.9。
T01/T02：只把 depth 改成 2/4；检验更浅或更深的树。
T03/T04：只把 min_child_weight 改成 4/8；检验抑制小叶节点是否改善泛化。
T05/T06/T07：只把 reg_lambda 改成 .1/10/30；检验弱/强 L2 叶权重正则。
固定 identity、PCA160、one_output_per_tree、seed42、max_bin256，其余沿用 baseline。四折结果完成后再决定下一阶段，不执行全网格。

余下 7 个配置预留给结构组合、learning_rate/rounds 联合测试和两种采样比例。实际安排依据短作业成本和阶段 1 结果，修改会在此记录。

阶段0排队补充：Slurm预计开始推迟到2026-11-17 16:12:36（集群PST，即19:12:36 EST）。短作业仍为PENDING；将同一job的TimeLimit由20分钟调为10分钟，尝试利用backfill空档。未取消/重复提交，未改模型参数或GPU数量。

阶段0通过（2026-11-17 23:51 UTC）：NK数据485训练/129验证，152维identity，PCA160，恢复预测129x18211。CUDA断言cuda:0通过，分数1.1541201221。训练10.073秒，模型阶段10.628秒，预处理1.531秒；采样本进程显存峰值432MiB。冷启动/共享存储加载约数分钟，不计作fit时间。无失败配置。
据此保留15组预算；先运行完整四折T00-T07。baseline也从头用同一入口重跑全部四折，短作业分数不用于排名。预计阶段1计算数分钟至十余分钟，为启动及负载留余量设置20分钟作业上限。

## Phase 1 complete / Phase 2 decision — 2026-11-18T00:03:01.081415+00:00

Stage1 job 55819987 COMPLETED 0:0; elapsed 4m10s. All eight trials have four folds. T00 1.032013213; T01 depth2 1.069300905; T02 depth4 1.012701066; T03 child4 1.198153079; T04 child8 1.719129359; T05 lambda0.1 1.014485977; T06 lambda10 1.098548083; T07 lambda30 1.129575539.
Decision: T08 combines depth4 and lambda0.1, retaining child1. The interaction is untested until T08 finishes; gains cannot be assumed additive. T09=(0.03,300), T10=(0.06,300), T11=(0.03,600), T12=(0.1,120) fix the T08 structure. T08 itself is (0.06,120). These permit fixed-rate rounds and fixed-rounds rate comparisons as well as joint pairs. Two slots remain for sampling. No full grid or early stopping. Estimated model time about 6 minutes; request 15 minutes including startup margin.
Operational incident: a Python string-newline SyntaxError prevented the first configuration write. Job 55820225 was submitted after that failed preparation and immediately cancelled while pending. No model trial was evaluated by this attempt. The setup was corrected and configuration validation is required before resubmission.

## Phase 2 complete / Phase 3 decision 2026-11-18T00:13:21.088298+00:00
Job 55820238 COMPLETED 0:0, elapsed 7m37s. T08=1.010803116 (36.92s); T09=1.007401617 (91.44s); T10=1.030623053 (91.74s); T11=1.029222030 (184.16s); T12=1.020850653 (40.02s).
Decision: keep T09 (lr0.03/300 rounds/depth4/lambda0.1/child1) as the parent. Higher learning rates or 600 rounds did not win among the tested configurations. T09 improves NK/CD4 relative to baseline but worsens CD8 and very slightly Treg; no universal cell-type improvement is claimed.
T13 changes only subsample 0.9 -> 0.7; T14 changes only colsample_bytree 0.9 -> 0.7. Lower sampling tests whether extra randomization reduces overfitting. Both are compared against T09. Request 10 minutes; estimated model time ~3 minutes. Exactly 15 distinct configurations; no more search.
Preparation note: the first stage3 setup attempt had the same string-newline formatting error before any file mutation or submission. Replaced that construction with print calls, validated saved configurations, then submitted.


Final-stage scheduling adjustment: while job 55820747 was pending for resources, its time limit was shortened from 10 to 7 minutes to improve backfill eligibility. Two trials are estimated at ~3 minutes of model work; 7 minutes includes margin for the cold startup observed in the smoke test. Same job ID, no resubmission and no parameter changes.

## Search complete 2026-11-18T00:23:28.285939+00:00
All 15 configurations completed four folds. Recommended T09 mean 1.00740161673428 baseline 1.0320132131701798 improvement_pct 2.3848140820112977
Final stage job 55820747. No additional tuning trials will run. See CONCLUSION.md for controlled comparisons and fold-level tradeoffs.

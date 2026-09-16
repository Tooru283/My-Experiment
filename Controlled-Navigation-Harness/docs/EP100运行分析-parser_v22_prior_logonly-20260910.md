# EP100 运行分析：parser v2.2 + candidate prior log-only（2026-09-10）

## 1. 结果边界

本报告分析完整运行 `parser_v22_prior_logonly_ep100_20260909_234512`：`val_unseen` 100/100，已写出正式 aggregate 文件并退出。

关键配置：

- 指令规划：`opennav.instruction_plan.v2.2.two_stage`
- progress provider：`acn_l1`
- candidate prior：`ENABLED=true`、`LOG_ONLY=true`、`MIN_PROB_GAP=0.0`
- TerminalRecheck：最多两次，每次把移动限制为 0.25m

正式统计文件：

- `logs/eval_results/ep100/20260909/parser_v22_prior_logonly_ep100_20260909_234512/stats_ckpt_val_unseen.json`
- `logs/eval_results/ep100/20260909/parser_v22_prior_logonly_ep100_20260909_234512/stats_ep_ckpt_val_unseen_r0_w1.json`

## 2. 总体指标

| 运行 | SR | OSR | SPL | nDTW | 最终距离 | steps | path length | collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parser v2.2 + prior log-only | **0.22** | 0.30 | **0.1244** | **0.3651** | **7.4834** | 10.04 | 13.2442 | 0.0690 |
| target_binding v3.3 retry | 0.14 | **0.31** | 0.0871 | 0.3361 | 7.8645 | 7.74 | 11.7639 | **0.0426** |
| terminal_track v3（2026-09-02） | 0.20 | **0.38** | 0.1098 | 0.3416 | 7.6843 | 9.70 | 14.8801 | 0.0620 |

相对上一轮 v3.3，SR 增加 8 个百分点，SPL 增加 0.0373，nDTW 增加 0.0290，最终距离缩短 0.381m；OSR 减少 1 个百分点，碰撞率增加 0.0263。相对 9 月 2 日运行，SR 增加 2 个百分点，OSR 减少 8 个百分点。

这不能解释为整体导航能力提高 8 个百分点。主要增量来自原来因解析失败而没有正常运行的 episode。

## 3. 解析恢复与行为变化分解

上一轮有 24 个 episode 发生 episode-local 解析失败，本轮 100 项全部解析成功。

| episode 集合 | 数量 | 本轮 SR | 上轮 SR | 本轮 OSR | 上轮 OSR |
|---|---:|---:|---:|---:|---:|
| 上轮解析失败、本轮恢复 | 24 | 6/24 | 0/24 | 9/24 | 0/24 |
| 上轮已能解析 | 76 | 16/76 | 14/76 | 21/76 | 31/76 |

SR 的 `+8` 可拆为解析恢复贡献 `+6`，原可运行集合贡献 `+2`。OSR 的 `-1` 可拆为解析恢复贡献 `+9`，原可运行集合退化 `-10`。

在两轮 actions 和 landmarks 文本完全相同的 62 个 episode 中，本轮 SR 从 10 增至 13，但 OSR 从 25 降至 17。该子集仍同时包含 candidate prior、TerminalRecheck 和其他运行时代码变化，不能作为单变量因果实验；它足以说明 OSR 下降不能只归因于解析文本变化。

固定十项 `244、218、226、371、602、1092、748、116、513、11` 的 SR/OSR 仍为 `0.20/0.40`，与上一轮相同，低于 9 月 2 日的 `0.40/0.70`。本轮固定十项最终距离从 7.276m 降至 5.340m，nDTW 从 0.302 升至 0.486，但没有转成更多成功。

## 4. 两阶段解析与确定性终点绑定错误

- 100/100 产生 `instruction_plan_parsed`，没有 `instruction_plan_failed`，也没有全局崩溃。
- action 阶段 100 项均首次通过。
- landmark 阶段 87 项首次通过，13 项第二次通过。
- 13 次重试全部因为无显式 stop/wait 的指令返回了非空 `terminal_target`，确定性校验要求改为 `null`。

100 项中有 14 项没有显式终止子句，全部接受 `terminal_target=null`；其中 11 项的 AnchorChain 终点与 VisualTargetVerifier 正式终点不一致。计入显式终点后，`terminal_policy.target` 与 verifier `required_landmark_terms` 共出现 **14/100** 项不一致：11 项隐式终点、3 项显式终点。

根因是第二阶段给出的 terminal target 没有作为独立字段传入 AnchorChain。系统只保留扁平 landmarks 文本，AnchorChain 随后又从动作字符串和地标顺序猜一次目标。

代表性错误：

- `Go alongside the pool towards the bar`：模型首次给出 `terminal_target=bar`，校验要求重试为 `null`；AnchorChain 取 `pool`，verifier 取 `bar`。
- `Stop when you get into this room`：第二阶段目标是 `gym`，AnchorChain 从 `this room` 重新猜成 `bathroom`。
- `wait there`：第二阶段目标是完整 frame 或 bathroom，AnchorChain 分别重新猜成 hallway 或 first door。

这是确定性的跨模块目标绑定错误，不是 Qwen confidence 问题。

## 5. SR、OSR 与正式 STOP

| 终止方式 | 数量 | SR | OSR |
|---|---:|---:|---:|
| step length limit | 97 | 21 | 29 |
| 正式 STOP | 3 | 1 | 1 |

22 个 SR 全部在终止时位于 3m 内，其中 21 个由步数上限结束，只有 1 个来自正式 STOP。OSR-SR 缺口由上一轮的 17 缩小为 8，说明进入成功半径后再离开的情况减少；这不能等同于 STOP 完成率提高。

正式 STOP 共 60 次提案：selector 提案 55 次并全部拒绝；progress-completion 提案 5 次，3 次提交、2 次拒绝。最终 `route_progress_complete=13`、`terminal_target_confirmed=6`、`goal_complete=3`。

3 次 goal STOP 提交只有 1 次成功：

| episode | step | 最终距离 | SR | 关键问题 |
|---|---:|---:|---:|---|
| 166 | 5 | 10.948m | 0 | 近处方向局部表面被当成 lamp 实例 |
| 1117 | 7 | 7.137m | 0 | AnchorChain 目标是 pool，verifier 目标是 bar；终点确认为空仍得到 goal complete |
| 824 | 10 | 2.847m | 1 | 正式提交成功 |

EP1117 在当前方向局部表面深度 6.034m、`distance_satisfied=false`、TerminalEvidenceMemory 未确认时仍提交。直接原因是该指令被标成隐式终点，ProgressProvider 允许 route complete 单独推导 goal complete。StopCoordinator 的 progress evidence 随后写成“路线进度和终点均确认”，与原始 ProgressUpdate 不一致。

EP166 属于另一层错误：内部证据链在错误目标实例上自洽。其局部表面深度约 1.01m、连续实例轨迹为 3 帧，但官方终点仍在 10.95m 外。全部 1004 个 `terminal_spatial_evidence` 继续使用 `direction_local_surface_not_instance_mask`，目标实例 mask 深度 P0 尚未解决。

## 6. JSON 重试、TerminalRecheck 与关系门控

STOP 当前视图视觉 JSON 共 1004 次：首次成功 758 次；触发一次精简重试 246 次，占 24.5%；重试救回 232 次，占重试项 94.3%；两次仍失败 14 次，占全部调用 1.39%。13 次最终失败为空响应，1 次为 JSON 缺少逗号。重试机制避免了崩溃，但首次输出稳定性没有改善。

TerminalRecheck 实际执行 8 次，覆盖 6 个 episode；每次执行距离均为 0.25m，第一次 hold 6 次、第二次 hold 2 次。6 项中最终 3 项 SR、4 项 OSR。样本太小且与其他改动同时发生，不能单独归因。

关系型终点覆盖 18 个 episode，共检查 182 次：64 次满足、118 次不满足。正式 STOP 场景中共 18 次关系检查，5 次满足、13 次不满足；18 次 STOP 全部拒绝。13 次不满足均作为正式 blocker 参与拒绝，因此关系字段已经进入正式门控。5 次关系满足后仍被其他当前方向证据阻止，尚无关系型正式 STOP 成功样本。

## 7. Candidate prior 状态

Candidate prior 在 harness trace 中完成 1004 次打分，全部为 `log_only=true`，没有动作覆盖。navigation JSONL 不含该事件，因为当前实现只写 harness trace；不能据此判断模块没有运行。

以本轮实际最终移动选择做反事实比较，992 次选择可与 prior 分数对齐：

| probability gap 阈值 | 预计覆盖数 | 覆盖率 |
|---|---:|---:|
| > 0 | 685 | 69.1% |
| > 0.05 | 452 | 45.6% |
| > 0.10 | 270 | 27.2% |
| > 0.20 | 52 | 5.2% |

所以 `MIN_PROB_GAP=0` 仍是强干预配置。当前日志没有未执行候选对应的真实目标距离收益，不能判断 prior top 是否优于 Qwen 选择。softmax gap 只是内部排序差，不是校准成功概率；waypoint 几何距离也不能替代目标距离。

关闭 prior 动作覆盖后，上轮已可运行的 76 项 SR 从 14 增至 16，OSR 从 31 降至 21。这与“prior 可能帮助进入成功半径、但不保证最终留在半径内”一致，但其他代码也有变化，仍需单变量配对实验。

## 8. 动作事件 P0

最终 ProgressUpdate 中仍有 30 个 episode 卡在未实现动作事件：`pass` 13 项、`cross` 9 项、`traverse` 8 项，去重并集 30 项。它们已影响正式进度，而不只是日志字段。

## 9. 下一轮优先级

1. 将 resolved terminal target 作为独立契约字段，从两阶段解析一直传到 AnchorChain、ProgressProvider 和 VisualTargetVerifier，停止从扁平字符串重新猜目标。
2. 无显式 STOP 的最后动作也必须形成隐式终点；有终点目标时，`goal_complete` 必须要求 route complete 和 terminal target confirmed 同时成立。
3. 加入不变量检查：有终点目标时，`goal_complete=true` 不允许 `terminal_target_confirmed` 为 false 或 null。
4. progress-completion STOP 必须要求当前空间距离满足或有效 TerminalEvidenceMemory 支持，修掉 EP1117 类型的远距离误停。
5. 让动作目标按语义绑定：`towards bar`、`into bathroom`、`until frame` 不能选择动作中的第一个 landmark。
6. 继续完成目标实例 mask 深度，以及 cross/pass/traverse 动作事件验证。
7. candidate prior 暂时保持 log-only。修复上述确定性错误后，再用同一 parser、同一代码和同一 episode 做单变量 A/B；不要以 `MIN_PROB_GAP=0` 直接恢复动作覆盖。

下一轮回归至少加入 EP1117、176、568、677，并保留原固定十项。

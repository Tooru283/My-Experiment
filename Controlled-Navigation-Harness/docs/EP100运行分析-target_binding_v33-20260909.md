# EP100 运行分析：target_binding_v33（2026-09-09）

## 1. 运行身份与结论

- 实验：`target_binding_v33_ep100_retry_20260909_113356`
- 数据：`val_unseen`，100 episodes，seed 0，单环境
- 运行时间：2026-09-09 11:34:25 至 21:23:29，共 9:49:04
- 导航记录：`logs/navigation_records/ep100/20260909/target_binding_v33_ep100_retry_20260909_113356_train_navigation_20260909_113412.jsonl`
- 官方汇总：`logs/eval_results/ep100/20260909/target_binding_v33_ep100_retry_20260909_113356/stats_ckpt_val_unseen.json`

本次 100 条全部完成。此前单条指令解析异常导致整次评测崩溃的问题已经消失：24 条无法通过计划校验的 episode 被局部终止并记为失败，其余 episode 继续运行。

当前结果首先受计划解析失败污染，其次暴露出目标实例距离、动作事件验证和候选先验三个问题。它不能单独用于判断 STOP 新链路已经有效。

## 2. 结果总览

| 口径 | episodes | SR | OSR | SPL | nDTW | 最终距离 | 路径长度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 官方全量 | 100 | 0.1400 | 0.3100 | 0.0871 | 0.3361 | 7.8645 m | 11.7639 m |
| 计划有效子集 | 76 | 0.1842 | 0.4079 | 0.1146 | 0.3439 | 7.6711 m | 15.4788 m |
| 固定 10 条回归集 | 10 | 0.2000 | 0.4000 | 0.0937 | 0.3018 | 7.2757 m | 14.7553 m |

固定回归集仍只有 EP244、EP116 成功；EP602、EP11 曾进入 3 m 后离开。EP513 本轮没有进入 3 m。固定 10 条与最近完整 EP10 的 SR=0.20、OSR=0.40 相同，没有显示出在线收益。

## 3. 两阶段指令解析

100 条中有 1 条命中缓存，另外 99 条重新解析：75 条成功，24 条失败。解析失败率为 24.24%，这 24 条均在第一步局部终止，SR/OSR 都为 0。

| 结果 | episodes | 调用情况 |
|---|---:|---|
| 成功 | 75 | 56 条两次调用成功；19 条经过一次阶段重试后成功 |
| 失败 | 24 | 19 条 landmark 阶段失败；5 条 action 阶段失败 |

landmark 阶段的 19 条失败可归为：`there` 指代 4 条，`once/when` 事件从句 4 条，句首介词位置短语 5 条，terminal target 必须位于 landmarks 末尾 2 条，其余目标绑定差异 4 条。action 阶段的 5 条均为“最终停止从句必须逐字复制”的校验失败。

这些错误发生在导航开始前，是生成结果与确定性校验器之间的语法/语义契约冲突，并非环境中“没有找到目标”。当前 episode 级隔离是正确的容错边界，但不能接受 24% 的样本被自动清零。下一步应让程序区分位置目标、关系参照物、时间触发条件和代词，并通过确定性归一化完成排序和指代解析；仍无法确定时才判该 episode 的计划无效。

## 4. 当前视图视觉 JSON 重试

共记录 774 次当前全景视觉证据调用：

- 599 次首次解析成功；
- 175 次触发一次精简重试，占 22.61%；
- 重试挽救 165 次，挽救率 94.29%；
- 10 次重试后仍失败，占全部调用 1.29%。

首次成功调用平均 6.67 秒；触发重试后平均 40.58 秒。重试机制有效避免了大部分证据丢失，但首次输出不稳定仍造成明显运行开销。整次运行中，普通视觉证据调用平均 17.01 秒，当前视图 STOP 证据调用平均 14.33 秒，是主要耗时来源。

## 5. STOP、ACN 与官方指标

62 次 STOP 决策包含：24 次计划失败的强制终止、34 次 selector STOP 提议、4 次 progress-completion STOP 提议。34 次 selector 提议全部被拒绝；4 次 progress-completion 提议全部提交。

4 次正式 goal STOP 中只有 EP259 的官方 SR 为 1。EP321 最终距离 3.645 m，EP166 和 EP643 的最终距离分别为 11.647 m 和 12.114 m。四次提交所用距离口径都是 `direction_local_surface_not_instance_mask`。这直接支持“目标实例 mask 深度”仍为 P0：方向局部表面很近，不能证明目标实例本身很近。

ACN 最终状态中，`route_progress_complete` 为真的有 9 条，`terminal_target_confirmed` 为真的有 7 条，`goal_complete` 为真的只有 4 条。官方有 14 条 success，但其中 13 条的 ACN route 尚未完成。官方 SR 只表示 episode 结束时距离目标点不超过阈值，不表示动作链和 STOP 已经完成；这 13 条不能解释成 ACN goal success。

TerminalRecheck 实际执行了 10 次 0.25 m 受限移动，说明拒绝合理 STOP 后的复查链已经在线生效。它只能改善观察位置，不能修复错误的实例距离口径。

## 6. 关系门控与动作事件

日志有 125 次关系检查，其中 77 次满足；进入正式 STOP 判定的 9 次关系检查全部满足。关系字段已经实际进入验证链，不是只停留在解析 JSON 中。正式样本只有 9 次，尚不足以证明关系判断准确。

episode 结束时仍有 27 条卡在未实现的动作事件验证上：

| 未实现验证器 | episodes |
|---|---:|
| `pass` | 12 |
| `cross` | 8 |
| `traverse` | 7 |

此外还有 `predicate_not_satisfied` 17 条、房间观察不可比较 8 条、要求的转向未到达 6 条。`cross/pass/traverse` 必须用动作前后状态变化和位移事件验证，不能只靠单帧目标可见性。

## 7. 候选先验

本次配置为 `CANDIDATE_PRIOR.LOG_ONLY=false`、`MIN_PROB_GAP=0.0`。候选先验改写了 526/774 个总步；排除 24 条计划失败的强制终止后，在 750 个正常导航步中改写 512 步，占 68.27%。其中 177 次的策略概率差不超过 0.05。

仅作离线诊断时，先验改写步的 oracle 距离增益均值为 -0.008 m，未改写步为 +0.490 m；正增益比例分别约为 49.8% 和 69.3%。两组步的状态分布不同，这不是因果结论，也不能把 waypoint distance 当目标距离，但足以说明当前零阈值先验过强。下一轮在线归因实验应先将候选先验设为 `LOG_ONLY=true`，或单独进行预注册 A/B，避免它继续覆盖大多数基础策略动作。

## 8. 修复与验证顺序

1. 修复两阶段解析的确定性契约：解析 `there`、时间从句和介词位置短语；程序化保证完整 terminal target 位于 landmarks 末尾；最终目的地晚于中间 `stop/wait` 时，不把中间停顿误认成最终目标。
2. 先对本次 100 条指令做 parse-only 回放，要求不再出现确定性 validator 拒绝，再跑固定 10 条。
3. 实现目标实例 mask 深度，正式 STOP 禁止使用方向局部表面距离替代实例距离。
4. 实现 `cross/pass/traverse` 的动作事件验证。
5. 将候选先验切到只记录模式做对照；关系门控与 TerminalRecheck 保持当前配置，避免一次改变多个决策面。
6. 上述检查通过后再跑新的 EP100；不要把本次 14% SR 当作新 STOP 链路的稳定结果。

## 9. 下一轮 EP100 准备状态（2026-09-09 夜间更新）

已根据本报告完成以下修改：

- 指令计划升级为 `opennav.instruction_plan.v2.2.two_stage`，缓存指纹随之变化，不读取 v2.1 旧计划；
- 程序从原指令确定性解析 `there`、`once/when`、句首介词位置和中间停顿，自动把完整 terminal target 放到 landmarks 末尾；
- STOP 校验按语义要求分组：允许 `Stop ... That's where you will wait` 合并为一个动作，仍拒绝把 EP513 的一个源 STOP 拆成两个 STOP 动作；
- `Walk across the floor` 的 motion 语义直接来自动作文本，不再依赖 Qwen 是否把 `floor` 放入 landmarks，首帧不会再以 `unknown + skip` 完成路线；
- 中间 `stop/wait` 后仍有路线动作时，不再把中间位置写入正式 terminal policy；
- unsupported action verifier 在选路反馈中改为明确的“继续实际执行原动作”，不再要求模型转向或返回以尝试修复不存在的验证器；
- `CANDIDATE_PRIOR` 改为 `ENABLED=true, LOG_ONLY=true`，保留诊断分数但不再改写导航动作。

用本机 Qwen 对同一批 100 条原始指令进行了不启动 Habitat 的实时两阶段预检：100/100 计划有效。action 阶段 100 条全部首次通过；landmark 阶段 87 条首次通过、13 条一次重试后通过，共 213 次调用。预检记录位于：

`logs/parse_only/ep100/20260909/instruction_plan_v22_preflight_20260909.jsonl`

现有完整单元测试为 155/155 通过，Python 语法检查和 `git diff --check` 通过。新缓存路径为：

`cache_files/R2R/actions_cache_9f036d67651f0a59.json`

该缓存当前不存在，固定 EP10 会按新提示和新解析器实时生成并记录计划。

目标实例 mask 深度仍未实现：当前 Habitat 配置没有 semantic/instance sensor，本地也没有分割模型。审查四次旧正式 STOP 后发现，启用“当前视图原文 corroboration”只会挡住唯一官方成功的 EP259，无法挡住另外三次错误提交，因此未把它作为替代修复。`cross/pass/traverse` 也继续明确记录为 unsupported，没有用单帧可见性或 waypoint distance 伪造动作完成。

下一步先运行固定 10 条。只有解析失败为 0、EP11 首帧不再 route complete、候选先验不产生 action-affecting override，且没有新增错误正式 STOP 时，才启动新的 EP100。由于实例深度和三类动作事件尚未解决，新 EP100 仍属于诊断轮次。

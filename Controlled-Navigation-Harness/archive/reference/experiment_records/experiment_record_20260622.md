# 2026-06-22 Open-Nav 代码改动日报

## 背景

昨日主要围绕两件事推进：

1. 整理运行日志目录结构，避免 ep20、ep100、不同日期的输出混在同一层。
2. 基于 9B ep100 运行结果，收紧 STOP rescue / fallback 相关逻辑，优先降低误停和不可解释恢复。

关联运行记录:

```text
/root/wjj/Open-Nav/logs/navigation_records/ep100/20260622/ep100_series_qwen_siglip_local_20260622_222705_train_navigation_20260622_222735.jsonl
```

该运行跨日完成：

- start: 2026-06-22 22:28:22
- end: 2026-06-23 17:37:10
- wall-clock: 约 19.15h

## ep100 运行结果

总体指标:

- episode: 100/100
- SR: 19/100 = 19%
- OSR: 29/100 = 29%
- SPL: 0.1295
- nDTW: 0.4189
- mean distance_to_goal: 7.10m
- mean steps_taken: 9.26

对比 4B 历史运行:

- 4B 0615: SR 16%，约 8.07h
- 4B 0616: SR 17%，约 8.33h
- 4B 0621: SR 16%，约 11.23h

判断:

- 9B 仅带来约 +2 到 +3 个 SR 点，但运行时间显著增加。
- 主要瓶颈不是模型大小，而是 STOP、rescue、fallback 的策略边界。

## 代码改动

### 1. 日志目录按 episode/date 分层

修改文件:

```text
run.py
```

新增逻辑:

- `_episode_group_name(episode_count)`
- `_apply_run_log_layout(base_dir, episode_group, run_date)`
- `_apply_trace_log_layout(trace_dir, episode_group, run_date)`

行为变化:

- `logs/running_log` 改为:

```text
logs/running_log/ep{N}/{YYYYMMDD}/...
```

- `RESULTS_DIR` 改为:

```text
logs/eval_results/ep{N}/{YYYYMMDD}/{exp_name}/...
```

- harness trace 改为:

```text
logs/harness_traces/ep{N}/{YYYYMMDD}/{variant}/...
```

- 通过环境变量向 trainer 传递:

```text
OPENNAV_RUN_DATE
OPENNAV_EPISODE_GROUP
```

目的:

- 让 ep20、ep100、跨日运行结果可追溯。
- 避免后续分析时误读旧日志。
- 支持 `logs/navigation_records/ep100/20260622/...` 这种结构化路径。

### 2. U1 phase-aware selector context 收敛

修改文件:

```text
vlnce_baselines/common/opennav_ext/evidence_scaffolder.py
```

主要行为:

- 按 phase 生成不同 context mode:
  - `search` -> `route_overview`
  - `approach` -> `subgoal_candidate`
  - `verify` -> `stop_verify`
  - `recover` -> `recovery`
- 只有 `route_overview` 和 `subgoal_candidate` 允许真正应用到 selector。
- `stop_verify` 阶段不拼接普通 visual summary，避免和 STOP verifier 的证据职责混在一起。
- context 中记录:
  - phase
  - mode
  - visual budget
  - phase confidence
  - current subgoal
  - recent failure / non-positive gain 信息

目的:

- 让 U1 只负责选择阶段的上下文增强。
- 不在 STOP 验证阶段引入额外 selector 偏置。
- 保持 U1 与 U2 的归因边界清楚。

### 3. U3 recovery policy 收紧

修改文件:

```text
vlnce_baselines/common/opennav_ext/recovery_policy.py
```

主要行为:

- 只有 `recovery_rank_trusted=true` 且 `fallback_strategy=visual_evidence_ranked` 时，才允许 recovery reselect。
- `observation_order` fallback 不再被当作可信恢复排序。
- 当前 candidate 仍有效且没有被 blocked 时，不强行改动作。
- 只处理 recoverable failure:
  - `empty_fallback_bad`
  - `progress_drift`
  - `stop_false_positive`
  - `candidate_missing`
- 保留 episode 级 recovery budget。

目的:

- 防止 U3 在没有可信 ranked evidence 时按原始候选顺序乱改。
- 让 recovery 的 action change 有可解释来源。

### 4. 严格 STOP rescue 策略

修改文件:

```text
vlnce_baselines/common/opennav_ext/stop_evidence_verifier.py
vlnce_baselines/common/base_il_trainer_llm.py
vlnce_baselines/config/default.py
run_OpenNav.yaml
```

昨日策略:

- rescue 必须来自 selector STOP gate。
- evidence mode 必须是 `current_pano`。
- stop relevant candidate 必须是 `__current_view__`。
- visual verdict 必须是 `allow`。
- selected candidate verdict 必须是 `allow`。
- 必须同时满足:
  - `final_target_visible=True`
  - `arrival_evidence=True`
  - no contradictions
  - no hard blockers
  - no allow warnings
  - no selected missing instruction terms
- rescue confidence threshold: 0.99
- rescue min step: 8
- recent gain 必须为正。
- recent non-positive gains 不超过 1。

目的:

- 先把误停风险压低。
- 避免视觉模型只看到远处目标就 rescue STOP。
- 保证 U2 当前仍是 selector visual rescue 的约束层，而不是完整 STOP gate 替代器。

备注:

- 该策略在 2026-06-23 的 ep20 回测中发现过严，已在 2026-06-23 日报中进一步调整为 verify-phase current-view rescue。

## 验证

昨日完成或保留的验证:

```text
python3 -m py_compile ...
```

结果:

- 相关 Python 文件语法检查通过。

离线 replay 结论:

- 严格 rescue 策略预计只在少量 episode 首次触发。
- 触发样例中多数处于 3m 内，说明 current-view + arrival evidence 的筛选方向基本正确。
- 但门槛过高会压制真实近点 STOP，后续需要小样本继续校准。

## 风险与后续

已知风险:

- 9B 成本太高，SR 提升不足。
- 严格 rescue 能压误停，但可能导致近点漏停。
- completion gate 仍可能被 `door/sink/room` 等通用目标词误导。
- recover 阶段的视觉 arrival 可能把远处目标误判为到位。

后续建议:

1. 回切 4B，优先提升迭代速度。
2. 用 ep20 小样本验证 STOP rescue 的边界，而不是直接跑 ep100。
3. 区分 completion auto-stop 与 selector visual rescue，二者使用不同保守程度。
4. 将每次 STOP/rescue/fallback 的触发原因写入日报，保证后续 SR 变化可以归因。

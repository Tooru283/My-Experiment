# 最新运行对比记录 2026-06-16 13:25

## 对比对象

新运行:

`/root/wjj/Open-Nav/logs/navigation_records/ep20_series_qwen_siglip_local_20260616_132553_train_navigation_20260616_132609.jsonl`

旧运行:

`/root/wjj/Open-Nav/logs/navigation_records/ep20_series_qwen_siglip_local_20260616_100333_train_navigation_20260616_100351.jsonl`

本记录只做分析归档，不包含代码修改。

## 总体指标

| 指标 | 旧运行 10:03 | 新运行 13:25 | 变化 |
|---|---:|---:|---:|
| episodes | 20 | 20 | - |
| SR | 0.25 | 0.25 | 持平 |
| OSR | 0.35 | 0.35 | 持平 |
| SPL | 0.153 | 0.162 | 小幅上升 |
| nDTW | 0.500 | 0.494 | 小幅下降 |
| avg distance_to_goal | 6.248 | 6.313 | 小幅变差 |
| avg steps | 9.35 | 8.45 | 更早终止 |
| avg path length | 12.219 | 10.779 | 更短 |
| step_length_limit | 17 | 11 | 减少 |
| stop_requested | 3 | 9 | 明显增加 |

结论: 新代码不是没有起作用，而是决策形态变了。STOP 明显更积极，路径更短，但 SR 没提升，因为新增 STOP 中有较多 false positive。

## 成功与失败变化

旧运行成功 episode:

`244, 218, 226, 1087, 265`

新运行成功 episode:

`244, 218, 1092, 513, 1087`

新增成功:

- `1092`: 旧运行 OSR=1 但走远到 6.39m；新运行 completion gate 在 step 7 STOP，distance=2.619，成功。
- `513`: 旧运行 OSR=1 但最后走远到 6.425m；新运行 selector STOP 被视觉 rescue，step 6 STOP，distance=1.552，成功。

丢失成功:

- `226`: 旧运行 step_length_limit 成功，distance=1.93；新运行早期移动路线变差，最终 distance=5.831。
- `265`: 旧运行 step_length_limit 成功，distance=2.951；新运行 selector STOP rescue 在 step 8 误停，distance=14.651。

## STOP 行为

旧运行:

- `stop_requested=3`
- STOP 成功率 `0/3`
- `visual_stop_allowed=3`
- `visual_stop_rescued=0`

新运行:

- `stop_requested=9`
- STOP 成功率 `3/9`
- `visual_stop_allowed=4`
- `visual_stop_rescued=5`
- `visual_stop_rejected=54`

STOP 成功来自:

- `1092`: completion gate allow，distance=2.619，success=1。
- `1087`: completion gate allow，distance=2.636，success=1。
- `513`: selector STOP visual rescue，distance=1.552，success=1。

STOP 失败来自:

- `748`: selector rescue，distance=8.852，false positive。`allow_warnings=["uncorroborated_final_target:dining room table"]`。
- `11`: selector rescue，distance=10.401，false positive。final target 为 `floor/archway` 弱泛化目标。
- `265`: selector rescue，distance=14.651，false positive。VLM 认为 bathroom visible，但实际远离目标。
- `140`: selector rescue，distance=12.276，false positive。VLM 把复杂 doorway relation 直接判为可达。
- `166`: completion gate allow，distance=3.621，失败但接近阈值边缘。
- `371`: completion gate allow，distance=3.516，OSR=1，但 SR=0，说明 near-goal 停得略早或阈值外。

## 新增规则效果

### V4 memory 降噪已生效

新运行:

- `multimodal_selector_context=169`
- `applied=169`
- `mem_suffix_nonempty=0`
- `include_memory_suffix=True` 次数为 0

说明 V3 episode memory 已不再拼到每个 selector candidate 文本里。

### selector_empty_prediction fallback 明显改善

旧运行:

- `selector_empty_prediction_fallback=15`
- mean distance gain `+0.021`
- positive/nonpositive = `8/7`

新运行:

- `selector_empty_prediction_fallback=12`
- mean distance gain `+0.765`
- positive/nonpositive = `9/3`
- score_policy 均为 `visual_goal_tracking`

说明 V4/fallback 的普通空预测补救有明显正向作用。

### stop_rejected_fallback 仍有问题

旧运行:

- `stop_rejected_fallback=42`
- mean distance gain `+0.035`
- positive/nonpositive = `20/22`

新运行:

- `stop_rejected_fallback=28`
- mean distance gain `-0.046`
- positive/nonpositive = `14/14`
- score_policy 均为 `stop_rejected_continue_movement`

说明 STOP 被拒后的 fallback 频次下降，但质量没有提升，甚至略差。后续需要重新设计 STOP rejected 后的移动策略，不能只靠当前视觉 candidate 排序。

### 弱终点 completion auto-stop 拦截生效，但副作用明显

新运行出现:

- `completion_gate_weak_final_target=26`
- episode 分布:
  - `244`: 8 次，最终 success=1
  - `218`: 9 次，最终 success=1
  - `602`: 8 次，最终 OSR=1，distance=3.712
  - `11`: 1 次，最终 false STOP

关键副作用:

- `602` 在 step 8 到达 `distance=0.951`，但由于 final target 是 `doorway`，completion auto-stop 被弱终点规则持续拦截，随后走远到 3.712。
- 这说明弱终点不能简单永久硬拦截；更合理的是“早期拦截，后期交给 V2 current-pano 严格验证”。

## 主要问题判断

### 1. selector STOP rescue 太松

5 次 `visual_stop_rescued` 中只有 1 次成功:

- 成功: `513`
- 失败: `748, 11, 265, 140`

失败模式:

- `uncorroborated_final_target` 仍被允许 rescue，例如 `748` 的 dining room table。
- 弱泛化目标仍可 rescue，例如 `11` 的 floor/archway。
- VLM 容易把 current pano 中的普通物体或方向关系升格成最终目标，例如 bathroom、doorway relation。

### 2. completion gate allow 精度比 selector rescue 好，但仍有边界误停

4 次 completion allow 中:

- 成功: `1092, 1087`
- near miss / OSR: `371`
- 失败但接近阈值边界: `166`

completion allow 不应大幅收紧，否则会损失真实 STOP；但可以增加“final target 具体性”和“current-view 文本距离/关系佐证”的轻量检查。

### 3. final landmark phrase matching 存在组合词问题

`643` 中 final landmark 被解析为 `chandelier billiards table`，而 VLM 输出 `chandelier` 与 `billiards table` 两个词。当前 verifier 将组合词整体视为 missing，导致:

- `final_target_visible=True`
- `arrival_evidence=True`
- 但 `matched_final_landmarks=[]`
- `missing_final_landmarks=["chandelier billiards table"]`

这类组合词需要在 landmark matching 层支持拆分或 AND 匹配，否则会误拒。

## 暂定后续方案，不在本记录中执行

1. 收紧 selector STOP rescue:
   - 若 `allow_warnings` 包含 `uncorroborated_final_target:*`，不允许 rescue。
   - 若 final target 全部是弱泛化词，selector rescue 需要更高门槛，例如 confidence=1.0 且 current-view 文本直接包含 final target。
   - 若 selected verdict 存在 `missing_instruction_terms`，不允许 rescue，避免 `265` 这种未完成路线却误停。

2. 调整弱终点 completion 规则:
   - 不再永久拦截弱终点。
   - 可以增加 `MIN_STEPS_FOR_WEAK_FINAL_TARGET_COMPLETION_STOP`，例如 step >= 8 后允许 completion STOP 进入 V2 验证。
   - 对 `doorway` 等弱终点，需要 current pano V2 allow 且 completion all_actions_completed。

3. 修复组合 final landmark 匹配:
   - 对 `chandelier billiards table` 支持拆为 `chandelier` + `billiards table` 的组合匹配。
   - 避免 current pano 已看到两个目标组件时仍判 missing。

4. 重新设计 STOP rejected fallback:
   - 当前 `stop_rejected_continue_movement` 平均 gain 为负。
   - 需要引入路线进度、历史回退惩罚或候选方向连续性，而不是只看 V1 视觉词。

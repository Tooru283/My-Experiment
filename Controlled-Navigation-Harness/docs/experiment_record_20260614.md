# 2026-06-14 V 系列运行分析与 schema 修复

## 目标

今天围绕 V2/V4 decision-effect 配置，分析最新运行数据，确认 Runtime Reduce、V1 compact JSON、V4 selector context、visual evidence fallback 是否稳定。

重点问题:

- 100 episode 运行是否已经可作为当前最优；
- `selector_empty_prediction_fallback` 是否真正摆脱 first-candidate fallback；
- V1 compact JSON 是否被 V2/V3/V4/fallback 正确消费；
- 是否存在 schema / prompt / runtime 参数导致的性能退化。

## 100 Episode 运行分析

记录文件:

```text
/root/wjj/Open-Nav/logs/navigation_records/v24_series_qwen_siglip_local_20260613_212115_train_navigation_20260613_212143.jsonl
```

总体指标:

- episode: 100
- step: 600
- success: 19/100
- oracle_success: 24/100
- SPL: 0.1708
- nDTW: 0.4797
- mean distance_to_goal: 7.42

运行时间:

- core runtime: 约 52.9s/step
- wall-clock: 约 71.2s/step
- `visual_evidence`: 约 15.18s/step
- `completion_estimation`: 约 21.01s/step
- `navigator_move_to_next_vp`: 约 16.70s/step

主要现象:

- Runtime Reduce 有效，`visual_evidence` 从早期 30s+ 降到约 15s/step。
- 导航质量下降，不能作为当前最优。
- `selector_empty_prediction_fallback` 出现 80 次，但 `changed=0`。
- `visual_stop_allowed=0`，全部 episode 走到 `step_length_limit`。
- 当前配置实际已经是 `VISUAL_EVIDENCE.MAX_TOKENS=768`，不是 512。

判断:

- 退化主要不是 token 不够。
- 更可疑的是 compact V1 输出和 V4/fallback 的证据信号质量不足。
- 需要先确认 V1 视觉证据是否被下游正确消费，再继续跑 100 episode。

## R3.1 小样本运行分析

记录文件:

```text
/root/wjj/Open-Nav/logs/navigation_records/v24_series_qwen_siglip_local_20260614_101818_train_navigation_20260614_101847.jsonl
```

总体指标:

- episode: 11
- success: 1/11
- oracle_success: 1/11
- SPL: 0.0693
- nDTW: 0.4777
- mean distance_to_goal: 6.98

运行时间:

- `visual_evidence`: 约 27.53s/step
- core runtime: 约 63.21s/step

主要现象:

- `selector_empty_prediction_fallback` 出现 6 次，仍然 `changed=0`。
- V2 出现 2 次 `visual_stop_allowed`，但对应 episode 未成功，STOP allow 仍存在 false positive 风险。
- 63 次 `visual_evidence` 中，49 次 `parsed` 是裸 `list`，14 次是 `dict`。
- V4/fallback 原实现只读取 `parsed["candidates"]`，导致裸数组输出虽然 `parse_error=null`，但下游得到 `candidate_evidence_count=0`。

## 问题定位

### V1 compact JSON 输出形状漂移

V1 compact prompt 要求返回:

```json
{"candidates": [...]}
```

但 Qwen-VL 在 compact 输出下经常返回:

```json
[...]
```

裸数组是合法 JSON，因此不会触发 `parse_error`。

问题出现在下游假设过硬:

```text
parsed 必须是 dict
parsed["candidates"] 必须存在
```

影响:

- V4 `candidate_evidence_count=0`；
- fallback `candidate_evidence_count=0`；
- V3 memory 少记录候选证据；
- V2 verifier 候选证据不足；
- 已经生成的视觉证据没有进入 selector context 或 fallback 排序。

### 为什么之前不明显

复查历史日志:

| 运行 | visual_evidence parsed 类型 |
|------|------------------------------|
| 20260613 15:08 | 55/55 dict |
| 20260613 17:55 | 72/72 dict |
| 20260613 21:21 | 581/600 list |
| 20260614 10:18 | 49/63 list |

结论:

- 这是 Runtime Reduce / compact JSON 之后暴露的问题。
- 早期非 compact 或较长输出时，模型更稳定地返回 dict。
- compact 输出下模型倾向于省掉顶层 `{"candidates": ...}` 包装。

## 代码修改

新增文件:

```text
/root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/visual_evidence_schema.py
```

新增工具函数:

```text
candidate_list_from_parsed()
normalize_visual_evidence_parsed()
candidate_evidence_from_result()
```

修改文件:

- `vlnce_baselines/common/opennav_ext/visual_evidence.py`
- `vlnce_baselines/common/opennav_ext/visual_target_verifier.py`
- `vlnce_baselines/common/opennav_ext/visual_evidence_memory.py`
- `vlnce_baselines/common/opennav_ext/multimodal_selector_context.py`
- `vlnce_baselines/common/opennav_ext/visual_fallback.py`

关键改动:

- V1 `VisualEvidenceLogger` 返回前把裸数组规范化为 `{"candidates": [...]}`。
- V2 `VisualTargetVerifier` 使用统一 schema 工具读取候选证据。
- V3 `VisualEvidenceMemory` 使用统一 schema 工具读取候选证据。
- V4 `MultimodalSelectorContext` 使用统一 schema 工具读取候选证据。
- `VisualEvidenceFallbackRanker` 使用统一 schema 工具读取候选证据。
- V1 prompt 增加顶层 object 约束:

```text
The top-level JSON value must be an object with a "candidates" key.
Do not return a bare array.
```

- V1 日志新增:

```text
parsed_root_type
```

用于后续直接统计模型原始返回是 `dict` 还是 `list`。

## 验证

已执行:

```bash
python -m py_compile \
  vlnce_baselines/common/opennav_ext/visual_evidence_schema.py \
  vlnce_baselines/common/opennav_ext/visual_evidence.py \
  vlnce_baselines/common/opennav_ext/multimodal_selector_context.py \
  vlnce_baselines/common/opennav_ext/visual_fallback.py \
  vlnce_baselines/common/opennav_ext/visual_evidence_memory.py \
  vlnce_baselines/common/opennav_ext/visual_target_verifier.py
```

结果:

```text
通过
```

隔离 smoke 验证:

- 裸数组 parsed 可被 schema 工具规范化；
- V4 可得到 `candidate_evidence_count=1`；
- fallback 可得到 `candidate_evidence_count=1` 并选中有证据候选；
- V3 memory 可记录候选；
- V2 verifier 可读取候选证据。

## 文档更新

更新:

```text
/root/wjj/Open-Nav/Controlled-Navigation-Harness/实验方案.md
/root/wjj/Open-Nav/Controlled-Navigation-Harness/V系列多模态视觉证据初步实现后调整方案.md
```

记录内容:

- 当前 `VISUAL_EVIDENCE.MAX_TOKENS=768`；
- 100 episode 运行结论；
- R3.1 fallback 修复；
- V1 schema 修复；
- 下一轮必须检查的字段。

## 当前结论

- Runtime Reduce 已经有效降低耗时，但仍未带来稳定导航收益。
- 100 episode 结果不能作为当前最优。
- 最新小样本退化的关键原因不是 VLM 无证据，而是 V1 compact 输出裸数组后，下游没有消费到证据。
- schema 修复后，需要重新跑小样本验证 V4/fallback 是否真正拿到候选级视觉证据。

## 后续建议

下一轮小样本必须检查:

```text
visual_evidence.parsed_root_type
multimodal_selector_context.candidate_evidence_count
selector_empty_prediction_fallback.candidate_evidence_count
selector_empty_prediction_fallback.changed
visual_target_verifier.visual_evidence_candidate_count
visual_stop_allowed case study
```

建议顺序:

1. 跑 10 到 12 episode 小样本，不直接跑 100。
2. 优先确认 `candidate_evidence_count` 不再大量为 0。
3. 检查 fallback 是否从 `changed=0` 恢复为正。
4. 如果 schema 修复后 V4/fallback 正常，再继续评估 V2 STOP allow 是否过宽。
5. 在 V2/V4 稳定前，暂缓 R4 adaptive candidate sampling。

## 12:25 Schema 修复后小样本复盘

日志：

```text
/root/wjj/Open-Nav/logs/navigation_records/v24_series_qwen_siglip_local_20260614_122559_train_navigation_20260614_122628.jsonl
```

规模：

- 11 episodes
- 60 navigation steps
- 2 个 `stop_requested`
- 9 个 `step_length_limit`

结果：

```text
success = 4 / 11
oracle_success = 4 / 11
SPL = 0.3416
nDTW = 0.5844
distance_to_goal = 5.7791
```

与 10:18 schema 修复前小样本相比：

```text
success: 1 / 11 -> 4 / 11
SPL:     0.0693 -> 0.3416
nDTW:    0.4777 -> 0.5844
dist:    6.9756 -> 5.7791
```

Schema 修复确认生效：

- V1 `parsed_root_type=dict` 为 60/60。
- V4 `candidate_evidence_count=0` 为 0。
- fallback 触发 7 次，`candidate_evidence_count` 均大于 0。
- fallback 7 次中 3 次改变了原首候选，5 次带来正向距离增益。

剩余问题：

- `visual_stop_allowed=2`。
- 2 次 STOP 放行均为失败终止：
  - episode 1092，`selector_stop_gate` 放行，终止距离约 5.43m。
  - episode 11，`completion_gate` 放行，终止距离约 10.40m。
- episode 11 的 final landmarks 为 `floor / archway`，属于弱泛化目标词，但旧逻辑只在 `selector_stop_gate` 上启用 `generic_final_terms` blocker，导致 `completion_gate` 被放行。

结论：

- V1 schema 问题已经不是主阻塞，V4/fallback 已能拿到候选级视觉证据。
- 当前主要风险转为 V2 STOP allow 过宽，尤其是 `completion_gate` 也可能被视觉证据误放行。

## V2 Generic STOP Allow 修复

修复文件：

```text
/root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/visual_target_verifier.py
```

修复内容：

- `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW=True` 时，`generic_final_terms` blocker 从仅作用于 `selector_stop_gate` 改为作用于所有 STOP gate source。
- 对 `completion_gate` 的 STOP 提案，如果 final landmarks 全是 `floor / archway / hallway` 这类泛化词，即使视觉候选为 allow，也会降为 `uncertain`，不直接放行 STOP。

验证：

```text
python -m py_compile vlnce_baselines/common/opennav_ext/visual_target_verifier.py
```

结果通过。

隔离 smoke：

- `completion_gate + floor/archway + confidence=0.95 + arrival=true` 输出 `verdict=uncertain`。
- `allow_blockers` 包含 `generic_final_terms:floor,archway`。
- 非泛化目标词 `dining room table` 在相同视觉证据下仍可输出 `verdict=allow`。

注意：

- 本次没有重新跑导航日志。
- 下一轮小样本需要确认 `completion_gate` 来源的泛化 STOP 不再进入 `visual_stop_allowed`，而是进入 `visual_stop_rejected` 或 `visual_stop_uncertain`。

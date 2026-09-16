---
tags:
  - 实验记录
  - Controlled-Navigation-Harness
  - Open-Nav
  - multimodal
  - visual-evidence
created: 2026-06-13
updated: 2026-06-13
status: active
source_repo: /root/wjj/Open-Nav
archived_from:
  - ./V系列多模态视觉证据实验记录-封存-20260613.md
---

# V 系列多模态视觉证据初步实现后调整方案

本文档承接已封存的 V 系列实验记录，记录 V0/V1/V2/V3/V4 全部初步实现后的问题、调整方案和下一轮实验顺序。

旧记录已封存为：

```text
/root/wjj/Open-Nav/Controlled-Navigation-Harness/V系列多模态视觉证据实验记录-封存-20260613.md
```

---

## 1. 当前实现状态

当前 V 系列已经形成完整链路：

| 版本 | 模块 | 当前状态 | 作用 |
|------|------|----------|------|
| V1 | `VisualEvidenceLogger` | 已实现 | 对候选图像抽取结构化视觉证据 |
| V2 | `VisualTargetVerifier` | 已实现并接入 decision-effect | 验证 STOP proposal，拦截不被视觉证据支持的 STOP |
| V3 | `VisualEvidenceMemory` | 已实现 | 聚合 episode 内视觉证据 |
| V4 | `MultimodalSelectorContext` | 已实现并接入 decision-effect | 将视觉摘要注入 selector 输入 |

当前默认运行配置：

```yaml
OPENNAV_HARNESS:
  ENABLED: true
  ENABLE_HARNESS_LOGGING: true
  ENABLE_DECISION_EFFECT: true

  VISUAL_EVIDENCE:
    ENABLED: true
    LOG_ONLY: true
    MAX_TOKENS: 768
    IMAGE_JPEG_QUALITY: 70
    METADATA_OBSERVATION_CHARS: 300
    COMPACT_JSON: true

  VISUAL_TARGET_VERIFIER:
    ENABLED: true
    LOG_ONLY: false
    CONFIDENCE_THRESHOLD: 0.9
    REJECT_ON_UNCERTAIN: true
    MIN_STEPS_BEFORE_ALLOW: 3
    REQUIRE_FULL_COVERAGE_FOR_ALLOW: true
    BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW: true

  VISUAL_EVIDENCE_MEMORY:
    ENABLED: true
    LOG_ONLY: true

  MULTIMODAL_SELECTOR_CONTEXT:
    ENABLED: true
    LOG_ONLY: false

  LLM_RUNTIME:
    COMPLETION_MAX_TOKENS: 384
    NAVIGATOR_MAX_TOKENS: 768
    THOUGHT_FUSION_MAX_TOKENS: 256
    DECISION_MAX_TOKENS: 64
```

---

## 2. 关键实验复盘

### 2.1 V4-only run

文件：

```text
/root/wjj/Open-Nav/logs/navigation_records/v_series_qwen_siglip_local_20260612_233128_train_navigation_20260612_233156.jsonl
```

结论：

```text
episode_count: 100
visual_evidence parse_error: 415/423
V4 applied: 423/423
success: 10/100
stop_requested: 71/100
```

问题：

- `MAX_TOKENS=384` 导致 V1 JSON 大量截断。
- V4 虽然注入 selector，但大多数 step 没有候选级视觉证据。
- 提前 STOP 仍由 completion gate 主导，V4 无法干预。

### 2.2 V2/V4 decision-effect run

文件：

```text
/root/wjj/Open-Nav/logs/navigation_records/v24_series_qwen_siglip_local_20260613_112044_train_navigation_20260613_112112.jsonl
```

结果：

```text
episode_count: 10
step_count: 57
visual_evidence parse_error: 0/57
V4 applied: 57/57
visual_stop_rejected: 28
stop_decision: 1
success: 3/10
oracle_success: 6/10
SPL mean: 0.2863
nDTW mean: 0.6699
distance_to_goal mean: 4.2091
```

同 10 个 episode 对比 V4-only：

```text
success:        1/10 -> 3/10
oracle_success: 1/10 -> 6/10
stop_requested: 8/10 -> 1/10
distance_to_goal mean: 6.39 -> 4.21
nDTW mean: 0.492 -> 0.670
```

结论：

- `MAX_TOKENS=1024` 解决了本轮 V1 解析失败。
- V2 拦截提前 STOP 明显有效。
- V4 context 已真实影响 selector 思考。

### 2.3 R2/R3 run：V2-first STOP gate + visual ranked fallback

文件：

```text
/root/wjj/Open-Nav/logs/navigation_records/v24_series_qwen_siglip_local_20260613_150848_train_navigation_20260613_150914.jsonl
```

对比 2.2 的同 10 个 episode：

| 指标 | V2/V4-current | R2/R3 |
|------|---------------|-------|
| step_count | 57 | 55 |
| visual_evidence parse_error | 0/57 | 0/55 |
| success | 3/10 | 5/10 |
| oracle_success | 6/10 | 7/10 |
| SPL mean | 0.2863 | 0.4863 |
| nDTW mean | 0.6699 | 0.7105 |
| distance_to_goal mean | 4.2091 | 3.4895 |
| selector_empty_prediction_fallback | 0 | 6 |
| visual_stop_allowed | 0 | 1 |
| visual_stop_rejected | 28 | 28 |
| sampled_all=false | 未单独记录 | 11/55 |

结论：

- R3 visual ranked fallback 是正向信号。6 次空预测 fallback 中，3 次改变旧 first-candidate fallback，且这些动作均为正 distance gain。
- R2 V2-first STOP gate 机制已经生效，但唯一一次 `visual_stop_allowed` 是失败 STOP。
- episode 11 step 1 中，VLM 因看到 `floor + archway` 判定 `final_target_visible=true`、`arrival_evidence=true`，系统在 `distance_to_goal=7.113m` 时停止，success=0。
- 因此 V2 allow 不能只靠“目标词可见 + arrival_evidence=true”，必须加入保守 allow 规则。

本轮 latency：

```text
step mean latency: ~104s
observation -> visual_evidence mean latency: ~37s
visual_evidence -> selector_raw mean latency: ~57s
```

R4 adaptive candidate sampling 仍然必要，但优先级低于修正 V2 allow 误停。

### 2.4 R2/R3-conservative run：收紧 V2 allow 后复测

文件：

```text
/root/wjj/Open-Nav/logs/navigation_records/v24_series_qwen_siglip_local_20260613_175536_train_navigation_20260613_175605.jsonl
```

本轮实际运行 12 个 episode。与 2.3 的共同 10 个 episode 对比：

| 指标 | R2/R3 | R2/R3-conservative |
|------|-------|--------------------|
| success | 5/10 | 5/10 |
| oracle_success | 7/10 | 7/10 |
| SPL mean | 0.4863 | 0.4863 |
| nDTW mean | 0.7105 | 0.6912 |
| distance_to_goal mean | 3.4895 | 3.7788 |
| stop_requested | 1 | 0 |
| visual_stop_allowed | 1 | 0 |
| selector_empty_prediction_fallback | 6 | 8 |

12 episode 全量结果：

```text
success: 5/12
oracle_success: 7/12
visual_evidence parse_error: 0/72
V4 applied: 72/72
visual_stop_allowed: 0
visual_stop_rejected: 36
selector_empty_prediction_fallback: 8
sampled_all=false: 12/72
```

结论：

- episode 11 的 step 1 误停被修掉，本轮全部以 `step_length_limit` 结束，没有 `stop_requested`。
- `allow_blockers` 正常生效。episode 11 的 selector STOP 被挡下的主要原因是 `generic_final_terms:floor,archway`，step 1 还叠加 `before_min_steps:1<3` 和 `sample_limited`。
- R3 fallback 收益保留。8 次 fallback 中 4 次改变旧 first-candidate fallback，6 次 positive distance gain。
- 新暴露日志缺口：selector STOP 被 V2 拦截时只写 `stop_rejected`，没有同步写 `visual_stop_rejected`。这会让自动统计低估 `selector_stop_gate` 来源的视觉 STOP 拦截。

已修复：

```text
selector_stop_gate verdict=reject/uncertain 且被 REJECT_ON_UNCERTAIN 拦截
-> 调用 apply_visual_stop_gate
-> 写入 visual_stop_rejected(source=selector_stop_gate, allow_blockers=...)
-> 再写入 stop_rejected 记录 fallback movement
```

---

## 3. 当前暴露的问题

### 3.1 V2 只能拦截 STOP，不能主动放行旧 gate 拒绝的 STOP

episode 11 多次出现：

```text
selector predicts STOP
visual evidence: final_target_visible=true, arrival_evidence=true
old should_stop(): false
```

旧逻辑把 V2 记录为 `not_applicable`，导致系统继续移动，最后 step_length_limit。

### 3.2 V2 allow 仍可能 metric 失败

episode 116 step 3：

```text
V2 verdict=allow
supporting_candidate_id=7
distance_to_goal=4.816
success=0
```

说明视觉目标可见与 VLN success 半径/真实目标点仍可能不一致。

### 3.3 selector 空预测 fallback 仍有风险

本轮：

```text
selector_raw.predictions=[]: 10
selected_equals_first_candidate: 10/10
```

旧 fallback 直接选第一个候选，未利用 V1/V4 视觉证据。

### 3.4 候选采样仍可能漏掉目标

本轮：

```text
sampled_all=false: 14/57
```

当候选数为 5、`MAX_CANDIDATES=4` 时，V1/V2/V4 可能漏掉目标候选。

### 3.5 latency 偏高

本轮粗略统计：

```text
step mean latency: ~101s
observation -> visual_evidence mean latency: ~35s
```

不宜简单永久提高 `MAX_CANDIDATES`。

---

## 4. 本轮已开始实现的调整

### 4.1 V2-first selector STOP gate

目标逻辑：

```text
selector predicts STOP
-> call VisualTargetVerifier with stop_proposal=true

if verdict == allow:
    write visual_stop_allowed
    accept STOP
elif verdict == reject:
    write visual_stop_rejected
    fallback movement
elif verdict == uncertain:
    if REJECT_ON_UNCERTAIN:
        write visual_stop_rejected
        fallback movement
    else:
        fallback old should_stop()
```

实现要点：

- selector STOP 不再先被旧 `should_stop()` 限制。
- 新增 `visual_stop_allowed` event。
- `visual_target_verifier` 输出增加：
  - `stop_relevant_candidate_id`
  - `selected_candidate_verdict`
  - `candidate_alignment`

### 4.2 selector 空预测视觉排序 fallback

新增：

```text
/root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/visual_fallback.py
```

核心类：

```text
VisualEvidenceFallbackRanker
```

排序依据：

```text
1. final_target_visible
2. arrival_evidence
3. matched_instruction_terms count
4. confidence
5. visible_landmarks count
6. missing_instruction_terms count
7. original candidate order
```

新增事件：

```text
selector_empty_prediction_fallback
```

该事件记录：

```text
fallback_strategy
selected_candidate
ranked_candidates
candidate_count
candidate_evidence_count
parse_error
```

### 4.3 V1 采样日志

`VisualEvidenceLogger` 返回新增：

```text
selection_reason_by_candidate
```

trainer 额外写入：

```text
visual_evidence_sampling
```

用于后续统计 `sampled_all=false` 时哪些候选被跳过。

### 4.4 V2 conservative STOP allow

R2/R3 暴露 `visual_stop_allowed` 误停后，本轮将 V2 allow 改成保守规则。

新增配置：

```yaml
OPENNAV_HARNESS:
  VISUAL_TARGET_VERIFIER:
    CONFIDENCE_THRESHOLD: 0.9
    MIN_STEPS_BEFORE_ALLOW: 3
    REQUIRE_FULL_COVERAGE_FOR_ALLOW: true
    BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW: true
```

含义：

- `MIN_STEPS_BEFORE_ALLOW=3`：禁止 step 1/2 这类过早 selector STOP 被视觉证据直接放行。
- `REQUIRE_FULL_COVERAGE_FOR_ALLOW=true`：当 V1 只检查候选子集时，不允许 V2 直接 allow STOP，避免未采样候选改变判断。
- `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW=true`：对 `selector_stop_gate` 生效；当 final landmark 全是 `floor`、`archway`、`doorway`、`hallway` 等低特异性词时，不允许 selector 仅凭可见性提前 STOP。
- 被这些规则挡下的 allow 会降级为 `uncertain`，在 `REJECT_ON_UNCERTAIN=true` 时拦截 STOP。

新增日志字段：

```text
visual_target_verifier.allow_blockers
visual_stop_allowed.allow_blockers
visual_stop_rejected.allow_blockers
```

下一轮重点看 episode 11 是否不再出现 step 1 误停，同时确认 218、602 的 fallback 收益是否保留。

### 4.5 selector STOP rejection logging

R2/R3-conservative 复盘发现 `selector_stop_gate` 的 V2 拦截没有进入 `visual_stop_rejected` 事件，只能通过 `visual_target_verifier` 与 `stop_rejected` 间接恢复。

修复后日志语义：

```text
visual_target_verifier:
  记录 verifier verdict、allow_blockers、候选证据

visual_stop_rejected:
  记录所有被 V2 decision-effect 拦截的 STOP
  source 可以是 completion_gate 或 selector_stop_gate

stop_rejected:
  只记录 STOP 被拦截后实际回退到哪个 movement candidate
```

该修复不改变决策，只修正统计和 case study 可读性。

### 4.6 Runtime Reduce P0/P1：低风险降时

目标：先压缩单步耗时中最明显的 token / 图像 / 日志负担，不改变候选集合、不改变 STOP 策略、不改变 V2/V4 决策链。

本轮已实现：

| 方法 | 配置 / 代码 | 预期降时 | 性能风险 | 说明 |
|------|-------------|----------|----------|------|
| 细粒度耗时日志 | 新增 `runtime_latency` 事件 | 无 | 无 | 记录 harness tool 与文本 LLM 调用耗时 |
| 文本 LLM token cap | `LLM_RUNTIME.*_MAX_TOKENS` | 中 | 低-中 | 限制长 thought / completion 输出 |
| V1 compact JSON | `VISUAL_EVIDENCE.COMPACT_JSON=true` | 中 | 低 | 字段名保持不变，V2/V4 不改解析 |
| V1 max tokens | `MAX_TOKENS=768` | 中 | 低-中 | 依赖 compact JSON 降低截断风险，并给 matched/missing 字段留余量 |
| JPEG quality | `IMAGE_JPEG_QUALITY=70` | 低-中 | 低 | 不改分辨率，先只降压缩质量 |
| candidate 文本截断 | `METADATA_OBSERVATION_CHARS=300` | 低-中 | 低-中 | 图像仍保留，减少 VLM prompt 文本 |

再次审查后调整：

```text
COMPLETION_MAX_TOKENS: 256 -> 384
NAVIGATOR_MAX_TOKENS: 512 -> 768
```

原因：

- `completion_estimation` prompt 要求先输出 Thought 再输出 `Executed Actions`，256 tokens 可能在关键字段前截断，影响 STOP gate。
- `NAVIGATOR` prompt 要求先长 Thought 再输出 `Prediction:`，512 tokens 可能导致 `Prediction:` 缺失，增加空预测 fallback。
- V1 compact JSON 调整为 `MAX_TOKENS=768`，因为 compact 输出虽然不再截断，但后续需要 matched/missing 字段有足够空间。

新增配置：

```yaml
OPENNAV_HARNESS:
  VISUAL_EVIDENCE:
    MAX_TOKENS: 768
    IMAGE_JPEG_QUALITY: 70
    METADATA_OBSERVATION_CHARS: 300
    COMPACT_JSON: true

  LLM_RUNTIME:
    COMPLETION_MAX_TOKENS: 384
    NAVIGATOR_MAX_TOKENS: 768
    THOUGHT_FUSION_MAX_TOKENS: 256
    DECISION_MAX_TOKENS: 64
```

新增日志：

```text
runtime_latency:
  operation
  elapsed_seconds
  category=harness_tool|text_llm
  max_tokens
```

下一轮重点统计：

```text
step_total mean
runtime_latency visual_evidence mean
runtime_latency completion_estimation mean
runtime_latency navigator_move_to_next_vp mean
runtime_latency test_decision mean
visual_evidence parse_error rate
selector_empty_prediction_fallback count
success / SPL / nDTW
```

---

## 5. 下一步实现顺序

1. 重新跑 Runtime Reduce P0/P1 小样本，验证 `runtime_latency` 事件和单步耗时下降。
2. 同时确认 `visual_stop_rejected(source=selector_stop_gate)` 正常出现。
3. 检查 compact V1 后 `visual_evidence.parse_error` 是否仍接近 0。
4. 检查 `selector_empty_prediction_fallback` 是否保留正向收益。
5. 若耗时仍偏高，再设计 adaptive candidate sampling。
6. 跑 R4 小样本：

```text
R4: Runtime Reduce P0/P1 + adaptive candidate sampling
```

对比指标：

```text
success / SPL / nDTW
stop_requested count
visual_stop_allowed count
visual_stop_rejected count
allow_blockers count
selector_empty_prediction_fallback count
sampled_all=false rate
step latency
runtime_latency by operation
```

---

## 6. 当前风险

- V2-first STOP 放行可能引入视觉误停，需要 case study 每个 `visual_stop_allowed`。
- V2 conservative allow 可能过度拦截真实 STOP，需要同时统计 `allow_blockers` 与最终 success。
- 视觉排序 fallback 可能放大 V1 误证据，需要保留完整 ranked candidates。
- Runtime Reduce P0/P1 可能造成 LLM 输出被截断，需要检查 selector parse 和 V1 parse。
- adaptive sampling 可能增加 latency，需要与收益同时评估。
- V2/V4 同时 decision-effect 后归因更复杂，后续需要拆分 R2/R3/R4。

---

## 7. 2026-06-13 21:21 最新 100 episode 复盘

样本：

`/root/wjj/Open-Nav/logs/navigation_records/v24_series_qwen_siglip_local_20260613_212115_train_navigation_20260613_212143.jsonl`

结论先行：

- 运行时间明显下降，`visual_evidence` 已从早期 30s+ 降到约 15s/step。
- 但导航质量回落，100 episode 结果不适合作为当前最优。
- `selector_empty_prediction_fallback` 出现 80 次，但原始实现几乎都回退到首候选，`changed=0`。
- `visual_stop_allowed=0`，V2 仍然过保守，整轮全部跑到 `step_length_limit`。

这次修正：

- 将 `VISUAL_EVIDENCE.MAX_TOKENS` 维持/修正为 `768`。
- 在 V1 compact prompt 中显式要求填充 `matched_instruction_terms` 与 `missing_instruction_terms`。
- 在 V4 上下文摘要中加入 `matched_instruction_terms`。
- 给 empty-prediction fallback 增加基于 `instruction/actions/landmarks` 的本地词项补分。

下一轮验证重点：

1. `selector_empty_prediction_fallback.changed` 是否恢复为正。
2. `visual_evidence` 中 matched/missing 字段是否不再频繁为空。
3. common 10 episode 上 success / oracle / nDTW 是否回到 2026-06-13 17:55 轮次附近。

### 7.1 Schema 修复

2026-06-14 10:18 小样本发现 V1 compact 输出存在两种合法形状：

```json
{"candidates": [...]}
```

以及：

```json
[...]
```

后者虽然 `parse_error=null`，但 V2/V3/V4/fallback 原实现只读取 `parsed["candidates"]`，导致大量 step 的 `candidate_evidence_count=0`，视觉证据生成后没有进入 selector context 或 fallback 排序。

修复：

- 新增 `visual_evidence_schema.py`，统一提供 `candidate_evidence_from_result()`。
- V1 `VisualEvidenceLogger` 返回前将裸数组规范化为 `{"candidates": [...]}`。
- V2 `VisualTargetVerifier`、V3 `VisualEvidenceMemory`、V4 `MultimodalSelectorContext`、`VisualEvidenceFallbackRanker` 均兼容 dict/list 两种 parsed 形状。

下一轮必须检查：

```text
multimodal_selector_context.candidate_evidence_count > 0
selector_empty_prediction_fallback.candidate_evidence_count > 0
visual_target_verifier.visual_evidence_candidate_count > 0
```

### 7.2 12:25 小样本复盘与 V2 Generic STOP 修复

样本：

`/root/wjj/Open-Nav/logs/navigation_records/v24_series_qwen_siglip_local_20260614_122559_train_navigation_20260614_122628.jsonl`

结果：

```text
episodes = 11
success = 4 / 11
oracle_success = 4 / 11
SPL = 0.3416
nDTW = 0.5844
distance_to_goal = 5.7791
stop_requested = 2
step_length_limit = 9
```

观察：

- Schema 修复后，V4/fallback 已能稳定拿到候选级视觉证据。
- `multimodal_selector_context.candidate_evidence_count=0` 不再出现。
- fallback 触发 7 次，其中 3 次改变原首候选，5 次候选距离增益为正。
- 这说明 10:18 轮次的主要问题确实是证据结构没有被下游消费，而不是 VLM 完全无效。

新暴露的问题：

- `visual_stop_allowed=2`，且 2 次均为失败终止。
- episode 11 的 `completion_gate` 使用 `floor / archway` 这类泛化 final landmarks 时被放行，终止距离约 10.40m。
- 旧版 `generic_final_terms` blocker 只覆盖 `selector_stop_gate`，没有覆盖 `completion_gate`。

修复：

- 在 `visual_target_verifier.py` 中，将 `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW` 扩展到所有 STOP gate source。
- 当 final landmarks 全部是泛化词时，`completion_gate` 也不能直接 visual allow。

隔离验证：

```text
completion_gate + floor/archway -> verdict=uncertain
allow_blockers includes generic_final_terms:floor,archway
completion_gate + dining room table -> verdict=allow
```

下一轮验证重点：

```text
visual_stop_allowed 是否下降
completion_gate 的 generic_final_terms 是否进入 blocker
真实终点 STOP 是否被过度拦截
fallback changed 是否继续保持为正
```

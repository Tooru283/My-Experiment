---
tags:
  - 实验
  - Controlled-Navigation-Harness
  - Open-Nav
  - A1
  - 代码设计
status: draft
created: 2026-06-10
updated: 2026-06-10
role: 第一阶段代码设计
source_repo: /root/wjj/Open-Nav
---

# 第一阶段代码设计

> 第一阶段只建立 `A0 -> A1` 的行为不变 Harness。A1 的新增能力全部是 logging-only / diagnostic-only，不能改变 waypoint 候选、prompt、LLM 决策、stop、fallback、模型、数据集、随机种子和评测脚本。

---

## 1. 阶段目标

第一阶段交付两个可对照运行的版本：

| 版本 | 含义 | 决策行为 |
|------|------|----------|
| A0 | 原始 Open-Nav baseline | 原样运行 |
| A1 | Instrumented / harnessed baseline | 与 A0 保持一致，只增加旁路状态和日志 |

A1 必须满足：

```text
enable_harness_logging = true
enable_decision_effect = false
```

验收条件：

- A0 原始 Open-Nav 可复现，固定配置、随机种子和日志路径；
- A1 与 A0 的 SR / SPL / NE / TL 基本一致；
- 每个 episode 有可回放 trace；
- 每步记录 waypoint、geometry-query、grounder、memory、selector、action、metrics；
- 新增模块异常时只写 `tool_failure`，不得中断或改变导航；
- 形成 50 个失败 episode 的人工标注输入表。

---

## 2. 现有 Open-Nav 接入点

实际代码仓库位于：

```text
/root/wjj/Open-Nav
```

当前导航主循环在：

```text
vlnce_baselines/common/base_il_trainer_llm.py
```

关键接入点：

| 位置 | 当前行为 | A1 插桩 |
|------|----------|---------|
| `construct_image_dicts()` | 将 waypoint angle / distance 映射到 0-11 方向视角 | 构造 `CandidateRecord`，记录 angle、distance、direction_id、raw_rank |
| `_eval_llm()` episode loop 开始 | 读取 episode、pose、heading、instruction | 初始化 `AgentState` 和 episode trace |
| `self.policy.net(mode="waypoint")` 后 | 生成候选 waypoint | 记录 waypoint 候选、candidate count、candidate mask / length |
| `navigator.observe_environment()` 后 | RAM + SpatialBot 生成方向描述 | 记录 observation text、RAM/SpatialBot 原始输出摘要 |
| `navigator.move_to_next_vp()` 后 | LLM 产生多次 prediction / thought | 记录 selector raw responses，不解析为新决策 |
| `navigator.thought_fusion()` 后 | 融合同方向 thought | 记录 fused prediction map |
| `navigator.test_decisions()` 后 | 得到最终 `next_vp` | 记录 selected_candidate 和 reason |
| `envs.step(env_actions)` 前 | 组装 Habitat action 4 | 记录将执行的 angle / distance |
| `envs.step()` 后 | 获得 next observation / done / info | 记录 step 后 pose、distance_to_goal、collision |
| episode done | 计算 SR / SPL / nDTW / path_length 等 | 关闭 episode trace，写 episode summary |

`Open_Nav` 文本决策逻辑位于：

```text
vlnce_baselines/common/navigator/spatialNavigator.py
```

A1 不修改该文件中的 prompt、retry、fusion、fallback random choice 逻辑。若需要记录 LLM 原始输出，优先在调用返回值外侧记录，避免改变函数签名；如果后续必须改签名，只增加可选返回诊断字段，并保证默认行为不变。

---

## 3. 新增模块布局

新增代码集中到：

```text
vlnce_baselines/common/opennav_ext/
  __init__.py
  agent_state.py
  metrics_logger.py
  geometry_query.py
  grounder_diagnostic.py
  visual_graph_memory.py
  context_builder.py
  oracle_metrics.py
  harness_config.py
```

第一阶段只实现 A1 必需能力：

| 文件 | 阶段 | 职责 | 是否影响决策 |
|------|------|------|--------------|
| `harness_config.py` | A1 | 读取配置开关，提供默认禁用策略 | 否 |
| `agent_state.py` | A1 | 定义每步状态 dataclass / dict schema | 否 |
| `metrics_logger.py` | A1 | 写 jsonl trace、episode summary、tool failure | 否 |
| `geometry_query.py` | A1 | 记录候选投影、深度有效性、occupancy risk 占位 | 否 |
| `grounder_diagnostic.py` | B1 logging | 从现有 observation text 提取匹配/失败约束占位分数 | 否 |
| `visual_graph_memory.py` | C1 logging | 记录 pose revisit、visit count、novelty 占位 | 否 |
| `context_builder.py` | A1 skeleton | 只生成 diagnostic context，不注入 LLM prompt | 否 |
| `oracle_metrics.py` | A1 | 计算 oracle candidate、distance gain、oracle rank | 否 |

---

## 4. 配置设计

在 `vlnce_baselines/config/default.py` 增加：

```python
_C.OPENNAV_HARNESS = CN()
_C.OPENNAV_HARNESS.ENABLED = False
_C.OPENNAV_HARNESS.ENABLE_HARNESS_LOGGING = False
_C.OPENNAV_HARNESS.ENABLE_DECISION_EFFECT = False
_C.OPENNAV_HARNESS.TRACE_DIR = "logs/harness_traces"
_C.OPENNAV_HARNESS.TRACE_FORMAT = "jsonl"
_C.OPENNAV_HARNESS.LOG_IMAGES = False
_C.OPENNAV_HARNESS.FAIL_OPEN = True

_C.OPENNAV_HARNESS.GEOMETRY_QUERY = CN()
_C.OPENNAV_HARNESS.GEOMETRY_QUERY.ENABLED = True
_C.OPENNAV_HARNESS.GEOMETRY_QUERY.LOG_ONLY = True

_C.OPENNAV_HARNESS.GROUNDER_DIAGNOSTIC = CN()
_C.OPENNAV_HARNESS.GROUNDER_DIAGNOSTIC.ENABLED = True
_C.OPENNAV_HARNESS.GROUNDER_DIAGNOSTIC.LOG_ONLY = True

_C.OPENNAV_HARNESS.MEMORY_DIAGNOSTIC = CN()
_C.OPENNAV_HARNESS.MEMORY_DIAGNOSTIC.ENABLED = True
_C.OPENNAV_HARNESS.MEMORY_DIAGNOSTIC.LOG_ONLY = True
```

A1 运行配置：

```yaml
OPENNAV_HARNESS:
  ENABLED: true
  ENABLE_HARNESS_LOGGING: true
  ENABLE_DECISION_EFFECT: false
  TRACE_DIR: logs/harness_traces/a1
  FAIL_OPEN: true
  GEOMETRY_QUERY:
    ENABLED: true
    LOG_ONLY: true
  GROUNDER_DIAGNOSTIC:
    ENABLED: true
    LOG_ONLY: true
  MEMORY_DIAGNOSTIC:
    ENABLED: true
    LOG_ONLY: true
```

硬约束：

```python
if cfg.OPENNAV_HARNESS.ENABLE_DECISION_EFFECT:
    raise ValueError("A1 does not allow decision-impact modules.")
```

---

## 5. 数据结构设计

### 5.1 AgentState

`AgentState` 用 dataclass 表达，写日志时转为 JSON-safe dict。

```python
@dataclass
class AgentState:
    episode: EpisodeState
    environment: EnvironmentState
    candidates: list[CandidateState]
    perception_state: PerceptionState
    memory_state: MemoryState
    selector_output: SelectorOutput
    action_state: ActionState
    verifier_output: VerifierOutput
    fallback_state: FallbackState
    metrics: StepMetrics
```

A1 必填字段：

| 区块 | 字段 |
|------|------|
| `episode` | `episode_id`, `split`, `instruction`, `step_id` |
| `environment` | `position`, `heading`, `done`, `distance_to_goal_before`, `distance_to_goal_after` |
| `candidates` | `candidate_id`, `direction_id`, `angle_rad`, `angle_deg`, `distance`, `raw_rank`, `image_ref` |
| `perception_state` | `observation_text_by_direction`, `geometry_queries`, `grounder_diagnostics` |
| `memory_state` | `visit_count`, `nearest_previous_step`, `revisit_score`, `candidate_novelty` |
| `selector_output` | `raw_predictions`, `raw_thoughts`, `fused_predictions`, `selected_candidate`, `reason`, `parse_error` |
| `action_state` | `habitat_action_id`, `angle_rad`, `distance`, `executed` |
| `metrics` | `oracle_candidate`, `oracle_rank_raw`, `distance_gain_selected`, `collision`, `tool_failures` |

### 5.2 CandidateState

```python
@dataclass
class CandidateState:
    candidate_id: str
    direction_id: int
    angle_rad: float
    angle_deg: float
    distance: float
    raw_rank: int
    image_ref: str | None = None
    geometry_score: float | None = None
    grounding_score: float | None = None
    novelty_score: float | None = None
    final_prompt_rank: int | None = None
```

A1 中 `geometry_score`、`grounding_score`、`novelty_score` 可以为空或 diagnostic 值，但不得用于排序。

### 5.3 Trace Event

每个 episode 一个 jsonl 文件：

```text
logs/harness_traces/a1/{split}/{episode_id}.jsonl
```

事件类型：

| event_type | 写入时机 |
|------------|----------|
| `episode_start` | episode 第一步开始前 |
| `step_start` | 每步读取 pose / heading 后 |
| `waypoint_candidates` | waypoint predictor 后 |
| `geometry_query` | GeometryQuery logging 后 |
| `grounder_diagnostic` | GrounderDiagnostic 后 |
| `memory_diagnostic` | VisualGraphMemory 后 |
| `observation` | `observe_environment()` 后 |
| `selector_raw` | `move_to_next_vp()` 后 |
| `selector_fused` | `thought_fusion()` 后 |
| `selector_final` | `test_decisions()` 后 |
| `action_pre_step` | `envs.step()` 前 |
| `action_post_step` | `envs.step()` 后 |
| `episode_end` | metric 汇总后 |
| `tool_failure` | 任意 diagnostic 模块异常时 |

单条事件格式：

```json
{
  "schema_version": "a1.trace.v1",
  "run_id": "a1_val_unseen_seed0",
  "episode_id": "12345",
  "step_id": 3,
  "event_type": "selector_final",
  "timestamp": 1781090000.0,
  "payload": {}
}
```

---

## 6. A1 工具接口

### 6.1 MetricsLogger

职责：

- 管理 episode jsonl 文件；
- 提供 `log_event()`、`log_tool_failure()`、`close_episode()`；
- 自动把 numpy / torch 标量转为 JSON-safe 类型；
- 所有写入失败只记录到普通 logger，不抛出到导航主循环。

接口：

```python
class MetricsLogger:
    def __init__(self, trace_dir: str, run_id: str, rank: int, fail_open: bool = True): ...
    def start_episode(self, episode_id: str, split: str, instruction: str) -> None: ...
    def log_event(self, event_type: str, step_id: int, payload: dict) -> None: ...
    def log_tool_failure(self, tool_name: str, step_id: int, error: Exception, context: dict | None = None) -> None: ...
    def end_episode(self, episode_id: str, metrics: dict) -> None: ...
```

### 6.2 GeometryQueryLogger

A1 不做真实几何裁决，只记录可稳定获取的几何证据。

输入：

- `candidates: list[CandidateState]`
- `position`
- `heading`
- `depth_by_direction` 或 depth image ref
- `distance_to_goal_before / after`，用于 oracle / distance gain

输出：

```python
@dataclass
class GeometryQueryResult:
    query_type: str
    candidate_id: str
    depth_valid: bool | None
    projected_pixel: tuple[int, int] | None
    world_point: tuple[float, float, float] | None
    confidence: float | None
    failure_reason: str | None
```

A1 最小实现：

- `query_type = "candidate_direction_depth"`
- `depth_valid`: 当前 candidate 对应方向深度图是否存在且非空；
- `projected_pixel`: 暂填中心点或 `None`，后续接真实投影；
- `world_point`: 使用 `position + heading + angle + distance` 的粗略水平面估计，标记为 `approx`;
- `confidence`: 只代表诊断数据完整度，不用于决策。

### 6.3 GrounderDiagnostic

A1/B1 只做文本诊断，不调用额外 LLM，不改变 prompt。

输入：

- `instruction`
- `actions`
- `landmarks`
- `observe_dict`
- `candidates`

输出：

```python
{
  "candidate_id": "3",
  "grounding_score": 0.0,
  "matched_constraints": [],
  "failed_constraints": [],
  "score_breakdown": {
    "object": 0.0,
    "room": 0.0,
    "direction": 0.0,
    "relation": 0.0,
    "distance": 0.0
  }
}
```

第一阶段允许先使用轻量关键词匹配：

- 将 `landmarks` 分行 / 逗号拆成短语；
- 在 `observe_dict[direction_id]` 中大小写无关匹配；
- `grounding_score = matched / total`；
- 未匹配短语进入 `failed_constraints`。

该分数只写 trace，不进入 `NAVIGATOR` 或 `DECISION_TEST` prompt。

### 6.4 VisualGraphMemoryDiagnostic

A1/C1 只维护 episode 内短期 pose 记录。

输入：

- 当前 `position`, `heading`
- 历史 step pose
- candidates

输出：

```python
{
  "visit_count": 2,
  "nearest_previous_step": 1,
  "revisit_score": 0.84,
  "candidate_novelty": {
    "0": 0.1,
    "1": 0.7
  },
  "loop_flag": false
}
```

实现规则：

- `revisit_score = exp(-euclidean_distance_to_nearest_history)`；
- `visit_count` 统计小于阈值半径的位置访问次数；
- `candidate_novelty` 用候选粗略 next position 到历史轨迹的最小距离归一化；
- 不写回 `observe_dict`，不影响 `next_vp`。

### 6.5 ContextBuilder Skeleton

A1 只生成 `diagnostic_context` 并写日志：

```python
{
  "candidate_count": 5,
  "has_geometry": true,
  "has_grounder": true,
  "has_memory": true,
  "selected_candidate": "3"
}
```

禁止行为：

- 不修改 `NAVIGATOR['user']`；
- 不修改 `DECISION_TEST['user']`；
- 不增加 LLM 输入字段；
- 不把 GeometryQuery / Grounder / Memory 结果注入 prompt。

### 6.6 OracleMetrics

A1 目标是诊断候选质量。

输入：

- step 前后 `info['position']['distance']` 或 Habitat distance measure；
- candidates 的 angle / distance；
- GT path / goal distance；
- selected candidate。

输出：

```python
{
  "oracle_candidate": "2",
  "oracle_rank_raw": 1,
  "distance_gain_selected": 0.42,
  "oracle_topk_recall": true,
  "query_oracle_agreement": null
}
```

实现分两档：

| 档位 | 方法 | 阶段 |
|------|------|------|
| v1 | 用 step 后实际 distance gain 只给 selected candidate 打分 | A1 最小可用 |
| v2 | 使用 simulator geodesic distance 估计所有候选 next pose 的 oracle rank | A1+ |

第一阶段可以先落 v1，把 v2 标记为 TODO，不阻塞 A1 行为不变验证。

---

## 7. 主循环插桩顺序

在 `_eval_llm()` 中按以下顺序插入。所有代码必须包在 Harness 开关下：

```python
harness_enabled = (
    hasattr(config, "OPENNAV_HARNESS")
    and config.OPENNAV_HARNESS.ENABLED
    and config.OPENNAV_HARNESS.ENABLE_HARNESS_LOGGING
)
```

步骤：

1. 初始化：

```python
if harness_enabled:
    harness_logger = MetricsLogger(...)
    geometry_query = GeometryQueryLogger(...)
    grounder_diag = GrounderDiagnostic(...)
    memory_diag = VisualGraphMemoryDiagnostic(...)
    context_builder = ContextBuilder(...)
```

2. episode loop 开始：

```python
if harness_enabled and episode_changed:
    harness_logger.start_episode(...)
```

3. waypoint predictor 后：

```python
candidates = build_candidate_records(radius_dict, distance_dict, images_dict)
harness_logger.log_event("waypoint_candidates", current_step, {"candidates": candidates})
```

4. diagnostic tools：

```python
geometry = geometry_query.run(candidates, position, heading, images_dict)
grounding = grounder_diag.run(instruction, actions, landmarks, observe_dict, candidates)
memory = memory_diag.update(position, heading, candidates)
diagnostic_context = context_builder.build_diagnostic(candidates, geometry, grounding, memory)
```

5. LLM selector 后：

```python
harness_logger.log_event("selector_raw", current_step, {
    "predictions": predictions,
    "thoughts": thoughts,
    "break_flag": break_flag
})
```

6. final decision 后：

```python
harness_logger.log_event("selector_final", current_step, {
    "selected_candidate": next_vp,
    "thought": thought,
    "error_number": error_number
})
```

7. `envs.step()` 前后：

```python
harness_logger.log_event("action_pre_step", current_step, env_actions[0])
outputs = envs.step(env_actions)
harness_logger.log_event("action_post_step", current_step, summarize_step_outputs(outputs))
```

8. episode done：

```python
harness_logger.end_episode(ep_id, metric)
```

---

## 8. 行为不变保护

A1 必须通过以下工程保护：

1. 决策保护：

```python
assert not config.OPENNAV_HARNESS.ENABLE_DECISION_EFFECT
```

2. 数据流保护：

- diagnostic 结果不写入 `observe_dict`；
- diagnostic 结果不写入 `observation`；
- diagnostic 结果不写入 `history_traj`；
- diagnostic 结果不改变 `fused_pred_thought`；
- diagnostic 结果不改变 `next_vp`；
- diagnostic 结果不改变 `env_actions`。

3. 异常保护：

```python
try:
    result = tool.run(...)
except Exception as exc:
    harness_logger.log_tool_failure("tool_name", current_step, exc)
    result = empty_diagnostic_result()
```

4. 随机性保护：

- 不新增随机采样；
- 不改变当前 `random.seed / np.random.seed / torch.manual_seed`；
- 不改变 LLM retry 次数；
- 不改变当前 `temperature=0`。

5. 性能保护：

- trace 写入采用 jsonl 追加；
- 图像默认不复制，只记录 direction id / image key；
- logging 失败不阻断 eval。

---

## 9. 输出文件

A1 输出建议：

```text
logs/harness_traces/a1/
  val_unseen/
    {episode_id}.jsonl
  summaries/
    stats_ep_ckpt_val_unseen_r0_w1_harness.json
    failed_episode_annotation_seed.csv
```

失败标注 CSV：

```csv
episode_id,success,spl,distance_to_goal,path_length,primary_failure,secondary_failure,notes
```

`primary_failure` 枚举：

```text
candidate_missing
geometry_bad
geometry_query_failed
grounding_wrong
memory_loop
progress_drift
selector_wrong
stop_false_positive
stop_false_negative
recovery_missing
recovery_bad
unknown
```

---

## 10. 最小实现清单

第一阶段实现顺序：

- [ ] 在 `default.py` 增加 `OPENNAV_HARNESS` 配置；
- [ ] 创建 `vlnce_baselines/common/opennav_ext/`；
- [ ] 实现 `MetricsLogger` 和 JSON-safe serializer；
- [ ] 实现 `AgentState` / `CandidateState` dataclass；
- [ ] 在 `construct_image_dicts()` 后构造 candidate records；
- [ ] 在 `_eval_llm()` 加 episode / step trace 插桩；
- [ ] 实现 GeometryQuery logging-only；
- [ ] 实现 GrounderDiagnostic logging-only；
- [ ] 实现 VisualGraphMemoryDiagnostic logging-only；
- [ ] 实现 ContextBuilder skeleton；
- [ ] episode done 时写 summary；
- [ ] 跑 A0 和 A1 同一 split / seed，对比 SR / SPL / NE / TL；
- [ ] 抽取失败 episode 标注 CSV。

---

## 11. A0 / A1 对照命令

A0 保持原始配置运行：

```bash
python run.py \
  --exp_name a0_opennav_val_unseen \
  --exp-config run_OpenNav.yaml \
  --llm gpt-4o-2024-08-06 \
  --api_key "$OPENAI_API_KEY"
```

A1 使用同一配置加 Harness 开关：

```bash
python run.py \
  --exp_name a1_harness_val_unseen \
  --exp-config run_OpenNav.yaml \
  --llm gpt-4o-2024-08-06 \
  --api_key "$OPENAI_API_KEY" \
  OPENNAV_HARNESS.ENABLED true \
  OPENNAV_HARNESS.ENABLE_HARNESS_LOGGING true \
  OPENNAV_HARNESS.ENABLE_DECISION_EFFECT false \
  OPENNAV_HARNESS.TRACE_DIR logs/harness_traces/a1
```

若只做 smoke test：

```bash
python run.py \
  --exp_name a1_harness_smoke \
  --exp-config run_OpenNav.yaml \
  --llm gpt-4o-2024-08-06 \
  --api_key "$OPENAI_API_KEY" \
  EVAL.EPISODE_COUNT 2 \
  OPENNAV_HARNESS.ENABLED true \
  OPENNAV_HARNESS.ENABLE_HARNESS_LOGGING true \
  OPENNAV_HARNESS.ENABLE_DECISION_EFFECT false
```

---

## 12. 不进入第一阶段的内容

以下内容必须留到 A2+：

- soft geometry score 影响候选；
- Grounder rerank；
- memory novelty penalty；
- Global Schema 注入 LLM prompt；
- IDG progress tracking 影响 ContextBuilder；
- TargetVerifier；
- FallbackController；
- stop 逻辑改写；
- prompt 大规模重写；
- LLM / VLM backbone 更换；
- waypoint-free 对照。

---

## 13. 第一阶段完成定义

第一阶段完成不是 SR 提升，而是证明 Harness 外壳可控：

```text
A1 指标 ~= A0 指标
且 A1 trace 能解释每个 episode 的候选、观察、选择、执行和结果。
```

只有该条件成立，后续 A2 / B2 / C2 / E / F 的收益才可以被归因到具体模块。

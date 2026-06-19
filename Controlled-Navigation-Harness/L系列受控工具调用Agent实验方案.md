---
tags:
  - 实验
  - Open-Nav
  - VLN-CE
  - agent
  - tool-calling
  - harness
  - L-series
created: 2026-06-15
updated: 2026-06-15
version: v0.1
role: L 系列受控工具调用 Agent 实验方案
status: draft
related:
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/项目总控]]"
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/实验方案]]"
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/自由工具调用Agent演进路线]]"
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/V系列多模态视觉证据初步实现后调整方案]]"
---

# L 系列受控工具调用 Agent 实验方案

## 0. 一句话目标

```text
L 系列研究的不是“直接把 Open-Nav 改成自由 Agent”，
而是在 Controlled Navigation Harness 内逐级开放 controller 自主性，
验证从 fixed pipeline 到 controlled tool-calling agent 的演进是否可控、可诊断、可复现，并且能改善 STOP、fallback 和失败恢复。
```

L 系列是 Controlled Navigation Harness 的 **controller-autonomy axis**：

```text
A 系列：证明 harness 外壳和诊断状态不污染 baseline
V 系列：建立视觉证据、视觉 STOP verifier、视觉记忆和视觉 selector context
L 系列：在已有 harness / evidence / verifier 基础上，研究工具调度自由度如何逐级增加
```

因此，L 系列的论文卖点不是“使用 LangChain / tool calling”，而是：

```text
Progressive controlled agentization for embodied VLN:
在每一级只增加一种 controller 自由度，并用 typed tools、budget、validator、trace replay 约束导航动作。
```

---

## 1. 核心研究问题

### 1.1 主问题

> 在 VLN-CE 中，能否把 Open-Nav 的固定导航流程逐步演进为受控工具调用 Agent，同时保留可诊断、可消融、可回放能力，并在 STOP 决策、空预测 fallback、局部失败恢复等场景获得可归因收益？

### 1.2 子问题

| 编号 | 问题 | 对应阶段 |
|------|------|----------|
| Q1 | 把当前 pipeline 工具化后，是否能保持原行为基本不变？ | L1 |
| Q2 | 使用 scripted controller 统一路由工具后，是否能让 STOP / fallback 路径更可解释，而不引入明显性能退化？ | L2 |
| Q3 | LLM 作为 shadow planner 时，它想调用的工具与 scripted controller 是否一致？它在哪些场景会过度调用或过早 STOP？ | L3 |
| Q4 | 允许 LLM 在白名单内真实调用工具后，是否能在预算内改善特定失败类型？ | L4 |
| Q5 | controller 自主性增加时，性能、安全性、成本和可复现性之间的 trade-off 如何变化？ | L0-L4 |

### 1.3 可检验假设

| 假设 | 内容 | 验证方式 |
|------|------|----------|
| H1 | L1 工具化是 behavior-preserving 的，指标与 L0 基本一致。 | action agreement、SR/SPL/nDTW paired comparison |
| H2 | L2 scripted controller 不一定显著提高 SR，但能降低 STOP / fallback 链路不可解释比例。 | trace completeness、fallback reason coverage、visual_stop event coverage |
| H3 | L3 shadow planner 能暴露 LLM 的工具偏好与风险，为 L4 白名单提供依据。 | plan-script agreement、unsafe proposal rate、over-call rate |
| H4 | L4 controlled tool-calling 不应追求全局无条件提升，而应在 stop false positive、empty prediction、local recovery 等失败类型上取得局部收益。 | failure-type stratified metrics |
| H5 | 自主性越高，成本和漂移风险越大；论文应报告 Pareto 曲线，而不是只报告最高 SR。 | performance / latency / tool-call budget curve |

---

## 2. L 系列与现有阶段的关系

### 2.1 与 A 系列的关系

A 系列回答：

```text
能不能先把 Open-Nav 变成可诊断 harness，并证明 logging / state / trace 不改变 baseline？
```

L 系列不重复 A0/A1 的 baseline 证明，但必须继承 A 系列原则：

- L1 必须先做 behavior-preserving tool registry；
- 每个 L 阶段都必须能关闭 controller，回到 L0 fixed pipeline；
- 工具化、controller、自主调度不能和模型升级、训练、waypoint-free 切换混在一起。

### 2.2 与 V 系列的关系

V 系列提供 L 系列要调度的主要工具：

| V 模块 | 在 L 系列中的角色 |
|--------|-------------------|
| V1 `VisualEvidenceLogger` | `extract_visual_evidence` 工具 |
| V2 `VisualTargetVerifier` | `verify_stop` 工具，STOP safety gate |
| V3 `VisualEvidenceMemory` | `update_visual_memory` / `query_visual_memory` 工具 |
| V4 `MultimodalSelectorContext` | `build_selector_context` 工具 |
| `VisualEvidenceFallbackRanker` | `rank_fallback` 工具 |

L 系列不得在 V2/V4 尚未稳定时直接进入 L4。最低前置条件：

- V1 schema 已稳定，`candidate_evidence_count=0` 不再由解析形状漂移导致；
- V2 STOP allow 已收紧，`visual_stop_allowed` case 能逐例解释；
- V4 / fallback 能消费候选级证据；
- runtime latency 有可观测字段。

### 2.3 与论文贡献的关系

L 系列可以作为论文主贡献之一，但表述应为：

```text
controlled agentization protocol
progressive tool exposure
budgeted and validator-guarded tool-calling for embodied navigation
```

不建议表述为：

```text
we use a free LLM agent
we integrate LangChain into VLN
LLM autonomously controls navigation
```

---

## 3. 不变量与实验边界

### 3.1 全阶段不变量

除明确标注的 controller 行为外，L0-L4 必须固定：

- 数据集 split、episode list、随机种子；
- waypoint predictor / policy checkpoint；
- LLM / VLM backbone 和推理参数；
- Habitat action space；
- 候选生成逻辑；
- V1/V2/V3/V4 的核心 prompt 和 schema；
- STOP verifier 安全规则；
- 评测脚本和 success radius；
- 最大 episode step 数。

### 3.2 禁止混入的变量

L 系列实验期间禁止同时引入：

- 更换 LLM / VLM；
- 训练或微调新 policy；
- waypoint-free 主链；
- 多 sub-agent 协作；
- 大规模 prompt 重写；
- beam search / 多次采样投票；
- 未进入 trace schema 的隐式 heuristic；
- LangChain / LangGraph 作为方法贡献。

### 3.3 安全硬约束

所有 L 阶段都必须满足：

| 约束 | 要求 |
|------|------|
| STOP | 任何 STOP final action 必须绑定 `verify_stop` 的 allow event |
| move | candidate id 必须来自当前有效候选集合 |
| env step | controller 不能直接调用 `env.step`，只能提交 `ActionProposal` |
| validator | 所有 `ActionProposal` 必须经过 `ActionValidator` |
| budget | 每 step 工具调用数和额外 LLM 调用数必须受配置限制 |
| trace | 每次工具调用、controller 决策、validator 拦截都必须写 jsonl |
| failure | 工具失败不得被解释成成功证据 |

---

## 4. 总体阶段设计

### 4.1 L 系列总览

| 阶段 | 名称 | controller 自主性 | 是否执行 LLM tool plan | 是否改变动作 | 论文作用 |
|------|------|-------------------|--------------------------|--------------|----------|
| L0 | Fixed Harness Pipeline | 无，固定流程 | 否 | 当前固定策略 | 内部对照 |
| L1 | Toolized Fixed Pipeline | 无，固定顺序调用工具 | 否 | 不应改变 | 证明工具化不污染行为 |
| L2 | Scripted Controller | 规则表调度 | 否 | 可以有受控条件分支 | 建立可解释 controller |
| L3 | LLM Shadow Tool Planner | LLM 只提出计划 | 否，只记录 | 不改变 | 分析 LLM 调度风险 |
| L4 | Controlled Tool-Calling Agent | LLM 在白名单内调度 | 是，受 validator / budget 约束 | 是 | 主方法或增强方法 |
| L5 | Runtime Adapter | 外部框架承载 controller | 可选 | 不作为核心变量 | 工程适配 / 未来工作 |

### 4.2 双基准设置

L 系列需要两个基准锚点，避免概念混乱：

| 基准 | 作用 | 使用场景 |
|------|------|----------|
| A0/A1 baseline | 证明 harness logging 不改变原始 Open-Nav | 论文 baseline / 方法前置验证 |
| L0 fixed pipeline | 与 L1-L4 使用同一批工具和 V 系列能力，但 controller 固定 | L 系列主对照 |

L0 不是原始 Open-Nav，而是 **当前稳定 fixed harness policy**。例如 V1/V2/V4/R3 已稳定后，L0 可以包含这些 decision-effect 模块，但工具调用顺序仍固定，controller 不具备自主调度能力。

---

## 5. 工具接口设计

### 5.1 ToolSpec

每个工具必须有稳定接口：

```yaml
ToolSpec:
  name: str
  version: str
  category: perception | memory | progress | selection | verification | fallback | action
  mutates_state: bool
  mutates_environment: bool
  llm_call: bool
  allowed_in_modes: [fixed_pipeline, scripted, llm_plan_log_only, controlled_tool_calling]
  input_schema: dict
  output_schema: dict
  failure_policy: fail_open | fail_closed
  timeout_seconds: float
  trace_fields: list
```

### 5.2 ToolResult

工具返回值统一封装：

```yaml
ToolResult:
  tool_name: str
  tool_version: str
  call_id: str
  episode_id: str
  step_id: int
  ok: bool
  result: dict
  error_type: str | null
  error_message: str | null
  latency_seconds: float
  llm_tokens: dict | null
  state_updates: dict
  trace_refs: dict
```

### 5.3 第一批工具白名单

| 工具 | 来源 | L1 | L2 | L3 | L4 | 说明 |
|------|------|----|----|----|----|------|
| `parse_instruction` | `Open_Nav.get_actions/get_landmarks` | 固定调用 | 固定调用 | 可见不可调 | episode start 固定调用 | 不允许 step 内重复调用 |
| `predict_waypoints` | policy waypoint predictor | 固定调用 | 固定调用 | 可见不可调 | 固定调用 | 候选集合不由 LLM 修改 |
| `observe_candidates` | `observe_environment` | 固定调用 | 固定调用 | 可见可建议 | 可调但受 budget | 初期建议固定，避免漏观察 |
| `extract_visual_evidence` | V1 | 固定调用或按配置 | 规则触发 | 可建议 | 可调 | 高成本工具，L4 必须限预算 |
| `update_visual_memory` | V3 | 固定调用 | 规则触发 | 可建议 | 可调 | 初期 log-only |
| `estimate_completion` | Open-Nav completion | 固定调用 | 规则触发 | 可建议 | 可调 | 文本 LLM 成本较高 |
| `build_selector_context` | V4 / ContextBuilder | 固定调用 | 固定调用 | 可建议 | 固定或可调 | 只构造上下文，不执行动作 |
| `select_waypoint` | selector / `test_decisions` | 固定调用 | 规则触发 | 可建议 | 可调 | 输出候选或 STOP proposal |
| `verify_stop` | V2 | STOP 时触发 | STOP 时强制触发 | 可建议 | STOP 前强制触发 | 不能被跳过 |
| `rank_fallback` | visual fallback | fallback 条件触发 | 规则触发 | 可建议 | 可调 | 只能在受限触发条件内使用 |
| `propose_action` | ActionRouter | 固定调用 | controller 调用 | 不执行 | controller 调用 | 只产生 proposal |
| `validate_action` | ActionValidator | 强制 | 强制 | 强制 | 强制 | 非 LLM 工具 |
| `step_environment` | envs.step | harness 内部 | harness 内部 | harness 内部 | harness 内部 | 不暴露给 LLM controller |

### 5.4 工具分级

为避免 L4 一步到位过宽，工具按风险分级开放：

| 等级 | 工具类型 | 示例 | L4 开放顺序 |
|------|----------|------|-------------|
| T0 | 只读低成本 | state query、trace lookup | 可最先开放 |
| T1 | 只读高成本 | `estimate_completion`、`extract_visual_evidence` | 预算内开放 |
| T2 | 状态更新 | `update_visual_memory` | 先 logging-only，再开放 |
| T3 | 决策建议 | `select_waypoint`、`rank_fallback` | 需要 validator |
| T4 | 安全关键 | `verify_stop` | STOP 前强制调用，但不由 LLM 决定是否绕过 |
| T5 | 环境动作 | `step_environment` | 不开放给 LLM |

---

## 6. AgentState 与 Trace Schema

### 6.1 L 系列 AgentState 最小字段

```yaml
AgentState:
  episode:
    episode_id: str
    instruction: str
    step_id: int
    split: str

  budget:
    max_tool_calls_per_step: int
    tool_calls_used_this_step: int
    max_extra_llm_calls_per_episode: int
    extra_llm_calls_used: int
    latency_budget_seconds: float | null

  candidates:
    candidate_ids: list
    selected_candidate: str | null
    invalid_candidate_attempts: list

  evidence:
    visual_evidence_event_id: str | null
    candidate_evidence_count: int
    schema_warnings: list

  memory:
    visual_memory_event_id: str | null
    revisit_score: float | null
    loop_risk: str | null

  progress:
    actions: list
    landmarks: list
    completion_estimate: dict | null

  decision:
    selector_event_id: str | null
    selector_prediction: dict | null
    stop_proposal: bool
    uncertainty_type: str | null

  verification:
    verifier_event_id: str | null
    stop_verdict: allow | reject | uncertain | not_applicable
    allow_blockers: list

  fallback:
    fallback_event_id: str | null
    trigger_reason: str | null
    ranked_candidates: list

  tool_history:
    calls_this_step: list
    calls_this_episode: int

  action:
    proposal: dict | null
    validator_verdict: allow | reject | repaired | not_run
    executed_action: dict | null
```

### 6.2 Planner-visible state 与 audit-only state

AgentState 中有些字段用于离线分析和失败归因，不能进入 L3/L4 planner prompt。否则 controller 会看到评测标签或 oracle 信息，导致实验泄漏。

| 字段类型 | 示例 | 是否给 planner | 用途 |
|----------|------|----------------|------|
| episode / instruction | instruction、step_id、actions、landmarks | 是 | 导航决策上下文 |
| candidate operational fields | candidate_id、angle、distance、raw observation summary | 是 | 合法动作选择 |
| compressed evidence | V1 candidate summary、V4 context、verifier verdict | 是 | 工具调度和决策 |
| budget fields | budget_left、tool_calls_used | 是 | 控制成本 |
| oracle metrics | oracle_candidate、oracle_rank、distance_gain | 否 | 离线诊断 |
| evaluation labels | success、distance_to_goal、SPL、nDTW | 否 | 评测 |
| post-step metrics | collision、distance_to_goal_after | 否，当前 step 不可见 | 事后分析 |
| full trace history | 全量 JSONL、全量原始 LLM/VLM 输出 | 否 | replay / audit |

因此需要维护两个 view：

```text
AgentStateFull:
  用于 trace、replay、offline audit，可以包含 oracle 和 metric 字段。

PlannerStateView:
  用于 L3/L4 controller prompt，只包含当前导航可观测信息、压缩证据、工具历史和预算。
```

所有 L3/L4 planner 输入必须由 `PlannerStateView` 生成，并在 trace 中记录 view schema version。

### 6.3 Trace 事件

L 系列新增 trace 事件：

| 事件 | 用途 |
|------|------|
| `agent_step_start` | 每 step 初始化 state 和 budget |
| `controller_decision` | 记录 controller 选择哪个工具或 final action |
| `llm_tool_plan` | L3/L4 记录 LLM planner 输出 |
| `tool_call_start` | 工具调用前记录输入摘要 |
| `tool_call_end` | 工具成功返回 |
| `tool_call_error` | 工具失败、timeout、schema error |
| `budget_exhausted` | 工具调用或 LLM 调用超预算 |
| `action_proposal` | controller 提交 move / stop |
| `action_validation` | validator allow / reject / repair |
| `safety_intervention` | STOP verifier、validator 或 budget policy 拦截 |
| `agent_step_end` | step 结束，汇总耗时、动作、指标 |

### 6.4 Replay 要求

每个 L 阶段都必须支持两类 replay：

| 类型 | 目的 | 要求 |
|------|------|------|
| tool-output replay | 固定工具输出，复跑 controller | controller 决策应可复现 |
| trace audit replay | 从 jsonl 恢复每步工具链 | 能解释最终动作来源 |

L4 若因 LLM nondeterminism 无法保证在线完全复现，至少必须保证：

- temperature 固定为 0 或最低可用值；
- planner prompt 和 tool list 版本写入 trace；
- 每次 planner 原始输出写入 trace；
- validator 拦截逻辑完全 deterministic；
- 对同一 trace 的离线审计结论一致。

---

## 7. 各阶段实验方案

### 7.1 L0：Fixed Harness Pipeline

#### 目标

建立 L 系列内部对照：

```text
同一批工具和证据模块，固定调用顺序，无 controller 自主调度。
```

L0 可以对应当前最稳定的 V 系列 decision-effect 配置，例如：

```text
V1 visual evidence
V2 conservative STOP verifier
V3 visual evidence memory logging-only
V4 multimodal selector context
visual ranked fallback
runtime latency logging
```

但必须等 V2 STOP allow、fallback、schema、runtime 先通过小样本复核。

#### 配置

```yaml
OPENNAV_HARNESS:
  AGENT_CONTROLLER:
    ENABLED: false
    MODE: fixed_pipeline
    TRACE_TOOL_CALLS: false
```

#### 记录指标

- SR / SPL / nDTW / NE / TL；
- `visual_stop_allowed` / `visual_stop_rejected`；
- `selector_empty_prediction_fallback.changed`；
- V1 schema warnings；
- tool latency by operation；
- STOP false positive case study；
- fallback positive distance gain。

#### 通过标准

- 当前 V 系列关键问题已可解释；
- 每个 STOP allow 都能回溯到 verifier evidence；
- fallback 不再退化为无解释 first-candidate；
- 形成固定 episode list，供 L1-L4 paired comparison。

---

### 7.2 L1：Toolized Fixed Pipeline

#### 目标

将 L0 的固定流程包装成 Tool Registry，但保持调用顺序和行为不变。

```text
L0 pipeline code
  -> L1 ToolSpec / ToolResult / ToolRegistry
  -> same order, same inputs, same outputs, same action
```

#### 改动

- 新增 `ToolSpec`、`ToolResult`、`ToolRegistry`；
- 每个已有模块注册为 typed tool；
- 主循环仍按 L0 顺序调用；
- trace 新增 `tool_call_start/end/error`；
- 工具失败按配置 `fail_open` 或 `fail_closed`，但默认行为必须匹配 L0。

#### 配置

```yaml
OPENNAV_HARNESS:
  AGENT_CONTROLLER:
    ENABLED: true
    MODE: toolized_fixed_pipeline
    TRACE_TOOL_CALLS: true
    REQUIRE_ACTION_VALIDATOR: true
    REQUIRE_VERIFIER_FOR_STOP: true
```

#### 对照

| 对照 | 差异 | 目标 |
|------|------|------|
| L0 vs L1-online | 真实运行 | 指标不应显著变化 |
| L0 trace vs L1 replay | 固定工具输出离线复跑 | 动作应一致 |
| L1 tool failure smoke | 人工模拟工具失败 | fail-open / fail-closed 记录正确 |

#### 指标

- action agreement；
- selected candidate agreement；
- STOP verdict agreement；
- fallback selected candidate agreement；
- tool trace completeness；
- tool latency overhead；
- schema error / tool failure rate。

#### 通过标准

```text
离线 replay action agreement >= 99%
在线 paired SR/SPL/nDTW 无明显退化
tool trace completeness = 100%
STOP bypass count = 0
invalid action executed count = 0
```

如果 L1 与 L0 出现差异，必须先定位是：

- 工具包装改变了输入；
- trace / serialization 改变了数据形状；
- LLM prompt 被意外改动；
- fail-open / fail-closed 策略改变了动作；
- nondeterministic LLM 调用导致自然漂移。

---

### 7.3 L2：Scripted Controller

#### 目标

引入 `AgentController`，但不让 LLM 调度工具。Controller 只按规则表和 AgentState 选择下一步工具。

```text
AgentState -> scripted controller -> tool call / action proposal
```

L2 的价值不是“更聪明”，而是：

- 把工具触发原因显式化；
- 把 STOP、fallback、budget、validator 变成统一路由；
- 为 L3/L4 提供 scripted reference policy。

#### L2 分两档

| 版本 | 内容 | 作用 |
|------|------|------|
| L2a strict scripted | 严格复刻 L1 固定顺序，只是由 controller 发出调用 | 验证 controller 外壳不改变行为 |
| L2b conditional scripted | 根据 state 条件触发 `verify_stop`、`rank_fallback`、memory update 等 | 形成真实规则 controller |

#### 规则表示例

| 条件 | controller 行为 |
|------|-----------------|
| step start 且无 candidates | `predict_waypoints` |
| candidates 已有但无 visual evidence | `extract_visual_evidence`，除非预算关闭 |
| selector 未运行 | `build_selector_context` -> `select_waypoint` |
| selector 提出 STOP | 强制 `verify_stop` |
| STOP allow | `propose_action(stop)` -> `validate_action` |
| STOP reject / uncertain | `rank_fallback` -> `propose_action(move)` |
| selector 空预测 | `rank_fallback` |
| candidate id 无效 | `rank_fallback` 或 fallback fail-closed |
| tool budget 用尽 | 降级为 L0 fixed fallback，并记录 `budget_exhausted` |

预算降级不能静默发生。任何从 scripted controller 降级到 L0 fixed fallback 的 step 都必须记录：

```yaml
budget_degradation:
  exhausted_budget_type: tool_calls | llm_calls | latency
  skipped_tools: list
  fallback_policy: l0_fixed | fail_closed | previous_safe_action
  selected_candidate_before_degradation: str | null
  selected_candidate_after_degradation: str | null
```

这类 step 在统计中需要单独分桶；否则 L2/L4 的性能可能实际来自 L0 fallback，而不是 controller 本身。

#### 配置

```yaml
OPENNAV_HARNESS:
  AGENT_CONTROLLER:
    ENABLED: true
    MODE: scripted
    SCRIPTED_VARIANT: strict  # strict | conditional
    MAX_TOOL_CALLS_PER_STEP: 4
    REQUIRE_VERIFIER_FOR_STOP: true
    REQUIRE_ACTION_VALIDATOR: true
```

#### 对照

| 对照 | 目的 |
|------|------|
| L1 vs L2a | controller 外壳是否 behavior-preserving |
| L2a vs L2b | 条件化路由是否改善 trace 和 fallback |
| L0 vs L2b | 规则 controller 对主指标和安全指标的影响 |

#### 指标

- controller reason coverage；
- STOP gate source coverage；
- fallback reason coverage；
- number of safety interventions；
- budget exhausted rate；
- budget degradation action delta；
- action agreement with L1；
- SR / SPL / nDTW；
- stop false positive / false negative；
- fallback positive distance gain。

#### 通过标准

- L2a 与 L1 基本一致；
- L2b 每个工具调用都有可解释 reason；
- STOP verifier bypass 为 0；
- fallback event 不允许缺少 trigger reason；
- 与 L1 相比，主指标不出现不可解释退化。

---

### 7.4 L3：LLM Tool Planner Logging-only

#### 目标

让 LLM 读取压缩 AgentState 和工具列表，输出它“想调用的下一个工具”或“想提交的 final action”，但实际执行仍由 L2 scripted controller 决定。

```text
L3 = L2 execution + LLM shadow plan logging
```

L3 是 L4 的安全前置实验。它回答：

- LLM 是否理解工具边界？
- LLM 是否倾向过度调用高成本工具？
- LLM 是否想绕过 STOP verifier？
- 哪些工具可以安全开放？
- 哪些 state 字段会诱导错误 STOP 或错误 fallback？

#### Planner 输入

只给 LLM 压缩状态：

```yaml
PlannerInput:
  instruction_summary
  step_id
  budget_left
  candidate_count
  evidence_summary
  memory_summary
  selector_summary
  verifier_summary
  fallback_summary
  allowed_tools
  safety_rules
```

不输入：

- 全量 RGB / depth；
- 全量历史日志；
- 未压缩 VLM 输出；
- Habitat oracle 信息；
- success / distance_to_goal 等评测标签。

#### Planner 输出

```json
{
  "thought": "STOP was proposed, but verifier has not run yet.",
  "tool": "verify_stop",
  "args": {
    "source": "selector_stop_gate",
    "selected_candidate": "0"
  },
  "risk": "stop_requires_verifier"
}
```

或：

```json
{
  "thought": "Candidate 3 matches the next landmark and no STOP is supported.",
  "final": {
    "type": "move",
    "candidate_id": "3"
  },
  "risk": "none"
}
```

#### 评估指标

| 指标 | 含义 |
|------|------|
| JSON parse rate | planner 输出是否结构化 |
| valid tool rate | 工具名是否在白名单 |
| valid args rate | 参数是否引用当前 state 中存在的 id |
| plan-script agreement | LLM plan 与 L2 下一步工具是否一致 |
| unsafe proposal rate | 是否试图 STOP without verifier、直接 step env、使用非法 candidate |
| over-call rate | 是否建议额外调用不必要高成本工具 |
| under-call rate | 是否漏掉必须工具，例如 STOP 前不 verify |
| premature stop proposal rate | 未满足 STOP evidence 时建议 STOP |
| budget awareness rate | 预算不足时是否仍建议高成本工具 |

#### 对照

| 版本 | 输入差异 | 目的 |
|------|----------|------|
| L3-min | 只给必要 state | 看最小 planner 能力 |
| L3+rules | 加 safety rules | 看规则显式化是否降低 unsafe |
| L3+examples | 加少量合法/非法例子 | 看 parse 和工具选择是否改善 |

#### 通过标准

L3 进入 L4 前至少满足：

```text
JSON parse rate >= 95%
valid tool rate >= 95%
STOP bypass proposal rate 接近 0，或能被 validator 稳定拦截
invalid candidate proposal rate 可被 validator 稳定识别
明确得到 L4 初始 allowlist 和 denylist
```

如果 L3 显示 LLM 大量过度调用 V1 或频繁建议提前 STOP，则 L4 只能先开放信息查询和 fallback ranker，不能开放完整 tool-calling。

---

### 7.5 L4：Controlled Tool-Calling Agent

#### 目标

允许 LLM controller 在白名单工具内真实选择下一步工具调用，但所有 final action 必须经过 ActionValidator，STOP 必须绑定 verifier allow。

```text
AgentState
  -> LLM controller chooses tool
  -> Tool executes
  -> state updates
  -> repeat within budget
  -> ActionProposal
  -> ActionValidator
  -> env.step or STOP
```

#### Controller 与 selector 的边界

L4 中有两个 LLM 相关角色，必须分开：

| 角色 | 职责 | 输出 | 是否直接执行动作 |
|------|------|------|------------------|
| LLM Controller / Planner | 决定下一步调用哪个工具，或在证据充分时提交 final action proposal | `ToolCall` 或 `ActionProposal` | 否 |
| LLM Selector | 在给定候选和上下文中做语义候选裁决 | candidate id / STOP proposal / uncertainty | 否 |

默认策略：

- L4a/L4b 中，LLM Controller 不直接替代 `select_waypoint`，而是决定是否调用 `select_waypoint`、`verify_stop`、`rank_fallback` 等工具；
- `select_waypoint` 仍是一个工具，它可以内部调用原 Open-Nav selector；
- Controller 提交的 final move 如果没有经过 `select_waypoint` 或 `rank_fallback` 支持，必须被标记为 `direct_controller_action`，单独统计；
- 初期不建议允许 `direct_controller_action` 进入执行链，除非作为 L4d 的单独 ablation。

这样可以避免 L4 同时改变“谁选择候选”和“何时调用工具”，导致归因混乱。

#### L4 分档开放

| 版本 | 开放范围 | 目的 |
|------|----------|------|
| L4a info-only | `estimate_completion`、`build_selector_context`、`verify_stop` 这类信息/验证工具 | 验证 LLM 调度不会破坏安全 |
| L4b fallback-aware | L4a + `rank_fallback` | 重点验证 STOP reject / empty prediction 恢复 |
| L4c evidence-budgeted | L4b + 受限 `extract_visual_evidence` / `update_visual_memory` | 验证高成本感知工具是否值得动态调用 |
| L4d full controlled | 白名单内完整调度，但不开放 `step_environment` | 作为最终 agent variant |

#### 默认预算

```yaml
OPENNAV_HARNESS:
  AGENT_CONTROLLER:
    MODE: controlled_tool_calling
    MAX_TOOL_CALLS_PER_STEP: 4
    MAX_LLM_TOOL_PLANS_PER_STEP: 1
    MAX_EXTRA_LLM_CALLS_PER_EPISODE: 12
    REQUIRE_VERIFIER_FOR_STOP: true
    REQUIRE_ACTION_VALIDATOR: true
    TRACE_TOOL_CALLS: true
```

预算建议从保守开始：

| 场景 | 工具调用预算 |
|------|--------------|
| 正常 move | 2-3 次 |
| STOP proposal | 3-4 次，必须包含 `verify_stop` |
| empty prediction | 3-4 次，允许 `rank_fallback` |
| tool failure | 不追加无限重试，最多一次 fallback |

#### ActionValidator 规则

| final action | 验证条件 |
|--------------|----------|
| move | candidate id 在当前候选集合中，未被标记 invalid，distance/angle 有效，并有 selector / fallback / allowed-direct-controller 来源 |
| stop | 存在当前 step 的 `verify_stop` allow event，且 source 与 proposal 对齐 |
| fallback move | 有 fallback trigger reason，ranked candidate 来源可回溯 |
| direct controller move | 默认拒绝；若在 L4d ablation 中开放，必须单独计数并和 selector-supported move 分开报告 |
| no-op | 默认不允许，除非环境 done 或显式 emergency policy |

#### 对照

| 对照 | 目的 |
|------|------|
| L2b vs L4a | LLM 调度信息工具是否带来收益或成本 |
| L4a vs L4b | LLM 是否能更好处理 STOP reject / empty prediction |
| L4b vs L4c | 动态视觉证据调用是否值得额外 latency |
| L0 vs L4d | 最终 controlled tool-calling 与 fixed pipeline 的整体差异 |

#### 主指标

- SR / SPL / nDTW / NE；
- stop false positive / false negative；
- fallback positive distance gain；
- failure-type recovery rate；
- tool calls per step；
- extra LLM calls per episode；
- latency per step；
- budget exhausted rate；
- validator intervention rate；
- replay audit completeness。

额外必须报告：

- `selector_supported_move_rate`；
- `fallback_supported_move_rate`；
- `direct_controller_action_rate`；
- `direct_controller_action_success_delta`，仅在 L4d ablation 开放时统计。

#### 通过标准

L4 不以“所有指标都涨”为唯一目标，而看是否满足：

```text
STOP verifier bypass = 0
invalid action executed = 0
direct controller action executed = 0，除非处于明确 L4d-direct ablation
tool-call budget violation = 0
trace completeness = 100%
在至少一个预定义失败类型上优于 L2b
性能/成本 trade-off 可解释
```

如果 L4 主指标不升，但显著降低 stop false positive 或改善 empty-prediction fallback，也可以作为论文中的 agentic-control 分析结果，而不是强行包装成全面提升。

---

### 7.6 L5：LangChain / LangGraph Runtime Adapter

#### 目标

L5 不是主实验阶段。只有当 L1-L4 的本地 Tool Registry、AgentState、ActionValidator 和 trace schema 稳定后，才考虑把 controller runtime 换成 LangGraph / LangChain。

#### 原则

- LangChain / LangGraph 只作为 runtime；
- ToolSpec / ToolResult / AgentState / ActionValidator 仍由项目内定义；
- trace 中不写第三方框架内部对象；
- L5 必须能回退到本地 controller；
- L5 不参与主论文收益归因。

#### 验收

```text
L5 与 L4d 在固定 trace replay 上 action agreement >= 99%
trace schema 不变
安全规则不变
无额外隐式 tool retry
```

---

## 8. 实验矩阵与运行顺序

### 8.1 分层运行规模

| 层级 | episode 数 | 用途 |
|------|------------|------|
| S0 unit / isolated smoke | 人工构造 case | schema、validator、tool failure |
| S1 focused smoke | 10-12 | 快速检查 STOP / fallback / schema |
| S2 balanced subset | 30-50 | 覆盖候选缺失、STOP、loop、fallback 等失败类型 |
| S3 report subset | 100 | 论文主表或附表 |
| S4 full split | 视资源决定 | 最终大规模验证 |

当前建议不要直接从 L0 跑到 L4 的 100 episode。应按：

```text
L0 S1 -> L1 S1 -> L1 replay -> L2a S1 -> L2b S1
-> L3 S1/S2 shadow analysis
-> L4a S1 -> L4b S1/S2 -> L4c if latency allows
-> 选择最稳定 variant 跑 S3
```

### 8.2 主矩阵

| 编号 | 配置 | 变化 | 运行规模 | 先决条件 |
|------|------|------|----------|----------|
| L0-fixed | 当前稳定 fixed harness | 无 controller | S1/S3 | V2/V4/R3 稳定 |
| L1-toolized | L0 + Tool Registry | 行为不应变 | S1/S2 + replay | L0 固定 |
| L2a-scripted-strict | L1 + strict controller | 外壳变化 | S1 | L1 agreement 通过 |
| L2b-scripted-conditional | L2a + 条件路由 | 规则 controller | S1/S2 | L2a 通过 |
| L3-shadow-min | L2b + LLM shadow plan | 不执行 plan | S1/S2 | L2b 通过 |
| L3-shadow-rules | L3 + safety rules | 不执行 plan | S1/S2 | L3-min parse 可用 |
| L4a-info-only | 执行低风险工具计划 | controlled tool-calling | S1 | L3 unsafe 可控 |
| L4b-fallback-aware | L4a + fallback rank | 重点恢复 | S1/S2 | L4a 安全 |
| L4c-evidence-budgeted | L4b + 动态 V1/V3 | 高成本工具调度 | S1/S2 | latency 可接受 |
| L4d-full-controlled | 白名单完整调度 | 最终 variant | S2/S3 | L4b/c case study 通过 |
| L5-runtime-adapter | runtime 替换 | 非主实验 | replay | L4d 稳定 |

### 8.3 近期最小可执行矩阵

若资源有限，最小矩阵是：

```text
L0-fixed
L1-toolized
L2b-scripted-conditional
L3-shadow-rules
L4b-fallback-aware
```

原因：

- L1 证明工具化无害；
- L2b 提供 scripted reference；
- L3 证明 LLM planner 的风险和可开放范围；
- L4b 聚焦最有希望的 failure recovery，不先开放高成本动态视觉调用。

### 8.4 L4 内部消融

L4 不能只报一个最终版本，否则无法判断收益来自工具调度、fallback、视觉证据还是 validator。至少保留以下内部消融：

| 消融 | 改动 | 观察 |
|------|------|------|
| L4b-no-direct | 禁止 controller 直接提交 move，只能通过 selector / fallback 支持 | 默认主版本 |
| L4b-direct-log-only | controller 可以提出 direct move，但不执行，只记录 | 评估 controller 候选选择能力 |
| L4b-no-fallback-tool | 移除 `rank_fallback`，其余不变 | 检查恢复收益是否来自 fallback tool |
| L4b-no-extra-completion | 禁止额外 `estimate_completion` | 检查额外文本 LLM 是否值得成本 |
| L4c-fixed-evidence | V1 固定调用，不允许动态增减 | 对比动态视觉工具调度收益 |
| L4c-budget-tight | 降低 `MAX_TOOL_CALLS_PER_STEP` | 画成本 / 性能曲线 |

如果资源只能跑一个 L4 主版本，默认选择：

```text
L4b-no-direct:
  controller 可调度 verify_stop / rank_fallback / selector，
  但不能绕过 selector 或 fallback 直接执行 move。
```

---

## 9. 指标体系

### 9.1 主导航指标

| 指标 | 说明 |
|------|------|
| SR | success rate |
| SPL | path efficiency weighted success |
| nDTW | path similarity |
| OSR | oracle success rate |
| NE | final navigation error |
| TL | trajectory length |

### 9.2 L 系列 controller 指标

| 指标 | 说明 |
|------|------|
| action agreement | 与 L0/L1 的动作一致率 |
| selected candidate agreement | 候选选择一致率 |
| controller reason coverage | controller 决策是否都有 reason |
| plan-script agreement | L3 planner 与 L2 scripted 下一步一致率 |
| unsafe proposal rate | planner 提出违反安全规则的比例 |
| invalid args rate | 工具参数非法比例 |
| validator intervention rate | validator 拦截或修正比例 |
| budget exhausted rate | 工具预算耗尽比例 |
| tool calls per step | 每步工具调用数 |
| extra LLM calls per episode | 每 episode 额外 LLM 调用数 |
| replay agreement | 固定 trace replay 的动作一致率 |

### 9.3 STOP 安全指标

| 指标 | 说明 |
|------|------|
| STOP verifier bypass | 未经 verifier allow 的 STOP，必须为 0 |
| visual_stop_allowed count | 被视觉 verifier 放行的 STOP 数 |
| visual_stop_rejected count | 被视觉 verifier 拦截的 STOP 数 |
| stop false positive | 未到目标却停止 |
| stop false negative | 到过目标附近但未停止 |
| generic blocker near-goal rate | generic blocker 是否误挡真实 STOP |
| allow blocker distribution | 哪些规则最常阻止 allow |

### 9.4 fallback / recovery 指标

| 指标 | 说明 |
|------|------|
| fallback trigger rate | fallback 触发比例 |
| fallback reason coverage | fallback 是否有结构化原因 |
| fallback changed rate | 是否改变旧 first-candidate fallback |
| fallback positive distance gain | fallback 动作是否接近目标 |
| recovery success rate | fallback 后 episode 是否成功或 OSR 改善 |
| over-trigger rate | 不需要 fallback 时是否过度触发 |

### 9.5 成本与稳定性指标

| 指标 | 说明 |
|------|------|
| step latency mean / p95 | 单步耗时 |
| tool latency by operation | 各工具耗时 |
| token cost | 文本 LLM / VLM token |
| parse error rate | planner / selector / V1 解析失败 |
| schema warning rate | 合法 JSON 但 schema 不可用 |
| tool failure rate | 工具异常或 timeout |
| trace size per episode | 日志体积 |

### 9.6 演进性指标

L 系列必须证明“演进”不是命名，而是 controller 自主性、工具使用方式和安全干预模式的可测变化。

| 指标 | 说明 | 预期趋势 |
|------|------|----------|
| autonomy level | L0=0、L1=1、L2=2、L3=shadow、L4=executed tool choice | 阶段递增 |
| controller choice points per step | 每步由 controller 做出的工具/动作选择点数量 | L0/L1 近 0，L2/L4 增加 |
| dynamic tool-call ratio | 非固定顺序触发的工具调用比例 | L2/L4 高于 L1 |
| tool diversity per episode | episode 内实际使用的工具类型数 | L4b/L4c 高于 L2 |
| scripted agreement | L3/L4 与 L2 scripted reference 的一致率 | 用于解释偏离是否合理 |
| safety intervention density | 每 100 step validator / verifier / budget 拦截次数 | L4 可能增加，但必须可解释 |
| cost per recovered failure | 每修复一个 failure case 带来的额外工具/LLM成本 | 用于判断 agent 化是否值得 |
| action source distribution | selector / fallback / verifier-stop / direct-controller 的动作来源分布 | 证明动作没有被黑箱替代 |

推荐绘制：

```text
x-axis: L0 -> L1 -> L2 -> L3 -> L4
y-axis-1: SR / nDTW / failure recovery
y-axis-2: tool calls per step / latency
y-axis-3: unsafe proposal / safety intervention
```

如果 L4 的 SR 没有超过 L2，但在 `selector_empty` 或 `stop_false_positive` 上有更好的 recovery / safety，并且成本可解释，仍然可以证明“受控演进”有研究价值。

---

## 10. 失败类型分层评估

L4 的收益很可能不是全局平均收益，而是对特定失败类型有效。因此每个 episode 至少标注一个主失败类型。

| 类型 | 判断依据 | L 系列关注点 |
|------|----------|--------------|
| `candidate_missing` | oracle candidate 不在 top-k | L 系列通常解决不了，只记录 |
| `selector_empty` | selector 输出空或无效 | L4b fallback-aware 重点 |
| `stop_false_positive` | 误停 | verify_stop / validator 必须降低 |
| `stop_false_negative` | 该停不停 | L4 是否能更合理触发 verify |
| `visual_schema_bad` | V1 schema 不可消费 | L1/L2 应 fail-open 并记录 |
| `grounding_wrong` | 选择语义不匹配候选 | L4 可能通过额外 evidence 改善 |
| `memory_loop` | 局部回环 | L4c 才考虑 memory tool |
| `fallback_bad` | fallback 选择使距离变差 | L4b 重点 |
| `budget_exhausted` | 工具预算不足导致降级 | 反映成本约束 |
| `planner_unsafe` | LLM shadow / controller 违反安全规则 | L3/L4 重点风险 |

报告结果时建议给出：

```text
overall metrics
+ failure-type stratified metrics
+ tool-call / latency cost
+ safety intervention analysis
```

---

## 11. 统计与报告规范

### 11.1 paired episode

L0-L4 必须使用相同 episode list。每个阶段输出 paired table：

```text
episode_id
L0 success / SPL / nDTW / stop reason
L2 success / SPL / nDTW / stop reason
L4 success / SPL / nDTW / stop reason
failure type
tool-call count
```

### 11.2 不确定性报告

对于 S2/S3，应使用 bootstrap confidence interval 或至少报告 episode-level paired wins/losses：

```text
L4 wins over L2b
L4 loses to L2b
ties
mean delta NE
mean delta nDTW
```

### 11.3 不能声明的结论

如果只跑 S1，不允许声明：

- L4 全局优于 Open-Nav；
- 工具调用 Agent 显著提升 VLN；
- 自由工具调用可靠。

S1 只能声明：

- trace 是否完整；
- STOP / fallback case 是否合理；
- 是否值得扩大到 S2/S3。

---

## 12. 论文呈现方式

### 12.1 推荐贡献表述

```text
We introduce a progressive controlled-agentization protocol for VLN-CE,
which refactors a fixed Open-Nav pipeline into typed tools and gradually
increases controller autonomy from scripted routing to shadow planning and
budgeted tool-calling, while guarding all physical actions with validators
and STOP decisions with visual verification.
```

中文：

```text
我们提出面向 VLN-CE 的渐进式受控 Agent 化框架，
先将固定 Open-Nav pipeline 工具化，再逐级提升 controller 自主性；
同时用预算、动作验证、STOP verifier 和可回放 trace 保证 embodied navigation 中的工具调用可控、可诊断、可消融。
```

### 12.2 推荐图表

| 图表 | 内容 |
|------|------|
| Figure 1 | L0-L4 controller autonomy ladder |
| Figure 2 | Tool Registry + AgentState + ActionValidator 架构图 |
| Figure 3 | 一条 episode 的 tool-call trace，可视化 STOP reject -> fallback |
| Figure 4 | 性能 / 安全 / 成本 Pareto 曲线 |
| Table 1 | L0-L4 主指标和 controller 指标 |
| Table 2 | failure-type stratified recovery |
| Table 3 | L3 shadow planner agreement / unsafe proposal |
| Case Study | 误停被 verifier 拦截、空预测由 fallback ranker 修复、LLM planner 被 validator 拦截 |

### 12.3 卖点边界

应该强调：

- embodied navigation 中 tool-calling 不能无约束；
- fixed pipeline 到 agent 的演进需要中间层；
- L3 shadow planning 是安全开放工具的前置评估；
- L4 的价值在于受限失败恢复和 STOP 安全，而不是炫技式自由调用。

避免强调：

- 使用某个 agent 框架；
- LLM 完全自主控制环境；
- 一次性 end-to-end agent；
- 没有成本约束的多轮工具调用。

---

## 13. 实现顺序

### P0：固定 L0

1. 先完成当前 V2 STOP allow / fallback / schema / runtime 的小样本复核。
2. 固定 L0 配置和 episode list。
3. 记录 L0 S1/S3 的主指标和 case study。

### P1：L1 Tool Registry

1. 新增 `opennav_agent/tool_spec.py`、`tool_registry.py`、`trace_schema.py`。
2. 包装 V1/V2/V3/V4/fallback/selector，但保持固定调用顺序。
3. 跑 L0 vs L1 replay 和 S1。

### P2：L2 Scripted Controller

1. 新增 `agent_state.py`、`controller.py`、`action_validator.py`。
2. 先实现 L2a strict，再实现 L2b conditional。
3. 检查 STOP / fallback / budget trace 是否完整。

### P3：L3 Shadow Planner

1. 新增 `llm_tool_planner.py`。
2. 只记录 planner 输出，不执行。
3. 统计 agreement、unsafe、over-call。
4. 形成 L4 allowlist / denylist。

### P4：L4 Controlled Tool-Calling

1. 先开放 L4a info-only。
2. 安全后开放 L4b fallback-aware。
3. 只有 latency 可接受时再开放 L4c evidence-budgeted。
4. 选择最稳定 variant 跑 S2/S3。

### P5：论文材料

1. 整理 paired episode table。
2. 整理 L3 planner 风险分析。
3. 整理 L4 case study。
4. 画 autonomy / safety / cost 曲线。

---

## 14. 风险与应对

| 风险 | 表现 | 应对 |
|------|------|------|
| 工具化改变行为 | L1 指标明显偏离 L0 | 先做 replay action agreement，禁止进入 L2 |
| LLM planner 不稳定 | L3 输出无法解析或频繁非法 | 加 rules / examples；仍不稳定则不进入 L4 |
| STOP 风险 | L4 提前 STOP | verifier allow 强绑定，validator 拦截 |
| 成本失控 | 每 step 多次 LLM/VLM | max tool calls、episode LLM budget、latency report |
| 归因混乱 | V2/V4/L4 同时变化 | L0 固定，L1/L2/L3/L4 逐级 paired comparison |
| trace 过大 | jsonl 难分析 | 压缩字段、图片只存 ref、长文本截断 |
| 工具失败被误用 | parse error 被当成 evidence | `ok=false` 和 schema warning 强制进入 state |
| 审稿质疑只是工程 | 没有清楚实验问题 | 报告 autonomy ladder、shadow planner、safety/cost Pareto |

---

## 15. Go / No-Go 决策

| 节点 | Go 条件 | No-Go 处理 |
|------|---------|------------|
| L0 -> L1 | L0 V 系列关键事件可解释 | 回到 V2/V4 修复 |
| L1 -> L2 | replay action agreement >= 99%，trace 完整 | 修 ToolSpec / ToolResult |
| L2 -> L3 | scripted controller 无安全绕过 | 修规则表和 validator |
| L3 -> L4 | unsafe proposal 可控，allowlist 明确 | L3 只作为分析，不执行 L4 |
| L4a -> L4b | info-only 无安全退化 | 收窄工具白名单 |
| L4b -> L4c | fallback 收益明确，latency 可接受 | 不开放高成本 V1 动态调用 |
| L4 -> S3 | S1/S2 case study 无硬安全问题 | 只报告 L3/L2，不宣称 L4 主方法 |

---

## 16. 当前建议

当前最稳的近期路线是：

```text
先固定 L0：
  V2 conservative STOP
  V4 selector context
  visual ranked fallback
  schema warnings
  runtime latency

再做 L1：
  tool registry + trace，不改变行为

再做 L2b：
  scripted controller，统一 STOP / fallback / budget reason

然后做 L3：
  LLM shadow planner，用来证明“为什么不能直接自由工具调用”

最后小步做 L4b：
  只开放 fallback-aware controlled tool-calling，
  不急着开放动态视觉证据和完整自由调度。
```

这条路线最适合作为论文卖点，因为它能同时回答：

- 为什么 embodied navigation 不能直接用自由 Agent；
- 如何把 fixed pipeline 逐步 agentize；
- 每增加一层自主性带来什么收益和风险；
- 工具调用在 VLN-CE 中到底改善了哪类失败。

---

## 17. 自审记录

### 17.1 第一轮自审：实验可执行性

检查项：

- L0-L4 是否每一级都有明确输入、输出、对照和验收；
- 是否避免把 A 系列、V 系列和 L 系列混成一条线；
- 是否给出了 L4 前的安全门槛；
- 是否能在资源有限时执行最小矩阵；
- 是否避免 LLM 自由调用 `env.step`。

发现的问题：

```text
1. L4 中 controller 和 selector 的边界还不够硬，容易变成双 LLM 黑箱。
2. L2/L4 的预算降级可能静默回到 L0 fallback，影响归因。
3. Planner 可能看到 oracle / distance_to_goal 等离线指标，存在信息泄漏风险。
4. L4 的收益来源需要内部消融，否则无法区分 fallback、视觉证据和 controller 调度。
```

已修订：

```text
1. 增加 PlannerStateView / AgentStateFull 区分，禁止 planner 看到 oracle 和评测标签。
2. 增加 L4 controller 与 selector 边界，默认禁止 direct controller move。
3. 增加 budget_degradation 记录和对应指标。
4. 增加 L4 内部消融表，默认主版本为 L4b-no-direct。
```

### 17.2 第二轮自审：论文表述与审稿风险

检查项：

- 是否把卖点落在 controlled agentization，而不是泛泛 tool-calling；
- 是否有足够指标证明“演进”而非工程包装；
- 是否报告成本和安全，而不是只报告 SR；
- 是否明确 LangChain / LangGraph 不是核心贡献；
- 是否避免把小样本结论写成显著提升。

发现的问题：

```text
1. L0-L4 仍可能被看成工程阶段列表，而不是“演进性”实验。
2. 需要更明确地度量自主性增加、动态工具调用增加和安全干预变化。
3. 需要说明即使 L4 平均 SR 不涨，也如何从失败类型、成本和安全曲线中得到论文结论。
```

已修订：

```text
1. 增加 9.6 演进性指标，包括 autonomy level、dynamic tool-call ratio、tool diversity、safety intervention density。
2. 明确建议画 L0->L4 的 performance / cost / safety 曲线。
3. 在演进性指标中说明 L4 若只改善特定失败类型，也可以形成受控 Agent 化结论。
```

最终判断：

```text
方案可以作为 L 系列实验设计草案使用。
它的强项是安全边界和归因链路清楚；
主要剩余风险是实现成本较高，且 L4 需要 S2/S3 结果才能支撑论文主张。
```

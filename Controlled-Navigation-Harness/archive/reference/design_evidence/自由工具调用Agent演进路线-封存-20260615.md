---
tags:
  - 实验
  - Open-Nav
  - VLN-CE
  - agent
  - tool-calling
  - harness
created: 2026-06-15
updated: 2026-06-15
version: v0.1
role: Controlled Navigation Harness 向自由工具调用 Agent 演进的设计路线
status: draft
related:
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/项目总控]]"
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/实验方案]]"
---

# 自由工具调用 Agent 演进路线

## 0. 一句话目标

```text
在不破坏当前 Harness 可诊断、可消融、可回放优势的前提下，
把固定 Open-Nav pipeline 逐步演进为受控 tool-calling navigation agent。
```

这里的目标不是立刻接入 LangChain，也不是让 LLM 完全自由控制环境，而是先把当前已存在的导航职责抽象成工具，再引入一个受约束的 Agent Controller，让它可以在明确预算和安全边界内选择下一步工具调用。

---

## 1. 为什么不能直接切到 LangChain Agent

VLN-CE 和普通问答/搜索类 Agent 不同，agent 的输出最终会变成环境动作。直接使用自由工具调用会带来几个问题：

| 风险 | 在导航任务中的表现 |
|------|--------------------|
| 不可复现 | 同一 episode 中工具调用路径可能漂移，难以复现实验结果 |
| 难消融 | prompt、工具顺序、工具选择和动作策略一起变化，收益难归因 |
| 成本失控 | 每一步可能多次调用 LLM/VLM，episode latency 难以控制 |
| STOP 风险 | LLM 可能绕过 verifier 直接要求停止 |
| 日志不完整 | 只记录最终动作时无法解释中间工具调用为什么发生 |

因此，本项目更适合采用：

```text
Controlled Tool-Calling Agent
  = Tool Registry
  + Agent Controller
  + Action Validator
  + Harness Trace
  + Budget / Safety Policy
```

LangChain 或 LangGraph 可以作为后续实现载体，但不应成为第一步。第一步应该先在现有代码中稳定工具接口和 trace schema。

---

## 2. 当前系统与目标系统的差异

### 2.1 当前系统

当前主链是固定流程：

```text
observe
  -> waypoint prediction
  -> visual evidence
  -> memory update
  -> completion estimation
  -> selector
  -> stop verifier
  -> fallback
  -> env.step
```

LLM 主要在固定位置工作：

- instruction -> actions / landmarks；
- history -> completion estimation；
- candidates -> next waypoint；
- fused thoughts -> final decision。

### 2.2 目标系统

目标系统把固定流程改成可选择工具的循环：

```text
AgentState
  -> AgentController chooses tool
  -> Tool executes and returns typed result
  -> Harness logs tool call
  -> AgentState updates
  -> repeat until ActionProposal
  -> ActionValidator
  -> env.step or STOP
```

差异不在于“是否使用 LLM”，而在于 LLM 从固定 selector 变成受控调度器。但所有动作仍必须经过 Harness 校验。

---

## 3. 工具化边界

第一批工具应该来自当前已有代码，避免先引入新能力。

| 工具名 | 来源 | 输入 | 输出 | 是否允许改行为 |
|--------|------|------|------|----------------|
| `parse_instruction` | `Open_Nav.get_actions/get_landmarks` | instruction | actions, landmarks | 否，episode start 只运行一次 |
| `predict_waypoints` | policy + TRM waypoint predictor | RGB-D observation | candidate list | 是，但候选生成本身保持原逻辑 |
| `observe_candidates` | `Open_Nav.observe_environment` | candidate images | observe_dict | 否 |
| `extract_visual_evidence` | `VisualEvidenceLogger` | candidates, images, instruction | candidate evidence | 初期否，先 log-only |
| `update_visual_memory` | `VisualEvidenceMemory` | visual evidence, pose | memory state | 初期否，先 log-only |
| `estimate_completion` | `Open_Nav.estimate_completion` | history, actions, landmarks | progress estimate | 否 |
| `select_waypoint` | `Open_Nav.move_to_next_vp/test_decisions` | candidates, context | candidate id or STOP | 是 |
| `verify_stop` | `VisualTargetVerifier` | STOP proposal, evidence | allow / reject / uncertain | 是，必须强制经过 |
| `rank_fallback` | `VisualEvidenceFallbackRanker` | candidates, evidence | ranked fallback | 是，仅在受限触发 |
| `propose_action` | ActionRouter | selected candidate | Habitat action | 是，但必须经 validator |
| `step_environment` | `envs.step` | validated action | observation, info, done | 是，唯一执行环境动作工具 |

关键原则：

- 工具不是 prompt 片段，而是有 typed input/output 的工程接口。
- 每次工具调用必须进入 jsonl trace。
- 任何会改变环境状态的工具必须经过 ActionValidator。
- STOP 不能由 AgentController 直接执行，必须经过 `verify_stop`。

---

## 4. Agent Controller 最小形态

最小 controller 不需要 LangChain。它只需要让 LLM 在固定工具列表中选择一个工具，并输出结构化 JSON。

### 4.1 单次工具调用格式

```json
{
  "thought": "Need visual evidence before deciding whether STOP is valid.",
  "tool": "verify_stop",
  "args": {
    "source": "selector_stop_gate",
    "stop_proposal": true,
    "selected_candidate": null
  }
}
```

### 4.2 最终动作格式

```json
{
  "thought": "Candidate 3 best matches the next landmark and STOP is not supported.",
  "final": {
    "type": "move",
    "candidate_id": "3"
  }
}
```

STOP 必须带 verifier 证据：

```json
{
  "thought": "The final target is visible and arrival evidence is strong.",
  "final": {
    "type": "stop",
    "verifier_event_id": "visual_target_verifier:episode42:step5"
  }
}
```

---

## 5. 受控策略

为了保持实验可控，Agent Controller 必须受这些规则限制：

| 约束 | 建议默认值 | 作用 |
|------|------------|------|
| 每 step 最大工具调用数 | 4 | 防止无限循环 |
| 每 episode 最大额外 LLM 调用数 | 受配置控制 | 控制运行成本 |
| 允许工具集合 | config 白名单 | 做 ablation |
| STOP 执行条件 | verifier allow | 防止提前 STOP |
| move 执行条件 | candidate id 必须有效 | 防止非法动作 |
| fallback 触发条件 | 空预测、无效候选、STOP reject | 保持受限恢复 |
| trace 记录 | 强制开启 | 保证可回放 |

Agent 不允许：

- 修改候选集合本身；
- 直接调用 `envs.step`；
- 跳过 `verify_stop` 执行 STOP；
- 自由创建新工具；
- 在没有 trace 的情况下调用工具；
- 把工具失败解释为成功证据。

---

## 6. 分阶段路线

### L0: 当前 Harness 固定流程

目标：稳定 V2/V4 decision-effect，完成 STOP allow 收紧和小样本复测。

验收：

- V1 schema 稳定；
- V4/fallback 能消费候选级证据；
- STOP allow case 能解释；
- 当前 README 和实验记录能对应日志字段。

### L1: Tool Registry 抽象

目标：不改变行为，只把现有函数包装成工具接口。

改动：

- 新增 `ToolSpec`、`ToolResult`、`ToolRegistry`；
- 把 V1/V2/V3/V4/fallback/selector 包成工具；
- 主循环仍按固定顺序调用；
- trace 中新增 `tool_call_start/tool_call_end/tool_call_error`。

验收：

- L1 与 L0 指标基本一致；
- 每个工具输入输出都能独立回放；
- 工具失败能 fail-open 或 fail-closed，行为由配置控制。

### L2: Scripted Controller

目标：引入 controller，但 controller 仍按规则表选择工具，不让 LLM 自由调度。

改动：

- `AgentState` 成为 controller 输入；
- controller 根据状态触发 `verify_stop`、`rank_fallback`、`select_waypoint`；
- 先不让 LLM 选择工具，只让它在 selector 内做候选裁决。

验收：

- 与 L1 行为接近；
- controller trace 能解释每次工具触发原因；
- STOP reject 后恢复路径更清晰。

### L3: LLM Tool Planner logging-only

目标：让 LLM 生成工具调用计划，但不执行它，只和 scripted controller 对比。

改动：

- 新增 `plan_tool_call`；
- 记录 LLM 想调用的工具、参数和理由；
- 实际执行仍由 scripted controller 决定。

验收：

- 统计 LLM plan 与 scripted plan 的一致率；
- 分析 LLM 是否倾向过度调用视觉工具或过早 STOP；
- 找到可安全开放的工具集合。

### L4: Controlled Tool-Calling Agent

目标：允许 LLM 在白名单工具内选择下一步调用，但最终动作仍由 validator 执行。

改动：

- 每 step 允许最多 N 次 tool call；
- 每次工具调用进入 trace；
- ActionProposal 必须经过 ActionValidator；
- STOP 必须绑定 `verify_stop` allow 事件。

验收：

- 成本、步数、工具调用数在预算内；
- 相同配置和 seed 下 replay trace 可复核；
- 相比固定 pipeline，能在特定失败类型上改善，不把所有收益混在一起。

### L5: LangChain / LangGraph 适配

目标：当本地 Tool Registry 和 trace schema 稳定后，再考虑替换 controller 实现。

适配原则：

- LangChain/LangGraph 只作为 controller runtime；
- ToolSpec、ToolResult、AgentState、ActionValidator、MetricsLogger 保持项目内定义；
- 不把第三方框架对象写进核心 trace；
- 能随时回退到本地 controller。

---

## 7. 推荐代码落点

建议新增目录：

```text
vlnce_baselines/common/opennav_agent/
  __init__.py
  tool_spec.py
  tool_registry.py
  agent_state.py
  controller.py
  action_validator.py
  trace_schema.py
```

与现有 `opennav_ext` 的关系：

| 目录 | 职责 |
|------|------|
| `opennav_ext` | 当前 Harness 工具实现和诊断模块 |
| `opennav_agent` | 将工具注册、编排、验证和未来 tool-calling controller 统一起来 |

短期内不建议把 `opennav_ext` 重命名或搬迁。先在 `opennav_agent` 中引用现有模块，等接口稳定后再决定是否合并。

---

## 8. 需要新增的配置

建议未来在 `OPENNAV_HARNESS` 下新增：

```yaml
OPENNAV_HARNESS:
  AGENT_CONTROLLER:
    ENABLED: false
    MODE: fixed_pipeline  # fixed_pipeline | scripted | llm_plan_log_only | controlled_tool_calling
    MAX_TOOL_CALLS_PER_STEP: 4
    MAX_LLM_TOOL_PLANS_PER_STEP: 1
    ALLOWED_TOOLS:
      - observe_candidates
      - extract_visual_evidence
      - update_visual_memory
      - estimate_completion
      - select_waypoint
      - verify_stop
      - rank_fallback
    REQUIRE_VERIFIER_FOR_STOP: true
    REQUIRE_ACTION_VALIDATOR: true
    TRACE_TOOL_CALLS: true
```

这样可以保持当前默认行为不变，只在明确打开 `AGENT_CONTROLLER.ENABLED` 后进入新路线。

---

## 9. 论文表述口径

如果后续写论文或报告，建议这样表述：

```text
We do not immediately adopt an unconstrained LLM-agent formulation.
Instead, we first refactor Open-Nav into a controlled navigation harness,
then progressively expose perception, memory, verification, and recovery
modules as typed tools under a budgeted tool-calling controller.
```

中文口径：

```text
本项目不是直接把 Open-Nav 改造成自由工具调用式 Agent，
而是先构建受控 Harness，再逐步开放工具调用能力。
这样可以在保留可诊断、可消融和可回放能力的同时，
研究 Agent 化调度是否真的改善 VLN-CE 的失败恢复和 STOP 决策。
```

---

## 10. 当前结论

当前最合理的路线是：

```text
先 Harness 化，再工具化，再 controlled tool-calling，
最后才考虑 LangChain/LangGraph runtime。
```

这条路线能让项目逐步靠近 LangChain 式自由工具调用 Agent，同时保留当前实验最重要的资产：可控变量、结构化 trace、STOP 安全约束和失败归因能力。

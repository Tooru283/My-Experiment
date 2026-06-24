---
tags:
  - 实验
  - Open-Nav
  - VLN-CE
  - agent
  - harness
  - U-series
  - 新论文创新点
created: 2026-06-19
updated: 2026-06-19
version: v0.1
role: U 系列新论文创新点实验方案
status: draft
derived_from:
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/新论文创新点引入]]"
related:
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/实验方案]]"
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/V系列多模态视觉证据初步实现后调整方案]]"
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/L系列受控工具调用Agent实验方案]]"
  - "[[Qwen-RobotNav]]"
  - "[[FORESIGHT]]"
  - "[[HSGM]]"
  - "[[Context-Nav]]"
  - "[[AgentVLN]]"
---

# U 系列新论文创新点实验方案

## 0. 一句话目标

```text
U 系列不是继续堆视觉模块，也不是开放自由工具调用，
而是把新论文启发收束为三个中粒度、可消融、可回滚的论文贡献单元：
阶段化证据组织、证据驱动 STOP 验证、失败类型条件恢复。
```

U 系列的核心原则：

```text
trace 可以细，method 不能碎；
日志字段可以多，实验变量必须少；
小机制作为实现细节，U1/U2/U3 才作为论文贡献和实验变量。
```

---

## 1. 系列定位

### 1.1 与现有系列的关系

| 系列 | 研究对象 | U 系列关系 |
|------|----------|------------|
| A 系列 | A0/A1 baseline 与 harness logging 是否行为不变 | U 系列必须继承 A0/A1 边界 |
| V 系列 | 视觉证据、视觉 STOP verifier、视觉 selector context | U 系列复用 V1/V2/V3/V4，但重新组织成论文级单元 |
| L 系列 | controller 自主性和受控工具调用 | U 系列不增加自由度，只增加结构化决策单元 |
| U 系列 | 新论文启发收束后的方法贡献 | 目标是形成最终论文的 2-3 个主机制 |

U 系列的论文叙事不是：

```text
我们引入了 8 个小模块。
```

而是：

```text
我们提出一个 phase-aware, evidence-grounded, ablation-safe controlled navigation harness。
```

### 1.2 当前实验背景

当前 V2/V4 已经进入 decision-effect，但最近 ep100 成功率仍然偏低：

| run | SR | OSR | SPL | nDTW | 关键现象 |
|-----|----|-----|-----|------|----------|
| `ep100_series_qwen_siglip_local_20260615_232107` | 0.16 | 0.24 | 0.116 | 0.453 | `step_length_limit=74`，STOP `6/26` 成功 |
| `ep100_series_qwen_siglip_local_20260616_162306` | 0.17 | 0.28 | 0.102 | 0.410 | STOP `9/48` 成功，`visual_stop_rescued=1/24` 成功 |

当前主要症状：

| 症状 | 观察 | U 系列对应单元 |
|------|------|----------------|
| 后半程 drift | step>=4 move action 负收益比例高 | U1 |
| V4 视觉上下文噪声 | 统一上下文可能在错误阶段注入错误证据 | U1 |
| STOP 误停 | `visual_stop_rescued` 成功仅 `1/24` | U2 |
| 弱终点漏停 | OSR=1 但 SR=0，episode 602 类 case | U2 |
| empty fallback 退化 | `selector_empty_prediction_fallback=79`，`41/79` 仍选原始第一候选 | U3 |
| 失败不可归因 | STOP reject、空输出、负收益、loop 混在一起 | U3 |

---

## 2. 研究问题与假设

### 2.1 主研究问题

> 能否把新论文启发中的阶段状态、证据组织、目标验证和失败恢复，收束为三个中粒度 controlled harness 单元，使 Open-Nav 的后半程决策、STOP 判断和 fallback 恢复更可诊断、可消融，并最终提升 VLN-CE 零样本导航表现？

### 2.2 子问题

| 编号 | 问题 | 对应单元 |
|------|------|----------|
| Q1 | 显式 phase 和 evidence slots 是否能减少 V4 统一上下文导致的后半程 drift？ | U1 |
| Q2 | 将 STOP 从普通候选选择中分离为 evidence-grounded verification，是否能同时降低误停和弱终点漏停？ | U2 |
| Q3 | fallback 先判断 failure_type 再受限恢复，是否优于 selector 空输出后机械 fallback？ | U3 |
| Q4 | U1/U2/U3 单元级收益是否能分别归因，而不是只在组合系统中看起来有效？ | 全部 |

### 2.3 可检验假设

| 假设 | 内容 | 验证方式 |
|------|------|----------|
| H1 | U1 phase-aware evidence 比 V4 unified context 更能降低后半程负收益动作比例。 | V4-unified vs U1-phase-aware paired comparison |
| H2 | U2 relation / trajectory / weak-target verifier 比当前 V2 rescue 更高 STOP precision。 | V2-original / rescue-off / U2 relation-aware 对照 |
| H3 | U3 failure-conditioned reselect 能降低 empty fallback 的 first-order 退化和 post-fallback 负收益。 | F0-original vs U3-reselect-only |
| H4 | U0-log 能一次记录 U1/U2/U3 trace，但不改变动作。 | action agreement 与 SR/SPL/nDTW 对齐 |
| H5 | 组合系统只用于最终展示，不能替代单元级归因。 | 单元级和组合级分开报告 |

---

## 3. 总体方法

U 系列将新论文启发收束为三个单元：

| 单元 | 名称 | 主要解决问题 | 改变位置 |
|------|------|--------------|----------|
| U1 | Phase-aware Evidence Scaffolding | 后半程 drift、视觉上下文噪声、候选证据组织混乱 | ContextBuilder / selector prompt slots |
| U2 | Evidence-grounded Stop Verifier | STOP 误停、弱终点漏停、目标关系判断不稳 | STOP proposal / near-goal gate |
| U3 | Failure-conditioned Recovery Policy | empty fallback 退化、走错后不会恢复、失败类型不可分 | fallback / recovery trigger |

总体流程：

```text
Open-Nav waypoint candidates
  -> U1: phase-aware evidence scaffolding
  -> LLM Selector
  -> move or stop proposal
  -> U2: evidence-grounded stop verifier if STOP / near-goal
  -> U3: failure-conditioned recovery if empty / rejected / negative / loop
  -> env.step or STOP
  -> trace / decision audit
```

三个单元的边界：

| 单元 | 改什么 | 不改什么 |
|------|--------|----------|
| U1 | 改 selector 看到的结构化证据和上下文槽位 | 不直接决定 STOP，不做 recovery |
| U2 | 验证当前是否应该 STOP | 不重排普通 move candidate |
| U3 | 在明确失败触发后做受限恢复 | 不每步接管 planner，不做自由多轮推理 |

---

## 4. 不变量与边界

### 4.1 全阶段不变量

U 系列所有阶段必须固定：

- 数据集 split、episode list、随机种子；
- waypoint predictor / policy checkpoint；
- LLM / VLM backbone 和推理参数；
- Habitat action space；
- 候选生成逻辑；
- success radius 和评测脚本；
- 最大 episode step limit；
- A0/A1 的行为不变边界；
- V1/V2/V3/V4 的基础 schema，除非实验明确标注。

### 4.2 禁止混入变量

U 系列期间禁止同时引入：

- 更换 LLM / VLM；
- 训练或微调新 policy；
- 直接替换为 Qwen-RobotNav；
- 复现 HSGM 完整地图 + A*；
- FORESIGHT 式每步 planner-critic-refine；
- Context-Nav 完整 ObjectNav pipeline；
- waypoint-free 主链；
- 多 sub-agent 在线协作；
- 大规模 prompt 重写；
- 未进入 trace schema 的隐式 heuristic。

### 4.3 U0-log 边界

U0-log 是 U 系列的诊断层，可以一次记录多个字段，但不能改变行为。它仍然属于 A1 行为不变边界内的 logging-only 扩展，但命名上归入 U 系列，避免再引入 M 或其他编号：

```text
enable_harness_logging = true
enable_decision_effect = false

允许:
  phase_evidence logging
  stop_verification logging
  recovery_state logging
  decision_audit logging

禁止:
  改 selector prompt
  改 candidate rank
  改 STOP gate
  改 fallback
  改 env action
```

---

## 5. U1 Phase-aware Evidence Scaffolding

### 5.1 目标

把当前统一 V4 视觉上下文改成按导航阶段选择证据槽位：

```text
search: 关注路线模式和可探索候选
approach: 关注 current_subgoal 和候选证据
verify: 交给 U2 验证 STOP evidence
recover: 交给 U3 使用 failure evidence
```

### 5.2 当前问题

V4 已经进入 decision-effect，但 ep100 未证明稳定收益。合理怀疑是：

```text
视觉证据本身有价值，
但统一上下文在错误阶段注入了错误证据，
导致 selector 后半程 drift。
```

### 5.3 最小状态

```yaml
phase_evidence:
  phase: search | approach | verify | recover | unknown
  current_subgoal: str
  phase_reason: str
  visual_budget: low | medium | high
  confidence: float
  evidence_slots:
    instruction_summary: str
    current_subgoal: str
    recent_progress: str
    candidate_evidence: list
    target_evidence: str
```

候选证据最小字段：

```yaml
candidate_evidence:
  candidate_id: int
  geometry:
    projection_valid: bool | unknown
    depth_valid: bool | unknown
    occupancy_risk: low | medium | high | unknown
  semantics:
    matched_constraints: list
    failed_constraints: list
    visual_support: str
  progress:
    matches_current_subgoal: yes | no | unknown
  risk:
    revisit_risk: low | medium | high | unknown
```

### 5.4 阶段策略

| phase | selector 主要看到 | 不应该看到 |
|-------|-------------------|------------|
| search | route pattern、top-k 候选概览、粗视觉证据 | 过多 target crop、完整历史 |
| approach | current_subgoal、candidate evidence、recent progress | 全量 episode memory |
| verify | 当前 observation 的 STOP evidence 摘要 | 普通 move prompt 噪声 |
| recover | failure_type、失败证据、受限候选 | 未分类的失败日志堆叠 |
| unknown | 回退到 V4-unified 的保守子集 | 自信地注入错误 phase |

### 5.5 实验阶段

| 编号 | 配置 | 是否改变动作 | 目的 |
|------|------|--------------|------|
| U1-log | 只记录 phase 和 evidence slots | 否 | 验证 phase classifier 可用 |
| U1-shadow | 构建 phase-aware prompt 但不执行 | 否 | 比较 shadow action 与原动作 |
| U1-phase-aware-V4 | 按 phase 注入 selector context | 是 | 验证是否减少 drift |

首轮只比较：

```text
V4-unified vs U1-phase-aware-V4
```

### 5.6 指标

| 指标 | 期望 |
|------|------|
| `phase=unknown` 比例 | 不高于 20%，否则 phase 不可靠 |
| step>=4 negative distance-gain ratio | 低于 V4-unified |
| `visual_context_noise` 失败比例 | 下降 |
| `progress_drift` 失败比例 | 下降 |
| SR/SPL/nDTW | 至少不低于 V4-unified |
| phase/action agreement | 人工 case study 可解释 |

---

## 6. U2 Evidence-grounded Stop Verifier

### 6.1 目标

把 STOP 从普通 action selection 中分离出来，作为目标满足性验证问题：

```text
STOP proposal
  -> intrinsic evidence
  -> relation evidence
  -> trajectory evidence
  -> weak target adjustment
  -> allow / reject / uncertain
```

### 6.2 当前问题

最新 ep100 中：

```text
stop_requested success: 9/48
visual_stop_rescued success: 1/24
OSR=1 but SR=0: 11 episodes
```

说明当前系统同时存在：

- rescue 过宽导致 false positive；
- 弱终点 gate 过严导致 false negative；
- 视觉目标可见性和 VLN success radius 不一致。

### 6.3 最小状态

```yaml
stop_verification:
  stop_proposed_by: llm | completion_gate | near_goal_gate | rescue
  target_type: object | room | area | landmark | weak_target | unknown
  intrinsic_support: yes | no | unknown
  relation:
    relation_type: near | left_of | right_of | before | inside | facing | unknown
    anchor: str
    support: yes | no | unknown
  trajectory_support: yes | no | unknown
  weak_target_adjustment: none | relax | block
  allow_stop: bool
  reject_reason: str
  confidence: float
```

### 6.4 Verifier 规则

| 证据 | 作用 |
|------|------|
| intrinsic evidence | 目标或目标区域是否存在 |
| relation evidence | 是否满足 near / left_of / inside / facing 等关系 |
| trajectory evidence | 是否已经到达应停阶段 |
| weak-target evidence | room / doorway / area 等弱目标是否需要特殊处理 |
| contradiction evidence | 当前证据是否明确反驳 STOP |

### 6.5 实验阶段

| 编号 | 配置 | 是否改变动作 | 目的 |
|------|------|--------------|------|
| U2-log | 只记录 relation / trajectory / weak-target evidence | 否 | 建立 STOP case study |
| U2-rescue-off | 关闭 visual rescue | 是 | 验证 rescue 是否负收益 |
| U2-relation-aware | 加入 relation blocker | 是 | 降低目标可见但关系不满足的误停 |
| U2-weak-target-adjusted | 弱目标 gate 调整 | 是 | 缓解 doorway / room / area 漏停 |

建议顺序：

```text
V2-original
  -> U2-rescue-off
  -> U2-relation-aware
  -> U2-weak-target-adjusted
```

不要一开始同时打开 relation-aware 和 weak-target-adjusted。

### 6.6 指标

| 指标 | 期望 |
|------|------|
| STOP precision | 高于当前 `9/48` |
| `visual_stop_rescued` precision | 明显高于当前 `1/24`，否则 rescue 默认关闭 |
| OSR=1 / SR=0 | 下降 |
| `stop_false_positive` | 下降 |
| `stop_false_negative` | 不上升或下降 |
| `completion_gate_weak_final_target` correctness | 能区分误停拦截和 near-goal 漏停 |

---

## 7. U3 Failure-conditioned Recovery Policy

### 7.1 目标

把 fallback 从机械异常处理改成受限失败恢复：

```text
failure event
  -> failure_type
  -> allowed recovery actions
  -> one-step constrained recovery
```

### 7.2 当前问题

最新 ep100 中：

```text
selector_empty_prediction_fallback: 79
first-order fallback ratio: 41/79
step>=4 negative move ratio: 227/490
```

说明 fallback 当前没有稳定利用失败证据，也没有足够改变动作质量。

### 7.3 最小状态

```yaml
recovery_state:
  trigger:
    type: selector_empty | negative_gain | loop | stop_rejected | candidate_missing
    step: int
  failure_type: progress_drift | selector_wrong | empty_fallback_bad | stop_false_positive | candidate_missing | unknown
  evidence:
    recent_distance_gain: list
    rejected_stop_reason: str
    repeated_view: bool
    current_subgoal: str
    candidate_evidence: list
  allowed_recovery:
    - reselect_from_topk
    - rotate_disambiguation
    - continue_current_subgoal
  selected_recovery: str
  budget_remaining: int
```

### 7.4 触发与允许动作

| 触发 | failure_type | 允许动作 |
|------|--------------|----------|
| selector empty | empty_fallback_bad | 从 top-k 重新选择，不扩展动作空间 |
| 连续负收益 | progress_drift / selector_wrong | 回到 current_subgoal 做受限 reselect |
| STOP rejected | stop_false_positive | 选择与 reject reason 不冲突的候选 |
| loop / repeated view | loop | 一次 rotate_disambiguation 或 top-k 替代 |
| candidate_missing | candidate_missing | 只记录或触发受限观察，不强行 hallucinate |

### 7.5 实验阶段

| 编号 | 配置 | 是否改变动作 | 目的 |
|------|------|--------------|------|
| U3-log | 只标注 failure_type | 否 | 验证失败分类可用 |
| U3-shadow | 生成 recovery proposal 但不执行 | 否 | 评估 recovery 是否合理 |
| U3-reselect-only | 按 failure_type 做 top-k reselect | 是 | 验证最小恢复是否改善 fallback |
| U3-progress-aware-recovery | 引入 current_subgoal / evidence notebook | 是 | 验证阶段化恢复 |

首轮只比较：

```text
F0-original-empty-fallback vs U3-reselect-only
```

如果 reselect-only 没有收益，不继续扩展复杂 recovery。

### 7.6 指标

| 指标 | 期望 |
|------|------|
| post-fallback distance gain | 提升 |
| fallback first-order ratio | 低于当前 `41/79` |
| fallback 后再次失败比例 | 下降 |
| `step_length_limit` | 不上升，最好下降 |
| late-stage drift | 下降 |
| recovery budget overrun | 为 0 |

---

## 8. 实验矩阵

### 8.1 Baseline 与诊断层

| 编号 | 配置 | 目的 | 是否改变动作 |
|------|------|------|--------------|
| A0 | Open-Nav original | 固定原始 baseline | 否 |
| A1 | Harness logging only | 验证外壳不改行为 | 否 |
| U0-log | U1/U2/U3 全部 trace 字段 log-only | 建立统一失败证据 | 否 |

U0-log 可以一次记录多个 U 字段，因为它不改变动作。但 A2+ 不能一次启用多个新决策单元。

### 8.2 单元级 decision-effect

| 单元 | 对照组 | 实验组 | 核心问题 |
|------|--------|--------|----------|
| U1 | V4-unified | U1-phase-aware-V4 | 阶段化证据是否减少 drift |
| U2 | V2-original / U2-rescue-off | U2-relation-aware / U2-weak-target-adjusted | STOP verifier 是否提高停靠质量 |
| U3 | F0-original-empty-fallback | U3-reselect-only | 失败类型恢复是否优于机械 fallback |

### 8.3 组合层

只有当单元级实验通过，才进入组合实验：

| 配置 | 目的 |
|------|------|
| U1 + U2 | 检查阶段化证据是否改善 STOP 前上下文 |
| U1 + U3 | 检查阶段化证据是否改善恢复触发 |
| U1 + U2 + U3 | 最终系统展示，不作为单元归因依据 |

组合系统只用于最终性能展示，不用于证明某个机制单独有效。

---

## 9. 当前优先顺序

结合最新 ep100 低 SR，U 系列不应同时推进三个完整单元。只使用 U 系列编号，推进顺序为：

| 顺序 | 任务 | 原因 |
|------|------|------|
| U0-log | 固定 A0/A1 行为等价验证，并建立 U 系列 trace | 没有这个，后续收益不可归因 |
| U2 | rescue-off 与 STOP case study | STOP 问题最局部，当前 `1/24` 最紧急 |
| U3 | failure_type 标注与 reselect-only | fallback 样本多，适合归因 |
| U1 | phase-aware V4 | U1 影响面最大，应在 phase 可靠后进入 decision-effect |
| U4-combined | 组合 U1+U2、U1+U3、全系统 | 只用于最终展示 |

推荐近期最小实验包：

```text
U0-log: 20 episodes
U2-rescue-off: paired 20 episodes
U3-log: replay / paired 20 episodes
U3-reselect-only: paired 20 episodes
U1-log: 20 episodes
U1-phase-aware-V4: paired 20 episodes after phase stable
```

ep100 只在小样本通过后使用：

```text
unit small-run pass
  -> fixed episode list paired run
  -> ep100 validation
  -> case study table
```

---

## 10. 日志与归因要求

### 10.1 必须新增或稳定的事件

```text
phase_evidence
candidate_evidence
stop_verification
weak_target_decision
recovery_state
failure_type_diagnostic
decision_audit
unit_override
```

### 10.2 decision audit

每个 decision-effect 单元必须记录：

```yaml
decision_audit:
  unit_enabled: none | U1 | U2 | U3 | combined
  original_action: str
  proposed_action: str
  final_action: str
  override: bool
  override_reason: str
  expected_failure_addressed: progress_drift | stop_error | fallback_error | none
```

### 10.3 归因规则

1. A0/A1 永远不改变动作。
2. U0-log 可以记录所有 trace，但不允许改 prompt、不允许 rerank、不允许改 STOP / fallback。
3. 每次 ep100 只开启一个 U decision-effect 单元。
4. 每条轨迹记录 `decision_effect_unit`。
5. 每次动作保留 Open-Nav 原始候选、原始 selector 输出、新单元建议、最终执行动作。
6. 如果一个收益只在组合系统中出现，而单元级实验没有收益，论文中不能把它写成单元贡献。

---

## 11. 指标体系

### 11.1 主指标

| 指标 | 作用 |
|------|------|
| SR | 成功率 |
| SPL | 路径效率 |
| nDTW | 轨迹相似度 |
| OSR | 是否曾到达目标附近 |
| NE | 最终导航误差 |
| TL | 路径长度 |

### 11.2 U1 指标

| 指标 | 作用 |
|------|------|
| phase unknown rate | phase classifier 是否可靠 |
| step>=4 negative gain ratio | 后半程 drift 是否下降 |
| progress_drift failure rate | 阶段状态是否减少漂移 |
| context length | phase-aware context 是否控制预算 |
| selector action agreement | U1 是否过度改变动作 |

### 11.3 U2 指标

| 指标 | 作用 |
|------|------|
| STOP precision | STOP 放行是否可靠 |
| STOP recall | 是否漏停 |
| visual rescue precision | rescue 是否应该保留 |
| weak-target miss rate | 弱终点是否被过度拦截 |
| relation blocker correctness | 关系验证是否误杀正确 STOP |

### 11.4 U3 指标

| 指标 | 作用 |
|------|------|
| failure_type coverage | 失败是否能被分类 |
| post-fallback distance gain | 恢复动作是否有效 |
| first-order fallback ratio | fallback 是否仍退化为原始第一候选 |
| recovery trigger precision | 是否过触发 |
| recovery budget usage | 成本是否可控 |

---

## 12. 代码集成建议

新增 U 系列模块建议放在：

```text
vlnce_baselines/common/opennav_ext/
  phase_evidence.py
  evidence_scaffolder.py
  stop_evidence_verifier.py
  failure_diagnostic.py
  recovery_policy.py
  decision_audit.py
```

接入点：

| 接入点 | 模块 |
|--------|------|
| episode start | phase 初始化、instruction schema |
| observation 后 | candidate evidence / visual evidence 汇总 |
| selector 前 | U1 evidence scaffolder |
| STOP proposal 后 | U2 stop evidence verifier |
| selector empty / STOP rejected / negative gain 后 | U3 failure diagnostic |
| fallback 前 | U3 recovery policy |
| final action 前 | decision audit / validator |
| episode end | failure attribution |

配置建议：

```yaml
OPENNAV_HARNESS:
  U_SERIES:
    ENABLED: true
    DECISION_EFFECT_UNIT: none  # none | U1 | U2 | U3 | combined

    PHASE_EVIDENCE:
      ENABLED: true
      LOG_ONLY: true

    STOP_EVIDENCE_VERIFIER:
      ENABLED: true
      LOG_ONLY: true
      ENABLE_RESCUE: false
      ENABLE_RELATION_CHECK: false
      ENABLE_WEAK_TARGET_ADJUSTMENT: false

    FAILURE_RECOVERY:
      ENABLED: true
      LOG_ONLY: true
      ENABLE_RESELECT: false
      MAX_RECOVERY_PER_EPISODE: 2
```

---

## 13. 与相关论文的对应关系

| 论文 | 吸收思想 | U 系列位置 | 不吸收内容 |
|------|----------|------------|------------|
| Qwen-RobotNav | task / observation config，阶段化视觉预算 | U1 | 不替换为训练式导航基础模型 |
| FORESIGHT | clue critique，失败线索诊断 | U3 | 不做每步自由 planner-critic-refinement |
| HSGM | subtask state，语义-几何证据 | U1，辅助 U2 | 不复现完整地图 + A* |
| Context-Nav | relation-aware target verification | U2 | 不迁移完整 ObjectNav pipeline |
| AgentVLN | tool / skill 边界，2D-3D evidence | U1 candidate evidence | 不训练 AgentVLN-Instruct，不改成技能策略 |

---

## 14. 当前待办

- [ ] 建立 U0-log trace schema：`phase_evidence`、`stop_verification`、`recovery_state`、`decision_audit`
- [ ] 从最新 ep100 中抽取固定 debug episode list：rescue-fail、weak-target-miss、empty-fallback、late-drift
- [ ] U2：跑 `visual_stop_rescue` off 的 paired 小样本
- [ ] U2：建立 relation / trajectory / weak-target STOP case table
- [ ] U3：对 79 个 empty fallback 标注 failure_type、selected rank、distance gain
- [ ] U3：实现 `U3-reselect-only` 最小策略并做 paired 小样本
- [ ] U1：实现 phase classifier log-only，统计 `phase=unknown` 比例
- [ ] U1：实现 phase-aware evidence slots shadow mode
- [ ] A0/A1：固定行为等价验证，避免 U 系列收益污染 baseline
- [ ] 单元级通过后，再进入 U1+U2、U1+U3、U1+U2+U3 组合实验

---

## 15. 一句话结论

U 系列把新论文启发转化为可落地实验轴：

```text
U1 解决“LLM 在什么阶段该看什么证据”；
U2 解决“什么时候真的应该 STOP”；
U3 解决“失败后如何受限恢复”。
```

最终论文不应写成模块堆叠，而应写成：

> 一个 phase-aware、evidence-grounded、ablation-safe 的 VLN-CE controlled navigation harness。

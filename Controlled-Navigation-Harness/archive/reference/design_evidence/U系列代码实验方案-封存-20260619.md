---
tags:
  - 实验
  - Open-Nav
  - VLN-CE
  - harness
  - U-series
  - code-design
created: 2026-06-19
updated: 2026-06-19
version: v0.1
role: U 系列代码实验方案
status: draft
derived_from:
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/U系列新论文创新点实验方案]]"
related:
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/实验方案]]"
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/U系列新论文创新点实验方案]]"
  - "[[07_Experiment/05_独立项目/Controlled-Navigation-Harness/V系列多模态视觉证据初步实现后调整方案]]"
---

# U 系列代码实验方案

## 0. 目标

本文档把 [[U系列新论文创新点实验方案]] 转成可落地的代码实现计划。

U 系列代码目标不是一次性实现完整系统，而是先建立 **U0 log-only 诊断层**，再逐步打开 U2 / U3 / U1 的单元级 decision-effect。这里不再引入 M 系列命名，避免和 U1/U2/U3 混淆。

```text
U0-log:       U 系列 trace 基础层，只记录，不改动作
U1:           phase-aware evidence scaffolding 单元实验
U2:           evidence-grounded STOP verifier 单元实验
U3:           failure-conditioned recovery 单元实验
U4-combined:  组合展示，不做单元归因
```

实现推进顺序按当前问题优先级执行：

```text
U0-log -> U2 -> U3 -> U1 -> U4-combined
```

---

## 1. 当前代码基线

### 1.1 已有扩展模块

当前 Open-Nav 扩展集中在：

```text
/root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/
```

已有模块：

| 文件 | 作用 |
|------|------|
| `harness_config.py` | harness / module 开关、A1 行为边界校验 |
| `metrics_logger.py` | harness trace jsonl |
| `visual_evidence.py` | V1 候选视觉证据 |
| `visual_target_verifier.py` | V2 STOP verifier |
| `visual_evidence_memory.py` | V3 episode 视觉记忆 |
| `multimodal_selector_context.py` | V4 selector 视觉上下文 |
| `visual_fallback.py` | empty / STOP rejected fallback ranker |
| `oracle_metrics.py` | distance gain 等 oracle 诊断 |

主循环接入点：

```text
/root/wjj/Open-Nav/vlnce_baselines/common/base_il_trainer_llm.py
```

关键现有事件：

```text
visual_evidence
visual_evidence_sampling
multimodal_selector_context
visual_target_verifier
visual_stop_allowed
visual_stop_rejected
visual_stop_rescued
completion_gate_weak_final_target
selector_empty_prediction_fallback
action_post_step
episode_end
```

### 1.2 现有问题对代码的直接要求

| 问题 | 当前事件 | U 系列代码需求 |
|------|----------|----------------|
| `visual_stop_rescued` 成功仅 `1/24` | `visual_stop_rescued` | U2 必须能关闭 rescue，并记录 relation / trajectory / weak-target 证据 |
| weak final target 漏停 | `completion_gate_weak_final_target` | U2 必须记录 weak-target decision，不先直接改 gate |
| empty fallback 退化 | `selector_empty_prediction_fallback` | U3 必须记录 failure_type、selected_rank、post-fallback distance gain |
| step>=4 drift | `action_post_step.distance_gain_selected` | U1/U3 必须记录 phase、current_subgoal、recent_progress |
| V4 上下文噪声 | `multimodal_selector_context` | U1 必须提供 phase-aware wrapper，而不是直接重写 V4 |

---

## 2. 配置设计

### 2.1 新增配置节点

在 `vlnce_baselines/config/default.py` 新增：

```python
_C.OPENNAV_HARNESS.U_SERIES = CN()
_C.OPENNAV_HARNESS.U_SERIES.ENABLED = False
_C.OPENNAV_HARNESS.U_SERIES.DECISION_EFFECT_UNIT = "none"

_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE = CN()
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.ENABLED = False
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.LOG_ONLY = True
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.UNKNOWN_CONFIDENCE_THRESHOLD = 0.4
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.LATE_STEP_THRESHOLD = 4

_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER = CN()
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.ENABLED = False
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.LOG_ONLY = True
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.ENABLE_RESCUE = False
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.ENABLE_RELATION_CHECK = False
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.ENABLE_WEAK_TARGET_ADJUSTMENT = False

_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY = CN()
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.ENABLED = False
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.LOG_ONLY = True
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.ENABLE_RESELECT = False
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.MAX_RECOVERY_PER_EPISODE = 2
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.NEGATIVE_GAIN_WINDOW = 2
```

### 2.2 decision-effect 规则

`DECISION_EFFECT_UNIT` 只能取：

```text
none | U1 | U2 | U3 | combined
```

规则：

| 配置 | 允许行为 |
|------|----------|
| `none` | 只记录 U trace，不改动作 |
| `U1` | 只允许 phase-aware context 改 selector 输入 |
| `U2` | 只允许 STOP gate / rescue / weak-target 策略改 STOP |
| `U3` | 只允许 failure recovery 改 fallback candidate |
| `combined` | 最终展示用，不用于单元归因 |

### 2.3 校验逻辑

扩展 `opennav_ext/harness_config.py`：

```python
def u_series_enabled(config) -> bool
def u_decision_effect_unit(config) -> str
def u_module_enabled(config, module_name: str) -> bool
def u_module_log_only(config, module_name: str) -> bool
```

并在 `validate_a1_harness_config()` 中加入：

```text
如果 U_SERIES.ENABLED=true 但 ENABLE_HARNESS_LOGGING=false -> error
如果 U module LOG_ONLY=false 但 ENABLE_DECISION_EFFECT=false -> error
如果 DECISION_EFFECT_UNIT=none，所有 U module 必须 LOG_ONLY=true
如果 DECISION_EFFECT_UNIT=U1，只允许 PHASE_EVIDENCE.LOG_ONLY=false
如果 DECISION_EFFECT_UNIT=U2，只允许 STOP_EVIDENCE_VERIFIER.LOG_ONLY=false
如果 DECISION_EFFECT_UNIT=U3，只允许 FAILURE_RECOVERY.LOG_ONLY=false
combined 只允许显式实验配置使用
```

---

## 3. 新增代码模块

### 3.1 文件结构

新增文件：

```text
vlnce_baselines/common/opennav_ext/
  phase_evidence.py
  evidence_scaffolder.py
  stop_evidence_verifier.py
  failure_diagnostic.py
  recovery_policy.py
  decision_audit.py
```

同时更新：

```text
vlnce_baselines/common/opennav_ext/__init__.py
vlnce_baselines/common/opennav_ext/harness_config.py
vlnce_baselines/config/default.py
run_OpenNav.yaml
vlnce_baselines/common/base_il_trainer_llm.py
```

### 3.2 `phase_evidence.py`

职责：

- 根据 instruction / actions / landmarks / current_step / history / estimation / recent gains 判断 phase；
- 输出 log-only 的 `phase_evidence`；
- 不直接改 selector 输入。

核心类：

```python
class PhaseEvidenceTracker:
    def __init__(self, late_step_threshold: int = 4, unknown_confidence_threshold: float = 0.4): ...

    def build(
        self,
        instruction: str,
        actions: str,
        landmarks: str,
        estimation: str,
        history: str,
        current_step: int,
        recent_distance_gains: list,
        stop_gate_metadata: dict,
    ) -> dict: ...
```

输出：

```yaml
phase_evidence:
  phase: search | approach | verify | recover | unknown
  current_subgoal: str
  phase_reason: str
  visual_budget: low | medium | high
  confidence: float
  recent_progress:
    recent_distance_gains: list
    non_positive_gain_count: int
```

首版 phase 规则保守实现：

| 条件 | phase |
|------|-------|
| selector empty / stop rejected / recent negative gain | recover |
| STOP proposal 或 final_action_completed | verify |
| current_step <= 2 且完成动作少 | search |
| estimation 有已完成动作但未完成 final | approach |
| 其他 | unknown |

### 3.3 `evidence_scaffolder.py`

职责：

- 包装已有 `MultimodalSelectorContext`；
- 根据 `phase_evidence.phase` 选择不同 evidence slots；
- U1 LOG_ONLY=true 时只产出 shadow context；
- U1 decision-effect 时替换 `selector_observe_dict`。

核心类：

```python
class PhaseAwareEvidenceScaffolder:
    def build(
        self,
        observe_dict: dict,
        visual_evidence_results: dict,
        visual_memory_results: dict,
        phase_evidence: dict,
        unified_context_results: dict,
    ) -> dict: ...
```

输出：

```yaml
phase_aware_context:
  applied: bool
  phase: str
  context_mode: route_overview | subgoal_candidate | stop_verify | recovery | conservative
  augmented_observe_dict: dict
  summaries: dict
  suppressed_slots: list
  candidate_count: int
  candidate_evidence_count: int
```

首版策略：

| phase | context_mode | 处理 |
|-------|--------------|------|
| search | route_overview | 使用短 visual summary，不注入 target/arrival 强词 |
| approach | subgoal_candidate | 保留 matched/missing/current_subgoal |
| verify | stop_verify | 不改普通 selector，交给 U2 |
| recover | recovery | 保留 failure_type / recent progress |
| unknown | conservative | 回退 V4 unified 的短摘要 |

### 3.4 `stop_evidence_verifier.py`

职责：

- 不替代 `VisualTargetVerifier`，而是生成 U2 层 evidence verdict；
- 对 current-view STOP evidence 增加 relation / trajectory / weak-target 字段；
- 控制 visual rescue 是否允许。

核心类：

```python
class StopEvidenceVerifier:
    def __init__(self, enable_rescue: bool, enable_relation_check: bool, enable_weak_target_adjustment: bool): ...

    def verify(
        self,
        source: str,
        visual_verifier_results: dict,
        stop_gate_metadata: dict,
        phase_evidence: dict,
        instruction: str,
        actions: str,
        landmarks: str,
        estimation: str,
        current_step: int,
    ) -> dict: ...
```

输出：

```yaml
stop_verification:
  source: completion_gate | selector_stop_gate
  target_type: object | room | area | landmark | weak_target | unknown
  intrinsic_support: yes | no | unknown
  relation_support: yes | no | unknown
  trajectory_support: yes | no | unknown
  weak_target_adjustment: none | relax | block
  allow_stop: bool
  allow_rescue: bool
  reject_reason: str
  confidence: float
```

首版 U2 decision 规则：

| 模式 | 规则 |
|------|------|
| `U2-rescue-off` | `allow_rescue=false`，任何 selector STOP rescue 都不放行 |
| `U2-relation-aware` | final target 可见但 relation unknown/negative 时不 rescue |
| `U2-weak-target-adjusted` | weak target all_actions_completed 且 current_step>=阈值时允许进入 stricter current-pano check |

### 3.5 `failure_diagnostic.py`

职责：

- 统一分类 selector empty、STOP rejected、negative gain、loop；
- log-only 阶段只写 `failure_type_diagnostic`；
- 为 U3 recovery policy 提供输入。

核心类：

```python
class FailureDiagnostic:
    def diagnose(
        self,
        trigger_type: str,
        current_step: int,
        phase_evidence: dict,
        recent_distance_gains: list,
        fallback_results: dict,
        stop_verifier_results: dict,
        selector_outputs: dict,
    ) -> dict: ...
```

输出：

```yaml
failure_type_diagnostic:
  trigger_type: selector_empty | stop_rejected | negative_gain | loop | candidate_missing
  failure_type: progress_drift | selector_wrong | empty_fallback_bad | stop_false_positive | candidate_missing | unknown
  confidence: float
  evidence: dict
```

### 3.6 `recovery_policy.py`

职责：

- 根据 failure_type 做受限 top-k reselect；
- 不扩展动作空间；
- 不做多步自由规划。

核心类：

```python
class FailureConditionedRecoveryPolicy:
    def propose(
        self,
        failure_diagnostic: dict,
        observe_dict: dict,
        visual_evidence_results: dict,
        fallback_results: dict,
        phase_evidence: dict,
        recovery_budget_remaining: int,
    ) -> dict: ...
```

输出：

```yaml
recovery_state:
  selected_recovery: none | reselect_from_topk | continue_current_subgoal | rotate_disambiguation
  selected_candidate: str
  original_fallback_candidate: str
  override: bool
  override_reason: str
  budget_remaining: int
```

首版只实现：

```text
U3-reselect-only:
  selector_empty 或 stop_rejected 时，从 fallback ranked_candidates 中选择:
    - 非原始 first-order 优先
    - matched_instruction_terms / local_instruction_hits 高
    - weak target false positive penalty
    - confidence 高
```

### 3.7 `decision_audit.py`

职责：

- 每次最终动作前写 audit；
- 用于证明单元级 override 是否发生。

核心函数：

```python
def build_decision_audit(
    unit_enabled: str,
    original_action: str,
    proposed_action: str,
    final_action: str,
    override_reason: str,
    expected_failure_addressed: str,
) -> dict: ...
```

事件：

```text
decision_audit
unit_override
```

---

## 4. 主循环插入点

### 4.1 初始化阶段

位置：`base_il_trainer_llm.py` 中现有 V 系列初始化附近。

新增对象：

```python
phase_evidence_tracker = None
phase_aware_scaffolder = None
stop_evidence_verifier = None
failure_diagnostic = None
recovery_policy = None
u_decision_unit = "none"
recovery_budget_remaining = 0
recent_distance_gains = []
```

初始化顺序：

```text
读取 OPENNAV_HARNESS.U_SERIES
  -> 初始化 PhaseEvidenceTracker
  -> 初始化 PhaseAwareEvidenceScaffolder
  -> 初始化 StopEvidenceVerifier
  -> 初始化 FailureDiagnostic
  -> 初始化 FailureConditionedRecoveryPolicy
```

### 4.2 episode start / reset

每个 episode 重置：

```text
recent_distance_gains = []
recovery_budget_remaining = MAX_RECOVERY_PER_EPISODE
phase_evidence_tracker.reset() if needed
```

写事件：

```text
u_series_episode_start
```

### 4.3 observation / visual evidence 后

现有流程已经生成：

```text
visual_evidence_results
visual_evidence_memory_results
multimodal_selector_context_results
```

新增：

```text
phase_evidence = PhaseEvidenceTracker.build(...)
log phase_evidence
```

注意：首版 phase_evidence 不依赖未来 oracle，只能使用当前可在线获得的信息。

### 4.4 selector 前 U1 插入

现有代码：

```text
selector_observation = observation
selector_observe_dict = observe_dict
if multimodal_selector_context_decision_effect:
    selector_observe_dict = augmented_observe_dict
```

新增逻辑：

```text
先照常构建 V4 unified context
再构建 U1 phase-aware context

if U1 LOG_ONLY:
    只记录 phase_aware_context

if DECISION_EFFECT_UNIT == U1:
    selector_observe_dict = phase_aware_context.augmented_observe_dict
    selector_observation = phase_aware_context.augmented_observation
```

约束：

- U1 不直接改 STOP；
- U1 不直接调用 fallback；
- U1 和 V4 不应同时都作为 decision-effect 竞争。U1 decision-effect 时，V4 unified 作为输入和对照记录。

### 4.5 STOP gate 后 U2 插入

现有 STOP 路径：

```text
completion gate:
  navigator.should_stop()
  record_visual_target_verifier()
  apply_visual_stop_gate()

selector STOP:
  navigator.should_stop()
  record_visual_target_verifier()
  visual_stop_can_rescue_selector_stop()
```

新增 U2：

```text
u2_stop_results = StopEvidenceVerifier.verify(...)
log stop_verification

if DECISION_EFFECT_UNIT == U2:
    if source == selector_stop_gate and current behavior wants rescue:
        rescue must pass u2_stop_results.allow_rescue
    if weak target adjustment enabled:
        allow / reject follows u2_stop_results weak_target_adjustment
```

首轮建议先做：

```text
ENABLE_RESCUE=false
ENABLE_RELATION_CHECK=false
ENABLE_WEAK_TARGET_ADJUSTMENT=false
```

也就是 U2-rescue-off，先验证关掉 `visual_stop_rescued` 是否减少误停。

### 4.6 empty fallback / STOP rejected 后 U3 插入

现有 empty fallback：

```text
if not predictions:
    fallback_results = rank_movement_fallback(...)
    predictions = [selected_fallback]
    log selector_empty_prediction_fallback
```

新增：

```text
failure_diag = FailureDiagnostic.diagnose(trigger_type="selector_empty", ...)
log failure_type_diagnostic

recovery_state = RecoveryPolicy.propose(...)
log recovery_state

if DECISION_EFFECT_UNIT == U3 and recovery_state.override:
    selected_fallback = recovery_state.selected_candidate
    log unit_override
```

STOP rejected fallback 也同样接：

```text
trigger_type="stop_rejected"
source_stage="stop_rejected"
```

### 4.7 action_post_step 后

现有 `action_post_step` 有：

```text
distance_gain_selected
distance_to_goal
```

新增：

```text
recent_distance_gains.append(distance_gain_selected)
保留最近 N=3
log post_action_progress
```

用于下一步 U1 phase 和 U3 failure diagnostic。

### 4.8 final action 前 decision audit

在每次 `action_pre_step` 前或 `selector_final` 后写：

```text
decision_audit:
  unit_enabled
  original_action
  proposed_action
  final_action
  override
  override_reason
```

如果发生 U1/U2/U3 override，额外写：

```text
unit_override
```

---

## 5. U 系列实现阶段

### U0-log：U 系列诊断层

目标：只记录，不改动作。

代码任务：

1. 新增配置节点 `U_SERIES`。
2. 新增 `phase_evidence.py`、`failure_diagnostic.py`、`decision_audit.py` 的 log-only 能力。
3. 主循环写事件：
   - `phase_evidence`
   - `failure_type_diagnostic`
   - `decision_audit`
4. 不改 selector prompt、不改 STOP、不改 fallback。

验证：

```bash
python -m py_compile \
  vlnce_baselines/common/opennav_ext/phase_evidence.py \
  vlnce_baselines/common/opennav_ext/failure_diagnostic.py \
  vlnce_baselines/common/opennav_ext/decision_audit.py \
  vlnce_baselines/common/base_il_trainer_llm.py \
  vlnce_baselines/config/default.py
```

小样本运行：

```bash
EPISODE_COUNT=5 EXP_NAME=u0_log_smoke bash run_OpenNav.bash
```

通过标准：

- 新事件存在；
- final action 与关闭 U_SERIES 时一致或可解释；
- SR/SPL 不作为本阶段收益判断；
- 无 `tool_failure` 或 fail-open 不改动作。

### U2：rescue-off / STOP evidence

目标：先解决 `visual_stop_rescued=1/24` 的误停风险。

代码任务：

1. 新增 `stop_evidence_verifier.py`。
2. 在 selector STOP rescue 前接入 U2。
3. `ENABLE_RESCUE=false` 时禁止 visual rescue 放行 STOP。
4. 写事件：
   - `stop_verification`
   - `weak_target_decision`
   - `unit_override` if STOP 被 U2 拦截或放行。

首轮配置：

```yaml
OPENNAV_HARNESS:
  ENABLE_DECISION_EFFECT: true
  U_SERIES:
    ENABLED: true
    DECISION_EFFECT_UNIT: U2
    STOP_EVIDENCE_VERIFIER:
      ENABLED: true
      LOG_ONLY: false
      ENABLE_RESCUE: false
      ENABLE_RELATION_CHECK: false
      ENABLE_WEAK_TARGET_ADJUSTMENT: false
```

小样本：

```bash
EPISODE_COUNT=20 EXP_NAME=u2_rescue_off_ep20 bash run_OpenNav.bash
```

通过标准：

- `visual_stop_rescued` 显著减少或为 0；
- STOP precision 高于当前 `9/48` 的趋势；
- OSR=1/SR=0 不明显恶化；
- 所有 STOP final action 有 verifier/audit 记录。

### U3：failure-type log + reselect-only

目标：让 empty fallback 不再机械退化为原始第一候选。

代码任务：

1. 新增 `recovery_policy.py`。
2. 在 `selector_empty_prediction_fallback` 后接入 failure diagnostic。
3. LOG_ONLY 阶段先标注：
   - selected rank；
   - original order；
   - failure_type；
   - post-fallback gain delayed join。
4. decision-effect 阶段只允许 top-k reselect。

首轮配置：

```yaml
OPENNAV_HARNESS:
  ENABLE_DECISION_EFFECT: true
  U_SERIES:
    ENABLED: true
    DECISION_EFFECT_UNIT: U3
    FAILURE_RECOVERY:
      ENABLED: true
      LOG_ONLY: false
      ENABLE_RESELECT: true
      MAX_RECOVERY_PER_EPISODE: 2
```

小样本：

```bash
EPISODE_COUNT=20 EXP_NAME=u3_reselect_ep20 bash run_OpenNav.bash
```

通过标准：

- fallback first-order ratio 低于 `41/79`；
- post-fallback distance gain 提升；
- recovery budget 不超限；
- 不引入自由动作或多步规划。

### U1：phase-aware V4

目标：减少后半程 drift 和 V4 上下文噪声。

代码任务：

1. 新增 `evidence_scaffolder.py`。
2. 复用现有 `MultimodalSelectorContext` 输出。
3. U1 LOG_ONLY 先生成 shadow context。
4. U1 decision-effect 时替换 selector 输入。

首轮配置：

```yaml
OPENNAV_HARNESS:
  ENABLE_DECISION_EFFECT: true
  U_SERIES:
    ENABLED: true
    DECISION_EFFECT_UNIT: U1
    PHASE_EVIDENCE:
      ENABLED: true
      LOG_ONLY: false
```

小样本：

```bash
EPISODE_COUNT=20 EXP_NAME=u1_phase_v4_ep20 bash run_OpenNav.bash
```

通过标准：

- `phase=unknown` 不高于 20%；
- step>=4 negative gain ratio 低于 V4-unified；
- SR/SPL/nDTW 不低于当前 V2/V4 小样本对照；
- prompt 长度不显著增加。

### U4-combined：组合系统

只有 U1/U2/U3 单元级通过后才做：

```text
U1 + U2
U1 + U3
U1 + U2 + U3
```

组合系统只用于最终展示，不能用于证明单个 U 单元有效。

---

## 6. 日志 schema

### 6.1 `phase_evidence`

```json
{
  "phase": "approach",
  "current_subgoal": "go to the left of the stairs",
  "phase_reason": "completed first action, final action incomplete",
  "visual_budget": "medium",
  "confidence": 0.72,
  "recent_progress": {
    "recent_distance_gains": [0.8, -0.1],
    "non_positive_gain_count": 1
  }
}
```

### 6.2 `stop_verification`

```json
{
  "source": "selector_stop_gate",
  "target_type": "weak_target",
  "intrinsic_support": "yes",
  "relation_support": "unknown",
  "trajectory_support": "no",
  "weak_target_adjustment": "block",
  "allow_stop": false,
  "allow_rescue": false,
  "reject_reason": "rescue_disabled_or_trajectory_not_supported",
  "confidence": 0.8
}
```

### 6.3 `failure_type_diagnostic`

```json
{
  "trigger_type": "selector_empty",
  "failure_type": "empty_fallback_bad",
  "confidence": 0.76,
  "evidence": {
    "fallback_selected_original_order": 0,
    "phase": "approach",
    "recent_non_positive_gain_count": 1
  }
}
```

### 6.4 `recovery_state`

```json
{
  "selected_recovery": "reselect_from_topk",
  "selected_candidate": "2",
  "original_fallback_candidate": "0",
  "override": true,
  "override_reason": "avoid_first_order_empty_fallback",
  "budget_remaining": 1
}
```

### 6.5 `decision_audit`

```json
{
  "unit_enabled": "U3",
  "original_action": "0",
  "proposed_action": "2",
  "final_action": "2",
  "override": true,
  "override_reason": "U3 reselect_from_topk",
  "expected_failure_addressed": "fallback_error"
}
```

---

## 7. 数据分析脚本

建议新增：

```text
Controlled-Navigation-Harness/scripts/
  summarize_u_series.py
  extract_u_debug_cases.py
```

### 7.1 `summarize_u_series.py`

输入：

```text
logs/navigation_records/*.jsonl
logs/harness_traces/*/*.jsonl
```

输出指标：

| 指标 | 来源 |
|------|------|
| STOP precision | `episode_end` + `stop_requested` |
| rescue precision | `visual_stop_rescued` + `episode_end` |
| phase unknown rate | `phase_evidence` |
| step>=4 negative gain | `action_post_step` |
| fallback first-order ratio | `selector_empty_prediction_fallback` / `recovery_state` |
| post-fallback gain | fallback step 的 `action_post_step` |
| unit override count | `unit_override` / `decision_audit` |

### 7.2 `extract_u_debug_cases.py`

输出四类固定 case list：

```text
rescue_fail
weak_target_miss
empty_fallback
late_drift
```

用于后续 paired run，避免每次换 episode 集合。

---

## 8. 验证命令

### 8.1 静态检查

```bash
git diff --check
python -m py_compile \
  vlnce_baselines/common/opennav_ext/phase_evidence.py \
  vlnce_baselines/common/opennav_ext/evidence_scaffolder.py \
  vlnce_baselines/common/opennav_ext/stop_evidence_verifier.py \
  vlnce_baselines/common/opennav_ext/failure_diagnostic.py \
  vlnce_baselines/common/opennav_ext/recovery_policy.py \
  vlnce_baselines/common/opennav_ext/decision_audit.py \
  vlnce_baselines/common/base_il_trainer_llm.py \
  vlnce_baselines/config/default.py
```

### 8.2 配置解析

```bash
python - <<'PY'
import yaml
with open('run_OpenNav.yaml') as f:
    yaml.safe_load(f)
print('yaml ok')
PY
```

### 8.3 smoke run

```bash
EPISODE_COUNT=5 EXP_NAME=u0_log_smoke bash run_OpenNav.bash
```

### 8.4 paired 小样本

```bash
EPISODE_COUNT=20 EXP_NAME=u2_rescue_off_ep20 bash run_OpenNav.bash
EPISODE_COUNT=20 EXP_NAME=u3_reselect_ep20 bash run_OpenNav.bash
EPISODE_COUNT=20 EXP_NAME=u1_phase_v4_ep20 bash run_OpenNav.bash
```

---

## 9. 风险与回滚

| 风险 | 表现 | 回滚 |
|------|------|------|
| U1 phase 误判 | `phase=unknown` 高或 selector drift 加重 | `PHASE_EVIDENCE.LOG_ONLY=true` |
| U2 过拒绝 | STOP 减少但 OSR=1/SR=0 增加 | `STOP_EVIDENCE_VERIFIER.LOG_ONLY=true` 或 `DECISION_EFFECT_UNIT=none` |
| U3 过触发 | recovery override 过多、路径变长 | `ENABLE_RESELECT=false` |
| 多单元混淆 | 无法判断收益来源 | 禁止 `combined` 进入单元实验 |
| 配置污染 A1 | A1 指标变化 | `U_SERIES.ENABLED=false` |

必须保留的全局回滚开关：

```yaml
OPENNAV_HARNESS:
  U_SERIES:
    ENABLED: false
    DECISION_EFFECT_UNIT: none
```

---

## 10. 当前待办

- [ ] 新增 `U_SERIES` 默认配置与 YAML 示例
- [ ] 扩展 `harness_config.py` 的 U 系列校验
- [ ] 实现 `phase_evidence.py`
- [ ] 实现 `failure_diagnostic.py`
- [ ] 实现 `decision_audit.py`
- [ ] 在主循环接入 U0-log 事件
- [ ] 实现 `stop_evidence_verifier.py`
- [ ] 接入 U2 rescue-off decision-effect
- [ ] 实现 `recovery_policy.py`
- [ ] 接入 U3 reselect-only decision-effect
- [ ] 实现 `evidence_scaffolder.py`
- [ ] 接入 U1 phase-aware selector context
- [ ] 新增 U 系列日志汇总脚本
- [ ] 跑 `u0_log_smoke` 5 episode
- [ ] 跑 U2/U3/U1 paired 20 episode

---

## 11. 一句话结论

U 系列代码实现应按以下顺序推进：

```text
U0-log 先让 U1/U2/U3 的证据都成为稳定 trace，
再只打开一个 U decision-effect 单元，
最后进入 U4-combined。
```

这样才能把当前 ep100 的低 SR 问题拆成可验证的代码实验，而不是继续堆模块。

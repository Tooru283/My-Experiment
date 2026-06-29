---
date: 2026-06-28
tags:
  - 日报
  - A0-baseline
  - arrival_gate
  - M1
  - 近目标终止
status: done
---

# 实验记录 2026-06-28

## 一、背景

本日工作围绕总方案（总方案-到达感知异构pipeline-20260626）的实验准备展开，目标是建立可归因的消融链：A0 anchor → 近目标失败信号分析 → M1 arrival_gate 实现 → 最强配置对比实验准备。

---

## 二、A0 Baseline 确认

**Run**：`ep100_series_qwen_siglip_local_20260628_125248`，val_unseen，100 episodes

**Config 状态**（确认为干净 A0）：
- `ENABLE_DECISION_EFFECT: False`
- `U_SERIES.ENABLED: False`
- V1/V2/V3/V4 全部 `LOG_ONLY: True`

**指标**：

| 指标 | 值 |
|---|---|
| SR | 16.0% (16/100) |
| OSR | 25.0% (25/100) |
| OSR→SR 转化率 | 64.0% (16/25) |
| OSR-SR gap | 9 episodes |
| SPL | 0.1015 |
| nDTW | 0.4458 |
| 平均步数 | 8.22 |

**近目标失败分类（9 个 OSR=1/SR=0 episodes）**：
- walk-through（T1 漏停）：7/9 = 77.8%，episodes：377, 513, 516, 531, 568, 586, 824
- off-goal-stop（T2/T4）：2/9 = 22.2%，episodes：602, 1084

**STOP 行为**：
- selector 请求 STOP：220 次
- 实际执行 STOP：44 次（精度 9/44 = 20.5%）
- stop_rejected_fallback：176 次

---

## 三、T1 信号分析

从 A0 JSONL 提取 7 个 walk-through episode 在 min_step 附近的 `stop_current_view_evidence` 信号，分析到达证据的分布。

**结果**（3 类子模式）：

| 类型 | Episodes | 特征 | M2 可救 |
|---|---|---|---|
| Group A：信号存在但 STOP 未执行 | 377, 513, 531 | arrival_evidence=True 在 min_step ±2 内出现，selector 也请求了 STOP，但 confidence 阈值未达到 | 是，M2 直接可救 |
| Group B：感知盲区 | 516, 824 | target_visible 全程 False，arrival_evidence 全程 False | 否，需要主动再观测 |
| Group C：时序错位 | 568, 586 | min_step 附近无采样，selector 的 STOP 请求出现在最近点之后 | 需 arrival_gate 提前触发 |

**M2 理论上限**：
- 仅靠现有 current_view 信号：可救 3/7 T1 → SR 最多 +3% → 19%
- 加主动再观测（Group B）+ arrival_gate 提前触发（Group C）：最多 5-6/7 → SR ~21-22%

---

## 四、代码修改

### 4.1 新增 `arrival_gate.py`（M1 实现）

路径：`vlnce_baselines/common/opennav_ext/arrival_gate.py`

触发条件：
```
estimated_goal_dist <= DIST_THRESHOLD (4.0m)
AND current_phase in {approach, verify, unknown}
AND current_step >= MIN_TRIGGER_STEP (1)
```

纯 log-only，不影响任何导航决策。每步记录 `arrival_gate` 事件到 JSONL。

### 4.2 Trainer 集成（4 处）

文件：`vlnce_baselines/common/base_il_trainer_llm.py`

1. 导入 `ArrivalGate`, `arrival_gate_enabled`, `arrival_gate_config`
2. 初始化 `arrival_gate = None` 和 `latest_goal_dist = None`
3. 每步执行后从 `step_output_summary` 提取并更新 `latest_goal_dist`
4. 在 `phase_evidence` 之后调用 `arrival_gate.check()` 并记录事件

### 4.3 Harness Config

文件：`vlnce_baselines/common/opennav_ext/harness_config.py`

新增 `arrival_gate_enabled()` 和 `arrival_gate_config()` 两个函数，从 YAML 读取 `ARRIVAL_GATE` 节点。

### 4.4 Analyze 脚本更新

文件：`scripts/analyze_navigation_jsonl.py`

新增 `arrival_gate` 事件处理，输出触发步数和触发 episode 数。

### 4.5 run_OpenNav.yaml — 两个配置变更

**A0 → M1 log-only**（今日 baseline 运行）：
```yaml
ARRIVAL_GATE:
  ENABLED: True
  DIST_THRESHOLD: 4.0
  ALLOWED_PHASES: [approach, verify, unknown]
  MIN_TRIGGER_STEP: 1
```

**A0 → 最强配置**（下一步对比实验）：

| 变更项 | A0 | 最强配置 |
|---|---|---|
| ENABLE_DECISION_EFFECT | False | True |
| TRACE_DIR | a0_baseline | max_config |
| V2 LOG_ONLY | true | **false** |
| V4 LOG_ONLY | true | true（约束不变）|
| U_SERIES.ENABLED | False | **True** |
| DECISION_EFFECT_UNIT | none | **combined** |
| U1 PHASE_EVIDENCE LOG_ONLY | true | **false** |
| U2 STOP_EVIDENCE_VERIFIER LOG_ONLY | true | **false** |
| U3 FAILURE_RECOVERY LOG_ONLY | true | **false** |

约束检查：`validate_a1_harness_config` 通过（U1 decision-effect 要求 V4 保持 log-only，已满足）。

---

## 五、V 系列与 U 系列的关系梳理

本日对两个系列的定位做了澄清（参见对话中的分析）：

| 系列 | 定位 | 与本研究的关系 |
|---|---|---|
| V1 | 视觉证据采集（log-only）| 所有视觉判断的数据源，必须开启 |
| V2 | 视觉 STOP 验证（可 decision-effect）| 直接控制 STOP 接受/拒绝，是 T1/T2 的核心 gate |
| V3 | 跨步证据记忆 | 辅助，保持 log-only |
| V4 | selector 多模态上下文（可 decision-effect）| 改善 selector 的 waypoint 选择，与 U1 互斥 |
| U1 | 阶段化证据脚手架 | 按 phase 注入不同上下文，approach/verify 阶段最关键 |
| U2 | 证据驱动 STOP 验证 | V2 的补充：rescue 被误挡的合理 STOP（直接对应 T1 Group A）|
| U3 | 失败条件恢复 | 处理 drift 和 empty fallback（对应 T5 和 U0 诊断的 fallback 退化）|

U 系列是 V 系列的「论文级重组」：不引入新的感知能力，而是把 V1-V4 的输出按阶段、按决策类型重新组织为可消融的方法贡献单元。

---

## 六、下一步

1. **立即**：跑 ep100 最强配置，与 A0 对比 SR/OSR/转化率
2. **分析**：用 analyze 脚本对比两次 run 的 arrival_gate 触发情况、STOP 行为变化、T1-T5 分类分布
3. **判断**：若最强配置的转化率没有提升甚至下降，重点排查 V2 的 REQUIRE_ARRIVAL_EVIDENCE 是否误挡了 Group B/C 的 STOP
4. **M2 实现**（后续）：arrival_gate triggered → 触发到达模式最小版（Group A 的自动终止）

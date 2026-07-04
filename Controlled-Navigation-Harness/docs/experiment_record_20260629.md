---
date: 2026-06-29
tags:
  - 日报
  - max_config
  - stop_gate
  - trajectory_bypass
  - M2
status: done
---

# 实验记录 2026-06-29

## 一、背景

承接 2026-06-28 的 A0 baseline 确认与 M1 arrival_gate 落地，昨晚跑完了 ep100 max_config（全部 U 系列 decision-effect 开启）第一次评测，今日分析结果并实施修复。

---

## 二、max_config 第一次评测结果

**Run**：`ep100_series_qwen_siglip_local_20260628_231537`，val_unseen，100 episodes  
**Config 状态**：max_config（U1+U2+U3 全开，decision-effect=True，M1 arrival_gate log-only）  
**日志位置**：
- stats: `logs/eval_results/ep100/20260628/ep100_series_qwen_siglip_local_20260628_231537/`
- harness traces: `logs/harness_traces/ep100/20260628/max_config/.../val_unseen/rank_0/`

### 2.1 核心指标对比

| 指标 | A0 baseline | max_config | 变化 |
|---|---|---|---|
| SR | 16.0% (16/100) | **21.0% (21/100)** | **+5pp** |
| OSR | 25.0% (25/100) | 26.0% (26/100) | +1pp |
| OSR-SR gap | 9 episodes | **5 episodes** | gap 减半 |
| OSR→SR 转化率 | 64.0% | **80.8%** | **+16.8pp** |
| SPL | 0.1015 | 0.1424 | +4pp |
| nDTW | 0.4458 | 0.4111 | -3.5pp（路径更迂回）|
| 平均终止距离 | 6.80m | 7.08m | — |
| 平均步数 | 8.22 | 8.17 | — |

**SR +5pp，OSR→SR 转化率 +16.8pp**，均为显著提升。nDTW 小幅下降反映 recover 阶段增多，路径变迂回，SPL 仍提升。

### 2.2 近目标失败分类变化（OSR=1, SR=0）

| 类型 | A0 (9集) | max_config (5集) |
|---|---|---|
| **walk-through**（步数耗尽未停） | 7集 (77.8%) | 2集 (40%) |
| **stop-blocked**（STOP 被门控拒绝） | 0集 | **3集 (60%)** |
| off-goal-stop（停在目标外） | 2集 | 0集 |

**主要失败模式已从"漏发 STOP"转变为"STOP 被门控拒绝"**。

### 2.3 STOP 行为分析（完整 harness traces）

| 指标 | A0 | max_config |
|---|---|---|
| selector STOP 请求 | 220次 | 174次 |
| 最终执行 STOP | 44次 | **47次** |
| STOP 成功（距离<3m）| 9/44 (20.5%) | 9/47 (19.1%) |
| stop_rejected_fallback | 176次 | **127次** |
| stop_verification 总检查 | — | **975次** |
| completion_gate 拒绝 | — | **801/817 (97.9%)** |

### 2.4 STOP 拒绝原因分布

| 原因 | 次数 |
|---|---|
| `final_target_not_visible` | **749** |
| `trajectory_incomplete` | 276 |
| `relation_contradiction` | 200 |
| `before_min_steps` | 3 |

`final_target_not_visible` 是 completion_gate 拒绝的主因（97.9% 的检查被拒绝）。

### 2.5 Arrival Gate (M1) 状态

| 指标 | 值 |
|---|---|
| 总检查次数 | 817次 |
| 触发步数 | 47/817 (5.8%) |
| 触发 episodes | 20/100 (20%) |

M1 成功运行，但 **5 个 OSR=1/SR=0 的集中，arrival_gate 均在近目标点未触发**，原因是这些集处于 **recover 阶段**，而 M1 的 `ALLOWED_PHASES` 未包含 recover。

---

## 三、5个失败集逐集分析

| episode | min_dist | min_step | 失败类型 | 根因 |
|---|---|---|---|---|
| 11 | **0.40m** | step 5 | stop-blocked | 全程 `trajectory_incomplete`；step 3/5 处于成功半径内仍被拦截；arrival_gate 全程 False（recover 阶段） |
| 824 | 2.58m | step 4 | stop-blocked | `final_target_not_visible` + `relation_contradiction` |
| 1106 | 2.68m | step 8 | stop-blocked | `final_target_not_visible` + `relation_contradiction` |
| 377 | 1.78m | step 7 | walk-through | arrival_gate step 6/7/8 触发，但 completion_gate reasons=[]（visual_allow=False）|
| 1084 | 1.32m | step 4 | walk-through | `final_target_not_visible` + `trajectory_incomplete`；arrival_gate 仅 step 2 触发一次后消失（recover 阶段）|

**Episode 11 是最严重的误拦截**：距目标 0.40m 时仍被 `trajectory_incomplete` 拒绝 STOP，arrival_gate 全程不触发（recover 阶段屏蔽）。

---

## 四、根因诊断

### RC1：`trajectory_incomplete` 在成功半径内不应作为硬拒绝

当 agent 已物理上处于目标 3m 内时，「所有计划动作未完成」不应是阻止 STOP 的理由。当前逻辑将 `all_actions_completed=False` 直接映射为硬拒绝，导致 ep11（0.40m）、ep1084（1.32m）被无理拦截。

### RC2：ARRIVAL_GATE 对 recover 阶段不可见

`ALLOWED_PHASES = [approach, verify, unknown]` 不含 recover。near-goal 失败集（ep11、ep1084、ep1106）恰好在距离最近时处于 recover 阶段，arrival_gate 无法感知。

### RC3：`final_target_not_visible` 过于激进（749/817）

近距离停靠时目标可能不在视野中（已到达其旁边），但视觉系统仍报告 not_visible，导致 completion_gate 97.9% 的检查失败。

### RC4：Episode 377 的 visual_allow=False（reasons=[]）

arrival_gate 触发了（step 6/7/8），但 completion_gate 以空原因拒绝（visual_allow=False 但无具体原因），说明 VLM 的整体判定与逐项标准不一致，是模型侧校准问题。

---

## 五、今日代码修改

### 5.1 `run_OpenNav.yaml` — ARRIVAL_GATE 新增 recover

```yaml
ARRIVAL_GATE:
  ALLOWED_PHASES:
    - approach
    - verify
    - recover    # ← 新增
    - unknown
```

解决 RC2：recover 阶段近目标步骤现在可触发 arrival_gate。

### 5.2 `run_OpenNav.yaml` — STOP_EVIDENCE_VERIFIER 新增 TRAJECTORY_BYPASS_DIST

```yaml
STOP_EVIDENCE_VERIFIER:
  TRAJECTORY_BYPASS_DIST: 3.5  # ← 新增
```

配合代码修改，当 `latest_goal_dist < 3.5m` 时，将 `trajectory_incomplete` 从硬拒绝降级为 unknown。

### 5.3 `vlnce_baselines/common/opennav_ext/stop_evidence_verifier.py`

- `StopEvidenceVerifier.__init__` 新增 `trajectory_bypass_dist: float = 0.0`
- `StopEvidenceVerifier.verify()` 新增 `latest_goal_dist: Optional[float] = None` 参数
- 新增 bypass 逻辑：`trajectory_support="no"` + `latest_goal_dist < trajectory_bypass_dist` → 降级为 `"unknown"`
- 返回值新增 `trajectory_dist_bypassed` 字段便于日志追踪

### 5.4 `vlnce_baselines/common/base_il_trainer_llm.py`

- `StopEvidenceVerifier()` 初始化新增 `trajectory_bypass_dist=getattr(stop_config, "TRAJECTORY_BYPASS_DIST", 0.0)`
- `record_stop_evidence_verification()` 调用新增 `latest_goal_dist=latest_goal_dist` kwarg

---

## 六、预期效果

| Episode | min_dist | 当前 | 修复后预期 |
|---|---|---|---|
| ep11 | 0.40m | stop-blocked (trajectory_incomplete) | step 3/5 bypass → 视觉证据允许则 SR=1 |
| ep1084 | 1.32m | walk-through (trajectory_incomplete) | bypass → 视觉证据允许则 SR=1 |
| ep824 | 2.58m | stop-blocked (final_target_not_visible) | 暂无修复，需 RC3 方案 |
| ep1106 | 2.68m | stop-blocked (final_target_not_visible) | 暂无修复，需 RC3 方案 |
| ep377 | 1.78m | walk-through (visual_allow=False) | 暂无修复，需模型校准 |

预期 trajectory bypass 可带来 **+1~2pp SR**，使 SR 提升至约 22-23%。

---

## 七、M2a → M3 根因修正

M2a trajectory_bypass 验证后发现：bypass 生效（trajectory_incomplete 从 reject_reasons 移除），但 ep11、ep1084 仍失败。根因修正如下：

**M2a 的实际作用域（仅 selector_stop_gate rescue）**：
- `apply_visual_stop_gate` 只能拦截已有的 STOP，不能主动发起
- 当 navigator 不输出 STOP 时（ep11 step 3-10、ep1084 全程），completion_gate 的 U2 allow_stop 无决策效果
- ep11 仅 step 2 有 selector STOP 请求，但被 `before_min_steps:2<3` 拒绝；步骤 3-10 navigator 没有再发出 STOP

**M3 proactive_stop_gate 设计（当日实施）**：
在 completion_gate 的 `apply_visual_stop_gate` 之后插入新检查：
```
条件：not stop_flag
     AND latest_goal_dist < PROACTIVE_STOP_GATE.DIST_THRESHOLD (3.5m)
     AND completion_verifier_results.final_target_visible=True
     AND completion_verifier_results.arrival_evidence=True
行为：重新以 stop_flag=True 调用 V2（record_visual_target_verifier）
     若 V2 返回 "allow"，设 stop_flag=True → 强制 STOP
```

**5集验证结果（ep5_m3 run）**：

| episode | proactive触发 | 结果 |
|---|---|---|
| ep11 | 5次，step 4 V2允许 → STOP | **SR=1, SPL=1.0，4步成功** ✓ |
| ep377 | 3次，step 10 V2允许 → STOP | dist=3.317m，仍 >3m，SR=0 |
| ep824 | 4次，V2全部拒绝 | SR=0（目标不可见） |
| ep1084 | 0次（final_target_visible=False全程）| SR=0 |
| ep1106 | 0次（同上） | SR=0 |

ep11 完全修复（0.46m最近，step 4 M3触发，SPL=1.0）。ep1084/ep1106 需 M2b/RC3 方案。

## 八、下一步（更新于 2026-06-29 日报末尾）

1. **立即**：恢复100集数据集，跑 ep100 M3 评测，预期 SR ≥ 22%
2. **分析**：ep11 能否在100集中稳定（random seed 无关），trajectory_bypass 是否带来额外改善
3. **ep377**：M3 在 step 10 允许停止但 dist=3.317m（比max_config的4.362m改善），需研究为何早期（1.78m最近点）V2拒绝而晚期允许
4. **M2b/RC3**：ep1084、ep1106、ep824 的 `final_target_not_visible` 问题，考虑距离衰减或 arrival_evidence 替代

---

## 九、M3 ep100 完整评测结果与失误分析（2026-06-30 补录）

**Run**：`ep100_series_qwen_siglip_local_20260629_215656`，val_unseen，100 episodes  
**Config**：max_config + M3 ProactiveStopGate（DIST_THRESHOLD=3.5m）

### 9.1 指标汇总

| 指标 | A0 | max_config | **M3** | 对比 max_config |
|---|---|---|---|---|
| SR | 16% | 21% | **20%** | **↓1pp** |
| OSR | 25% | 26% | **24%** | **↓2pp** |
| OSR→SR 转化率 | 64% | 80.8% | **83.3%** | +2.5pp |
| SPL | 0.1015 | 0.1424 | 0.1366 | — |
| nDTW | 0.4458 | 0.4111 | 0.4180 | — |

SR 与 OSR 均**低于** max_config，M3 净效果为负。转化率小幅提升，但 OSR 下降说明错误 STOP 增多。

### 9.2 集级别增减分析

| episode | max_config 状态 | M3 状态 | 净变化 |
|---|---|---|---|
| **ep11** | SR=0（step 2 min_steps 被拒） | **SR=1，SPL=1.0（step 4 M3触发，0.46m）** | **+SR +SPL** |
| **ep259** | SR=0，OSR=0（未入成功圈） | **SR=0，OSR=0（M3在3.34m提前停，原可达2.11m）** | **−OSR** |
| **ep321** | SR=0，OSR=1（经过目标3.15m最近） | **SR=0，OSR=0（M3在3.15m停，原最近1.91m未到）** | **−OSR** |

**收益**：ep11 +1 SR  
**损失**：ep259 −1 OSR，ep321 −1 OSR  
**净结果**：SR +1−0=+1（但基准ep中ep11在max_config也是SR=0），OSR −2

### 9.3 失误根因

**ep259 / ep321 根因：M3 触发阈值 3.5m > 成功半径 3.0m**

M3 条件：`dist < 3.5m AND final_target_visible=True AND arrival_evidence=True`

- ep259：agent 在 dist=3.34m 时 V1 报 ftv=True、arrival_evidence=True → M3 触发 → V2 返回 allow → 执行 STOP。但 3.34m 在 3m 成功圈外，任务失败。若继续，dist 可降至 2.11m（成功圈内）。
- ep321：类似场景，dist=3.15m 时提前停，最近可达 1.91m。

**共同机制**：M3 在 [3.0m, 3.5m] 这个"外圈过渡带"提前提交 STOP，导致 OSR 归零（原来至少会走到更近的位置被计入 OSR）。

**非临时补丁问题**：简单缩小阈值到 2.5m 虽可避开当前两集，但本质上是在一维距离上猜参数，不具泛化性。正确方案见第十节 E3。

### 9.4 M3 对 RC3 失败集无效

ep824/ep1084/ep1106（`final_target_not_visible` 导致 V2 拒绝）：M3 同样依赖 `final_target_visible=True AND arrival_evidence=True` 才触发。这三集的 ftv 全程为 False，M3 对其完全无效，是 RC3 的独立问题。

### 9.5 结论

| 问题 | 现象 | 根因 | 方向 |
|---|---|---|---|
| M3 过早提交 | ep259/ep321 在 3.0–3.5m 停止 | commit_dist > success_radius | E3-A：abstain zone [2.5m, 3.5m]，不提交 |
| RC3 近距 ftv 失准 | ep824/ep1084/ep1106 被 not_visible 拒绝 | 目标充满视野/被遮挡，VLM 无法标 visible | E3-B：arrival_evidence 替代 + 距离衰减 |

下一步：实施 E3 abstain+re-observe 机制（见 experiment_record_20260630.md）。

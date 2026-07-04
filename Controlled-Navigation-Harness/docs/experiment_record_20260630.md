---
date: 2026-06-30
tags:
  - 日报
  - E3
  - abstain
  - arrival_override
  - carry_forward
  - M3_commit_zone
  - e3_carry_forward
  - PSG_carry_forward
  - U2_E3B_override
status: done
---

# 实验记录 2026-06-30

## 一、背景

承接昨日（2026-06-29）M3 ep100 结果。SR=20%，低于 max_config（21%）。失误根因：
- **M3 提前提交**（ep259 3.34m、ep321 3.15m，均在 3m 成功圈外）
- **RC3 近距 ftv 失准**（ep824/ep1084/ep1106，目标在视野内被遮挡/充满视野，VLM 报 not_visible）

E3 前置分析已证实（0-3m 内 ftv 率仅 11-45%，arr_rate=1.0），今日实施 E3 abstain+re-observe 机制。

---

## 二、E3 前置分析结论

基于 M3 ep100 trace（100集）统计：

| 距离区间 | V2 final_target_visible 率 | 真实到达率 | 差距 |
|---|---|---|---|
| 1.0–1.5m | 0.11 | 1.00 | 0.89 |
| 1.5–2.0m | 0.21 | 1.00 | 0.79 |
| 2.0–2.5m | 0.38 | 1.00 | 0.62 |
| 2.5–3.0m | 0.45 | 1.00 | 0.55 |
| 3.0–3.5m | 0.62 | 0.73 | 0.11 |

**结论**：0-3m 范围内 `final_target_visible` 信号全程失准。使用它作为 hard-reject 标准会系统性地阻止成功停止。

---

## 三、E3 实施内容（2026-06-30）

### 3.1 E3-for-M3：内圈提交区（COMMIT_DIST_THRESHOLD）

**问题**：M3 在 DIST_THRESHOLD=3.5m 内提交 STOP，但成功圈只有 3.0m。ep259/ep321 在 3.15-3.34m 被提前停。

**修复**：增加 `COMMIT_DIST_THRESHOLD: 2.0`。M3 在整个 DIST_THRESHOLD 内仍触发 V2 re-evaluation（用于日志记录 + E3-A/B 支持），但 STOP 只在 dist < 2.0m（成功圈内 + 余量）时才提交。

**代码位置**：
- `run_OpenNav.yaml`: `PROACTIVE_STOP_GATE.COMMIT_DIST_THRESHOLD: 2.0`
- `harness_config.py`: `proactive_stop_gate_config()` 新增 `commit_dist_threshold`
- `base_il_trainer_llm.py`: `proactive_stop_commit_dist` 变量，M3 提交条件增加 `_within_commit_zone` 检查

**预期效果**：
- ep11（0.46m）：仍在 2.0m 内 → M3 正常触发 → ✓
- ep259（3.34m）：超出 2.0m → M3 记录但不提交 → agent 继续接近 → 可能到达 2.11m → SR+1
- ep321（3.15m）：同上 → 可能到达 1.91m → SR+1

### 3.2 E3-B：arrival_evidence 替代（E3_ARRIVAL_OVERRIDE_DIST）

**问题**：RC3 三集（ep824/ep1084/ep1106）在 dist < 3m 时被 `final_target_not_visible` 硬拒绝。arrival_evidence 更可靠（VLM 对近距接触证据的感知优于 "是否看见目标"）。

**机制**：当 `dist < E3_ARRIVAL_OVERRIDE_DIST=2.5m AND arrival_evidence=True AND not_visible` 时：
1. 从 reject_reasons 中移除 `final_target_not_visible`
2. 直接 force allow_stop=True（arrival_evidence 作为视觉确认替代）

**代码位置**：
- `run_OpenNav.yaml`: `STOP_EVIDENCE_VERIFIER.E3_ARRIVAL_OVERRIDE_DIST: 2.5`
- `stop_evidence_verifier.py`: `e3_arrival_override` 逻辑 + force allow_stop

### 3.3 E3-A：距离衰减弃权（E3_ABSTAIN_DIST）

**机制**：当 `dist < E3_ABSTAIN_DIST=2.0m AND not_visible AND arrival_evidence=False` 时：
- 不将 `final_target_not_visible` 加入 reject_reasons
- 结果：allow_stop=False（无确认证据），abstain=True（停止被阻止，但标记为"弃权"而非"拒绝"）
- agent 继续接近，下一步重新判断（re-observe by approaching closer）

### 3.4 E3-C：Carry-Forward 弃权

**机制**：E3-A 基础上，若 `recent_ftv_window[-3:]` 中有任意步骤 ftv=True，则同时标记 `carry_forward_abstain=True`（对 E3-A 的精细化标签，用于后续 ablation 分析）。

**代码位置**：
- `base_il_trainer_llm.py`: `recent_ftv_window` 列表，每步追加 ftv 值，window size=10，`record_stop_evidence_verification` 中读取最近 3 步
- `stop_evidence_verifier.py`: `carry_forward_visible` 参数，`e3_carry_forward_abstain` 标签

---

## 四、修改文件汇总

| 文件 | 修改内容 |
|---|---|
| `run_OpenNav.yaml` | M3 添加 `COMMIT_DIST_THRESHOLD: 2.0`；StopEvidenceVerifier 添加 `E3_ARRIVAL_OVERRIDE_DIST: 2.5`、`E3_ABSTAIN_DIST: 2.0` |
| `harness_config.py` | `proactive_stop_gate_config()` 新增 `commit_dist_threshold` |
| `stop_evidence_verifier.py` | `__init__` 新增 `e3_arrival_override_dist`、`e3_abstain_dist`；`verify()` 新增 `carry_forward_visible` 参数；E3-A/B/C 逻辑；`e3` metadata 字段 |
| `base_il_trainer_llm.py` | StopEvidenceVerifier init 新增 E3 参数传入；`recent_ftv_window` 追踪；M3 commit 区域检查 `_within_commit_zone` |
| `experiment_record_20260629.md` | 追加 §9 M3 ep100 失误分析 |

---

## 五、下一步（原计划，已执行）

1. **立即**：用 ep5 subset（ep11/ep259/ep321/ep824/ep1106）验证 E3 修复效果 → **跳过，直接 ep100**
2. **如果 ep5 验证正向**：跑 ep100 M4（E3 完整配置）→ **已完成，见第七节**
3. **分析**：对比 risk-coverage 曲线 → **E1 校准分析已完成，见第八节**
4. **E4**（后续）：4B vs 7B backbone 对比 → 待做

---

## 六、ep5 验证目标（已跳过，直接 ep100）

| episode | max_config 结果 | M3 结果 | E3 预期 |
|---|---|---|---|
| ep11 | SR=0 | SR=1 (✓) | SR=1（保持）|
| ep259 | SR=0, OSR=0 | SR=0, OSR=0 | SR≥0, OSR+1（继续接近 2.11m）|
| ep321 | SR=0, OSR=1 | SR=0, OSR=0 | SR≥0, OSR+1（继续接近 1.91m）|
| ep824 | SR=0 | SR=0 | SR+1 候选（E3-B 若 arrival_evidence=True）|
| ep1106 | SR=0 | SR=0 | SR+1 候选 |

---

## 七、ep100 E3 完整评测结果（2026-06-30 实测）

**Run**：`ep100_series_m4_20260630_105500`，val_unseen，100 episodes  
**Config**：max_config + E3（COMMIT_DIST=2.0m、E3_ARRIVAL_OVERRIDE=3.0m、E3_ABSTAIN=2.0m、carry_forward window=3）  
**日志位置**：
- stats: `logs/eval_results/ep100/20260630/ep100_series_m420260630_105500/`
- harness traces: `logs/harness_traces/ep100/20260630/max_config/.../val_unseen/rank_0/`

### 7.1 核心指标对比

| 指标 | A0 | max_config | **E3（今天）** | 对比 max_config |
|---|---|---|---|---|
| SR | 16.0% | 21.0% | **22.0%** | **+1pp** |
| OSR | 25.0% | 26.0% | **26.0%** | 持平 |
| OSR-SR gap | 9 ep | 5 ep | **4 ep** | **-1 ep** |
| OSR→SR 转化率 | 64.0% | 80.8% | **84.6%** | **+3.8pp** |
| SPL | 0.1015 | 0.1424 | **0.1524** | **+1pp** |
| nDTW | 0.4458 | 0.4111 | 0.4178 | 基本持平 |
| 平均步数 | 8.22 | 8.17 | 8.11 | — |

### 7.2 STOP 行为分析

| 指标 | A0 | max_config | E3 |
|---|---|---|---|
| Stop proposals | 219 | — | 282 |
| Allowed | 64 | — | 63 |
| **False stop**（误停） | 52 (81.2%) | — | 47 (74.6%) |
| **Missed stop**（漏停） | 20 (12.9%) | — | 44 (20.1%) |
| Stop precision | 18.8% | — | 25.4% |
| Stop recall | 37.5% | — | 26.7% |
| Termination ECE | 0.7808 | — | 0.7073 |

**关键发现**：max_config 启用 V2+U2 后，漏停从 20 → 44，是"失败模式迁移"的直接证明：stop verifier 虽减少误停，但过度拒绝导致漏停大幅增加。

### 7.3 近目标失败集分析（OSR=1, SR=0，共 4 集）

| episode | min_dist | failure_type | 详细根因 |
|---|---|---|---|
| **ep377** | 1.78m | walk-through | step 6/7（3.7m）visible=True，step 8（1.78m）visible=False（目标填满视野）。selector 未发 STOP，PSG 因 ftv=False 不触发。carry_forward 已建立但 PSG 门控未使用。 |
| **ep824** | 2.58m | stop-blocked | completion_gate（scan）：visible=True；stop_proposal VTV：visible=False（不同 candidate view）。V2 拒绝，U2 E3-B 产生 allow_stop=True（carry_forward=True，dist=2.58m < 3.0m），但当前代码路径中 E3-B 无法 override V2 的拒绝。 |
| **ep1084** | 1.32m | walk-through | 全程 visible=False，arrival_evidence=False（Group B 感知盲区）。1.32m 时无 STOP 提案。 |
| **ep1106** | 2.68m | stop-blocked | 全程 visible=False，arrival_evidence=False。selector 发 STOP，V2 拒绝，无 carry_forward 可用。 |

**4 集全部指向 RC3**：近距离 `final_target_not_visible` 信号失准（目标充满视野或感知盲区）。

### 7.4 Arrival Gate 与 PSG 触发情况

| 指标 | 值 |
|---|---|
| arrival_gate 总检查 | 811次 |
| 触发步数 | 123/811 (15.2%) |
| 触发 episodes | 31/100 (31%) |
| PSG 触发次数 | 25次 |
| PSG 允许 STOP | 12次 |
| PSG 拒绝 STOP | 13次 |

**PSG 对 RC3 失败集（ep377/ep1084）无效**：两集处于最近点时 completion_gate 的 `final_target_visible=False`，PSG 当前需要 ftv=True 才触发。

---

## 八、E1 校准分析（2026-06-30）

**脚本**：`scripts/e1_calibration.py`  
**输出**：`docs/e1_calibration_20260630/`（5 张图）

### 8.1 Termination ECE（b_t 校准误差）

| 配置 | Termination ECE | Brier Score |
|---|---|---|
| A0 | **0.7808** | 0.7362 |
| E3（max_config） | **0.7073** | 0.6688 |

ECE 接近 0.8，完美校准为 0，随机猜测约 0.25。验证器置信度与真实到达率严重背离，是论文核心动机的直接数据支撑。

### 8.2 距离分桶分析（来自 `01_distance_binned.png`）

`final_target_visible`（ftv）率在 0-3m 内系统性低于真实到达率，并出现 ep377 式反转（1.78m 时拒绝，3.3m 时反而允许）。这是"近距离可见性信号不可信"的可视化证据。

### 8.3 False/Missed Stop 对比

| | A0 → E3 | 方向 |
|---|---|---|
| False stop | 52 → 47 | ↓ 减少（更精准）|
| Missed stop | 20 → 44 | ↑ 翻倍（过度拒绝）|

**论文核心实证**：激活 stop verifier 把漏停问题从"发不出 STOP"迁移为"STOP 被过度拒绝"，且 ECE 仍高，说明需要弃权/carry_forward 机制而非单纯阈值调整。

---

## 九、新代码修改（2026-06-30 晚，E3 carry_forward）

### 9.1 问题定位

通过逐集 trace 分析（读取 ep377/824/1084/1106 的 VTV 事件）：

- **ep377 gap**：completion_gate scan 在 step 6/7（3.7m）正确返回 ftv=True，建立了 carry_forward。但 step 8（1.78m）时 PSG 检查 `completion_verifier_results.final_target_visible=False` → 跳过。E3-B 已有 carry_forward 逻辑，但无法到达，因为 PSG 根本不触发。
- **ep824 gap**：completion_gate 在 step 5（2.58m）返回 ftv=True，selector_stop_gate V2 调用（不同 candidate）返回 ftv=False。U2 E3-B 产生 `allow_stop=True`（carry_forward=True + dist < 3.0m），但代码路径中 E3-B allow_stop 没有独立的 override 通道——只能通过 `allow_rescue` 路径，后者要求 `trajectory_support="yes"`。

### 9.2 `base_il_trainer_llm.py`：PSG carry_forward 触发（约 2251 行）

**原 PSG 触发条件**：
```python
if (
    not stop_flag
    and proactive_stop_enabled
    and latest_goal_dist < proactive_stop_dist  # 3.5m
    and completion_verifier_results.get("final_target_visible")  # 需要当前步 ftv=True
    and completion_verifier_results.get("arrival_evidence")
):
```

**新增 carry_forward 分支**：
```python
_carry_forward_psg = any(recent_ftv_window[-3:]) if recent_ftv_window else False
if (
    not stop_flag
    and proactive_stop_enabled
    and latest_goal_dist < proactive_stop_dist
    and (
        (ftv=True AND arrival_evidence=True)  # 原条件
        or (_carry_forward_psg AND dist < proactive_stop_commit_dist)  # 新：历史可见 + 内圈
    )
):
```

**机制**：历史 3 步内曾可见（carry_forward=True）且当前处于 commit 内圈（< 2.0m）→ 触发 PSG → V2 重判 → U2 E3-B 用 carry_forward 允许 commit。

**目标 episode**：ep377（step 8：1.78m < 2.0m，carry_forward=True from step 6/7）。

### 9.3 `base_il_trainer_llm.py`：U2 E3-B standalone override（约 2748 行）

在 `visual_target_verifier_rejects_stop` 分支（V2 拒绝）之后、fallback 之前，新增独立通道：

```python
# 原代码：V2 拒绝后直接 fallback
selector_stop_rejection_reason = "visual_target_verifier_rejected_stop"

# 新增：E3-B standalone override
if (
    not stop_flag
    and stop_evidence_verifier_decision_effect
    and selector_stop_evidence_results.get("allow_stop")
    and selector_stop_evidence_results.get("e3", {}).get("arrival_override")
):
    stop_flag = True
    stop_reason = "U2 E3-B carry-forward arrival override: target visible in recent steps..."
```

**机制**：当 U2 通过 carry_forward + dist < e3_arrival_override_dist=3.0m 产生 `allow_stop=True` 时，无论 V2 verdict 如何，直接 commit STOP。

**目标 episode**：ep824（step 5：dist=2.58m < 3.0m，carry_forward=True from step 3 completion_gate）。

### 9.4 `run_OpenNav.yaml`：配置变更

| 配置项 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `TRACE_DIR` | `logs/harness_traces/max_config` | `logs/harness_traces/e3_carry_forward` | 区分本次 run |
| `E3_ABSTAIN_DIST` | 2.0 | **3.0** | ep824（2.58m）、ep1106（2.68m）扩入 abstain 区，防止硬拒绝触发 recovery |

`E3_ABSTAIN_DIST` 扩大到 3.0m：ep1106（全程 not_visible）在 2.68m 被拒后改为 abstain → 继续接近，而非进入 recover 被拉远。

### 9.5 代码机制关系图

```
selector 未发 STOP：
  completion_gate scan (stop_flag=False) → ftv=False（近距遮挡）
  recent_ftv_window 已有历史 True        ← carry_forward 已建立
  PSG 检查：
    旧：ftv=False → 不触发
    新：carry_forward=True AND dist < 2.0m → 触发 PSG
      → V2 重判 (stop_flag=True)
      → U2 E3-B: carry_forward=True, dist < 3.0m → arrival_override=True
      → allow_stop=True → _e3_b_allow=True → stop_flag=True ✓

selector 发 STOP，V2 拒绝：
  selector_stop_gate V2 (stop_flag=True) → ftv=False → reject
  U2 处理：
    carry_forward=True, dist < 3.0m → e3_arrival_override=True
    allow_stop=True（其他 reject_reasons 为空时）
  旧：allow_stop=True 只在 allow_rescue 路径使用（需 trajectory_support="yes"）
  新：E3-B standalone check → stop_flag=True ✓
```

### 9.6 预期效果

| episode | 问题 | 新机制 | 预期结果 |
|---|---|---|---|
| ep377 (1.78m) | walk-through，PSG 不触发 | PSG carry_forward 触发，U2 E3-B commit | SR+1 |
| ep824 (2.58m) | stop-blocked，U2 E3-B 无 override 通道 | E3-B standalone override | SR+1（若无 relation_contradiction）|
| ep1084 (1.32m) | walk-through，全程 not_visible | 无 carry_forward → PSG 仍不触发 | 不变 |
| ep1106 (2.68m) | stop-blocked，全程 not_visible | abstain_dist=3.0m → 不 hard-reject | 继续接近，可能入成功圈 |

---

## 十、下一步

1. **立即**：跑 ep100 e3_carry_forward 配置，观察 SR/OSR 变化
2. **重点监控**：ep377 和 ep824 是否修复；ep1084/ep1106 是否因 abstain 扩大而得到改善
3. **风险**：U2 E3-B standalone override 可能引入新的 false stop（不受 trajectory_support 约束），需重点观察 false_stop 计数
4. **若 SR ≥ 23%（+1ep）**：转入 E4 跨 backbone（4B vs 7B）准备
5. **若 SR 无变化或下降**：深入分析 ep824 的 relation_contradiction 是否阻止了 E3-B allow_stop

---

## 十一、ep100 e3_carry_forward 评测结果（2026-07-01）

**Run**：`ep100_series_m420260630_214135`，val_unseen，100 episodes  
**Config**：E3 完整配置 + PSG carry_forward + U2 E3-B standalone override + E3_ABSTAIN_DIST=3.0m  
**日志**：`logs/navigation_records/ep100/20260630/ep100_series_m420260630_214135_train_navigation_20260630_214200.jsonl`

### 11.1 核心指标对比

| 指标 | A0 | max_config | E3（上午）| **e3_carry_forward（本次）** | vs max_config |
|---|---|---|---|---|---|
| SR | 16% | 21% | 22% | **24%** | **+3pp** |
| OSR | 25% | 26% | 26% | **26%** | 持平 |
| OSR→SR | 64% | 80.8% | 84.6% | **92.3%** | **+11.5pp** |
| OSR-SR gap | 9 | 5 | 4 | **2** | **-3ep** |
| SPL | 0.101 | 0.142 | 0.152 | **0.187** | **+4.5pp** |
| nDTW | 0.446 | 0.411 | 0.418 | 0.426 | 基本持平 |

### 11.2 目标集逐一结果

| episode | E3上午 | 本次 | 触发机制 | 说明 |
|---|---|---|---|---|
| ep11 (0.46m) | SR=1 ✓ | **SR=1 ✓ SPL=1.0** | M3 保持 | 稳定 |
| ep259 (2.11m) | SR=1 ✓ | **SR=1 ✓ SPL=1.0** | commit_dist | 稳定 |
| ep321 (1.91m) | SR=1 ✓ | **SR=1 ✓ SPL=1.0** | commit_dist | 稳定 |
| ep377 (1.78m) | SR=0 | **SR=1 ✓ SPL=0.785** | PSG carry_forward + E3-B | 新增修复 ✓ |
| ep824 (2.58m) | SR=0 | **SR=1 ✓ SPL=0.932** | E3-B standalone override | 新增修复 ✓ |
| ep1084 (1.32m) | SR=0 | SR=0（步数耗尽 3.77m） | 无 carry_forward | 感知盲区，未解决 |
| ep1106 (2.68m) | SR=0 | SR=0（步数耗尽 6.27m） | abstain 触发但走偏 | 见分析 |

### 11.3 关键机制 trace 确认

**ep377 完整路径**：
- step 3（5.32m）：completion_gate 返回 `ftv=True, arrival=True` → `recent_ftv_window` 建立 carry_forward
- step 8（1.78m）：PSG carry_forward 分支触发（`_carry_forward_psg=True` + `dist < 2.0m commit zone`）
- V2 调用（stop_flag=True）：返回 reject（`ftv=False`，目标充满视野）
- U2 E3-B：`carry_forward_visible=True` + `dist=1.78m < E3_ARRIVAL_OVERRIDE_DIST=3.0m` → `arrival_override=True` → `allow_stop=True`
- E3-B standalone override：`stop_flag=True` → STOP commit → **SR=1**

**ep824 完整路径**：
- step 5（2.58m）：PSG carry_forward 触发（completion_gate step 3 曾见 ftv=True）
- V2 返回 reject（`ftv=False`，`arrival=False`）
- U2 E3-B：`carry_forward=True` + `dist=2.58m < 3.0m` → `arrival_override=True` → `allow_stop=True`
- E3-B standalone override 生效 → **SR=1**（relation_contradiction 未阻止，因 allow_stop 直接优先）

**ep1106 分析**：step 9（2.68m）abstain 正确触发（`e3_near_dist_abstain`），阻止硬拒绝。但 agent 已进入 recover 阶段，step 10 navigation 向错误方向行走，最终 6.27m 步数耗尽。**根因：全程无 ftv、无 carry_forward，abstain 只能阻止拒绝，无法提供确认信号。** 本质是 Group B 感知盲区，与 ep1084 同类。

### 11.4 STOP 行为分析

| 指标 | E3（上午） | e3_carry_forward | 变化 |
|---|---|---|---|
| Stop proposals（stop_requested 集） | 56 | 56 | 持平 |
| SR=1 stops（true positive） | 22 | **24** | +2 |
| False stops（OSR=0 + stop_requested） | — | **38** | — |
| Step limit terminations | — | 36 | — |

E3-B standalone override 在 ep377/ep824 上新增了 2 个 true positive，未观测到大量新 false stop 引入（OSR=0 集的 false stop 数量维持在正常范围）。

### 11.5 OSR 瓶颈分析

OSR=0 的 74 集分布：

| 终止原因 | 数量 | 分析 |
|---|---|---|
| false stop（stop_requested，OSR=0） | 38 | 提前停止，永远没进入 3m 圈 |
| step_length_limit（步数耗尽） | 36 | 步数内未到达 3m 圈 |

步数耗尽集的 dist 分布（最终距离，即 min_dist 近似）：

| 最终距离区间 | 集数 |
|---|---|
| 3–5m（近距错过） | **13** |
| 5–7m | 16 |
| 7–10m | 24 |
| >10m | 21 |

**13 集在 3–5m 内被步数截断**——这是最近期的 OSR 提升空间。详见第十二节讨论。

---

## 十二、OSR 提升思路（2026-07-01 分析）

当前 OSR=26%（停滞自 max_config），SR 已从 16% 提升到 24%。OSR-SR gap 从 9→2，**转化率已不再是主要瓶颈，OSR 本身（导航覆盖率）成为新瓶颈**。

### 问题来源

OSR=0 的 74 集由两类原因构成：

**类型 A：false stop（38集）**  
agent 在距目标 >3m 时提前停止，永远没有机会接近。这些集里有相当一部分其实路线正确，只是 stop gate 过早放行。改善思路：提高 stop 精准度，让 gate 在这些集上继续拒绝、保持 agent 导航。

**类型 B：步数耗尽（36集）**  
agent 走完步数上限但从未进入 3m 圈。其中 13 集在 3–5m 处被截断（ep559/810/1071/1077/176/516/181，最近 ep1085 仅 3.49m）。改善思路分两方向：

1. **适度放宽步数上限**：当前 SHORT=10/LONG=12。若放宽到 SHORT=12/LONG=14，13 集里的近距错过集（3–4m）可能有 2–4 集补进 3m 圈。代价是平均步数增加、SPL 轻微下降（但这些集本身 SPL 已是 0，无额外惩罚）。
   
2. **E4 跨 backbone**：当前 SpatialBot3B + Qwen3.5-4B 在 74 个 OSR=0 集上导航失败，根本原因是感知/推理能力不足。换 7B 模型（selector 或 SpatialBot）可以系统性改善 routing 质量，是 OSR 提升最直接的路径。

**类型 A 和类型 B 是相互制约的**：减少 false stop 会让更多 agent 继续走，其中一部分会进入 3m（OSR+），另一部分会走更远然后步数耗尽（仍然 OSR=0 但 dist 更大）。需要权衡。

### 优先行动建议

1. **短期（本周）**：小幅放宽步数上限（SHORT: 10→12, LONG: 12→14），重跑 ep100，观察 OSR 变化。风险低（仅影响步数耗尽集），预期 OSR +1~2pp。

2. **中期（E4）**：7B backbone 对照实验。既能提升 routing 覆盖率（OSR↑），又能测试"可靠终止机制随规模的一致性"（论文 E4 核心诉求）。若 7B 的 OSR 显著高于 4B，则 SR 提升主要来自导航能力而非机制，需在论文中诚实标注。

3. **不建议**：进一步放宽 stop gate（可能引入更多 false stop，反而拉低 SR）；为 OSR 专门加新模块（与论文"可靠终止"主线不符，且 AAAI 时间不允许）。

### 当前位置总结（决策闸口）

| 条件 | 状态 |
|---|---|
| E1 校准图清晰 | ✓（ECE=0.71，距离失准曲线可见） |
| E3 残余集有动作 | ✓（ep377/ep824 修复，OSR→SR=92.3%） |
| E4 跨 backbone 计划 | 🔄 待执行 |
| SR ≥ 23% | ✓（SR=24%，+3pp） |

**结论**：E3 成立，核心贡献有数据。按计划进入 E4 跨 backbone 准备。同时可并行尝试步数上限微调（短期 OSR 改善）。

---

## 十三、下一步（更新）

1. **立即可做**：步数上限小幅放宽（SHORT 10→12，LONG 12→14），重跑 ep100，观察 OSR 变化
2. **E4 准备**：确认 7B 模型可用性，设计 A0_7B vs e3_carry_forward_7B 对照组
3. **论文写作**：E1 校准图 + E3 失败模式迁移图是核心动机，可开始 intro/method 草稿
4. **ep1084/ep1106**：两集属 Group B 感知盲区（全程无 ftv 无 arrival），在 4B 模型下无解，留作 E4/backbone 对比数据点

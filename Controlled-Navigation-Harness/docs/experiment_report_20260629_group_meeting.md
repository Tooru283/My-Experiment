# Open-Nav / Controlled Navigation Harness 近两周实验报告

汇报时间：2026-06-29  
统计范围：2026-06-24 至 2026-06-29  
数据来源：
- `logs/eval_results/**/stats_ckpt_val_unseen.json`
- `logs/eval_results/**/stats_ep_ckpt_val_unseen_r0_w1.json`
- `logs/harness_traces/**/*.jsonl`（harness 事件追踪）
- `Controlled-Navigation-Harness/docs/experiment_record_*.md`

---

## 1. 摘要

本周围绕一个核心目标推进：**把 OSR（Oracle Success Rate，曾到过目标附近）转化为 SR（Success Rate，最终停在目标内）**。

三件主要工作：

1. **固定 A0 anchor**：以 A0（原始 Open-Nav + 本地 Qwen3.5-4B）作为固定基准，在 100 集 val_unseen 上确认 SR=16%，消除历史结果中的环境和模型版本噪声。
2. **完成 max_config 首次 ep100 评测**：全量开启 U 系列 decision-effect（U1 phase scaffold + U2 stop verifier + U3 failure recovery），SR 提升至 21%，OSR→SR 转化率从 64% 提升到 80.8%，OSR-SR gap 从 9 集缩减到 5 集。
3. **实施 M2a+M3 近目标门控修复**：针对最严重的 stop-blocked 失败模式，实现 trajectory_bypass（M2a）和 proactive_stop_gate（M3），5 集目标验证中 ep11（距目标 0.40m）完全修复（SPL=1.0，仅 4 步）。

**当前核心结论**：max_config 将 SR 从 16% 提升至 21%，创历史最高；主要失败模式已从"走过目标不停"转为"STOP 被门控过度拒绝"，修复方向清晰，但存量 OSR 集中仍有 4 集待解。

---

## 2. 核心指标演进

| 配置 | SR | OSR | OSR→SR | SPL | nDTW | 平均终止距离 |
|---|---:|---:|---:|---:|---:|---:|
| A0 baseline（2026-06-28） | 16% | 25% | 64.0% | 0.1015 | 0.4458 | 6.80m |
| max_config（2026-06-28） | **21%** | 26% | **80.8%** | **0.1424** | 0.4111 | 7.08m |
| M2a+M3 验证（5 集，2026-06-29） | — | — | — | — | — | ep11 修复 |

与上次汇报（2026-06-24）最优结果 SR=19% 相比：
- **max_config 达到 SR=21%，超越之前最好成绩 2pp，且为稳定的 100 集结果。**
- SPL 0.1424 也是历史最高，说明路径质量没有因多步恢复而退化。

与原文 Open-Nav 对比：

| 方法 | SR ↑ | OSR ↑ | SPL ↑ | nDTW ↑ |
|---|---:|---:|---:|---:|
| Open-Nav-Llama3.1（原文） | 16 | 23 | 12.90 | 44.99 |
| Open-Nav-GPT4（原文） | 19 | 23 | 16.10 | 45.79 |
| **本地 max_config（2026-06-28）** | **21** | **26** | **14.24** | **41.11** |

SR 和 OSR 均超过原文 GPT4，是项目首次稳定超越原文结果。SPL 略低于原文 GPT4（recover 路径增多），nDTW 偏低（路径迂回），两者均有改善空间，但不是当前主要矛盾。

---

## 3. 方法：近目标终止系统（M 系列）

### 3.1 问题定位

上次汇报（2026-06-24）的核心遗留问题是 **OSR-SR gap 大**：agent 经常走到目标附近，但最终没有停在 3m 内。当时分析认为主因是"走过不停"（walk-through）。

本周 max_config 首测揭示了问题的完整结构：

**失败模式迁移（OSR=1 但 SR=0 的集）**：

| 类型 | A0（9 集） | max_config（5 集） |
|---|---:|---:|
| walk-through（走过目标，没停） | 7 集（77.8%） | 2 集（40%） |
| stop-blocked（STOP 被门控拒绝） | 0 集 | **3 集（60%）** |

max_config 激活了 U2 stop verifier 后，walk-through 从 7 减到 2，**主失败模式已从"不发 STOP"转为"发了 STOP 但被门控拒绝"**。净结果：gap 从 9 降到 5。

### 3.2 STOP 门控拒绝根因分析

对 5 个失败集逐步复盘（通过 harness trace JSONL）：

| episode | 最近距离 | 失败类型 | 根因 |
|---|---|---|---|
| ep11 | **0.40m** | stop-blocked | `trajectory_incomplete` 强拒绝：指令动作未全完成，离目标 0.40m 仍被拦截 |
| ep1084 | 1.32m | walk-through | selector 全程未发 STOP；`final_target_visible=False` |
| ep824 | 2.58m | stop-blocked | `final_target_not_visible` + `relation_contradiction` |
| ep1106 | 2.68m | stop-blocked | 同上 |
| ep377 | 1.78m | walk-through | V2 在 1.78m 处拒绝（目标识别失败），退远后重新允许（3.3m），超出成功阈值 |

从 100 集全量 STOP 拒绝原因分布：

| 拒绝原因 | 次数（/817 次检查） |
|---|---:|
| `final_target_not_visible` | 749（91.7%） |
| `trajectory_incomplete` | 276（33.8%） |
| `relation_contradiction` | 200（24.5%） |

`final_target_not_visible` 是最主要的拒绝原因：近距离时目标已移出视野（太近看不到全貌），但 V2 verifier 仍报告不可见，导致 97.9% 的 completion_gate 检查被拒绝。

### 3.3 M2a：trajectory_bypass

**问题**：`trajectory_incomplete`（指令动作序列未全完成）被作为硬拒绝条件，无视物理距离。ep11 在 0.40m 处仍被此条件拦截。

**修复**：当 `latest_goal_dist < 3.5m` 时，将 `trajectory_support` 从 "no" 降级为 "unknown"，不再追加 `trajectory_incomplete` 到 reject_reasons。

**实验发现**：bypass 技术上生效（reject_reasons 中 trajectory_incomplete 消失），但 ep11/ep1084 仍失败——根本原因是 navigator（LLM selector）在近目标步骤根本不发出 STOP，而 M2a 只能救援已被拦截的 STOP 请求，无法主动发起。

这暴露了一个更深的架构问题：`apply_visual_stop_gate` 只能**拦截**现有 STOP，不能**主动发起**。

### 3.4 M3：proactive_stop_gate（本周核心修复）

**设计**：在 completion_gate 每步触发时，当以下条件同时满足，以 `stop_flag=True` 重新驱动 V2 做真实判断，不依赖 navigator 输出：

```
not stop_flag（navigator 未请求 STOP）
AND latest_goal_dist < 3.5m
AND final_target_visible = True（来自上一步 V2 的一般视觉证据）
AND arrival_evidence = True
```

若重新评估后 V2 返回 "allow"，则强制执行 STOP。

**5 集验证结果**：

| episode | M3 触发 | V2 结果 | 最终 |
|---|---|---|---|
| ep11 | step 3 reject，step 4 allow ✓ | allow（V2 正确识别目标） | **SR=1，SPL=1.0，4 步完成** |
| ep377 | step 10 allow | allow，但此时 dist=3.317m>3m | SR=0（已退离成功半径）|
| ep824 | 4 次触发，全 reject | V2 拒绝（目标不可见） | SR=0 |
| ep1084 | 0 次（final_target_visible=False 全程）| — | SR=0 |
| ep1106 | 0 次（同上）| — | SR=0 |

**ep11 修复详情**：step 3 时目标不在画面中（V2 reject，正确），step 4 目标重新出现在当前视角（V2 allow）→ 强制 STOP，agent 停在 0.46m 处，SPL=1.0。M3 的双重过滤（距离阈值 + V2 二次确认）有效避免了误停。

---

## 4. 当前残余失败模式（4 集）

```
RC3 — final_target_not_visible（3 集）
  ├─ ep824（min_dist=2.58m）— 近距离目标识别持续失败
  ├─ ep1106（min_dist=2.68m）— 同上，relation_contradiction 共存
  └─ ep1084（min_dist=1.32m）— 全程 final_target_visible=False，M3 无法触发

near-goal visibility reversal（1 集）
  └─ ep377（min_dist=1.78m）— V2 在 1.78m 拒绝，退至 3.3m 后允许
                              根因：极近距离时目标占满视野，模式匹配失效
```

这 4 集都指向同一个底层问题：**极近距离时视觉验证器无法正确识别目标**。可能原因：

1. 目标实体太近导致图像截断，VLM 无法识别。
2. arrival_evidence 信号（如门把手、地板接近）与 final_target_visible 不一致。
3. 指令中的目标描述词（如"the television"）与近距画面特征无法匹配。

---

## 5. 下阶段计划

### 5.1 RC3：近距离目标不可见修复（M2b）

**候选方案**：

| 方案 | 做法 | 风险 |
|---|---|---|
| 距离衰减 | `dist < 2m` 时降低 `final_target_not_visible` 为软拒绝 | 增加误停（T2） |
| arrival_evidence 替代 | 有强 arrival 信号时跳过 visibility check | arrival 信号本身可靠性待评估 |
| 视野切换 | 近距改为底部 / 水平视角再采一次图 | 需改感知层，成本较高 |

当前策略：先从 100 集 harness trace 中统计 `final_target_visible` 与距离的分布（2m 内有多少集目标实际可见），决定衰减阈值。

### 5.2 ep100 M3 完整验证

M3 的 ep100 评测正在运行，预期 SR ≥ 22%（ep11 稳定贡献 +1pp）。重点观测：
- M3 误停率（有多少非目标附近的集被强制停住）
- ep377 在 100 集中的行为（是否在最近点附近有 V2 allow）

### 5.3 跨 backbone 对照（SR ≥ 24% 后启动）

当前所有实验基于 Qwen3.5-4B 本地部署。M 系列修复完成后，计划加一组 7B/8B 对照，验证 OSR→SR 转化率的提升是否与 backbone 规模解耦。

---

## 6. 系统数据流说明

### 6.1 总体结构

系统每步执行两条平行路径：**感知路径**（图像→文字，给 selector 用）和 **V 系列验证路径**（图像→结构化证据，给 harness 用），最终汇入 Qwen 文字 LLM 做候选决策。

```
Habitat 环境
  └─ 每步输出全景 12 方向的 RGB（224×224）+ Depth（256×256）图

候选航点生成（WaypointBert，本地预训练 Transformer，参数冻结）
  └─ RGB 特征 + Depth 特征 → 120角度×12距离 概率热图
       → top-k 候选航点（每个候选：角度 + 距离 + 对应图像）

感知路径（图像→文字，每候选各跑一次）
  ├─ RAM（本地 SwinL）: RGB → 物体 tag 列表
  └─ SpatialBot3B（本地 3B VLM）: RGB + Depth → 空间场景文字描述
       合并 → "Direction 3: Scene Description: ... Scene Objects: chair, table..."

文字决策（Qwen3.5-4B，lmdeploy 本地服务，纯文字 API）
  ├─ 指令解析: 指令 → 动作列表 → 地标列表
  ├─ 完成度估计: 历史 + 动作 + 地标 → 已执行动作
  └─ Selector（NAVIGATOR）: 候选 ID + 指令 + 历史 + SpatialBot 文字 → 候选 ID 或 STOP

V 系列验证（Qwen3.5-4B 多模态，发送 base64 图片）
  ├─ V1 VisualEvidenceLogger（每步，LOG_ONLY）
  │    候选 RGB 图 → {final_target_visible, arrival_evidence, confidence...} × 每候选
  └─ V2 VisualTargetVerifier（STOP 时触发，decision-effect）
       当前视野全景拼图 → {verdict: allow/reject, reason}

U 系列门控（纯 Python 逻辑，无模型调用）
  ├─ U1 PhaseEvidenceTracker: V1 证据 → 阶段判断 → 修改 selector 输入文字
  ├─ U2 StopEvidenceVerifier: V1+V2+U1 → allow_stop（控制 STOP 是否执行）
  └─ U3 RecoveryPolicy: 失败类型 → recovery 候选
```

### 6.2 逐模块说明

**第零层：WaypointBert 候选生成（本地预训练神经网络）**

在所有 VLM 调用之前，首先要解决"能走到哪里"的问题。Habitat 提供的是连续 3D 空间，不是离散图节点，所以需要一个候选航点生成器：

- **模型**：`BinaryDistPredictor_TRM`（WaypointBert，Transformer 2层，参数冻结，不参与训练）
- **输入**：全景 12 方向的 RGB 特征（ResNet，2048×7×7）+ Depth 特征（128×4×4）
- **输出**：120个角度 × 12个距离 的概率热图，每个位置表示"此方向此距离是否为可行航点"
- **结果**：从热图中提取 top-k 候选航点，每个候选有角度 + 距离，映射到对应的图像（候选0~11，对应 0°~330° 每 30° 一个扇区）

**这一步决定了 selector 能选择的候选集合。** 后续所有模块（SpatialBot、V1、NAVIGATOR）都是对这 12 个候选航点进行描述、分析和选择，没有这一步就没有"候选"的概念。

**第一层：Habitat Simulator**

每步提供当前位置全景 12 方向的 RGB（224×224）和 Depth（256×256）图像，作为所有后续处理的原始输入。

**第二层：感知模块（本地推理，输出纯文字）**

| 模块 | 输入 | 输出 | 部署方式 |
|---|---|---|---|
| RAM（Recognize Anything） | RGB 图（224×224） | 物体 tag 列表（逗号分隔文字） | 本地 GPU，SwinL |
| SpatialBot3B | RGB + Depth | 空间场景描述（"The chair is 1.2m away..."） | 本地 GPU，3B VLM |

每个候选方向各跑一次，12 方向合并成一段文字 `observation` 传给 selector。**这一层输出的是文字，不是图片。**

**第三层：Qwen3.5-4B 文字 LLM（仅接收文字）**

通过 lmdeploy 本地服务（`http://127.0.0.1:23333/v1`，OpenAI-compatible）：

| 调用 | 输入 | 输出 |
|---|---|---|
| 指令解析（ACTION_DETECTION） | 导航指令（文字） | 动作步骤列表 |
| 地标提取（LANDMARK_DETECTION） | 动作列表（文字） | 地标词列表 |
| 完成度估计（COMPLETION_ESTIMATION） | 历史轨迹 + 地标 + 动作（文字） | 哪些动作已执行 |
| **Selector（NAVIGATOR）** | 候选 ID + 指令 + 历史 + estimation + **SpatialBot 文字** | 候选 ID 或 STOP |
| 思路融合（THOUGHT_FUSION） | 多个 thought（文字） | 融合后的 thought |
| 决策验证（DECISION_TEST） | 融合 thought + observation（文字） | 最终候选 ID |

**Selector 看到的是 SpatialBot/RAM 生成的文字描述，不是原始图片。**

**第四层：V 系列（向 Qwen 发送图片，输出结构化证据）**

V1 和 V2 与 selector 并行运行，使用同一个 Qwen3.5-4B 但通过多模态接口发送图片：

- **V1 VisualEvidenceLogger**（每步运行，LOG_ONLY 不影响动作）：
  - 输入：候选 RGB 图（base64 编码）+ 文字 prompt（指令、动作、地标）
  - 输出：每候选的结构化 JSON：`{final_target_visible, arrival_evidence, confidence, ...}`

- **V2 VisualTargetVerifier**（STOP 提案时触发，decision-effect 影响 STOP）：
  - 输入：当前视野所有方向拼成的全景 contact sheet（base64）+ 文字 prompt
  - 输出：`{verdict: allow/reject/uncertain, final_target_visible, arrival_evidence, reason}`
  - M3 proactive_stop_gate 即在 completion_gate 每步调用 V2（以 `stop_flag=True`），若 V2 返回 allow 则强制 STOP

**第五层：U 系列（Python 逻辑，无模型调用）**

U 系列是纯 Python 的决策层，读取 V1/V2 输出的结构化 JSON，做规则判断或候选重排，不额外调用任何模型。

**U1 — PhaseAwareEvidenceScaffolder（阶段感知证据注入）**

根据当前步数、已完成动作数、近期距离收益变化，将导航阶段归类为 `search / approach / verify / recover` 之一，并把对应的阶段提示（含 V4 视觉摘要）附加到 selector 的每个候选观测文字后面：

```
{SpatialBot 场景描述} [U1: phase=approach; mode=subgoal_candidate; budget=medium; subgoal=walk to the door]
```

让 selector 在不同导航阶段有不同的关注重点——搜索阶段只看路标，验证阶段还附带视觉摘要，恢复阶段附加失败历史。

**U2 — StopEvidenceVerifier（证据驱动 STOP 核查）**

当 selector 输出 STOP、V2 stop gate 返回结果后，U2 对 STOP 做最终裁决。它读取 V1 置信度、V2 verdict、U1 阶段、轨迹支持情况，综合判断"这次 STOP 是否应该执行"：

- **正常路径**：STOP 被 V2 拒绝时，U2 也输出"blocked"
- **Rescue 路径**（`ENABLE_RESCUE=true`）：当 V2 拒绝但 V1 置信度 ≥ 0.95 且处于 verify 阶段、近期距离有正增益时，U2 可以覆盖 V2 的拒绝，允许 STOP 执行
- **M2a trajectory_bypass**：当智能体已在目标 3.5m 内时，将"轨迹不完整"从硬拒绝降级为"未知"，减少因轨迹计算误差导致的误拒

**U3 — RecoveryPolicy（失败条件候选重选）**

当检测到确认的导航失败（selector 输出为空、STOP 误触发、连续负距离增益、路径环路）时，U3 从候选集中重新选择一个备用候选方向，代替原先的选择。每集最多触发 2 次（`MAX_RECOVERY_PER_EPISODE: 2`），只处理可恢复的失败类型。

**U 系列与 V 系列的分工**：V 系列负责感知（调用模型从图片提取信息），U 系列负责决策（用规则从结构化数据得出行动）。U 系列引入的额外推理成本为零。

### 6.3 多模态输入实际起了什么作用

系统内部存在**两条平行决策链**，而不是一条统一的图文决策链：

| 决策类型 | 处理模块 | 输入形式 | 直接使用图片 |
|---|---|---|---|
| **候选方向选择**（routing） | NAVIGATOR → Qwen 文字 LLM | 纯文字 | ✗ |
| **STOP 是否允许**（stop gate） | V2 VisualTargetVerifier → Qwen 多模态 | 图片 + 文字 | ✓ |

**selector 实际看到的文字**包含两层图片衍生内容：

```
{SpatialBot3B 场景描述}
[U1: phase=approach; subgoal=walk to the door; {V4 visual summary}]
```

- **第一层（SpatialBot3B + RAM）**：每步对 12 个候选方向分别推理，RAM 输出物体 tag 列表，SpatialBot3B 输出空间距离描述，两者合并成文字传给 selector。
- **第二层（V1 → V4 → U1）**：V1 将候选图片以 base64 发给 Qwen 多模态，得到结构化证据 JSON；V4 将 JSON 压缩成文字摘要；U1 将摘要作为 suffix 附加在每个候选的 SpatialBot 文字后面，再交给 selector。

图片本身从未进入 selector 的 API 调用——**selector 始终只调用文字接口**。多模态贡献的是"图片理解之后提炼出来的文字"。

**唯一让图片直接影响决策（不经过文字转换）的地方是 V2 stop gate**：当前视野全景图拼成 contact sheet，以 base64 直接发给 Qwen 多模态，返回 allow/reject 裁决。M3 的核心机制正是在 completion_gate 强制触发这一路，绕过了"selector 不主动发 STOP"的问题。

**结论**：当前系统是"文字决策为主、图片验证为辅"的架构。routing 决策的图片信息已经过两次降维（图片→SpatialBot 文字，图片→V1 JSON→V4 文字），STOP 决策是唯一端到端使用图片的路径。这也是为什么 OSR→SR 转化率的提升空间主要在 stop gate 侧，而不在 routing 侧。

### 6.4 V4 为什么保持 LOG_ONLY（未开启决策效果）

V4（`MULTIMODAL_SELECTOR_CONTEXT`）在设计上可以直接替换 selector 的 observe_dict，让文字观测中携带图片衍生的视觉摘要，即：

```
{SpatialBot3B 场景描述} [V4: cid=3; match=door,hallway; conf=medium]
```

但当前配置 `LOG_ONLY: true`，V4 不直接作用于 selector。原因有三：

**1. 架构设计约束：DECISION_MODE=phase_gated_u1**

V4 的 `decision_mode` 字段写为 `phase_gated_u1`，意思是"V4 的输出应由 U1 进行阶段门控后再注入，而非直接应用"。V4 计算出每个候选的 `selector_safe_summaries`，交给 U1 读取，U1 再根据当前导航阶段（search/approach/verify/recover）决定是否附加到候选文字后面。如果 V4 自行 apply，U1 的阶段门控就被绕过了。

**2. 同时开启会导致双重注入**

若 V4 decision-effect 开启，它先将观测改写为 `{SpatialBot} [V4: ...]`；随后 U1 读取这份已经改写的观测，再附加 `[U1: phase=...; {V4 summary again}]`。同一份 V4 视觉摘要被注入两次，人为放大了视觉证据的权重。代码中用 `u1_shadow_matches_ablation_context = not v4_context_applied` 标志位追踪这种情况。

**3. 安全约束：SUPPRESS_TARGET_ARRIVAL_FOR_SELECTOR=true**

V4 的 `selector_safe_summary` 主动屏蔽 `final_target_visible` 和 `arrival_evidence` 两个字段，不让 selector 直接看到"目标可见"信号。原因：如果 selector 看到"目标可见"就可能输出 STOP，而这一判断应由专门的 stop gate（V2/M3）负责，不应由 routing selector 直接做。即使将来 V4 打开，这一设计仍保留。

**当前实际路径（V4 LOG_ONLY + U1 decision-effect）：**

```
V1（图片→Qwen→JSON）→ V4（JSON→文字摘要，计算但不直接应用）
                                 ↓
                          U1 读取 V4 摘要
                          按导航阶段决定是否附加
                                 ↓
         selector 观测 = {SpatialBot} [U1: phase=...; {V4 摘要（阶段门控后）}]
```

若未来要让 V4 独立承担 routing 决策效果，需要先将 U1 切换为"V4 已应用时跳过视觉摘要注入"的模式，避免双重计数。

### 6.5 关于 SigLIP 的说明

实验目录名中出现的 `qwen_siglip_local` 是历史命名约定，**SigLIP 当前未被任何代码调用**。`OPENNAV_SIGLIP_PATH` 环境变量仅在 `run_OpenNav.bash` 中导出，但没有 Python 模块读取它。当前图像理解能力由 SpatialBot3B（场景描述）、RAM（物体标注）和 Qwen3.5-4B 多模态（V1/V2 视觉证据）三者分工承担。

---

## 7. 附：指标解释与实验方法说明

### 7.1 A0 锚点的重要性

上次汇报（2026-06-24）的历史最好 SR=19% 无法确认是"配置改进"还是"episode 随机性"，因为缺少固定的基准配置。

本周在同一组 100 集上同时跑 A0（所有 harness 模块 log-only，不影响行为）和 max_config（全开 decision-effect），两者使用相同随机种子、相同模型、相同 episode 顺序：

- A0: SR=16%，这是 Open-Nav baseline 在本地 Qwen3.5-4B 上的真实性能。
- max_config: SR=21%，+5pp 完全来自 harness decision-effect 的贡献，排除 episode 采样噪声。

### 7.2 OSR→SR 转化率是比 SR 更稳定的诊断指标

SR 容易因 episode 随机性波动（±1%），但 OSR→SR 转化率衡量的是"已到过目标附近的集中有多少能成功终止"，对近目标终止能力的描述更稳定：

| 配置 | OSR→SR 转化率 |
|---|---:|
| A0 | 64.0%（16/25） |
| max_config | **80.8%（21/26）** |

这意味着在所有曾进入 3m 的集里，max_config 能成功留住的比例从 64% 提升到 81%，是本周工作的核心贡献。

### 7.3 STOP 分析：gate 是瓶颈，不是 OSR

max_config 有 174 次 selector STOP 请求，最终只有 47 次执行，其中成功停在 3m 内的仅 9 次（成功率 19%）。这说明 STOP 的质量（而非数量）是核心问题——gate 拒绝了大量有效的近目标 STOP，而放行的却不都是好的。

M3 proactive_stop_gate 的价值正在于此：它不是增加 STOP 请求，而是在"有强视觉证据但 navigator 没说要停"的情况下主动触发，填补 gate 架构的盲区。

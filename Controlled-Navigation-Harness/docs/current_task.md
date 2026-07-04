---
date: 2026-07-01
tags:
  - current_task
  - living_document
status: active
updated: 2026-07-01
---

# Current Task — 导航实验当前状态与下一步

> 本文档是滚动更新的任务追踪，每次实验迭代后更新。不记录完整实验过程，只记录**当前问题、方案设计、预计下一步**。完整过程见对应日报。

---

## ⭐⭐ 目标协调（2026-07-02，AAAI 仍是目标）

**目标 = AAAI-27（全文 7/27）。终止与 OSR 不是两条路，是同一机制在两个决策点：**

| | 终止决策 | 导航/路由决策 |
|---|---|---|
| 机制 | 弃权+再观测（E3/E5,已做）| 弃权+再观测（P1,待做）|
| 理论 | VoI/选择性预测 | **同一个** |

**统一 AAAI 命题**：零样本 VLN-CE 小 VLM 的核心失败=**高代价决策点（路由+停止）上基于失准信念的过度提交**。training-free 校准驱动的选择性再观测（信念落不确定带→弃权取证）在两决策点一致提升 SR。
- 终止(E3/E5)=已验证锚点；**P1=推广+新OSR杠杆+论文核心卖点**（分水岭）
- 贡献三点：①OSR→SR分解+小VLM信念校准分析(method-agnostic,现有trace可算)②统一机制③跨backbone一致
- **SOTA gap 姿态**：不 beat SOTA（我们24<SmartWay29<Spatial-VLN33）；走"正交/互补(决策层training-free,可叠加)+分析贡献+泛化"
- **优雅降级**：P1有效→强论文；P1无效→退回终止-only(旧计划)仍可投
- 护城河 vs scoop：training-free + 校准驱动 + 双决策（区别 AwareVLN训练/Fast-SmartWay自问/FSR-VLN建图/Spatial-VLN多专家）

> 注：性能工作(P0/E5)**服务**AAAI贡献，非竞争——P0是补课(不写贡献)，P1是核心。OSR 是提升 SR 的手段，机制是论文的点。

**可执行清单见 → [[aaai_execution_plan_20260702]]**（贡献→图表映射、实验矩阵、周计划到7/27、优雅降级、审稿预案、写作大纲）。

**✅ E1 已做（2026-07-02，见 [[e1_calibration_20260702/README]]）**：4B e3_carry_forward trace 上——近距(0-3m,已到达)可见率仅 **43.2%**(严重欠检测)、远距(8+m)仍 16.9% 假阳；**Termination ECE=0.3236**(高置信桶经验到达仅20%,近似反校准)；false stop 38 主导。**论文 C1 动机图就绪，零实验依赖。** 待补：A0/9B 跨配置一致性、路由决策校准(扩到统一命题)、risk-coverage。

---

## ⭐ 前序结论（2026-07-01 文献对比）

**转化率 92.3% 已封顶（全场最高），SR 天花板是 OSR。攻 OSR 的机制（P1）同时是论文贡献。**

**架构优化方案与 P0 实现设计见 → [[architecture_optimization_20260701]]。关键发现：WaypointBert 真实几何 (`distance_dict`/`radius_dict`) 已算好但从未注入 navigator，后者靠 SpatialBot 幻觉距离盲选——P0 = 注入真实几何，training-free、改动小、最高 ROI。**

---

## 一、当前实验进度（2026-07-01）

### 1.1 已完成里程碑

| 实验 | SR | OSR | OSR→SR | 关键机制 |
|---|---|---|---|---|
| A0 基线 | 16% | 25% | 64% | 无 harness |
| max_config | 21% | 26% | 80.8% | V1+V2+V3+V4+U1+U2+U3 全开 |
| E3（首轮） | 22% | 26% | 84.6% | E3 abstain + arrival_override |
| **e3_carry_forward（4B）** | **24%** | **26%** | **92.3%** | PSG carry_forward + U2 E3-B standalone |

当前最优：**SR=24%，OSR=26%，gap=2**（4B 模型，ep100 val_unseen）

### 1.2 正在进行

- **ep100 9B 测试跑**（RTX 4090）：进度约 60/100，SR≈18%，**低于** 4B
  - 结论定性为"9B 零调优基线"，不作为正式结果
  - 根因：`NAVIGATOR_MAX_TOKENS=1024` 截断 9B 长输出（空预测 fallback 22.6% vs 4B 6.9%）+ 过度停止

---

## 二、文献对标（R2R-CE val_unseen, zero-shot）

| 方法 | 时间 | 模型 | SR | OSR | SPL | OSR→SR | 核心机制 |
|---|---|---|---|---|---|---|---|
| Open-Nav (原论文) | 2024 | GPT-4 | 19 | — | 16.1 | — | 我们的 base |
| CA-Nav | 2024.12 | — | 25.3 | 48.0 | — | ~53% | 约束感知子指令 |
| **SmartWay** | 2025.03 | GPT-4o | **29** | **51** | 22.5 | 56.9% | occupancy航点+回溯 |
| **Spatial-VLN** | 2026.01 | DeepSeek-v3 | **33** | 高 | — | — | 显式空间感知+冲突探索 |
| VLN-Zero* | 2025.09 | GPT-4.1/5 | 42.4 | 51.6 | 26.3 | 82% | 两阶段探索建图+缓存 |
| **我们** | 2026.07 | **Qwen 4B** | **24** | **26** | **18.7** | **92.3%** | 可靠终止（V2/U2/E3） |

\* VLN-Zero 两阶段（先探索建图），任务设定不同，仅供参考。

**三条关键读数：**
1. **SmartWay 是我们的直系后继**（同一实验室 Qiao & Wu，论文明确基于 Open-Nav）。它 OSR=51%，我们 26%，**几乎两倍**。
2. **Spatial-VLN 直接在 Open-Nav 上做对照**，SR 33%（+17pp over Open-Nav），核心论点是"**LLM 缺的是空间感知，不是终止能力**"——与我们总方案的核心论点**正面冲突**。
3. **我们的 OSR→SR 转化 92.3% 是全场最高**，甚至超过 SmartWay（56.9%）。**转化机制是我们的优势，缺的是喂给它更多 OSR。** 推算：若保持 92.3% 转化、OSR 提到 51%，SR ≈ 47%。

---

## 三、与"总方案-可靠终止决策"的张力（须正视）

**总方案的核心主张**：小 VLM 在零样本 VLN-CE 的剩余瓶颈是"近目标终止的可靠性"，而非到达能力。据此把 OSR→SR 转化作为中心目标。

**问题**：
1. **转化已到极限**。总方案自己已观测"OSR 几乎不动（25→26）"，且转化率已 92.3%，继续做 E3/E5 是在 26% 的 OSR 上抠个位数，边际收益趋零。
2. **总方案自己承认终止不是最大杠杆**（第7节诚实注记：AwareVLN 消融显示 stopping-error 是三节点中最小的 −5.4，远小于 subtask −13.1、deviation −10.3）。选终止是因为它对 AAAI-27 "最干净可测、可校准"，是**论文策略决定，不是 max-SR 决定**。
3. **撞车风险评估（2026-07-01 已读 DV-VLN / AgenticNav）**：
   - **Spatial-VLN（2026.01）**：直接在 Open-Nav 上、小模型、同样失败分类"doorway/multi-room/ambiguous"，用 OSR-focused 更强故事占了"小 VLM 零样本 VLN-CE"坑——**最大撞车风险**。
   - **DV-VLN（2601.18492）**：**不撞车**。它是训练式（LLaMA-2 微调）、离散 R2R/RxR/REVERIE、验证每步**动作**（TFV+MEV 重采样+实体恢复）。我们是零样本+VLN-CE+**校准驱动弃权**，正面区分。反而它的 generate-then-verify **支持**我们"验证从终止扩到导航"的转向（第四节 4.1）。
   - **AgenticNav（2606.09）**：工具调用 harness，action tool 让 VLM **直接选像素目标绕过航点预测器**，明确"优于传统航点预测器"——继 SmartWay 后**第三个独立证据**证明 WaypointBert 是瓶颈。
   - **Fast-SmartWay（2511.00933）**：**部分撞车 + 强力验证**。SmartWay 直系后继（同实验室，Qiao=Open-Nav 作者）。其 Uncertainty-Aware Reasoning（Disambiguation 自报困惑→重扫 + FPBR 预测-比对）= 我们提议的**导航级弃权+再观测**，且是其**最大消融杠杆 +8pp SR**（19.75→27.75）。**关键区分**：它的再观测触发是"**问 MLLM 你困惑吗**"（未校准的 LLM 内省），我们是"**校准驱动的不确定带/VoI**"——总方案的理论内核恰好成为对它的差异化锚点。它还**端到端消除航点预测器**（第4个航点瓶颈证据）。

**两个目标需要分清：**
- **目标A（用户本轮明确）：更高的 SR** → 必须转向攻 OSR。
- **目标B：AAAI-27 可辩护贡献** → 总方案的"校准感知可靠终止"叙事仍成立，但风险在上升。

**已裁决（2026-07-02）**：用户明确"先做性能，暂不考虑论文"→ 走目标A。旧总方案（可靠终止决策）标记 superseded，新建 [[总方案-性能优先-执行版-20260702]]，以攻 OSR / max-SR 为中心。V2/U2/E3 终止机制冻结（已到极限），E5 是唯一补丁。

---

## 四、更优的方案：把"弃权+再观测"从终止扩展到全导航

**核心洞察**：我们已经有的 E3"弃权+主动再观测"原语（当终止信念落入不确定带时，不硬拒绝，而是调累积证据/换视角再判），在理论上**正是 Spatial-VLN 用来解决导航歧义的"冲突驱动探索"同一个原语**——但我们只把它用在了**终止决策**这一个点上。

**方案**：把这个原语从"终止决策"泛化到"**导航决策**"，用同一套 VoI/序贯最优停止框架同时攻 OSR。

具体三个实例（对应总方案第4节的 COMMIT/CONTINUE/REOBSERVE 框架）：

**4.1 导航级 REOBSERVE —— 攻"迷路"集【外部已验证为最大杠杆】→ 详细设计见 [[p1_design_20260702]]（2026-07-02）**
- **落点已勘定**：`move_to_next_vp`(spatialNavigator.py:567) 每步只推理1次，`thought_fusion`(:612)/`test_decisions`(:637) 多路径基础设施**已建好但休眠**；P1=触发时多采样K次唤醒它。
- **触发方案升级（2026-07-02，见 [[p1_design_20260702]] §9 Path B）**：核心创新是"**校准回答何时再观测**"，触发信号从启发式 A∪B **升级为校准后的 vote-dispersion**（诱导信念，self-consistency/semantic-entropy 血统）。A∪B 降为降级对照臂。**go/no-go 闸门**：离线校准（记录真实prompt+共享采样器，零重建、活过重构）验证 dispersion→贪心走错率是否单调；不单调则触发方案推倒重来，省一整轮 online。**决胜消融**=预算配平 risk-coverage 四臂（校准dispersion / confusion-prompt / A∪B启发式 / 匹配预算随机），赢下它创新点才成立。脚本 `scratchpad/p1_dispersion_calib.py` 就绪，代码 diff 见 §9，待 Run2 完释放 GPU 落地。`api.py:83 temperature=0` 是前提（须参数化）。
- 当 selector 对下一航点的信念落入不确定带（进度停滞、或 phase=recover）时，触发主动再观测/换视角，而非直接选最高分候选
- 复用现有 E3 再观测基础设施，只是触发点从"stop gate"移到"selector"
- **外部验证**：Fast-SmartWay 的 Uncertainty-Aware Reasoning（=此机制）是其最大消融杠杆 **+8pp SR**；DV-VLN 的 generate-then-verify、Spatial-VLN 的 conflict-driven exploration 均同向。这是全场公认的 OSR/SR 最大杠杆。
- **我们的差异化（须守住）**：触发信号用**校准驱动的不确定带/VoI**，而非 Fast-SmartWay 的"问 MLLM 你困惑吗"（未校准内省）、也非 Spatial-VLN 的多专家投票、DV-VLN 的训练式验证。training-free + calibration 是唯一护城河。

**4.2 Backtracking 回溯 —— 更丰富的 CONTINUE，攻死胡同**
- 动作空间加 "move back to last good viewpoint"（SmartWay 核心，真机 +12pp SR）
- 走错路/负增益连续 k 步时可退回，直接对应总方案框架里"CONTINUE 的额外代价 T5"
- navigator prompt + 动作空间改动，**不需重训**

**4.3 arrival-gate STOP 封锁（E5）—— 攻假停38集**
- episode 内 arrival_gate 从未触发（从未 < 4m）则硬拒 STOP
- 防止在 8m 处误停浪费 episode
- 低成本防御补丁

**为什么这个方案更优（同时服务目标A和B）：**
- ✅ **攻 OSR**（真正的 SR 杠杆）：4.1+4.2 直击 36 集迷路 + 死胡同
- ✅ **保住理论内核**：仍是 VoI / 选择性预测 / 序贯最优停止，总方案第4节框架原封不动，只是把决策点从"终止"扩到"导航"——叙事从"可靠终止"升级为"**校准驱动的可弃权序贯导航决策**"
- ✅ **复用现有基础设施**：E3 再观测代码已在，四周窗口可行
- ✅ **差异化**：区别于 Spatial-VLN（多专家+启发式）和 SmartWay（重训航点），我们是 training-free + 校准驱动

**航点瓶颈（P3）—— 诊断后大幅降级，见 [[architecture_optimization_20260701]] §5b：**
- ⚠️ **重要修正**：前沿论文点名 WaypointBert 是 OSR 瓶颈，但那基于**真机/低层控制撞墙**场景。我们自测（4B trace）：**航点 ~98% 精确到达（位移/预测=0.971），"被挡"仅 2-4%**——我们是 teleport setup，可达性没问题。
- ~~路线A 重训 occupancy 预测器~~ / ~~占用后过滤~~ → **关闭**（改善的是可达性/%Open，我们已 98%，无靶点）。
- ~~路线B/C 绕过/消除~~ → 暂缓（同理，收益靶点不在可达性）。
- ~~残余待测：候选覆盖~~ → **已证伪（2026-07-02，见 §7.1）**：用真实目标坐标在72迷路集测得前向候选 ~84% 都存在、真死胡同仅11–16%，失败集是随机游走（48%走远）。覆盖不是瓶颈，选择层是。SmartWay 强化版预测器（DINOv2+occupancy，同 `BinaryDistPredictor_TRM` 类，非 drop-in 需换 RGB 前端）SR 上限低。
- **净结论**：ResNet/占用/覆盖整条 waypoint 路线**正式关闭**（有直接反证，非仅无靶点），火力集中选择层（P1）。

**4.6 新增杠杆（2026-07-02 读 MSNav + Progress-Think）**

两篇都直击"选择/推理层"，给出 training-free 可迁移想法：

- **P0.5 目标房间空间布局注入（MSNav，新杠杆，training-free，直攻假停）**
  - MSNav 的 Spatial Module：从指令**推断目标房间的物体与布局**（"目的地是厨房，布局是几把椅子在中间+壁炉"），注入决策。**可插拔、纯 prompt**，在 NavGPT/MapGPT 上 +2.3~2.8 SR。
  - 直击我们**38 集假停**的核心："stop in hallway instead of kitchen"（MSNav/NavBench 都点名）——agent 误把走廊当目标。给它"目标长什么样"能改善 endpoint 识别 → 减少假停。
  - 实现：completion/stop 决策前，先让 LLM 从指令推断目标房间布局，注入 V2/completion 上下文。与 P0（航点距离注入）正交，可叠加。

- **进度单调约束（Progress-Think，稳定 completion estimator）**
  - Progress-Think 核心发现：**语义进度**（哪段子指令完成）>> 数值进度（SR 43.8 vs 33.4）——我们的 completion estimator 已是语义的 ✓。
  - 但它每步重估、会震荡/回退（NavBench 说我们进度估计接近随机）。Progress-Think 的 **monotonic co-progression**：进度只增不减。**training-free 迁移**：让"已执行动作"单调累积（carry forward 最大进度），不许回退。稳定"agent 不知道走到哪"。
  - 注：Progress-Think 主增益来自 RL 训练（不可用），只借单调约束这个便宜稳定器。

**优先级更新**：P0.5（目标布局注入）性价比高且直攻最大失败类（假停），建议提到 P1 之前或并行。

---

## 五、预计下一步

### 立即（当前 ep100 9B 跑完后）
1. 分析 9B ep100 完整结果，写日报，确认 9B 零调优基线
2. **决策**：主线回 4B（配置成熟）实施新方案；9B 作为 E4 跨 backbone 数据点保留

### 短期（下一轮实验，按投入产出）
1. **E5 arrival-gate STOP 封锁**（最低成本，先做）
   - `base_il_trainer_llm.py`：追踪 `_arrival_gate_ever_triggered`
   - `run_OpenNav.yaml`：`ARRIVAL_GATE.NO_APPROACH_STOP_BLOCK: True`
2. **Backtracking 回溯**（4.2，中成本高价值）
   - navigator 动作空间加 "move back"，负增益 k 步触发
3. **导航级 REOBSERVE**（4.1，复用 E3 infra）
4. 每轮结果记录日报 + 更新本文档

### 中期
- E-final：occupancy-aware 航点预测器（需重训，最高 OSR 天花板）
- 读 DV-VLN / AgenticNav / MSNav，确认撞车边界

### 待用户决策的岔路口
- **主目标是 max-SR 还是 AAAI-27 论文？** 决定是否放弃"纯终止"叙事、全面转 OSR。本方案（第四节）试图两者兼顾，但若时间紧张需二选一。

---

## 六、关键配置快照（当前生效）

| 参数 | 当前值 | 说明 |
|---|---|---|
| 模型 | Qwen3.5-9B（ep100 测试中） | 正常主线用 4B |
| SHORT/LONG_ACTION_STEP_LIMIT | 10 / 12 | e3_carry_forward |
| E3_ARRIVAL_OVERRIDE_DIST | 3.0m | U2 E3-B override |
| E3_ABSTAIN_DIST | 3.0m | abstain zone |
| PSG COMMIT_DIST | 2.0m | PSG 提交阈值 |
| TRACE_DIR | `e3_carry_forward` | 当前 trace 目录 |

---

## 七、失败分布（4B e3_carry_forward, 100集）—— OSR 瓶颈证据

| 类型 | 数量 | 说明 | 攻击方案 |
|---|---|---|---|
| SR=1 ✓ | 24 | 成功 | — |
| OSR=0 假停 | **38** | 从未接近目标（avg 8m）就停 | E5 (4.3) |
| OSR=0 步数耗尽 | **36** | 迷路耗尽步数 | REOBSERVE (4.1) + Backtrack (4.2) |
| OSR=1 SR=0 步数耗尽 | 2 | 接近过但步数不够 (ep1084/1106) | 步数上限 |

OSR 天花板=26，SR 天花板=26。**攻 OSR 才能抬高 SR 天花板。**

### 7.1 覆盖率 vs 选择 诊断（2026-07-02，决定性证据，关闭 waypoint 路线）

用 Run1 trace（P0-off 9B，完整100集）+ 数据集真实目标坐标（`OpenNav_R2R-CE_100_bertidx.json.gz` 的 `goals[0].position`），在 **72 迷路集（=100−OSR28）的 640 决策点**上分离"生成层 vs 选择层"：

**测地 ground-truth（用 trace 现成的 `post_action_progress.distance_gain_selected`，无代理偏差）：**
- 所选候选 **48% 走远目标**（gain<−0.1m）、48% 走近、4% 不动；每步净增益 **均值 +0.07m / 中位 +0.03m ≈ 随机游走**。

**欧氏覆盖（候选 world_point vs 目标，交叉印证）：**
- 84% 的步**存在**能靠近目标的候选（生成层给了好选项）；真死胡同（零前向选项）仅 **11–16%**；但 selector 只在 **31%** 的步选了靠近候选。
- 阈值稳健性：selection≫coverage 在 PROG≤0.5m 成立；PROG≥1.5m 时 coverage 反升（65%），因水平欧氏无法捕捉"绕行/穿门"——故以**测地 ground-truth 为准**。

**裁决**：候选生成/覆盖**不是**瓶颈（前向选项 ~84% 都在），**选择/路由是瓶颈**（失败集随机游走、48% 主动走远）。→ **换 waypoint 预测器（含 SmartWay 强化版 DINOv2+occupancy）SR 上限很低，路线正式关闭**（第四节 §4.5 P3 从"无靶点暂关"升级为"有直接反证"）。火力压 P1（选择层再观测）。诊断脚本见 scratchpad。

### 7.2 Run1（9B+调参+E5，P0off）ep100 完成（2026-07-02）

SR=**24%** / OSR=**28%** / SPL=**0.200** / nDTW=0.438 / 转化 85.7%。**调参(NAV_TOK2048/COMP768)+E5 把 9B 从 SR20/OSR23/SPL0.149 救回**，OSR 与 SPL 反超 4B（26/0.187）。终止分解：stop_requested 41（真停21+假停20），step_limit 59（够到<3m 但没停3 + 从未靠近56）。**假停 52→20 砍半**（E5 生效，全程仅2次 rescue）。主导失败已从假停转为 **56 集从未靠近**（=OSR 瓶颈=P1 靶心）。

### 7.3 Run2（P0on）前34集初步（2026-07-02，进行中）

同34集对比：Run1 P0off SR20/OSR23 vs Run2 P0on SR14/OSR17，**P0 净负 −2**（翻正1、翻负3；3集翻负全是终点飘远6.7–7.8m=路由被带偏）。假设：navigator 把 `[Waypoint distance]`（步长距离）误读为到目标距离。**黄灯偏红，待满100集定论**；若确认净负，P0.1=修正注入语义再跑。与 §7.1 一致——选择层脆，塞信息反乱，治本靠 P1。

---

## 八、历史决策记录

| 日期 | 决策 | 结果 |
|---|---|---|
| 2026-06-28 | M2a trajectory_bypass + M3 PSG | SR 20%→21% |
| 2026-06-29 | max_config ep100 | SR 21%，OSR→SR 80.8% |
| 2026-06-30 | E3 abstain+arrival_override | SR 22%，OSR→SR 84.6% |
| 2026-07-01 | e3_carry_forward | SR **24%**，OSR→SR **92.3%** |
| 2026-07-01 | 切 9B 模型测试（100集完成） | **SR 20% / OSR 23%**，全面低于 4B；假停80%、空预测fallback22.4%；**9B OSR<4B 证明瓶颈是架构非模型** |
| 2026-07-01 | **文献对比→战略转向：终止转化已封顶，转攻 OSR** | 方案见第四节 |
| 2026-07-01 | **架构对标+P0设计** | 见 [[architecture_optimization_20260701]]；P0=真实几何注入 |
| 2026-07-01 | P0 实现 + 9B 调参（NAV_TOK 2048/COMP 768） | 见 model_switch_guide 六b |
| 2026-07-01 | P0 ep1 冒烟(9B) | 格式✓ token截断修复✓(fallback 0%)；但 ep244 SR1→0 回归 |
| 2026-07-01 | 定位回归根因 | **非P0**：navigator停止推理0次引用几何；真凶=U2 rescue override 无距离守卫，6.2m/step3 假停（也是38集假停机制之一）|
| 2026-07-01 | 实现 E5 rescue 距离守卫 + P0 开关 | `RESCUE_MAX_GOAL_DIST=4.0`（守卫2661/2681分支）；`GEOMETRY_INJECTION` 开关做干净消融 |

---

## 九、参考文献（本轮新增，强相关）

- **SmartWay** (arXiv 2503.10069)：Open-Nav 直系后继，occupancy航点+回溯，OSR 51/SR 29
- **Spatial-VLN** (arXiv 2601.12766)：直接对照 Open-Nav，显式空间感知，SR 33，论点"缺空间感知非终止"
- **VLN-Zero** (arXiv 2509.18592)：两阶段探索建图+缓存，SR 42.4
- **CA-Nav** (arXiv 2412.10137)：约束感知，SR 25.3
- **DV-VLN** (2601.18492)：已读，训练式动作双重验证，不撞车，支持"验证扩到导航"转向
- **AgenticNav** (2606.10577)：已读，工具调用harness，VLM直接选像素绕过航点预测器
- **Fast-SmartWay** (2511.00933)：已读，SmartWay直系后继，端到端消除航点+Uncertainty-Aware Reasoning(+8pp)，部分撞车但校准触发差异化，SR 27.75/SPL 24.95
- **MSNav** (2508.16654)：已读，动态记忆图+**目标空间布局推断**(可插拔+2.3~2.8SR)+Qwen-Sp。离散R2R/REVERIE。→ P0.5 来源
- **Progress-Think** (2511.17097)：已读，语义进度推理(语义>数值 SR43.8vs33.4)+单调co-progression。训练式NVILA-2B，R2R-CE SR60.1。→ 单调约束想法来源
- **AwareVLN** (2605.22816)：已读，训练式 SR73.5。稀疏推理3节点(子任务/偏离/停止)。**消融证实转向：子任务−13.1>偏离−10.3>停止−5.4(最小)**；稀疏>密集(65.4vs63.8)验证P1触发式设计。可借三元结构"场景→进度→计划"。
- **FSR-VLN** (2509.13733)：已读，离线HMSG+Fast-Slow(快筛→慢验证仅难例触发,省82%)。任务差异大不迁移，但佐证"触发式昂贵推理"。
- **文献survey 收敛（2026-07-02）**：强相关8篇已覆盖(SmartWay/Fast-SmartWay/Spatial-VLN/VLN-Zero/DV-VLN/AgenticNav/MSNav/Progress-Think + AwareVLN/FSR-VLN)。共识=选择/推理层是瓶颈，稀疏触发式再观测是杠杆；我们差异化=零样本+校准触发。新 training-free 杠杆已尽(P0/P0.5/P1/P2)。剩余(Mem2Ego/SE-VLN/CMMR-VLN/ActiveVLN/CLOSER-VLN/LightZeroNav)边际低，需要时再查。

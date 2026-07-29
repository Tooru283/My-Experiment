---
name: project_oracle_leak_20260704
description: 火警定论20260704 — 整个E3/E5/PSG终止侧keyed于单一oracle变量latest_goal_dist(=测地目标距离GT)；污染面=E3锚点全体不是单条E5支路；干净重跑前SR=24不能报training-free
metadata: 
  node_type: memory
  type: project
  originSessionId: d7e660c1-ead5-4844-8398-9823f61067f3
---

例行 wiring 前血统检查抓到的 oracle 泄漏（发生于 20260704，来源 SR=24 headline run `ep100_series_m420260702_230800` val_unseen trace）。血统：`recent_distance_gains`/`latest_goal_dist` 源自 `oracle_metrics.selected_distance_gain = distances[-2]−distances[-1]`，`distances = info["position"]["distance"]` = 模拟器到目标测地距离（SR/NE 同源 GT）。**参与决策，非仅日志。**

**Path A — 增益 oracle（`stop_evidence_verifier._rescue_blockers`，branch 于 non_positive_gain_count / latest_gain>0；headline `RESCUE_REQUIRE_POSITIVE_RECENT_GAIN: true`）**：出现 594/1232 rescue 事件、全 100 集，但**决定性=0**。`allow_rescue` 需 blocker 列表为空；凡有增益 blocker 必同时有非 oracle blocker，单删增益翻转 0 决策。挂着不咬人，仍须摘干净。同一增益也散见 failure_diagnostic:88-95(progress_drift→U3) 与 U3 recovery，当前均不决定性。

**Path B — 绝对目标距离 oracle（`base_il_trainer_llm.py:2688-2715` `_rescue_dist_blocked = latest_goal_dist > RESCUE_MAX_GOAL_DIST(=4.0)`）**：直白明文比较，读绝对测地目标距离阻止接受 STOP→强制 movement fallback。**决定性，精确命中 168 次/39 集**（导航日志 `E5: rescue/override STOP blocked`=168，trace `stop_rescue_block`/`movement_fallback`=168 对上）。这是 E5"假停 52→20"远距离抑制器的真牙齿。

**判决**：SR=24 / 转化 / 假停52→20 被 Path B 污染；干净重跑前不能声称 training-free 零样本。确切 SR delta 只能靠干净重跑（RESCUE_MAX_GOAL_DIST=0 + 增益门全关），离线上界=168决策/39集。心理准备：干净 SR 大概率 <24，掉幅=E5 里 oracle 真实贡献（该掉的水分，现在掉比审稿时掉便宜百倍）。

**Step0 audit 定论（20260704 完成）：污染面=整个 E3 锚点，不止 E5 一条支路。** 根因=单一变量 `latest_goal_dist`（`base_il_trainer_llm.py:3194 = step_output_summary[0]["distance_to_goal"] = oracle_metrics.latest_distance_to_goal = info["position"]["distance"][-1]` = 测地目标距离 GT）喂给**七个**消费者，均在推理路径 branch：
1. E5 far-block `_rescue_dist_blocked`(>4.0) — 决定性 168 occ/39 eps（Path B）。
2. E3 arrival_override(`<3.0` ∧ arrival_ev/carry_fwd) — 决定性，剥 final_target_not_visible+relation_contradiction 让 STOP 过 — TRUE 14 occ/7 eps。
3. E3 abstain(`<3.0` ∧ 无 arrival_ev) — 剥 final_target_not_visible 降级为 abstain 续行 — TRUE 24 occ/7 eps。
4. E3 trajectory_bypass(`<3.5`) — 绕过轨迹支持要求 — TRUE 34 occ/13 eps。
5. PSG proactive_stop(`<proactive_stop_dist`)+commit(`<2.0`) — 86 occ/22 eps。
6. Path A 增益门(recent_distance_gains) — 594 present / **0 决定性**（不咬人，仍摘）。
7. ArrivalGate(`<=4.0`) — **仅日志**（trainer:1644-1662 只 write record，不 branch），无害。
→ E3 carry_forward（[[project_e3_20260630]] 的 SR=24 验证锚点、转化 92.3%）本身就在读 GT 判到达。近区(<3m)全程仅 92 records/28 eps，故 E3 near-gate 真实触发是该区可观机会的显著比例。
**Step1 修复窗口**：所有 gate 的 `latest_goal_dist<X` 合取项换可观测（同 verdict 里已有的 `arrival_evidence`/`carry_forward_visible` 视觉信号；E5"没接近过就别停"→"从未视觉确认过目标"）；分级距离带(3.0/2.0/3.5/4.0)无法用二值"见过没"复现，需设计——PSG commit(<2.0)最紧最难，或用 P0 waypoint 距离传感器(可观测)。Path A 全关；backtracking hunks1-3 同窗口进（默认关不改行为）。
**Step1 已实施（20260704，待干净重跑验证）**：#1-#5 删 latest_goal_dist 决策合取项完成——E5 far-block 换 `not any(recent_ftv_window)`(从未视觉确认)；E3 arrival_override/abstain 换纯视觉(e3_intrinsic_absent + arrival_evidence/carry_forward_visible)；trajectory_bypass 换 arrival_evidence/carry_forward；PSG entry 去距离门、commit 换 `_persistent_visual_confirm`(连续≥2步 final_target_visible)。#6 config 关(RESCUE_REQUIRE_POSITIVE_RECENT_GAIN:false, RESCUE_MAX_NON_POSITIVE_GAINS:99)。**统一规则(定稿)**：oracle 距离合取项 → 持续(≥2 连续步 final_target_visible)视觉证据，三个 STOP-enabling 门(#2 arrival_override/#4 trajectory_bypass/#5 PSG commit)一体适用；#2 是最强门(剥 blocker 放 STOP)故不吃单帧 arrival_evidence 或 any-of-3 carry_forward(远处 16.9%假阳放大器)。信号在 record_stop_evidence_verification 单一 choke point(1868)算一次 persistent_visual_confirm 传入 verifier。#3 abstain(不置 allow_stop)与 _rescue_blockers(用可观测 arrival_evidence 加限制)不动。#7 ArrivalGate 不动(仅日志)。U3 gain 路径核实=仅日志(触发只 selector_empty/stop_rejected,progress_drift 分支不可达)。latest_goal_dist 仅存 verdict 日志,不参与决策。backtracking hunks1-3 同窗接线(offer_move_back=False 默认,byte-identical)。5 文件 ast 通过、self-test 绿、无悬空引用。
**认知重写**：memory 里"V2/U2/E3 冻结/已到极限"是错的——它们没到极限，是被 GT 抬到了极限；干净基线掉多少=oracle 在终止侧的真实贡献,是 #2 章一个数字("无 GT 兜底时小 VLM 终止差多少")。E3 oracle 触发集中 <3m(92 records 占 38)恰是 E1 说 VLM 检测最烂(43%)的区间——GT 一直替模型兜它最失准那段。
**Smoke(3集:244/377/550,20260705)通过全 checklist**：无崩溃、MOVE_BACK offered=0、新门全触发(E5"never visually confirmed"块、E3 abstain、PSG、STOP拒绝)、schema完整、latest_goal_dist仅日志。3集全fail但是设计上的最坏样本(2近目标门集+1迷路)非SR估计。**关键机制发现**：ep377(旧oracle成功)现经 PSG 在 5.42m 假停——9B completion-verifier 置信 1.00 谎报 ftv+ae 且≥2连续步,持续性守卫拦不住"自信且持续错"的VLM。这是诚实的水分,也正当化 v2 waypoint 几何否决(几何能杀掉视觉杀不掉的5.4m"到达")。已提交 e0b55d7(主体)+6455cb2(post-smoke:TRACE_DIR+PSG reason串诚实化),打 annotated tag **clean_baseline_v1**;metadata 落 logs/harness_traces/clean_baseline_v1/RUN_METADATA.txt。serve_qwen.sh 启停 9B(transformers serve greedy)。**全量100集 clean_baseline_v1 已挂**(20260705_103023,PID待重跑,dataset=100全集,whitelist未设),等 SR。新增 env 门控 OPENNAV_EPISODE_IDS 白名单(未设=零行为差异)。
**Step2**：干净重跑=新 baseline（RESCUE_MAX_GOAL_DIST=0 + 所有 latest_goal_dist gate 换掉/关），confusion replay 让路搁置（[[project_p1_design_20260702]]），闸门线冻结至干净基线立起。804 校准集采于污染轨迹但三探针全负、对轨迹分布不敏感，结论保留，metadata 记"采于修复前配置"，不重采。

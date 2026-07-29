---
name: project_clean_baseline_decision_tree
description: 预注册SR落点决策树(20260705,数字出来前锁定,防现场发明反应)+ v2改用深度传感器否决(非航点绑定) + backtracking hunk4 rev2待接
metadata:
  node_type: memory
  type: project
  originSessionId: d7e660c1-ead5-4844-8398-9823f61067f3
---

clean_baseline_v1 全量100集跑于 20260705_103023，等 [[project_oracle_leak_20260704]] 干净 SR。**数字出来前预注册反应，防止落点后现场发明叙事。**

## 干净数落地 20260705_1912（进程 19:12:41 退出，最终 aggregate）
**SR=0.16 / OSR=0.21 / 转化=76.2% / SPL=0.128 / nDTW=0.420 / NE=7.61m**。落**15–19 中段** → 执行预案：v2 深度 veto 升关键路径，对标表改 "honest observable-only baseline"，掉幅入 #2 分析章。
- **24→16 的 8 点全是 oracle**：E3/E5/PSG 终止侧增益拿掉 `latest_goal_dist`(测地GT) 后整段蒸发，SR 打回 A0 水平(=16)。7/4「SR=24带毒待干净重跑」结论=毒占8点。
- **但转化是干净涨的：A0 64% → 16/21=76.2%**(+12点，不靠GT)。SR 没跟涨因 **OSR 反降 25→21**：瓶颈从"停不准"移到"到不了目标邻域"，印证 [[project_arch_pivot_20260701]]「SR天花板=OSR」。
- depth veto 从可选增量→**关键路径支点**(治可观测错停 ep377 类 + 零oracle)。下一步：单开 smoke，先 depth(`bash scripts/smoke/smoke_depth_veto.sh`)。

**SR 落点决策树（三段，写死）**：
- **≥20**：水分可控，故事照旧——干净基线 + v2 depth veto / backtracking 做增量，掉幅（24→X）进 #2 分析章交代。
- **15–19**：v2 深度 veto 升为**关键路径**（假停回归是主因，深度直接治它），backtracking 排第二；对标表姿态改写为 "honest observable-only baseline"，掉幅本身作为核心分析发现浓墨写。
- **<15**：终止侧没有 GT 就基本不成立——这是比"涨3点"更重的论文命题：**"零样本小 VLM 的终止能力被 oracle 评测惯例系统性高估"**。叙事重心整体转 #2 分析章 + 诚实机制修复。这个结局论文照样成立，不慌。

**纪律**：waiter 报最终 aggregate 前，不看分集中间数、不做任何反应（Run2 34集教训已交学费）。

**v2 veto 机制定稿 = 深度传感器，不是航点绑定**（用户 20260705 拍板）。理由：航点绑定绕圈（目标绑哪个候选？绑错？依赖航点生成器覆盖）；RGB-D 深度图在目标方向的读数 = 可观测目标距离，所有 VLN-CE 方法合法用深度传感器，零 oracle 嫌疑。
- **规则**：`v2_stop = persistent_visual_confirm ∧ (depth_at_claimed_target ≤ 阈值)`。
- ep377 的 conf-1.00@5.42m 被深度直接击毙（depth 读 5.4m > 3m → veto），不依赖航点覆盖。
- **实现约束**：additive、默认关（GEOMETRY/DEPTH_VETO switch off），baseline 一落地就挂对照跑。深度是治 ep377 那类"自信且持续错的 VLM 到达谎报"的唯一可观测手段（持续性守卫拦不住一致错误）。

**backtracking hunk4 rev2 三处修正**（干净基线后第一个增量实验大概率 = hunk4 + v2 veto，代码现在就位）：
1. STOP 回退改**最佳真实候选**（不是 MOVE_BACK 占位）。
2. backtrack 后 `last_executed_move = None`。
3. 显式 flag 流控 + trace 形态定义。

**已实施 + 提交 20260705**（commit `cc77a8e`，tag `increments_v2_staged_20260705`，父=`clean_baseline_v1`/6455cb2=正在跑的基线代码）。两开关默认 OFF → 基线 byte-identical。py_compile 绿、backtrack self-test 绿、yaml 解析两开关=False。
- **DEPTH_STOP_VETO**（`VISUAL_TARGET_VERIFIER.DEPTH_STOP_VETO.ENABLED`）：`_depth_stop_ok`/`_metric_depth_center_median` 读**原始** sim depth（不是 generate_input 第220行重归一化的 JPEG——那个毁了度量），前视中心区中值，米=v×(MAX−MIN)+MIN（habitat 默认 0/10, NORMALIZE True）。接到 verify() 的 `depth_confirm` 合取 #2/#4 + PSG commit(#5) `_psg_depth_ok`。unavailable→fail-open。每次调用 log `depth_stop_veto`(depth_reading_m/view_id/vetoed/fail_open)——**离线评估命中/误杀的唯一通道**。
- **BACKTRACK**（`OPENNAV_HARNESS.BACKTRACK.ENABLED`）：offer 于 ego stall 或 dead_end(`len(radius_dict)==0`，从<=1收紧)；apply 反向180°。fix#1/#3 失败 backtrack→ranked best real candidate(None守卫)；fix#2 last_executed_move 在**唯一 grep 确认的 action-4 发射点**(env_actions[0])捕获，path-agnostic；fix#2b backtrack 后置 None。
- **关键接线修复（否则机制静默死亡）**：`test_decisions` 原会 pop 掉非 observe_dict 的候选→MOVE_BACK 永远到不了 next_vp。已加 `offer_move_back` 参数在两处 test_decisions 调用点放行 MOVE_BACK。thought_fusion 按 predictions 建 key，MOVE_BACK 天然穿过（无需改）。
- **A/B smoke 必须核对（我无法离线验证的两跳）**：① MOVE_BACK 真的走到 next_vp 并触发 backtrack_apply（"新门真的会触发"）；② depth_stop_veto 事件 depth_reading_m **非 null**（若全 null=helper 没找到 depth 键，机制 inert，需修 key 解析）。**分开跑，别两开关齐上**（归因）。

**smoke 工具已就位 20260705**（commit `3a33779`，tag `increments_v2_staged_20260705` 已前移至此；`clean_baseline_v1` 仍 6455cb2 不动）：
- **先修的雷**：`depth_stop_veto` 原走 `log_u_event`，而它 `if not u_series_active: return` —— 单开 DEPTH_STOP_VETO(U_SERIES off) 的 A/B 会 fire 机制但**零事件**，和"helper 找不到 depth 键"无法区分，毁掉唯一离线通道。已改走 `log_fallback_event`(无条件)。
- `scripts/smoke/{smoke_depth_veto,smoke_backtrack}.sh` + `verify_smoke.py`：各只开一个开关；PID 6378 存活即**拒跑**(单GPU);经 `OPENNAV_EPISODE_IDS` 白名单定向(深度=ep377 保证击毙;backtrack=游走集 1084/1106);经 `OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.DEPTH_STOP_VETO.ENABLED`/`OPENNAV_HARNESS.BACKTRACK.ENABLED` yacs CLI 覆写。
- verifier 判决：depth → PASS(有事件且 reading 非 null)/FAIL(全 null=key 解析坏)/INCONCLUSIVE(没触发 stop→加集);backtrack → PASS(offered→applied)/FAIL(offered 但没 applied=MOVE_BACK 没到发射点)/INCONCLUSIVE(没 offered→换游走集)。事件落 harness_traces jsonl `event_type`+`payload`;verifier 只数事件不碰 SR。已对 live baseline traces 干跑：解析干净、正确报 INCONCLUSIVE(开关 off→0事件)、守卫正确拒跑。

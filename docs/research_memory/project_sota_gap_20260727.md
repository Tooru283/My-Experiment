---
name: project-sota-gap-20260727
description: "20260727文献对照:本项目SR16=Open-Nav忠实复现,但zero-shot R2R-CE已走到53.3;差距是架构代际(持久空间记忆+分层规划),非实现质量;M1的null适用范围因此被高估"
metadata: 
  node_type: memory
  type: project
  originSessionId: d0625f31-50ec-45e1-94ee-c02015204fdf
  modified: 2026-07-27T09:01:24.859Z
---

**20260727d 查文献后的结论：差距是架构代际，不是实现质量。**
全文 `Controlled-Navigation-Harness/docs/架构对照-SOTA差距与改造路径-20260727.md`。

**定位**：本项目 SPL 12.8 vs Open-Nav(ICRA2025) 论文 12.9 → **忠实复现**。
而 zero-shot R2R-CE val-unseen 榜：SpaceVLN **53.3** / GTA 48.8 / HSGM 47.9 /
VLN-Zero 42.4 / Three-Step Nav 34.0 / SmartWay 29.0 / **Open-Nav ~16（本项目在此）**。
赛道 18 个月涨了 3 倍，而本项目一直在给 16 那版做增量。

**所有 40%+ 方法的共同点 = 持久空间记忆 + 分层规划**，本 pipeline 两者皆无
（无记忆逐步航点选择，预算 10–12 跳，实测 6.56 步/集）。
SpaceVLN = Spatial Waypoint 图(位姿/楼层/区域/地标+可达边+已走链边) + 局部地标记忆 +
planner(每~30原语1次)/executor(每步) 分离 + 锚点链取最远已验证锚点 + 三重合取终止；
动作是原语 {Stop,Forward,TurnL,TurnR}、**171.7 步/集**；模型 = Qwen3.5-Plus + Qwen3.5-Flash
（**两者本账号百炼可用**）。

**消融**：w/o空间记忆 **−14.4** / w/o地标记忆 −10.4 / w/o planner-CoT −8.5 / w/o exec-CoT −6.2。
⚠ **剥掉全部记忆与CoT仍有 37.3** → **骨架(分层+原语动作+锚点链)本身价值 ≥ 记忆价值**。

**对既有结论的两条修正（重要）**：
1. [[project_m1_20260727]] 的 null **只界定"无记忆单步选择"上界(43%)**，
   未界定"有空间记忆时"的上界——SpaceVLN planner 是在记忆图上选未验证锚点，同家族模型。
2. **"选择层近随机"需重新归因**：无记忆策略在局部选择本就接近随机，是架构没给状态、
   非模型缺陷。本项目自己的"失败集随机游走48%走远"正是该症状，当时被读作选择层瓶颈
   （见 [[project_selector_bottleneck_20260719]]、[[project_arch_pivot_20260701]]）。

**已有死代码正对应文献关键组件**：`visual_graph_memory.py` 的
`VisualGraphMemoryDiagnostic`(109行，revisit_score/candidate_novelty/loop_flag)
被 `MEMORY_DIAGNOSTIC.LOG_ONLY=True` 锁在日志侧；`landmark_matching.py`(135行)有零件、
无跨步地标池。⚠ **不得假设翻开关=+14.4**：现有模块只有位置历史+距离，无区域/楼层/可达边。

**改造路径**：阶段0(几小时)=禁停跑一轮测真实OSR上界 + 加大 `SHORT/LONG_ACTION_STEP_LIMIT`(现10/12)，
判据 OSR≈23→须动骨架 / OSR≈40+→先修停止；阶段1=`MEMORY_DIAGNOSTIC.LOG_ONLY→False`
（⚠ Arm C 已证"步内重排"不转化，但 visit-info 是"跨步避免重访"，机制不同）；
阶段2(几周重写)=planner/executor分离+锚点链(替掉已证伪的completion_estimation)+
持久航点图+跨步地标池+三重合取终止(直打假停56)；阶段3=plus/flash 分工。

⚠ **限定**：复现SOTA打折；榜单数字未逐篇核对口径(航点预测器/步数预算/集划分)，引用前须复核。

参见 [[project_m1_20260727]] [[reference_full_chain_doc]] [[feedback_no_autonomous_runs]]。

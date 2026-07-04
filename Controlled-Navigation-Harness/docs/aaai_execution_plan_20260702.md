---
date: 2026-07-02
tags:
  - AAAI
  - execution-plan
  - minimal-publishable-path
status: active
related:
  - "[[current_task]]"
  - "[[总方案-性能优先-执行版-20260702]]"
  - "[[总方案-可靠终止决策-执行版-20260629]]"
---

# AAAI-27 最小可发表路径（执行清单）

把 [[current_task]] 顶部"目标协调"落成可执行清单。截止：摘要 7/20（18天）、全文 7/27（25天）。

## 0. 一句话命题（论文卖点）

**零样本 VLN-CE 中，小 VLM 的核心失败是高代价决策点（路由 + 停止）上基于失准信念的过度提交；一个 training-free、校准驱动的选择性再观测机制在两个决策点一致提升 SR。**

标题候选：*Know When to Look Again: Calibration-Driven Selective Re-observation for Zero-Shot VLN-CE*

## 1. 贡献 → 证据 artifact 映射（写论文前先锁定要产出哪些图表）

| 贡献 | 证据 artifact | 数据来源 | 状态 |
|---|---|---|---|
| C1 失败分解：OSR→SR 缺口是独立瓶颈 | 表：A0/max_config/E3 的 SR/OSR/转化 | 现有 trace | ✅ 有数据 |
| C1 信念失准（路由+停止都失准）| 图：距离分桶 接受率/可见率/真到达率；reliability diagram；ECE | 现有 trace（E1）| ⏳ 待算，**无需新实验** |
| C2 统一机制：校准驱动选择性再观测 | 方法图 + risk-coverage 曲线（硬拒绝 vs 软衰减 vs 弃权再观测）| E3 trace + P1 新跑 | ⏳ P1 待实现 |
| C3 跨 backbone 一致 | 表：4B vs 9B 上机制的转化/SR 增益 | Run1/Run2 + 4B 对照 | ⏳ 部分在跑 |
| C-辅 正交/互补 | 论证：机制作用于决策层，与航点/感知改进(SmartWay/Spatial-VLN)可叠加 | 论证+引用 | 写作 |

**关键**：C1 的分析图**现在就能做**（E1，纯现有 trace），是无风险的论文骨架。C2 的 P1 是分水岭。

## 2. 实验矩阵（需要哪些跑）

| 实验 | 配置 | 目的 | 状态 |
|---|---|---|---|
| E1 校准分析 | 现有 4B/9B trace | C1 动机图 | 可立即做 |
| Run1 | 9B+调参+E5，P0off | 基线 | 🔄 进行中 |
| Run2 | 9B+调参+E5，P0on | P0 净效果 | 队列 |
| P1-abl | 硬拒绝/软衰减/弃权再观测 三档 | C2 核心 risk-coverage | 待 P1 实现 |
| E4 | 4B vs 9B 最佳配置 | C3 跨 backbone | 部分 |

注：P0（几何注入）是补课不写贡献，但作为"决策层信息增强"的对照仍有分析价值（消融行）。

## 3. 周计划（到 7/27）

- **W1（7/2–7/8）**：E1 全套算完（动机图+校准+risk-coverage on 现有 trace）；Run1/Run2 出结果；**P1 实现 + ep1 验证**。→ 决定 P1 是否有效（分水岭）。
- **W2（7/9–7/15）**：P1 三档消融 ep100；E4 跨 backbone；开始写 Method + Analysis。摘要（7/20）用 W1-W2 结果。
- **W3（7/16–7/22）**：补跑 + 主结果表定稿；写 Intro/Related/Exp。摘要 7/20 提交。
- **W4（7/23–7/27）**：全文打磨、图表、附录、rebuttal 预案。7/27 提交。

## 4. 优雅降级（P1 是超集，可回退）

- **P1 有效**（risk-coverage 支配 + 跨 backbone 一致）→ 强论文：统一的双决策选择性提交。
- **P1 无效**（不动或伤 SPL）→ 退回**终止-only**（旧 [[总方案-可靠终止决策-执行版-20260629]]）：仍可投，卖点收窄为"小 VLM 终止信念失准的刻画与校准修复"，靠 C1+终止侧 C2/C3 支撑。中等强度。
- **兜底**：即便机制增益有限，"OSR→SR 分解 + 小 VLM 决策信念校准分析"本身是 method-agnostic 的可写贡献。

## 5. 审稿风险 → 预案

| 风险 | 预案 |
|---|---|
| "SR 24 < SmartWay 29 / Spatial-VLN 33" | 不 claim SOTA；强调 training-free + 决策层正交（可叠加）+ 分析贡献 + 跨 backbone 泛化 |
| "被 Spatial-VLN/AwareVLN/Fast-SmartWay scoop" | 护城河：training-free + **校准驱动**触发（非训练/非LLM自问/非多专家）+ 作用于**双决策** |
| "只是 prompt 工程/调 gate" | 用 VoI/选择性预测理论落点 + risk-coverage/ECE 把"调 gate"抬成"校准问题" |
| "oracle 距离用于 gate 是否作弊" | 已是 harness 既有设定（arrival_gate/PSG/E3 都用）；论文声明并作近似讨论 |
| "增益靠多停换" | false-stop/risk-coverage 兜住；E5 已把远假停压下 |

## 6. 写作大纲（骨架先立）

1. Intro：零样本 VLN-CE 小 VLM 瓶颈=过度提交（非感知/生成），引出选择性提交。
2. Related：区分 SmartWay/Spatial-VLN(生成层)、AwareVLN/DV-VLN(训练)、Fast-SmartWay(LLM自问)、FSR-VLN(建图)——我们=training-free 校准驱动双决策。
3. Analysis(动机)：OSR→SR 分解 + 距离分桶失准 + reliability/ECE。
4. Method：VoI/选择性预测框架 → 路由+停止统一的弃权再观测。
5. Exp：主结果 + risk-coverage 消融 + 跨 backbone + 正交性论证。
6. Discussion：诚实边界（终止非最大杠杆、training-free 近似）。

## 7. 立即可做（不等任何跑）
**E1 校准分析脚本**：在现有 A0/max_config/e3_carry_forward/9B trace 上算：
- 距离分桶（0-3/3-4/4-5/5+m）的 P(STOP allow)、P(final_target_visible)、经验 P(到达)
- reliability diagram + Termination ECE / Brier
- risk-coverage 曲线（扫 STOP 阈值）
- false/missed stop 分解
产出即论文第 3 节动机图。**这是把工作从"调 gate"抬成"科学问题"的最便宜一步，且完全无实验依赖。**

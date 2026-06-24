# Open-Nav / Controlled Navigation Harness 近两周实验报告

汇报时间: 2026-06-24  
统计范围: 2026-06-10 至 2026-06-24  
数据来源:

- `logs/eval_results/**/stats_ckpt_val_unseen.json`
- `logs/eval_results/**/stats_ep_ckpt_val_unseen_r0_w1.json`
- `logs/navigation_records/**/*.jsonl`
- `Controlled-Navigation-Harness/docs/experiment_record_*.md`
- 原文 Open-Nav 对比表: `Method / TL / NE / nDTW / OSR / SR / SPL`
- HiMemVLN 外部强基线: arXiv:2603.14807 与用户提供的对齐表格数据

## 1. 汇报摘要

近两周主要完成了三件事:

1. 完成 Controlled Navigation Harness 的总体方案设计: 明确 A0/A1/A2+ 分层，区分 logging-only 与 decision-effect，定义 Tool / State / Verifier / Trace 的职责边界，避免把诊断能力误写成方法收益。
2. 完成 V 系列多模态视觉证据实验轴设计与初步落地: 从 V0 到 V4 覆盖视觉服务验证、Visual Evidence Logging、Visual TargetVerifier、VisualEvidenceMemory、Multimodal Selector Context 和视觉 fallback/rerank，并通过 schema 修复解决视觉证据未被下游消费的问题。
3. 完成 U 系列受控决策单元设计与归因修复: 围绕 phase-aware selector context、STOP rescue scope、trusted recovery、decision audit、unit override trace 和 phase parser，建立可解释的动作改写与失败恢复链路。

支撑性工作包括: 本地 Qwen3.5 / SigLIP 运行链路、navigation record、日志目录分层、ep20/ep100 评测与 SR/OSR 复盘。这些不是本阶段方法主线，但保证了 U/V 系列可以被稳定验证和归因。

当前最重要的结果:

- ep100 完整评测最高 SR 仍为 19%: 2026-06-13 `v24` 和 2026-06-22 `ep100_series` 均达到 19/100。
- 与原文 Open-Nav-GPT4 的 SR 19% 持平，但当前最佳完整 ep100 仍不是稳定超越: 6/13 V24 的 nDTW/SPL 较高，6/22 的 OSR 较高，但 6/23 最新结果 SR 为 18%。
- 最新 2026-06-23 ep100 为 18/100，比 2026-06-22 净少 1 条成功；不是整体几何质量变差，平均 final distance 从 7.10m 降到 7.02m。
- OSR 长期高于 SR: 2026-06-22 为 29/100 OSR、19/100 SR；2026-06-23 为 27/100 OSR、18/100 SR。说明已有一批 episode 曾到过 3m 内，但最终没有稳定留在成功范围内。
- 小样本最高到过 50% SR，但波动较大，不能直接当作整体收益。6/14 schema 修复后 11-episode 从 1/11 提升到 4/11，证明 trace/schema 问题会显著影响下游决策。

## 2. 实验目标与路线

本阶段目标不是训练新 policy，而是把 Open-Nav 从原来的 LLM-centered pipeline 改造成可控、可诊断、可消融的 navigation harness。

方法路线:

- A0: 原始 Open-Nav baseline，用于固定本地 Qwen / SigLIP 运行环境。
- A1: harnessed / instrumented baseline，只加日志、状态和 trace，理论上不改变动作。
- V 系列: 引入多模态视觉证据，包括 visual evidence logging、visual target verifier、visual memory、multimodal selector context。
- U 系列: 引入 phase-aware context、STOP/recovery 边界、decision audit 和可归因 override。
- 当前优化重点: 把 OSR 转成 SR，减少近目标后继续移动、误停、recover 阶段漂移。

### 2.1 整体框架

当前整体框架可以概括为:

```text
Habitat / VLN-CE environment
  -> candidate waypoint observation
  -> text observation / SpatialBot / RAM
  -> V-series visual evidence tools
  -> U-series phase / STOP / recovery controllers
  -> selector / fallback / action executor
  -> metrics + navigation record + harness trace
```

更具体地说，系统不再把所有判断都交给一个 LLM prompt，而是把导航过程拆成几类可控组件:

| 层级 | 组件 | 职责 | 输出 |
|---|---|---|---|
| 环境层 | Habitat / waypoint candidate generator | 给出当前全景候选方向、RGB/Depth、可执行 waypoint | candidate ids、candidate images、distance/collision 等环境信息 |
| 感知层 | SpatialBot / RAM / text observation | 将候选图像转成原 Open-Nav 使用的文本 observation | scene description、objects、candidate text |
| V 系列视觉证据层 | V1/V2/V3/V4/V-fallback | 直接读取候选图像，抽取目标可见性、到达证据、候选视觉记忆和 selector 视觉上下文 | visual evidence、target verdict、visual memory、selector context、fallback rank |
| U 系列受控决策层 | U0/U1/U2/U3 | 根据 phase、STOP proposal、failure type 和 recent progress 控制 selector 输入、STOP gate 和 recovery | phase evidence、STOP decision、recovery proposal、unit override |
| 决策层 | LLM selector + fallback controller | 从候选 waypoint 或 STOP 中选择下一步动作 | selected candidate 或 STOP |
| 执行层 | action executor | 将 candidate id 转成 Habitat action，执行移动或 STOP | env step result、distance gain、termination reason |
| 记录层 | navigation record / harness trace / eval stats | 保存每一步 observation、证据、决策、override 和最终指标 | JSONL trace、episode metrics、aggregate stats |

框架设计的核心边界:

- LLM selector 仍负责高层候选选择，但不再独占 STOP、验证、恢复和归因。
- V 系列负责“视觉证据是否支持这个候选 / STOP”，不是直接替代导航 policy。
- U 系列负责“何时允许这些证据影响动作”，通过 phase、scope 和 decision-effect 开关保持可控。
- Trace 是一等产物，每个动作改写都必须能回溯到 unit、reason、original action 和 final action。

### 2.2 端到端实验闭环

本阶段的完整实验流程不是“改一个 prompt 后看 SR”，而是按固定闭环推进:

```text
实验假设
  -> 配置一个最小 decision-effect 单元
  -> 模型和数据 preflight
  -> ep20 诊断运行
  -> trace / navigation record / stats 三路落盘
  -> episode-level 失败归因
  -> 必要时修 schema、gate 或 scope
  -> ep100 完整验证
  -> 写入日报和组会报告
  -> 进入下一轮假设
```

具体步骤如下:

| 阶段 | 输入 | 操作 | 输出 | 判断标准 |
|---|---|---|---|---|
| 1. 提出实验假设 | 上一轮 SR/OSR、STOP case、recover case | 明确本轮只解决一个问题，例如 STOP 漏停、visual rescue 误停、fallback 漂移或 phase 误判 | 实验假设与预期影响面 | 不能同时打开多个无法归因的改动 |
| 2. 固定配置边界 | `run_OpenNav.yaml`、`run_OpenNav.bash` | 标注 A0/A1/V/U 版本、模型、episode 数、decision-effect 单元、log-only 开关 | 可复现配置 | 必须知道本轮是否改变动作 |
| 3. 运行前检查 | Qwen 服务、SigLIP、Habitat split、checkpoint、episode count | 执行模型 preflight，确认服务端模型名和配置一致 | 通过或提前失败 | 避免跑完才发现 4B/9B 或路径不一致 |
| 4. 小样本诊断 | ep10/ep20 固定或半固定 episode | 先观察 SR/OSR、STOP 请求、phase 分布、fallback 触发和异常日志 | 小样本 stats 与 trace | 只用于定位问题，不直接宣称整体收益 |
| 5. 单 episode 复盘 | `navigation_records`、`harness_traces`、per-episode stats | 对 OSR=1/SR=0、STOP fail、success flip、distance drift 做逐步回放 | 失败类型表 | 能解释是视觉证据、phase、STOP、fallback 还是执行导致 |
| 6. 修复与消融 | schema、gate、scope、ranker、parser | 修最小代码或配置，并尽量保留 log-only 对照 | 新配置与变更记录 | 修复必须能落到 trace 字段，不能只凭主观判断 |
| 7. ep100 验证 | 稳定后的配置 | 完整跑 100 episode，统计 SR/OSR/SPL/nDTW/final distance | ep100 聚合结果 | 作为主要结论依据 |
| 8. 报告归档 | stats、trace 结论、代码改动 | 写入 `experiment_record_YYYYMMDD.md` 和本组会报告 | 时间线、结果表、失败模式、下一步计划 | 结论区分“已验证结果”和“下一步假设” |

当前实际执行节奏:

```text
ep20 用来发现问题:
  schema 是否稳定
  phase 是否误判
  STOP 是否过宽或过严
  recovery 是否可信

ep100 用来验证趋势:
  SR 是否稳定提升
  OSR-SR gap 是否缩小
  SPL 是否被额外步数拖低
  success/failure flip 是否集中在同一类失败
```

因此，本报告中的小样本结果只作为诊断证据；真正用于组会结论的是 ep100 表格、OSR-SR gap 和逐 episode flip 分析。

实际运行入口在 Open-Nav 根目录:

```text
cd /root/wjj/Open-Nav
EPISODE_COUNT=20 EXP_NAME=ep20_series_qwen_siglip_local_YYYYMMDD_HHMMSS ./run_OpenNav.bash
EPISODE_COUNT=100 EXP_NAME=ep100_series_qwen_siglip_local_YYYYMMDD_HHMMSS ./run_OpenNav.bash
```

运行前由 `run_OpenNav.bash` 做本地 LLM preflight，确认 `run_OpenNav.yaml` 中的模型配置和服务端返回模型一致。运行后主要产物分三类:

| 产物 | 路径模式 | 用途 |
|---|---|---|
| 聚合指标 | `/root/wjj/Open-Nav/logs/eval_results/ep{N}/{YYYYMMDD}/{exp_name}/stats_ckpt_val_unseen.json` | 看 SR、SPL、nDTW、mean distance 等整体指标 |
| 单 episode 指标 | `/root/wjj/Open-Nav/logs/eval_results/ep{N}/{YYYYMMDD}/{exp_name}/stats_ep_ckpt_val_unseen_r0_w1.json` | 找 success flip、OSR=1/SR=0、min distance 和 final distance |
| navigation record | `/root/wjj/Open-Nav/logs/navigation_records/ep{N}/{YYYYMMDD}/*.jsonl` | 复盘每一步 selector、STOP、fallback、distance 变化 |
| harness trace | `/root/wjj/Open-Nav/logs/harness_traces/ep{N}/{YYYYMMDD}/{variant}/.../*.jsonl` | 复盘 V/U 证据、phase、verifier、decision audit 和 unit override |
| 运行日志 | `/root/wjj/Open-Nav/logs/running_log/ep{N}/{YYYYMMDD}/*.log` | 查异常、耗时、模型服务失败和环境错误 |

每轮实验归档时必须记录:

- `EPISODE_COUNT`、`EXP_NAME`、运行日期和模型版本。
- 本轮启用的 V/U 模块及其 `LOG_ONLY` / decision-effect 状态。
- 是否修改了 selector context、STOP gate、fallback/recovery 或 parser。
- ep20 诊断结论和 ep100 验证结论是否一致。

### 2.3 单 episode 复盘流程

每个失败或翻转 episode 按同一套顺序复盘，避免只看最终 SR:

```text
1. 先读 per-episode stats:
   success、oracle_success、spl、ndtw、final_distance、trajectory length。

2. 如果 oracle_success=1 但 success=0:
   找 min_distance、min_distance_step、final_distance，判断是否“到过但没停住”。

3. 打开 navigation record:
   按 step 查看 selector_raw、selector_final、stop_requested、fallback、distance_gain。

4. 打开 harness trace:
   对齐 phase_evidence、visual evidence、stop_verification、failure_recovery、decision_audit。

5. 判断失败类型:
   STOP false positive、STOP false negative、near-goal drift、empty fallback、visual schema failure、phase parser failure、ranker untrusted。

6. 回到配置或代码:
   只调整对应 gate / scope / parser / schema，并保留 trace 字段用于下一轮验证。
```

组会中重点展示三类 case:

- `success -> failure`: 找本轮改动破坏了什么，例如 recover 阶段把近点拉远。
- `failure -> success`: 找本轮改动真正解决了什么，例如 STOP 被正确放行或 fallback 选到更可信候选。
- `OSR=1, SR=0`: 这是当前主瓶颈，优先分析 near-goal STOP、hold 和 recovery。

### 2.4 单步运行流程

当前每个 navigation step 的主流程如下:

```text
1. 读取当前 observation 和候选 waypoint。
2. 生成原始文本 observation，保留 Open-Nav baseline 的输入形式。
3. V1 对候选图像做 visual evidence extraction。
4. V3 将当前视觉证据写入 episode-level visual memory。
5. U0 计算 phase evidence: search / approach / verify / recover / unknown。
6. U1 根据 phase 构建 phase-aware selector context。
7. completion estimation 判断指令动作完成情况。
8. selector 基于文本 observation 和可选 U1/V4 context 选择 candidate 或 STOP。
9. 如果 selector 提出 STOP:
   - 原始 STOP gate 先判断是否允许；
   - V2/U2 对 current-view / final target / weak target 做二次验证；
   - 若不通过，记录 stop_rejected 并回退到移动候选。
10. 如果 selector 空预测或 STOP 被拒后候选不足:
    - V-fallback / U3 根据可信视觉排序和 failure type 选择 recovery candidate；
    - 若 ranker 不可信，则不做 action override。
11. decision_audit 汇总 original action、proposed action、final action 和 unit overrides。
12. 执行 Habitat action，记录 distance gain、collision、done、termination reason。
13. episode 结束时写入 SR / OSR / SPL / nDTW / final distance 等指标。
```

这个流程的重点不是增加更多 prompt，而是把原来混在 prompt 里的职责拆开:

- “我现在走到哪一步了”由 U0 phase evidence 记录。
- “图像是否支持到达目标”由 V1/V2 提供证据。
- “是否允许视觉证据影响 STOP”由 U2 控制。
- “selector 失败后能否改候选”由 V-fallback/U3 控制。
- “这一步到底是谁改了动作”由 decision audit 记录。

### 2.5 与原始 Open-Nav 的差异

原始 Open-Nav 流程更接近:

```text
candidate images -> text observation -> LLM selector -> candidate action
```

当前 Harness 流程变为:

```text
candidate images
  -> text observation for baseline compatibility
  -> visual evidence for verifier / memory / fallback
  -> phase and failure state for controlled intervention
  -> audited selector / STOP / recovery decision
```

主要差异:

- 原始流程只知道最终选了哪个 candidate；现在能知道为什么选、谁改了、改动是否生效。
- 原始流程中 STOP 和 fallback 容易和 selector prompt 混在一起；现在 STOP、fallback、recovery 有独立 gate 和日志。
- 原始流程无法区分“视觉证据生成失败”和“视觉证据未被消费”；现在通过 schema、candidate count、decision-effect 标记可追踪。
- 原始流程很难解释 SR 变化；现在可以分解为 OSR、STOP precision、recover drift、fallback first-candidate 等具体问题。

### 2.6 V 系列设计

V 系列是多模态视觉证据实验轴，核心目标是把“候选图像真实看到了什么”从纯文本 observation 中拆出来，形成可记录、可验证、可用于 selector/STOP/fallback 的结构化证据。

设计原则:

- V0/V1 不改变导航行为，只验证图文服务能力并记录视觉证据。
- V2 先做 log-only STOP / final target verifier，再显式打开 decision-effect。
- V3/V4 必须建立在 V1/V2 证据稳定、schema 稳定之后，避免把错误视觉证据注入 selector。
- 所有 V 系列模块都要记录 `LOG_ONLY` / `decision_effect_enabled`，避免把诊断结果误当作动作收益。

| 版本 | 名称 | 设计目标 | 是否影响动作 | 当前状态 |
|---|---|---|---|---|
| V0 | Qwen3.5-4B 多模态服务验证 | 验证本地 Qwen 服务能接收图文输入 | 否 | 已完成 |
| V1 | Visual Evidence Logging | 对候选视角图像抽取 `visible_landmarks`、`final_target_visible`、`arrival_evidence` 等结构化证据 | 否 | 已完成初版 |
| V2 | Visual TargetVerifier | 对 STOP proposal 和 final target 可见性做视觉验证，拦截明显误停或放行强证据 STOP | 默认否，显式配置后影响 STOP | 已完成初版并多轮校准 |
| V3 | VisualEvidenceMemory | 在 episode 内聚合“看见过什么、在哪里看见、是否支持到达”的视觉记忆 | 否 | 已完成初版 |
| V4 | Multimodal Selector Context | 将压缩后的视觉证据注入 selector context，辅助候选选择 | 默认否，显式配置后影响 selector | 已完成初版，正在收紧使用边界 |
| V-fallback | VisualEvidenceFallbackRanker | 在 selector 空预测或 STOP rejected 后，用候选级视觉证据替代 first-candidate fallback | 配置后影响 fallback | 已接入并审查可信排序条件 |

V 系列当前经验:

- 视觉证据本身有价值，但不能直接等价于“到达目标点”。
- V1 compact JSON schema 是关键基础设施；6/14 修复裸数组输出后，小样本从 1/11 恢复到 4/11。
- V2/V4 的收益必须分开归因: V2 影响 STOP，V4 影响 selector，不能混在同一个指标里解释。

### 2.7 U 系列设计

U 系列是受控决策单元实验轴，目标不是再加一个自由 agent，而是把 phase、STOP、recovery 和动作 override 拆成可审计的单元实验。

设计原则:

- 先做 U0 log-only，确保 U1/U2/U3 的证据都能稳定写入 trace。
- 每次只打开一个 decision-effect 单元，避免无法归因。
- U1 只改 selector 输入，不直接 STOP 或 fallback。
- U2 只管 STOP / rescue / weak-target 策略，不改普通移动候选。
- U3 只管 failure-conditioned recovery，不在 ranker 不可信时乱改动作。
- U4-combined 只作为组合展示，不能作为单元归因依据。

| 版本 | 名称 | 设计目标 | 允许影响面 | 当前状态 |
|---|---|---|---|---|
| U0-log | U 系列诊断层 | 记录 phase、failure type、decision audit、unit override，不改动作 | 无 | 已接入 |
| U1 | Phase-aware Evidence Scaffolding | 根据 `search/approach/verify/recover` 生成不同 selector context | 只允许改 selector 输入 | 已实现并增加 context source 标记 |
| U2 | Evidence-grounded STOP Verifier | 对 selector STOP rescue、completion weak target、current-view evidence 做受控验证 | 只允许改 STOP gate / rescue | 已实现，当前 scope 明确为 `selector_visual_rescue_only` |
| U3 | Failure-conditioned Recovery | 对 empty fallback、stop false positive、progress drift 等失败类型做可信 recovery reselect | 只允许改 fallback/recovery candidate | 已实现，并收紧为 trusted visual rank 才能 reselect |
| U4-combined | 组合展示 | 在 U1/U2/U3 单元验证后做组合实验 | 组合影响 selector、STOP、recovery | 尚不作为主要结论 |

U 系列当前经验:

- decision audit 是后续论文归因的必要条件，否则无法证明动作级 override 是否真的发生。
- phase parser 的可靠性直接影响 U1/U2；6/23 已修复 `1. None` 被误计为已完成动作的问题。
- U3 不能按原始候选顺序做 recovery，否则会把 fallback 噪声伪装成方法收益。

## 3. 时间线

| 日期 | 主要工作 | 结论 |
|---|---|---|
| 2026-06-10 | 本地 Qwen3.5-4B 和 SigLIP 接入；A0/A1 1 episode smoke | 本地模型和导航链路跑通，A1 与 A0 指标一致 |
| 2026-06-12 | 修 STOP 控制链路；降低 thought fusion 成本；V0/V1 多模态服务和视觉证据日志接入 | STOP 从不可控变成可记录、可拒绝；Qwen-VL 图文输入可用 |
| 2026-06-13 | V2/V4 decision-effect 早期版本；V24 小样本和 ep100 | 小样本最高 5/10，ep100 达 19/100，但多数 success 仍依赖 step limit |
| 2026-06-14 | 定位 V1 compact JSON schema 漂移；修复裸数组导致下游 candidate evidence 为 0 | 11-episode 从 1/11 提升到 4/11，schema 修复有效 |
| 2026-06-15 至 2026-06-16 | 多轮 ep20/ep100 验证，补充 current-view evidence、STOP rejected fallback 等 | SR 在 16%-25% 区间波动，STOP 与近目标漂移仍是主问题 |
| 2026-06-19 至 2026-06-21 | U 系列代码审查与归因修复: decision audit、U3 trusted recovery、U1 context source、U2 scope | trace 可信度提升，避免把 log-only / rescue-only 误读为完整方法收益 |
| 2026-06-22 | 日志目录按 episode/date 分层；严格 STOP rescue；长时 ep100 运行 | ep100 达 19/100、OSR 29/100，但耗时显著，且 OSR-SR gap 仍大 |
| 2026-06-23 | 回切 4B，增加模型 preflight；修 phase 对 `None` 的误解析；调整 verify-phase STOP rescue | ep20 退到 3/20，随后完成修复；最新 ep100 为 18/100 |

## 4. 完整 ep100 结果

| 日期 | 实验 | SR | OSR | OSR-SR gap | SPL | nDTW | final dist | steps | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-12 | `v_series_qwen_siglip_local_20260612_233128` | 10/100 | 11/100 | 1 | 0.0921 | 0.4311 | 7.86 | 4.23 | 早期 V 系列，stop_requested 过多 |
| 2026-06-13 | `v24_series_qwen_siglip_local_20260613_212115` | 19/100 | 24/100 | 5 | 0.1708 | 0.4797 | 7.42 | 6.00 | 当前 SPL/nDTW 最好的一次 ep100 |
| 2026-06-15 | `ep100_series_qwen_siglip_local_20260615_232107` | 16/100 | 24/100 | 8 | 0.1160 | 0.4531 | 7.19 | 7.14 | STOP 链路更复杂后 SR 回落 |
| 2026-06-16 | `ep100_series_qwen_siglip_local_20260616_162306` | 17/100 | 28/100 | 11 | 0.1017 | 0.4102 | 7.37 | 8.27 | OSR 提高但 SR 转化差 |
| 2026-06-21 | `ep100_series_qwen_siglip_local_20260621_225137` | 16/100 | 21/100 | 5 | 0.1099 | 0.3984 | 7.77 | 8.75 | U 系列 trace 接入后完整回测 |
| 2026-06-22 | `ep100_series_qwen_siglip_local_20260622_222705` | 19/100 | 29/100 | 10 | 0.1295 | 0.4189 | 7.10 | 9.26 | OSR 最高，运行耗时长 |
| 2026-06-23 | `ep100_series_qwen_siglip_local_20260623_215922` | 18/100 | 27/100 | 9 | 0.1122 | 0.4180 | 7.02 | 9.17 | 最新结果，SR 净少 1 条 |

### 4.1 与原文 Open-Nav 结果对比

原文表格口径为百分制 `nDTW/OSR/SR/SPL`，本报告内部表格中的 `nDTW/SPL` 是 0-1 小数。下面将本地 ep100 的 `nDTW/SPL` 乘以 100 后对齐展示。`TL` 原文是 trajectory length，本地统计表当前只记录平均 step 数，不是同口径 TL，因此这里不填。

| Method | TL | NE ↓ | nDTW ↑ | OSR ↑ | SR ↑ | SPL ↑ | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| Open-Nav-Llama3.1 (Ours, 原文) | 8.07 | 7.25 | 44.99 | 23 | 16 | 12.90 | 原文结果 |
| Open-Nav-GPT4 (Ours, 原文) | 7.68 | 6.70 | 45.79 | 23 | 19 | 16.10 | 原文结果 |
| HiMemVLN-Qwen2-VL-72B | 7.55 | 6.65 | 52.79 | 36 | 30 | 26.85 | 层次化记忆 + 72B MLLM，作为外部强基线 |
| 2026-06-13 V24 local | - | 7.42 | 47.97 | 24 | 19 | 17.08 | 本地 ep100，SR/SPL/nDTW 最接近或略高于原文 GPT4，但 NE 更差 |
| 2026-06-22 ep100 local | - | 7.10 | 41.89 | 29 | 19 | 12.95 | 本地 ep100，OSR 最高，SR 与原文 GPT4 持平 |
| 2026-06-23 latest local | - | 7.02 | 41.80 | 27 | 18 | 11.22 | 最新本地 ep100，NE 优于 Llama3.1、弱于 GPT4，SR 低于 GPT4 1 点 |

对比结论:

- 当前本地最好 SR 已达到原文 Open-Nav-GPT4 的 19%，但还没有稳定超过；最新完整 ep100 是 18%。
- 本地 6/22 和 6/23 的 OSR 高于原文两行，说明“到过目标附近”的能力不弱，主要差距仍在 OSR 到 SR 的转化。
- 本地最新 NE 为 7.02，比原文 Llama3.1 的 7.25 好，但仍弱于 GPT4 的 6.70。
- 本地最新 SPL 明显低于原文 GPT4，说明路径效率和主动 STOP/hold 仍是主要短板。

### 4.2 与 HiMemVLN 的对比

HiMemVLN-Qwen2-VL-72B 的 SR 到 30%，不能简单解读为“只要加 memory 就能涨 10 个点”，也不能直接说明我们的记忆机制无效。两边的 memory 进入决策闭环的深度不同:

- HiMemVLN 的记忆是方法主体: 短期 Localer 使用视觉图记忆做当前位置回忆、重访检测和候选方向软约束；长期 Globaler 抽取全局导航 schema，并持续用历史轨迹校准当前动作。也就是说，它的 memory 直接影响 waypoint 选择和长程方向一致性。
- 我们当前 V3 `VisualEvidenceMemory` 第一版主要是 episode 内视觉证据聚合和 trace 诊断；V4 中 `include_memory_suffix` 默认关闭，避免把“历史看见过目标”误注入到每个候选里，导致 selector 把旧证据当成当前方向证据。因此当前实验并没有真正测试一个强 decision-effect memory policy。
- HiMemVLN 使用 Qwen2-VL-72B，视觉理解、指令跟随、长上下文推理和对 memory 文本的使用能力都显著强于我们当前主跑的本地 4B/9B 配置。记忆模块本身依赖基座模型正确读取和执行，模型规模差异会放大收益差距。
- 从指标看，HiMemVLN 相对本地 2026-06-22 最好 OSR 是 36 vs 29，提升 7 点；SR 是 30 vs 19，提升 11 点。也就是说它不仅更容易到过目标附近，还更能把 near-goal 状态保持到最终 STOP。我们的主要瓶颈仍是 OSR-to-SR 转化、STOP/hold 和 recovery 漂移。

因此当前结果说明我们的“视觉证据记忆记录”还不是 HiMemVLN 式的“层次化决策记忆”。它不是完全不行，而是还停留在可观测、可诊断、弱注入阶段；下一步应该把 memory 做成 phase-gated、空间锚定、只在可信场景影响动作的决策模块，而不是简单打开全局 memory suffix。

后续可做的对齐消融:

- `M0`: 当前 log-only / weak-injection memory，作为对照。
- `M1`: 只加入短期 visual graph memory，用于重访检测和候选方向降权。
- `M2`: 加入 near-goal hold memory，专门减少 OSR=1 但 SR=0 的漂移。
- `M3`: 加入长期 instruction schema / CameFrom / last-k trajectory summary，约束长程方向漂移。
- `M4`: 与 U2 STOP verifier 联动，只在 current-view 或近两步高置信证据支持时允许 memory 影响 STOP。

解读:

- SR 未稳定突破 20%，但 OSR 多次达到 24%-29%，说明 agent 有能力走到目标附近。
- 2026-06-22 与 2026-06-23 对比: SR 从 19 到 18，OSR 从 29 到 27；final distance 反而略好，说明 SR 下降来自阈值附近 episode 重新洗牌，而非整体距离变差。
- SPL 最高的是 2026-06-13 V24，主要因为步数更短、路径更短；后续版本加入更多 STOP/recovery 机制后，路径长度和步数增加，SPL 被拉低。

## 5. 小样本验证结果

| 日期 | 实验 | 规模 | SR | OSR | SPL | nDTW | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| 2026-06-13 | `v24_series_112044` | 10 | 3/10 | 6/10 | 0.2863 | 0.6699 | V24 初步有效，但样本太小 |
| 2026-06-13 | `v24_series_150848` | 10 | 5/10 | 7/10 | 0.4863 | 0.7105 | 小样本最高，但不能外推 |
| 2026-06-13 | `v24_series_175536` | 12 | 5/12 | 7/12 | 0.4053 | 0.6311 | 仍有较高波动 |
| 2026-06-14 | `v24_series_101818` | 11 | 1/11 | 1/11 | 0.0693 | 0.4777 | schema 漂移导致下游未消费视觉证据 |
| 2026-06-14 | `v24_series_122559` | 11 | 4/11 | 4/11 | 0.3416 | 0.5844 | schema 修复后明显恢复 |
| 2026-06-16 | `ep20_series_100333` | 20 | 5/20 | 7/20 | 0.1530 | 0.5002 | STOP/recovery 调整后中等表现 |
| 2026-06-16 | `ep20_series_132553` | 20 | 5/20 | 7/20 | 0.1621 | 0.4938 | 与上一轮接近 |
| 2026-06-22 | `ep20_series_183509` | 20 | 8/20 | 10/20 | 0.2794 | 0.5411 | 小样本表现最好的一轮 ep20 |
| 2026-06-23 | `ep20_series_183532` | 20 | 3/20 | 7/20 | 0.1306 | 0.4987 | 严格 STOP 与 phase 问题暴露，后续已修 |

小样本结论:

- 小样本最高到 40%-50% SR，但 ep100 未复现，说明当前 sample variance 很大。
- 小样本仍有价值: 它帮助定位了 V1 schema、completion weak target、phase `None` 误解析、verify/recover STOP 边界等问题。
- 后续仍应采用“ep20 诊断 -> ep100 验证”的节奏，但不能只看 ep20 SR。

## 6. 关键技术进展

### 6.1 本地化运行链路

已完成:

- Qwen3.5-4B 本地 OpenAI-compatible API 接入。
- `run_OpenNav.bash` 增加模型 preflight，防止服务端模型与配置不一致。
- 日志目录按 `ep{N}/{YYYYMMDD}/{exp_name}` 分层，便于复盘。

意义:

- 实验可以离线、可重复运行。
- ep20/ep100 不再混在同一目录中，减少误读旧结果的风险。

### 6.2 Navigation record 与 Harness trace

已记录事件包括:

- `episode_start` / `episode_end`
- `observation`
- `completion_estimation`
- `selector_raw` / `selector_fused` / `selector_final`
- `stop_verification` / `visual_stop_allowed` / `visual_stop_rejected`
- `phase_evidence` / `phase_aware_context`
- `failure_type_diagnostic` / `failure_recovery`
- `decision_audit` / `unit_override`

意义:

- 每次 SR 变化可以追到 episode、step、phase、candidate、STOP gate 和 fallback。
- U 系列修复后，trace 可以区分 log-only、selector context override、action-level override，归因更可靠。

### 6.3 V 系列视觉证据

主要产物:

- V1: Visual Evidence Logging。
- V2: Visual TargetVerifier。
- V3: VisualEvidenceMemory。
- V4: Multimodal Selector Context。
- VisualEvidenceFallbackRanker。

关键修复:

- 6/14 定位 compact JSON 输出形状漂移: Qwen-VL 有时返回裸数组而不是 `{"candidates": [...]}`。
- 新增统一 schema 工具，将裸数组规范化为 candidates object。
- 修复后，11 episode 从 1/11 提升到 4/11，`candidate_evidence_count=0` 问题消失。

当前判断:

- 视觉证据对诊断和小样本有帮助。
- 但直接把视觉 allow 用于 STOP 会有误停风险，尤其是 recover 阶段和通用目标词场景。

### 6.4 STOP 与 phase/recovery

已完成:

- 允许 navigator 输出 STOP，并加硬门控。
- STOP gate 拒绝后移除 STOP，回退到合法移动候选。
- completion gate 对 `door/sink/room/hallway` 等弱目标词更保守。
- U2 当前 scope 明确为 `selector_visual_rescue_only`，避免误读成完整 STOP verifier。
- phase parser 不再把 `1. None` / `not completed` 误判为已完成动作。
- U3 只有在 visual evidence ranking 可信时才允许 recovery reselect。

当前判断:

- 过宽 STOP 会误停，过严 STOP 会错过 near-goal 成功。
- 下一阶段应专门做 current-view verify auto-stop，而不是继续扩大通用 STOP rescue。

## 7. 失败模式分析

### 7.1 OSR 到 SR 转化不足

定义:

- SR: 最终距离 `distance[-1] <= 3m`。
- OSR: 轨迹中任意一步 `distance <= 3m`。

现象:

- 6/16 ep100: OSR 28/100，SR 17/100，gap 11。
- 6/22 ep100: OSR 29/100，SR 19/100，gap 10。
- 6/23 ep100: OSR 27/100，SR 18/100，gap 9。

说明:

- agent 经常可以接近目标，但没有稳定停住。
- 成功不一定来自主动 STOP，很多成功是 step limit 时刚好仍在 3m 内。

### 7.2 STOP false positive 与 false negative 同时存在

false positive:

- completion gate 对 `door`、`sink`、`room` 这类通用词容易误判。
- visual verifier 看到远处目标时可能把 `final_target_visible` 误当成 `arrival_evidence`。

false negative:

- 6/23 ep20 中 `513/265/810/1092` 曾到过 3m 内但最终失败。
- 旧 U2 策略过严，`allow_rescue=0`，近点 STOP 被压制。

判断:

- STOP 不应做成一个全局阈值，而应区分 source 和 phase。
- verify 阶段 current-view evidence 可适当放开；recover/search 阶段继续保守。

### 7.3 Recovery 阶段容易漂移

最新 6/23 ep100 与 6/22 对比:

- 9 个 episode 从 success 变 failure。
- 丢失成功的 episode: `226, 371, 1092, 11, 1087, 804, 1307, 377, 7`。
- 这些 episode 最终平均距离从上一版的 1.38m 变为 5.37m。
- 它们在 6/23 日志中都进入 `recover`，常见 `stop_false_positive` 或 `empty_fallback_bad`。

说明:

- 近目标后如果进入 recover，selector/recovery 仍可能继续拉远。
- recovery policy 不能只靠候选顺序，需要可信视觉排序和最近距离收益约束。

### 7.4 小样本高分不可直接外推

证据:

- 6/13 ep10 曾到 5/10。
- 6/22 ep20 到 8/20。
- 但 ep100 长期停在 16%-19%。

说明:

- 当前 episode 分布差异很大。
- 需要固定 episode subset 或做分层分析，否则 ep20 只适合 debugging，不适合宣称整体提升。

## 8. 最新 SR 下降的解释

6/23 ep100 相比 6/22:

| 指标 | 2026-06-22 | 2026-06-23 | 变化 |
|---|---:|---:|---:|
| SR | 0.19 | 0.18 | -0.01 |
| OSR | 0.29 | 0.27 | -0.02 |
| SPL | 0.1295 | 0.1122 | -0.0173 |
| nDTW | 0.4189 | 0.4180 | -0.0009 |
| final distance | 7.0999 | 7.0173 | -0.0826 |
| steps | 9.26 | 9.17 | -0.09 |

逐 episode:

- 9 条从成功变失败。
- 8 条从失败变成功。
- 净变化为 -1 条成功。

结论:

- 这不是整体导航距离变差，而是阈值附近样本重新洗牌。
- 真正需要关注的是 OSR 也下降了 2 条，说明到达目标附近的样本略少。
- 失败集中在 recover 阶段，说明当前修复方向仍应聚焦 near-goal hold、verify auto-stop 和 recovery 约束。

## 9. 下阶段计划

### 9.1 固化分析脚本

已新增:

```text
scripts/analyze_navigation_jsonl.py
```

典型用法:

```text
python3 scripts/analyze_navigation_jsonl.py \
  logs/eval_results/ep20/20260623/ep20_series_qwen_siglip_local_20260623_183532 \
  logs/navigation_records/ep20/20260623/ep20_series_qwen_siglip_local_20260623_183532_train_navigation_20260623_183557.jsonl \
  --top 10
```

输出:

- SR / OSR / SPL / nDTW / final distance。
- `oracle_success=1 but success=0` 列表。
- 每个 episode 的 `min_distance -> final_distance`。
- STOP 成功、STOP 失败、STOP 被拦截。
- high-confidence current-view evidence 分布。
- selector empty prediction、fallback changed、recovery override。
- phase 分布和 recover 触发原因。
- schema / parse health，包括 schema error、schema warning、parse error。

目的:

- 每轮 ep20/ep100 后自动回答“为什么 SR 没升”。
- 避免继续依赖手工 jq 和主观观察。

已用 2026-06-23 ep20 验证:

- 复现 SR 3/20、OSR 7/20、SPL 0.1306、nDTW 0.4987。
- 自动列出 `265/513/810/1092` 为 `OSR=1 but SR=0`。
- 自动定位最大 drift 为 `810: min_distance 0.045m -> final_distance 5.1767m`。
- 统计 recover 分布: `recover=96`，主要触发为 `stop_false_positive` 和 `empty_fallback_bad`。

### 9.2 current-view verify auto-stop

方案:

- 只在 `phase == verify` 时允许。
- `current_step >= 3`。
- 只看 `__current_view__` contact sheet。
- visual verdict 和 selected candidate verdict 均为 allow。
- `final_target_visible=true` 且 `arrival_evidence=true`。
- confidence >= 0.95。
- 无 hard blockers、contradictions、missing instruction terms。
- recover/search/unknown 阶段禁止触发。

预期:

- 把 `513/810` 这类“已经到过近点但继续走远”的 episode 转成 SR。

### 9.3 weak target 二次确认

针对:

- `door`
- `doorway`
- `room`
- `hallway`
- `sink`
- `stairs`

策略:

- 第一次 weak target allow 只记录，不直接 STOP。
- 连续两步 current-view allow 且 arrival=true 后再允许。
- 或 selector 连续两次提出 STOP 且 current-view allow，再允许。

目的:

- 降低 `door/sink` 类误停，同时保留 doorway/room 类型目标的成功可能。

### 9.4 completion 输出 schema 化

将 completion estimation 从自由文本改为 JSON:

```json
{
  "executed_actions": [
    {"index": 1, "action": "...", "completed": true, "evidence": "..."}
  ],
  "final_action_completed": false,
  "all_actions_completed": false
}
```

目的:

- 减少 `1. None`、否定句和动作词复现造成的 phase 误判。
- 为 U1/U2 phase-aware logic 提供稳定输入。

### 9.5 条件式额外步数，而不是全局延长 step limit

不建议直接把 step limit 加大，因为已经成功的 episode 可能会走出 3m。

建议:

- 最近 2 步距离收益为正，且 current-view 未 allow 时，额外给 1-2 步。
- current-view allow 且 verify phase 时优先 STOP。
- 最近 2 步负收益时不延长，防止漂移。

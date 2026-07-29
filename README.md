# Controlled Navigation Harness for Open-Nav

本仓库记录一个基于 Open-Nav 的 VLN-CE 实验项目。当前重点不是复现原论文主页，而是把 Open-Nav 的 LLM-centered navigation pipeline 改造成可诊断、可消融、可回放的 controlled navigation harness。

原始 Open-Nav 作为 waypoint-based baseline 保留。本项目在其外层逐步加入显式状态、视觉证据、STOP 验证、selector context、fallback 排序和结构化 trace，用来分析零样本连续环境导航中的失败来源。

## 实验目标

本项目要回答的问题是:

> 能否将 Open-Nav 中由 LLM 耦合承担的导航职责，重构为一个受控 navigation harness，使进度、候选、记忆、验证和恢复都成为显式状态与工具接口，从而提高 VLN-CE 零样本导航的可诊断性、可消融性，并为后续受限恢复机制提供可靠触发依据？

当前不做:

- 不训练或微调新 policy。
- 不切换到 waypoint-free 主链。
- 当前主实验不做自由工具调用式 LLM Agent；后续可按“先 Harness 化、再工具化、再 controlled tool-calling”的路线单独推进。
- 不引入在线多 sub-agent 协作。
- 不把模型升级和模块收益混在同一组实验里。

## 项目框架流程图

下图按当前代码主链整理，展示从 Habitat 环境到 STOP/动作决策的完整数据流，以及 V/U 系列 harness 的介入位置。

```mermaid
flowchart TD
  A["运行入口\nrun_OpenNav.bash"] --> B["run.py\n加载 run_OpenNav.yaml / vlnce_task.yaml"]
  B --> C["baseline_registry\nschedulesampler-OPENNAV"]
  C --> D["SSTrainer\nBaseVLNCETrainerLLM.eval"]

  subgraph Inputs["外部依赖（本地部署，不提交至仓库）"]
    I1["R2R / VLN-CE 数据集 + MP3D 场景\nHabitat Simulator 运行环境"]
    I2["WaypointBert（BinaryDistPredictor_TRM）\n本地预训练 Transformer，参数冻结\n输出：top-k 候选航点（角度 + 距离）"]
    I3["Qwen3.5-4B（lmdeploy 本地服务）\n文字 LLM（selector / completion）\n多模态 VLM（V1 / V2 视觉证据）两用"]
    I4["SpatialBot3B（本地 3B VLM）\n内嵌 SigLIP-so400m 作为视觉编码器\n输出：空间场景描述（含距离估计）"]
    I5["RAM（Recognize Anything，SwinL）\n输出：物体 tag 列表"]
  end

  I1 --> D

  D --> E["Episode start\ninstruction → 动作列表 / 地标缓存（Qwen 文字）"]
  E --> F["候选生成\n全景 RGB+Depth → WaypointBert\n→ top-k 候选航点"]

  F --> G["感知层（每候选各一次，输出纯文字）\nRGB → RAM → 物体 tag\nRGB+Depth → SpatialBot3B(SigLIP) → 空间描述\n合并 → 候选观测文字"]

  subgraph Harness["V / U 系列 Harness"]
    V1["V1 VisualEvidenceLogger\n候选 RGB base64 → Qwen 多模态\n→ {final_target_visible, arrival_evidence, confidence}"]
    V2["V2 VisualTargetVerifier\n全景拼图 base64 → Qwen 多模态\n→ {verdict: allow/reject, reason}"]
    U1["U1 PhaseAwareEvidenceScaffolder（纯 Python）\nV1 证据 → 阶段归类（search/approach/verify/recover）\n→ 候选文字追加阶段 suffix 和 V4 视觉摘要"]
    U2["U2 StopEvidenceVerifier（纯 Python）\nV1+V2+U1 结构化证据 → 最终 allow/blocked 裁决\nM2a：dist<3.5m 时将 trajectory_incomplete 降级为软拒绝"]
    U3["U3 RecoveryPolicy（纯 Python）\n失败检测 → 备用候选重选（每集 ≤2 次）"]
    M3["M3 ProactiveStopGate（纯 Python）\ndist<3.5m + 上步 final_target_visible=True\n→ 强制触发 V2 → 若 allow 则强制 STOP"]
  end

  G --> V1
  V1 --> U1
  G --> J

  U1 --> J["完成度估计\n历史 + 动作 + 地标 → Qwen 文字 LLM → 已执行动作"]
  J --> K{"completion gate\n提议 STOP?"}

  K -- "否，dist<3.5m\n且 final_target_visible" --> M3
  K -- "否" --> L
  K -- "是" --> V2
  M3 --> V2

  V2 --> U2
  U2 -- "allow" --> S["STOP action"]
  U2 -- "blocked" --> U3
  U3 --> L

  L["Selector（Qwen 文字 LLM）\n候选 ID + 指令 + 历史 + U1 注入文字 → 候选 ID 或 STOP"]
  L -- "候选 waypoint" --> O["环境动作\naction=4, angle + distance"]
  L -- "STOP" --> V2
  L -- "空预测 / 无效" --> U3

  O --> P["envs.step\n更新位置、碰撞、history"]
  P --> Q{"episode 结束?"}
  Q -- "否" --> F
  Q -- "是" --> R["episode metrics\nSR / SPL / nDTW / TL 等"]
  S --> R

  R --> T["输出记录\nnavigation_records / harness_traces\neval_results / running_log"]

  I2 --> F
  I3 --> J
  I3 --> L
  I3 --> V1
  I3 --> V2
  I4 --> G
  I5 --> G
```

关键读法:

- **感知层**：候选生成后，每个候选方向经 RAM（物体 tag）和 SpatialBot3B 处理，输出纯文字给 selector。SpatialBot3B 内嵌 SigLIP-so400m 作为视觉编码器（`mm_vision_tower`），随模型一起加载，不是独立外部模块。图片本身不进入 selector 的 API 调用。
- **两条决策链**：routing（候选选择）由 Qwen **文字** LLM 完成，看到的是 SpatialBot3B / RAM 生成的文字；stop gate（STOP 是否允许）由 Qwen **多模态** VLM 完成，图片以 base64 直接发送。
- **U 系列零模型调用**：U1/U2/U3 是纯 Python 规则层，读取 V1/V2 的结构化 JSON 做决策，不额外调用任何模型，推理成本为零。
- **M3 主动触发**：completion gate 每步检查距离阈值和上步 V2 的 `final_target_visible` 信号，条件满足时以 `stop_flag=True` 重跑 V2，填补"selector 未发 STOP"的盲区。
- `run_OpenNav.yaml` 控制各模块的 `LOG_ONLY` / decision-effect 模式；V4（MultimodalSelectorContext）当前保持 `LOG_ONLY`，其输出由 U1 经阶段门控后注入 selector 观测，避免双重注入。

## 当前状态

更新时间: 2026-06-29

当前主线: U 系列门控（U1 阶段感知 + U2 STOP 核查 + U3 恢复）+ M3 主动 STOP 触发。

核心指标（100 集 val_unseen）:

| 配置 | SR | OSR | OSR→SR | SPL | nDTW |
|---|---:|---:|---:|---:|---:|
| A0 baseline（2026-06-28） | 16% | 25% | 64.0% | 0.1015 | 0.4458 |
| max_config（2026-06-28） | **21%** | 26% | **80.8%** | **0.1424** | 0.4111 |

max_config 首次稳定超越原文 Open-Nav-GPT4（SR=19%，OSR=23%），M3 已验证修复 ep11（SPL=1.0，仅 4 步）。

核心判断:

- **主失败模式已迁移**：A0 中 walk-through 占 OSR-SR gap 的 77.8%；max_config 激活 U2 stop verifier 后，walk-through 降至 40%，主因转为 stop-blocked（60%，STOP 被门控过度拒绝）。
- **残余 4 集**均指向近距离目标识别失败（`final_target_not_visible`），SpatialBot3B/V2 在极近距离时无法正确识别已进入视野的目标。
- **M3 ep100 验证正在运行**（exp: `ep100_series_qwen_siglip_local_20260629_215656`，预计 SR ≥ 22%）。
- 下阶段优先修复 RC3（M2b 距离衰减或 arrival_evidence 替代方案），待 SR ≥ 24% 后再加 7B/8B backbone 对照。

## 阶段划分

阶段顺序:

```text
A0/A1 固定可比基线
  -> B/C 只记录证据和状态
  -> D/E/F 让证据进入决策链
  -> G 降低运行成本
  -> H 单独评估模型升级
```

A 阶段用于保证可比性。B/C 阶段主要产生日志和诊断证据。D/E/F 阶段才允许改变导航行为。G/H 属于后续优化，不能混入当前 V2/V4 主实验收益。

| 阶段 | 类型 | 核心问题 | 是否改行为 | 当前状态 |
|------|------|----------|------------|----------|
| A0 Baseline | 原始对照 | 原始 Open-Nav 在同一批 episode 上表现如何 | 否 | 保留为参照 |
| A1 Instrumented Baseline | 行为不变的 Harness 外壳 | 加日志、状态、trace 后是否仍与 A0 一致 | 否 | 已有初版，仍需 A0/A1 指标对齐 |
| B Evidence Logging | 候选级证据采集 | 每个 waypoint 候选是否有可解析语义/视觉证据 | 否 | V1 已实现，schema 修复已生效 |
| C Memory & State | episode 内状态聚合 | 单步证据能否沉淀为 seen target、arrival evidence、revisit/novelty | 主要 log-only | V3 已实现初版 |
| D Selector Context | 证据进入候选选择 | 候选级视觉摘要能否帮助 selector 选更合理的 waypoint | 是 | V4 已进入 decision-effect |
| E STOP Verification | STOP 验证 | STOP 是否有足够视觉证据支持，能否减少提前 STOP | 是 | V2 已接入，正在收紧 allow 规则 |
| F Fallback Policy | 失败恢复 | 空预测或 STOP 被拒后，能否替代 first-candidate fallback | 是 | visual ranked fallback 已接入，需继续复测 |
| G Runtime Policy | 运行成本控制 | 如何降低 VLM/LLM latency，同时不破坏证据覆盖率 | 是，但不应改变候选集合 | compact JSON、JPEG、token cap 已接入；adaptive sampling 暂缓 |
| H Model Upgrade | 模型升级对照 | 更强 depth encoder / waypoint predictor 是否带来独立收益 | 是，必须单独消融 | 待调研 |

阶段验收口径:

- A1 必须先证明 SR/SPL/NE/TL 与 A0 基本一致，才能把后续收益归因给 Harness 模块。
- B/C 的产物首先是日志质量，不直接声明导航性能收益。
- D/E/F 的收益必须分别看 selector 变化、STOP allow/reject case、fallback 后 distance gain。
- G/H 必须单独做对照，避免把运行策略或模型升级收益混入 V2/V4。

## V / U 系列模块

**V 系列**（调用 Qwen 多模态，输出结构化证据 JSON）:

| 模块 | 代码入口 | 当前作用 |
|------|----------|----------|
| V1 VisualEvidenceLogger | `vlnce_baselines/common/opennav_ext/visual_evidence.py` | 每步对候选 RGB base64 → Qwen 多模态 → `{final_target_visible, arrival_evidence, confidence}` |
| V2 VisualTargetVerifier | `vlnce_baselines/common/opennav_ext/visual_target_verifier.py` | STOP 提案时触发；全景拼图 → Qwen 多模态 → `{verdict: allow/reject, reason}` |
| V4 MultimodalSelectorContext | `vlnce_baselines/common/opennav_ext/multimodal_selector_context.py` | V1 JSON → 候选级视觉摘要文字；当前 LOG_ONLY，输出交 U1 阶段门控后注入 selector |
| Schema tools | `vlnce_baselines/common/opennav_ext/visual_evidence_schema.py` | 统一兼容 dict/list 形式的 V1 输出 |

**U 系列**（纯 Python 规则逻辑，零模型调用）:

| 模块 | 代码入口 | 当前作用 |
|------|----------|----------|
| U1 PhaseAwareEvidenceScaffolder | `vlnce_baselines/common/opennav_ext/` | V1 证据 → 阶段归类（search/approach/verify/recover）→ 候选文字 suffix 注入 |
| U2 StopEvidenceVerifier | `vlnce_baselines/common/opennav_ext/` | V1+V2+U1 → allow/blocked 最终裁决；含 M2a trajectory_bypass（dist<3.5m 时降级轨迹拒绝） |
| U3 RecoveryPolicy | `vlnce_baselines/common/opennav_ext/` | selector 空预测 / STOP 误触发 / 连续负距离 → 备用候选重选（每集 ≤2 次） |
| M3 ProactiveStopGate | `vlnce_baselines/common/base_il_trainer_llm.py` | completion gate 每步检查 dist<3.5m + final_target_visible → 强制触发 V2 → 若 allow 则强制 STOP |

## 数据集与日志

当前仓库已包含 OpenNav_R2R-CE_100 的 `val_unseen` 小规模评估数据:

| 文件 | 用途 |
|------|------|
| `data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz` | 100 个 R2R-CE / VLN-CE episode 的指令、路径和 BERT index 预处理数据，用于当前小样本与 100 episode 评估 |
| `data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/val_unseen_gt.json.gz` | `val_unseen` ground truth，用于 success、oracle_success、SPL、nDTW、distance_to_goal 等指标计算 |

已提交的实验日志目录:

| 目录 | 内容 | 汇报用途 |
|------|------|----------|
| `logs/navigation_records/` | 每轮导航的 JSONL 事件日志和 plain log | 复盘每个 episode 的动作、STOP、fallback、V1/V2/V4 事件 |
| `logs/harness_traces/` | 按 episode 拆分的 structured trace | 定位 selector context、STOP verifier、fallback 对具体 step 的影响 |
| `logs/eval_results/` | Habitat 评估输出的 aggregate / per-episode 指标 | 快速读取 SR、SPL、nDTW、distance 等指标 |
| `logs/running_log/` | shell / trainer 运行记录 | 检查启动配置、进程运行和异常终止 |
| `logs/downloads/` | 模型或依赖下载记录 | 追踪本地模型准备过程 |

## 已观察结果

### 日志索引

| ID | 日志文件 |
|----|----------|
| V4-100 | `logs/navigation_records/1/v_series_qwen_siglip_local_20260612_233128_train_navigation_20260612_233156.jsonl` |
| V24-10 | `logs/navigation_records/1/v24_series_qwen_siglip_local_20260613_112044_train_navigation_20260613_112112.jsonl` |
| R23-10 | `logs/navigation_records/v24_series_qwen_siglip_local_20260613_150848_train_navigation_20260613_150914.jsonl` |
| R23C-12 | `logs/navigation_records/v24_series_qwen_siglip_local_20260613_175536_train_navigation_20260613_175605.jsonl` |
| RR-100 | `logs/navigation_records/v24_series_qwen_siglip_local_20260613_212115_train_navigation_20260613_212143.jsonl` |
| PRE-11 | `logs/navigation_records/v24_series_qwen_siglip_local_20260614_101818_train_navigation_20260614_101847.jsonl` |
| FIX-11 | `logs/navigation_records/v24_series_qwen_siglip_local_20260614_122559_train_navigation_20260614_122628.jsonl` |

### 主要指标

| ID | 实验含义 | Ep | success | oracle | SPL | nDTW | mean DTG | 关键诊断 |
|----|----------|----|---------|--------|-----|------|----------|----------|
| V4-100 | 早期 V4-only 100 episode | 100 | 10/100 | 11/100 | 0.0921 | 0.4311 | 7.8584 | `parse_error=415/423`，视觉证据大多没有进入 selector |
| V24-10 | V2/V4 decision-effect 小样本 | 10 | 3/10 | 6/10 | 0.2863 | 0.6699 | 4.2091 | `parse_error=0/57`，`visual_stop_rejected=28`，V4 已真实进入 selector |
| R23-10 | V2-first STOP gate + visual ranked fallback | 10 | 5/10 | 7/10 | 0.4863 | 0.7105 | 3.4895 | 6 次 fallback 触发，10 episode 中表现最好，但仍有误 STOP 风险 |
| R23C-12 | conservative STOP / fallback 小样本 | 12 | 5/12 | 7/12 | 0.4053 | 0.6311 | 4.5635 | `visual_stop_allowed=0`，`visual_stop_rejected=36`，全部到 step limit |
| RR-100 | Runtime Reduce 100 episode | 100 | 19/100 | 24/100 | 0.1708 | 0.4797 | 7.4200 | latency 降低，但 compact schema 漂移导致 `mm_ctx zero=581/600` |
| PRE-11 | schema 修复前小样本 | 11 | 1/11 | 1/11 | 0.0693 | 0.4777 | 6.9756 | `mm_ctx zero=49/63`，fallback 缺少候选证据 |
| FIX-11 | schema 修复后小样本 | 11 | 4/11 | 4/11 | 0.3416 | 0.5844 | 5.7791 | `parsed_root_type=dict 60/60`，`mm_ctx zero=0/60`，fallback evidence 7/7 |

### 诊断结论

- 从 V4-100 到 V24-10，核心变化不是单纯加模块，而是把 V1 视觉证据从“记录到日志”推进到“能被 V4 selector context 消费”。`parse_error` 从 415/423 降到 0/57 后，success 从 10% 提升到 30%，oracle_success 从 11% 提升到 60%。
- R23-10 是当前小样本中最强的正向信号: success 5/10、SPL 0.4863、nDTW 0.7105、mean DTG 3.4895。它说明 visual ranked fallback 有价值，但样本量太小，不能直接写成最终结论。
- RR-100 说明“运行成本下降”和“导航效果提升”不能混为一谈。该轮 100 episode 的 visual evidence latency 均值约 15.18s/step，但 compact JSON 产生 schema 漂移，导致 V4/fallback 大量拿不到候选证据。
- PRE-11 到 FIX-11 是最清晰的 schema 修复对照: success 从 1/11 到 4/11，SPL 从 0.0693 到 0.3416，nDTW 从 0.4777 到 0.5844；同时 `candidate_evidence_count=0` 问题被清掉。
- STOP 仍是当前最大风险点。V2 能显著减少提前 STOP，但 `visual_stop_allowed` 的 evidence threshold 仍需收紧，尤其是 `completion_gate` 来源、泛化目标词、只看到相似类别但没有 arrival evidence 的情况。

## 下一步

优先级从高到低:

1. **等待 M3 ep100 结果**（`ep100_series_qwen_siglip_local_20260629_215656`，预计 SR ≥ 22%）；重点观测 M3 误停率和 ep377 的 V2 行为。
2. **RC3 修复（M2b）**：统计 100 集 harness trace 中 `final_target_visible` 与距离的分布（2m 内有多少集目标实际可见），决定距离衰减阈值或 arrival_evidence 替代方案。
3. **跨 backbone 对照**（SR ≥ 24% 后启动）：加一组 7B/8B 本地模型对照，验证 OSR→SR 转化率提升与 backbone 规模是否解耦。
4. **V4 decision-effect 准备**：将 U1 切换为"V4 已应用时跳过视觉摘要注入"模式，避免双重计数后再开启 V4 独立效果消融。

## 待改进事项

这些方向不直接替换当前主线 baseline，应作为独立 ablation 或后续分支评估，避免把模型升级收益和 Harness 模块收益混在一起。

| 方向 | 当前判断 | 注意事项 |
|------|----------|----------|
| DD-PPO depth encoder 替代 | Habitat DD-PPO 官方还有 SE-ResNeXt50 / SE-ResNeXt101 等更强深度编码器可调研 | 当前代码按 `VlnResnetDepthEncoder` + ResNet-50 权重结构加载，不能直接替换 checkpoint，需要改 backbone 和加载逻辑 |
| SmartWay-style waypoint predictor | 2025 SmartWay 方向用 DINOv2、masked cross-attention、occupancy-aware loss 强化 waypoint prediction，适合作为候选生成升级路线 | 这会改变 waypoint 候选质量和错误分布，应单独做 `Waypoint Predictor Upgrade` 对照，不应混入 V2/V4 主实验 |
| 自由工具调用 Agent 演进 | 可以把现有 V1/V2/V3/V4、selector、fallback、env action 包装为 typed tools，再引入受控 Agent Controller | 不应直接切到 LangChain 自由 Agent；必须保留 Tool Registry、Action Validator、STOP Verifier、预算限制和 trace |

## Agent 化演进方向

当前项目已经有 embodied navigation agent 的基本闭环，但不是 LangChain 式自由工具调用 Agent。更稳妥的后续路线是把现有固定 pipeline 逐步改造成受控 tool-calling agent:

```text
L0 当前 Harness 固定流程
  -> L1 Tool Registry 抽象
  -> L2 Scripted Controller
  -> L3 LLM Tool Planner logging-only
  -> L4 Controlled Tool-Calling Agent
  -> L5 LangChain / LangGraph 适配
```

核心原则:

- 先把现有 `visual_evidence`、`visual_target_verifier`、`visual_evidence_memory`、`multimodal_selector_context`、`visual_fallback`、selector 和 env action 抽象成 typed tools。
- 先让工具调用可记录、可回放、可消融，再允许 LLM 在白名单工具内选择下一步调用。
- STOP 仍必须经过 `VisualTargetVerifier`，环境动作必须经过 Action Validator。
- LangChain/LangGraph 只作为后续 controller runtime，不替代项目内的 ToolSpec、AgentState、MetricsLogger 和安全约束。

详细路线见:

```text
Controlled-Navigation-Harness/自由工具调用Agent演进路线.md
```

下一轮必须检查的日志字段:

```text
visual_evidence.parsed_root_type
multimodal_selector_context.candidate_evidence_count
selector_empty_prediction_fallback.candidate_evidence_count
selector_empty_prediction_fallback.changed
visual_target_verifier.visual_evidence_candidate_count
visual_stop_allowed
visual_stop_rejected
visual_stop_uncertain
```

## 运行方式

基础运行:

```bash
bash run_OpenNav.bash
```

当前默认配置在:

```text
run_OpenNav.yaml
habitat_extensions/config/vlnce_task.yaml
```

当前关键配置:

```yaml
OPENNAV_HARNESS:
  ENABLED: true
  ENABLE_HARNESS_LOGGING: true
  ENABLE_DECISION_EFFECT: true

  VISUAL_EVIDENCE:
    ENABLED: true
    LOG_ONLY: true
    MAX_CANDIDATES: 4
    MAX_TOKENS: 768
    IMAGE_JPEG_QUALITY: 70
    COMPACT_JSON: true

  VISUAL_TARGET_VERIFIER:
    ENABLED: true
    LOG_ONLY: false
    CONFIDENCE_THRESHOLD: 0.9
    REQUIRE_ARRIVAL_EVIDENCE: true
    REQUIRE_FULL_COVERAGE_FOR_ALLOW: true
    BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW: true

  VISUAL_EVIDENCE_MEMORY:
    ENABLED: true
    LOG_ONLY: true

  MULTIMODAL_SELECTOR_CONTEXT:
    ENABLED: true
    LOG_ONLY: false
```

## 文档索引

| 文档 | 用途 |
|------|------|
| `Controlled-Navigation-Harness/项目总控.md` | 项目边界、阶段定义和文档入口 |
| `Controlled-Navigation-Harness/实验方案.md` | 完整实验方案和 harness 设计 |
| `Controlled-Navigation-Harness/自由工具调用Agent演进路线.md` | 从当前 Harness 向受控 tool-calling Agent 演进的阶段路线 |
| `Controlled-Navigation-Harness/V系列多模态视觉证据初步实现后调整方案.md` | V0-V4 初步实现后的问题复盘和下一轮矩阵 |
| `Controlled-Navigation-Harness/V系列多模态视觉证据实验记录-封存-20260613.md` | V 系列初步实现过程记录 |
| `Controlled-Navigation-Harness/docs/experiment_record_20260610.md` | 本地 Qwen / SigLIP A0 smoke、A1 logging 接入 |
| `Controlled-Navigation-Harness/docs/experiment_record_20260612.md` | STOP 控制链路、Thought Fusion、V0/V1/V2 前置验证 |
| `Controlled-Navigation-Harness/docs/experiment_record_20260614.md` | 100 episode / 小样本复盘、schema 修复和 STOP allow 修复 |
| `Controlled-Navigation-Harness/docs/code_review_issues.md` | 当前代码风险、fallback 和 schema 待修问题 |
| `MIGRATION.md` | 迁移到新机器的完整步骤:第三方库链接、habitat 补丁、环境重建、模型下载 |

## 依赖和大文件

本仓库已提交当前实验日志和 OpenNav_R2R-CE_100 的 `val_unseen` 小规模评估数据。第三方源码库、模型快照和场景资产只作为本地依赖保留，不作为仓库内容提交。以下路径需要按本地环境自行准备(**逐条获取链接、版本号和校验和见 [`MIGRATION.md`](MIGRATION.md)**):

```text
data/scene_datasets/mp3d/
external/
recognize_anything/
SpatialBot/
SpatialBot3B/
waypoint_prediction/checkpoints/check_val_best_avg_wayscore
data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth
recognize_anything/pretrained/ram_swin_large_14m.pth
SpatialBot3B/model-00001-of-00002.safetensors
SpatialBot3B/model-00002-of-00002.safetensors
```

当前 `.gitignore` 已排除:

- `cache_files/`
- `image_show/`
- `navigator_log.log`
- `logs/checkpoints/`
- `__pycache__/`
- `data/scene_datasets/mp3d/`
- `*.safetensors`
- `*.pth`
- `waypoint_prediction/checkpoints/`
- `SpatialBot/`
- `SpatialBot3B/`
- `recognize_anything/`
- `external/`
- `rag/`

## 原 Open-Nav 信息

本项目基于 Open-Nav:

- Project website: <https://sites.google.com/view/opennav>
- Paper: <https://arxiv.org/pdf/2409.18794>
- Original repository remote in this workspace: `https://github.com/YanyuanQiao/Open-Nav`

原 Open-Nav 论文:

```bibtex
@inproceedings{qiao2025opennav,
  author    = {Yanyuan Qiao and Wenqi Lyu and Hui Wang and Zixu Wang and Zerui Li and Yuan Zhang and Mingkui Tan and Qi Wu},
  title     = {Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025}
}
```

## Acknowledgements

This repository builds on Open-Nav and includes or references components from DiscussNav, Discrete-Continuous-VLN, Habitat-Lab, SpatialBot, and Recognize Anything Model. Third-party licenses and original source references should be preserved when using or redistributing this code.

# Controlled Navigation Harness for Open-Nav

本仓库记录一个基于 Open-Nav 的 VLN-CE 实验项目。当前重点不是复现原论文主页，而是把 Open-Nav 的 LLM-centered navigation pipeline 改造成可诊断、可消融、可回放的 controlled navigation harness。

原始 Open-Nav 作为 waypoint-based baseline 保留。本项目在其外层逐步加入显式状态、视觉证据、STOP 验证、selector context、fallback 排序和结构化 trace，用来分析零样本连续环境导航中的失败来源。

## 实验目标

本项目要回答的问题是:

> 能否将 Open-Nav 中由 LLM 耦合承担的导航职责，重构为一个受控 navigation harness，使进度、候选、记忆、验证和恢复都成为显式状态与工具接口，从而提高 VLN-CE 零样本导航的可诊断性、可消融性，并为后续受限恢复机制提供可靠触发依据？

当前不做:

- 不训练或微调新 policy。
- 不切换到 waypoint-free 主链。
- 不做自由工具调用式 LLM Agent。
- 不引入在线多 sub-agent 协作。
- 不把模型升级和模块收益混在同一组实验里。

## 当前状态

更新时间: 2026-06-15

当前主线: V 系列多模态视觉证据 + V2/V4 decision-effect 修正。

核心判断:

- V1 visual evidence schema 问题已经修复，裸数组输出会被规范化为 `{"candidates": [...]}`。
- V4 selector context 和 visual ranked fallback 已经能消费候选级视觉证据。
- 小样本中 V2/V4 有正向信号，但 100 episode 结果还不能作为稳定改进结论。
- 当前主要风险从“视觉证据未被消费”转为 “V2 STOP allow 过宽”，尤其是泛化目标词和 `completion_gate` 来源的 STOP。
- 下一步应先跑 10 到 12 episode 小样本验证 STOP 修复，不应直接跑 100 episode。

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

## V 系列模块

| 模块 | 代码入口 | 当前作用 |
|------|----------|----------|
| V1 VisualEvidenceLogger | `vlnce_baselines/common/opennav_ext/visual_evidence.py` | 对候选 RGB 视角抽取结构化视觉证据 |
| V2 VisualTargetVerifier | `vlnce_baselines/common/opennav_ext/visual_target_verifier.py` | 验证 STOP proposal，拦截或放行 STOP |
| V3 VisualEvidenceMemory | `vlnce_baselines/common/opennav_ext/visual_evidence_memory.py` | 聚合 episode 内视觉证据 |
| V4 MultimodalSelectorContext | `vlnce_baselines/common/opennav_ext/multimodal_selector_context.py` | 将候选级视觉摘要注入 selector 输入 |
| Visual fallback | `vlnce_baselines/common/opennav_ext/visual_fallback.py` | selector 空预测时用视觉证据排序候选 |
| Schema tools | `vlnce_baselines/common/opennav_ext/visual_evidence_schema.py` | 统一兼容 dict/list 形式的 V1 输出 |

## 已观察结果

### V2/V4 decision-effect 小样本

记录文件:

```text
logs/navigation_records/v24_series_qwen_siglip_local_20260613_112044_train_navigation_20260613_112112.jsonl
```

10 episode 结果:

| 指标 | 数值 |
|------|------|
| success | 3/10 |
| oracle_success | 6/10 |
| SPL mean | 0.2863 |
| nDTW mean | 0.6699 |
| distance_to_goal mean | 4.2091 |
| visual_evidence parse_error | 0/57 |
| visual_stop_rejected | 28 |

结论: V2 拦截提前 STOP 有效，V4 context 已真实进入 selector。

### R2/R3: V2-first STOP gate + visual ranked fallback

记录文件:

```text
logs/navigation_records/v24_series_qwen_siglip_local_20260613_150848_train_navigation_20260613_150914.jsonl
```

同 10 episode 对比 V2/V4-current:

| 指标 | V2/V4-current | R2/R3 |
|------|---------------|-------|
| success | 3/10 | 5/10 |
| oracle_success | 6/10 | 7/10 |
| SPL mean | 0.2863 | 0.4863 |
| nDTW mean | 0.6699 | 0.7105 |
| distance_to_goal mean | 4.2091 | 3.4895 |

结论: visual ranked fallback 是正向信号。6 次空预测 fallback 中，3 次改变旧 first-candidate fallback，且这些动作均为正 distance gain。

### 100 episode 运行复盘

记录文件:

```text
logs/navigation_records/v24_series_qwen_siglip_local_20260613_212115_train_navigation_20260613_212143.jsonl
```

100 episode 结果:

| 指标 | 数值 |
|------|------|
| success | 19/100 |
| oracle_success | 24/100 |
| SPL | 0.1708 |
| nDTW | 0.4797 |
| mean distance_to_goal | 7.42 |

结论: Runtime Reduce 有效降低耗时，但该 100 episode 结果不能作为当前最优。主要问题是 compact V1 输出发生 schema 漂移，导致 V4/fallback 没有真实消费候选级视觉证据。

### Schema 修复后小样本

记录文件:

```text
logs/navigation_records/v24_series_qwen_siglip_local_20260614_122559_train_navigation_20260614_122628.jsonl
```

11 episode 结果:

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| success | 1/11 | 4/11 |
| SPL | 0.0693 | 0.3416 |
| nDTW | 0.4777 | 0.5844 |
| distance_to_goal | 6.9756 | 5.7791 |

修复确认:

- `visual_evidence.parsed_root_type=dict`: 60/60。
- V4 `candidate_evidence_count=0`: 0。
- fallback 触发 7 次，候选证据数量均大于 0。
- fallback 7 次中 3 次改变原首候选，5 次带来正向距离增益。

剩余问题:

- 2 次 `visual_stop_allowed` 都是失败终止。
- 已修复 `completion_gate` 上泛化 final landmarks 的 STOP allow 过宽问题。
- 需要下一轮小样本验证该修复是否真正降低误停。

## 下一步

优先级从高到低:

1. 跑 10 到 12 episode 小样本，验证 `completion_gate` 来源的泛化 STOP 不再被 V2 误放行。
2. 检查 `visual_stop_allowed` case study，确认 allow 必须有更强的 final target / arrival evidence。
3. 统计 `selector_empty_prediction_fallback.changed` 和 fallback 后的 distance gain。
4. 补充 schema warning: 合法 JSON 但非预期结构时，日志必须区分 parse error、schema error、empty evidence。
5. 在 V2/V4 稳定后，再恢复 R4 adaptive candidate sampling。
6. 最后再重跑 100 episode，不把小样本正向信号直接写成最终结论。

## 待改进事项

这些方向不直接替换当前主线 baseline，应作为独立 ablation 或后续分支评估，避免把模型升级收益和 Harness 模块收益混在一起。

| 方向 | 当前判断 | 注意事项 |
|------|----------|----------|
| DD-PPO depth encoder 替代 | Habitat DD-PPO 官方还有 SE-ResNeXt50 / SE-ResNeXt101 等更强深度编码器可调研 | 当前代码按 `VlnResnetDepthEncoder` + ResNet-50 权重结构加载，不能直接替换 checkpoint，需要改 backbone 和加载逻辑 |
| SmartWay-style waypoint predictor | 2025 SmartWay 方向用 DINOv2、masked cross-attention、occupancy-aware loss 强化 waypoint prediction，适合作为候选生成升级路线 | 这会改变 waypoint 候选质量和错误分布，应单独做 `Waypoint Predictor Upgrade` 对照，不应混入 V2/V4 主实验 |

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
| `Controlled-Navigation-Harness/V系列多模态视觉证据初步实现后调整方案.md` | V0-V4 初步实现后的问题复盘和下一轮矩阵 |
| `Controlled-Navigation-Harness/V系列多模态视觉证据实验记录-封存-20260613.md` | V 系列初步实现过程记录 |
| `Controlled-Navigation-Harness/docs/experiment_record_20260610.md` | 本地 Qwen / SigLIP A0 smoke、A1 logging 接入 |
| `Controlled-Navigation-Harness/docs/experiment_record_20260612.md` | STOP 控制链路、Thought Fusion、V0/V1/V2 前置验证 |
| `Controlled-Navigation-Harness/docs/experiment_record_20260614.md` | 100 episode / 小样本复盘、schema 修复和 STOP allow 修复 |
| `Controlled-Navigation-Harness/docs/code_review_issues.md` | 当前代码风险、fallback 和 schema 待修问题 |

## 依赖和大文件

本仓库没有提交本地大模型权重、Matterport3D 场景数据和运行日志。它们需要按路径自行准备:

```text
data/scene_datasets/mp3d/
waypoint_prediction/checkpoints/check_val_best_avg_wayscore
data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth
recognize_anything/pretrained/ram_swin_large_14m.pth
SpatialBot3B/model-00001-of-00002.safetensors
SpatialBot3B/model-00002-of-00002.safetensors
```

当前 `.gitignore` 已排除:

- `logs/`
- `cache_files/`
- `__pycache__/`
- `data/scene_datasets/mp3d/`
- `*.safetensors`
- `*.pth`
- `waypoint_prediction/checkpoints/`

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

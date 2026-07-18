# V 系列改进方案 2026-06-16

## 背景

`ep20_series_qwen_siglip_local_20260616_100333` 的最新日志说明，V 系列模块已经稳定接入，但对 SR 的正向贡献不足:

- V1 候选视觉证据稳定产出，schema 基本可用。
- V2 STOP verifier 参与决策，但 3 次实际 STOP 均失败；多数成功仍来自 `step_length_limit`。
- V3 memory 有记录能力，但 target/arrival 记忆稀疏，且 episode 级记忆可能把 selector 带向曾经看过但当前不该去的目标。
- V4 selector context 每步都生效，但后半程 distance gain 下降，说明视觉上下文没有足够稳定地转化为正确移动。
- STOP rejected / empty prediction fallback 的平均距离增益接近 0，排序策略还不够像“继续执行路线”，更像“找视觉上像终点的候选”。

因此，本轮不再继续扩大视觉模块，而是做收敛改造: 降低噪声、提高 STOP 精度、让 fallback 更像导航策略。

## 目标

1. V4 只给 selector 低噪声候选证据，不把 V3 episode 记忆默认拼到每个候选文本里。
2. V2 保持高精度 STOP，尤其避免 `floor / archway / room / hallway` 这类弱终点词触发 completion 直停。
3. fallback 在 STOP 被拒后应继续移动，而不是再次被 `final_target_visible=True` 的视觉信号诱导到假终点。
4. 新增日志字段能直接回答“视觉模块是否提升了移动/STOP”。

## 改动方案

### P0: V4 selector context 降噪

现状:

- 每个候选后拼接 V1 summary 和统一 memory suffix。
- memory suffix 是 episode 级的 `seen_ever / missing_recent / target_seen`，对每个候选都相同，容易让 selector 把历史目标当作当前方向证据。

修改:

- `MultimodalSelectorContext` 增加 `include_memory_suffix`，默认关闭。
- 候选摘要改成更短的结构化 evidence table:
  - `cid`
  - `match`
  - `miss`
  - `target`
  - `arrival`
  - `conf`
  - `note`
- 不再默认输出 `visible_landmarks` 的长列表，减少 Qwen text selector 被视觉枚举词带偏。

预期:

- selector prompt 更短，候选之间视觉差异更突出。
- V3 保留日志和后验分析能力，但默认不参与在线选择。

### P1: fallback 排序分场景

现状:

- fallback 统一优先 `final_target_visible` 和 `arrival_evidence`。
- 在 `stop_rejected_fallback` 场景中，这会与 STOP rejection 的意图冲突: 当前不能停时，fallback 仍可能追逐“像终点”的候选。

修改:

- `VisualEvidenceFallbackRanker.rank()` 增加 `source_stage` 和 `reason` 参数。
- `stop_rejected` 阶段:
  - 降低 `final_target_visible / arrival_evidence` 的直接权重；
  - 优先 `matched_instruction_terms`、local instruction hits、confidence；
  - 对只有弱泛化终点词命中的候选增加 penalty。
- 普通 selector fallback:
  - 保留 final target / arrival 的正向权重，但加入弱泛化目标 penalty。

预期:

- STOP 被拒后 fallback 更偏向“继续推进路线”，减少原地确认式错误。
- 空预测 fallback 仍可利用视觉证据找有指令 landmark 的方向。

### P1: completion STOP 对弱终点词保守

现状:

- `should_stop()` 只要 completion estimator 判断全动作完成且文本观察匹配 final landmark，就允许 STOP。
- 日志中 `floor / archway`、`sink / tub` 类 case 会在 current pano VLM allow 后变成错误 STOP。

修改:

- 在 `spatialNavigator.should_stop()` 中识别弱终点词:
  - `area`
  - `archway`
  - `doorway`
  - `entryway`
  - `floor`
  - `hall`
  - `hallway`
  - `room`
  - `stair`
  - `stairs`
  - `staircase`
- 如果 final landmark 全部是弱终点词，completion gate 不直接 STOP。
- selector 明确选择 STOP 时，仍可走 V2 current pano rescue；也就是弱词不禁止 STOP，只禁止 completion gate 自动直停。

预期:

- 减少 completion gate false positive。
- 有效 STOP 主要通过 selector STOP + V2 高置信 current-view evidence 救回。

## 验证指标

下一轮小样本至少统计:

- `visual_stop_allowed` 按 source 的成功率和 final distance。
- `visual_stop_rescued` 数量、成功率、final distance。
- `completion_gate_weak_final_target` 拦截次数。
- `stop_rejected_fallback` 的 selected candidate distance gain。
- `selector_empty_prediction_fallback` 的 selected candidate distance gain。
- `multimodal_selector_context.include_memory_suffix` 是否为 false，prompt 是否明显变短。

## 实验建议

优先跑 20 episode 验证:

```bash
EPISODE_COUNT=20 bash run_OpenNav.bash
```

通过后再跑 100 episode。若 20 episode 中 `stop_requested` 仍全部失败，应继续收紧 V2 allow；若 `step_length_limit` 过多且 near-goal 不 STOP，则放宽 selector STOP rescue，而不是放宽 completion gate。

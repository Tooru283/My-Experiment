# SR 下降后续问题汇总与修复方案 2026-06-16

## 背景

前一轮针对 `ep100_series_qwen_siglip_local_20260615_232107` 的分析已经确认，SR 下降不是单一模型能力问题，而是导航策略在后半程大量进入低收益移动和 `step_length_limit`。今天审查最新修改后，新的主要风险集中在 STOP gate 被收得过紧。

本文件记录本轮需要解决的问题、设计取舍和对应修改计划。

## 当前关键问题

### P1: current-view 文本佐证成为硬拦截，可能拒绝正确 STOP

当前 `VisualTargetVerifier` 会对 `stop_evidence_mode="current_pano"` 的 allow candidate 追加 `uncorroborated_final_target:<term>` blocker。由于配置中 `REJECT_ON_UNCERTAIN=True`，一旦出现 blocker，整体 verdict 会从 `allow` 降成 `uncertain` 并被拒绝。

这个规则原本是为了解决 `dining room table` 被 VLM 从普通 `table` 误升格的问题，但它有两个副作用:

- 当前文本 observation 由另一个视觉/文本摘要链路产生，可能漏写房间修饰词或最终目标短语。
- 图像证据已经支持 `final_target_visible=True` 和 `arrival_evidence=True` 时，文本缺词不一定说明 STOP 错误。

结论: 这条规则适合作为诊断和后验统计，不适合作为默认硬拦截。

### P1: selector STOP 不能被视觉证据救回

selector 选择 STOP 后，当前逻辑要求 `navigator.should_stop()` 也先通过，视觉 verifier 的 allow 才能真正放行 STOP。如果 completion parser 或 final landmark gate 漏判，视觉证据即使高置信支持当前位置到达目标，也只能记录日志，然后 fallback 到 movement。

这会把“防早停”变成“该停不停”。在最新 SR 下降样本里，大量 episode 后期移动收益接近 0，并最终触发 `step_length_limit`，因此这条逻辑有较高概率继续压低 SR。

### P2: episode count 配置与实验命名不一致

`run_OpenNav.yaml` 中 `EVAL.EPISODE_COUNT` 是 100，但 `run_OpenNav.bash` 会用 shell 变量覆盖该值。当前脚本默认值和实验名前缀也容易漂移，例如 `ep20` 名称但默认 episode count 不是 20。

结论: 需要让默认 episode count、实验名前缀和 YAML 基线一致。短跑 smoke test 应通过显式环境变量覆盖。

### P2: 旧审查文档中的配置状态已经过期

`code_review_issues.md` 中仍有 `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW=True` 的描述，但当前 YAML 和默认配置均为 `False`。这会影响后续问题归因。

## 修复原则

1. STOP 的执行仍必须基于当前位置证据，而不是任意 waypoint candidate。
2. `__current_view__` current-pano 证据继续作为 STOP verifier 的 selected candidate。
3. current-view 文本佐证默认只作为诊断，不作为 hard blocker。
4. selector STOP 可以被视觉证据救回，但必须满足:
   - verifier 使用 `current_pano`。
   - selected candidate 是 `__current_view__` 且 verdict 为 allow。
   - selected candidate 同时有 `final_target_visible=True` 与 `arrival_evidence=True`。
   - completion estimation 没有明确否定短语。
   - 不存在 min-step、sample coverage、generic blocker 等其他硬 blocker。
5. 如果 completion 明确说动作未完成，则不允许视觉救回 STOP。
6. 每一次视觉救回 STOP 都必须写结构化日志，便于复盘 false positive。

## 修改计划

### 1. VisualTargetVerifier

- 新增配置 `REQUIRE_CURRENT_VIEW_TEXT_CORROBORATION_FOR_ALLOW`，默认 `False`。
- 保留 `current_view_corroboration` 报告。
- 当该配置为 `False` 时，`uncorroborated_final_target` 只进入 `allow_warnings`，不进入 `allow_blockers`。
- 当该配置为 `True` 时，恢复硬 blocker 行为，用于后续专项 ablation。

### 2. Base Trainer STOP 决策

- 新增 selector STOP 视觉救回判断。
- 如果 `old_stop_flag=False` 但视觉证据满足救回条件，则:
  - `stop_flag=True`
  - `stop_reason="Navigator selected STOP and current-view visual evidence rescued STOP."`
  - 写入 `visual_stop_rescued` 事件
  - 同时保留原始 verifier 结果与旧 stop gate 状态
- 视觉救回不绕过明确否定 completion 和其他硬 blocker。

### 3. 配置

- `run_OpenNav.yaml` 增加 `REQUIRE_CURRENT_VIEW_TEXT_CORROBORATION_FOR_ALLOW: False`。
- `default.py` 增加相同默认值。
- `run_OpenNav.bash` 默认 episode count 与 YAML 保持一致，实验名前缀从 episode count 自动生成。

### 4. 文档

- 更新旧审查汇总中 `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW` 的当前状态描述。
- 新增本文件作为 2026-06-16 修复入口。

## 验证计划

本轮先做轻量验证:

- `git diff --check`
- 相关 Python 文件 `py_compile`
- YAML 能被 `yaml.safe_load` 解析
- 用轻量 stub 验证 `VisualTargetVerifier` 在默认配置下不会因 `uncorroborated_final_target` 把 allow 降为 uncertain

完整效果验证需要后续固定 episode 子集复跑，重点看:

- `visual_stop_rescued` 数量和对应 final `distance_to_goal`
- `visual_stop_rejected` 的 blocker 分布
- `step_length_limit` 是否下降
- selector STOP 后 fallback movement 是否减少
- SR / OSR / SPL 是否相对同一 episode 子集改善

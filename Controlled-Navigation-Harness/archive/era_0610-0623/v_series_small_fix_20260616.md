# V 系列小修记录 2026-06-16

## 背景

基于 `latest_run_comparison_20260616_1325.md` 的对比，13:25 新运行相对 10:03 旧运行:

- SR 持平 `0.25`。
- STOP 从 `3` 次增加到 `9` 次。
- STOP 成功从 `0/3` 变为 `3/9`。
- selector visual rescue 出现 `5` 次，但只有 `1` 次成功。

说明视觉模块已经影响决策，但 selector STOP rescue 太松，带来了新的 false positive。

## 本轮目标

只做小改，不改 V4/fallback 大结构:

1. 降低 selector STOP rescue false positive。
2. 保留 completion gate 的有效 STOP。
3. 修复弱终点 completion gate 永久拦截导致 near-goal 走远的问题。

## 修改内容

### 1. selector STOP rescue 遇到 warning 不救

修改位置:

`vlnce_baselines/common/base_il_trainer_llm.py`

规则:

如果 `visual_stop_can_rescue_selector_stop()` 中的 verifier 结果存在 `allow_warnings`，不允许 rescue。

原因:

- `748` 的错误 STOP 带有 `uncorroborated_final_target:dining room table`。
- 这类 warning 表示 VLM 把视觉对象或场景关系升格成 final target，但 current-view 文本没有足够佐证。

预期影响:

- 阻止 `748` 类 false rescue。
- 可能也会阻止 `513` 类带 warning 的正确 rescue；这是 precision/recall 取舍。本轮优先降低 false positive。

### 2. selector STOP rescue 要求 missing_instruction_terms 为空

修改位置:

`vlnce_baselines/common/base_il_trainer_llm.py`

规则:

如果 selected current-view verdict 中存在 `missing_instruction_terms`，不允许 rescue。

原因:

- `265` 的错误 STOP 中，VLM 认为 bathroom visible，但 `missing_instruction_terms=["oven", "turn right"]`，说明路线证据不完整。
- selector rescue 不应在路线关键项缺失时绕过 completion gate。

预期影响:

- 阻止未完成路线但 VLM 看到 final object 的误停。
- 不影响 completion gate allow。

### 3. 弱终点 completion auto-stop 从永久拦截改为前期拦截

修改位置:

- `vlnce_baselines/common/navigator/spatialNavigator.py`
- `vlnce_baselines/common/base_il_trainer_llm.py`
- `vlnce_baselines/config/default.py`
- `run_OpenNav.yaml`

新增配置:

```yaml
OPENNAV_HARNESS:
  NAVIGATION_RUNTIME:
    MIN_STEPS_FOR_WEAK_FINAL_TARGET_COMPLETION_STOP: 8
```

规则:

- 当 final target 属于弱泛化词，例如 `doorway / archway / room / hallway / stairs` 时:
  - `current_step < 8`: 拦截 completion auto-stop。
  - `current_step >= 8`: 允许 completion gate 进入 V2 current-pano verifier。

原因:

- `11` 在 step 3 因 `floor/archway` 早停，应该被挡住。
- `602` 在 step 8 已经到 `distance=0.951`，但旧弱终点永久拦截导致没有 STOP，最后走远到 `3.712`。

预期影响:

- 保留早期弱目标防误停。
- 给后期 near-goal 的 weak final target 一个通过 V2 验证后 STOP 的机会。

## 新增日志字段

`completion_gate_weak_final_target` payload 新增:

- `current_step`
- `weak_final_target_min_steps`

下一轮可直接检查是否符合:

- step < 8 才出现弱终点拦截。
- step >= 8 的弱终点如果 still no STOP，则看 V2 reject 原因。

## 下一轮验证重点

对 20 episode 新运行检查:

1. `visual_stop_rescued`
   - 数量应下降。
   - false positive 应减少，尤其是 `748 / 265 / 140 / 11` 类样本。

2. `completion_gate_weak_final_target`
   - 应只出现在 `current_step < 8`。
   - `602` 类 near-goal step 8 不应再被该规则直接拦截。

3. STOP 成功率
   - 比 13:25 的 `3/9` 更重要。
   - 如果 STOP 数下降但 STOP 成功率提升，是预期结果。

4. SR / OSR
   - 期望 SR 不低于 `0.25`。
   - 重点看是否能恢复 `602`、避免 `748/265/140/11` 的 false rescue。

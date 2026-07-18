# 2026-06-16 今日修改审查记录

## 背景

用户要求审查 2026-06-16 当前工作区修改，重点关注前一轮针对 SR 下降所做的 STOP gate、visual target verifier、completion parsing、fallback 与运行配置调整。

本次审查是静态代码审查加轻量校验，不包含完整 VLN 复跑。

## 审查时间

- 时间: 2026-06-16 12:01 CST
- 仓库: `/root/wjj/Open-Nav`
- 主要输入: 当前 worktree、`run_OpenNav.bash`、`run_OpenNav.yaml`、导航与视觉验证相关代码 diff

## 审查范围

重点审查文件:

- `run_OpenNav.bash`
- `run_OpenNav.yaml`
- `vlnce_baselines/common/base_il_trainer_llm.py`
- `vlnce_baselines/common/navigator/prompts.py`
- `vlnce_baselines/common/navigator/spatialNavigator.py`
- `vlnce_baselines/common/opennav_ext/visual_evidence.py`
- `vlnce_baselines/common/opennav_ext/visual_target_verifier.py`
- `vlnce_baselines/common/opennav_ext/visual_evidence_memory.py`
- `vlnce_baselines/common/opennav_ext/visual_evidence_schema.py`
- `vlnce_baselines/common/opennav_ext/visual_fallback.py`
- `vlnce_baselines/common/opennav_ext/landmark_matching.py`
- `vlnce_baselines/config/default.py`
- `Controlled-Navigation-Harness/docs/code_review_issues.md`

同时查看了 worktree 状态，发现存在大量已跟踪日志删除和未跟踪运行产物。

## 执行过的检查

1. 查看工作区状态与 diff 规模:

```bash
git -C /root/wjj/Open-Nav status --short
git -C /root/wjj/Open-Nav diff --stat
git -C /root/wjj/Open-Nav diff --name-only
```

2. 核对启动脚本与 YAML 配置是否一致:

```bash
sed -n '1,80p' /root/wjj/Open-Nav/run_OpenNav.bash
nl -ba /root/wjj/Open-Nav/run_OpenNav.yaml | sed -n '1,180p'
rg -n "episodes_to_eval|EPISODE_COUNT" /root/wjj/Open-Nav
```

3. 审查主导航链路:

- `completion_gate`
- `selector_stop_gate`
- STOP 被拒后的 movement fallback
- invalid `next_vp` 的 env action fallback
- `step_length` 配置化逻辑

重点查看范围:

```bash
nl -ba vlnce_baselines/common/base_il_trainer_llm.py | sed -n '780,1905p'
```

4. 审查 STOP 视觉验证链路:

- `__current_view__` current-pano evidence
- `_allow_blockers()`
- `_verdict()`
- `REJECT_ON_UNCERTAIN`
- current-view text corroboration

重点查看范围:

```bash
nl -ba vlnce_baselines/common/opennav_ext/visual_target_verifier.py | sed -n '1,780p'
nl -ba vlnce_baselines/common/opennav_ext/visual_evidence.py | sed -n '1,620p'
```

5. 审查 completion parser 与 STOP gate:

- `Executed Actions:` marker 解析
- unstructured response recovery
- `_parse_executed_actions()`
- `_final_landmark_gate()`
- `should_stop()`

重点查看范围:

```bash
nl -ba vlnce_baselines/common/navigator/spatialNavigator.py | sed -n '1,640p'
nl -ba vlnce_baselines/common/navigator/prompts.py
```

6. 审查辅助模块:

- visual evidence schema diagnostics
- visual fallback ranker
- shared landmark matching
- visual evidence memory

7. 运行轻量校验:

```bash
git -C /root/wjj/Open-Nav diff --check
python -m py_compile \
  /root/wjj/Open-Nav/vlnce_baselines/common/base_il_trainer_llm.py \
  /root/wjj/Open-Nav/vlnce_baselines/common/navigator/prompts.py \
  /root/wjj/Open-Nav/vlnce_baselines/common/navigator/spatialNavigator.py \
  /root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/visual_evidence.py \
  /root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/visual_evidence_memory.py \
  /root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/visual_evidence_schema.py \
  /root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/visual_fallback.py \
  /root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/visual_target_verifier.py \
  /root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/landmark_matching.py \
  /root/wjj/Open-Nav/vlnce_baselines/config/default.py
```

结果:

- `diff --check` 通过。
- 上述 Python 文件 `py_compile` 通过。
- 尝试直接实例化 `VisualTargetVerifier` 做局部行为探针，但项目包导入会先加载 Habitat 依赖，当前 shell 缺少 `habitat_baselines`，该探针未跑通。

## 主要审查结论

### P1: STOP 放行可能过度保守

`VisualTargetVerifier._allow_blockers()` 在 `stop_evidence_mode == "current_pano"` 且 selected candidate 为 allow 时，会根据 current-view 文本观察增加 `uncorroborated_final_target:<term>` blocker。

随后 `_verdict()` 遇到 allow candidate 但存在 blocker 时返回 `uncertain`。当前配置 `REJECT_ON_UNCERTAIN: True`，因此这类 STOP 会被拒绝。

风险:

- 如果 VLM 图像证据正确，但文本 observation 没有写出完整修饰词，例如只写了 `table` 而没有 `dining room table`，正确 STOP 也会被拒。
- 这会增加 near-goal episode 继续移动并最终 `step_length_limit` 的概率，可能继续拉低 SR。

相关位置:

- `vlnce_baselines/common/opennav_ext/visual_target_verifier.py:653`
- `vlnce_baselines/common/opennav_ext/visual_target_verifier.py:717`
- `run_OpenNav.yaml:85`

### P1: selector STOP 无法被视觉证据单独救回

selector 选中 STOP 后，逻辑要求 `old_stop_flag` 也通过，视觉 verifier 的 `allow` 才真正放行 STOP。

风险:

- completion parser 或 landmark gate 一旦漏判，视觉证据即使高置信支持当前位置到达目标，也只能记录日志并 fallback 到移动。
- 这会把“防早停”变成“该停不停”，与 SR 下降分析中大量 `step_length_limit` 失败相符。

相关位置:

- `vlnce_baselines/common/base_il_trainer_llm.py:1587`
- `vlnce_baselines/common/base_il_trainer_llm.py:1601`
- `vlnce_baselines/common/base_il_trainer_llm.py:1615`

### P2: 评测 episode 数配置不一致

`run_OpenNav.yaml` 中 `EVAL.EPISODE_COUNT: 100`，但 `run_OpenNav.bash` 默认 `EPISODE_COUNT=20`，并通过命令行覆盖 YAML。

风险:

- 直接运行脚本时实际是 20 episode，不是 YAML 里的 100 episode。
- 与之前 11/100 episode 运行结果比较时，可能误把样本规模差异当作策略变化。

相关位置:

- `run_OpenNav.yaml:15`
- `run_OpenNav.bash:10`
- `run_OpenNav.bash:38`

### P2: 文档与当前配置漂移

`Controlled-Navigation-Harness/docs/code_review_issues.md` 仍描述 `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW=True`，但当前 `run_OpenNav.yaml` 是 `False`。

风险:

- 后续按文档复现实验会误判当前 STOP 策略。
- 审查记录和实际配置不一致，会影响问题归因。

相关位置:

- `Controlled-Navigation-Harness/docs/code_review_issues.md:126`
- `Controlled-Navigation-Harness/docs/code_review_issues.md:148`
- `run_OpenNav.yaml:88`

### P2: worktree 混入大量日志删除和运行产物

当前工作区存在:

- 多个已跟踪 `logs/navigation_records`、`logs/running_log` 文件被删除。
- 多个新的 eval results、harness traces、navigation records 未跟踪。

风险:

- 如果直接提交，会把代码修改、实验数据、日志清理混在一起。
- 审查、回滚和复现实验都会变困难。

## 建议处理顺序

1. 先处理 P1 STOP 过度保守问题:
   - 将 current-view text corroboration 从硬 blocker 改成诊断字段，或只对明确高风险目标启用。
   - 对 `uncorroborated_final_target` 的样本单独统计 near-goal 成功/失败后再决定是否作为硬规则。

2. 调整 selector STOP 救回逻辑:
   - 当 `current_pano` selected candidate 高置信 allow、`arrival_evidence=True`、且 completion estimation 没有明确否定时，允许视觉证据救回 STOP。
   - 保留结构化日志，区分 `completion_gate_stop`、`visual_rescued_stop`、`visual_rejected_stop`。

3. 固定 episode count:
   - 要么把脚本默认改成与 YAML 一致。
   - 要么在实验记录里明确写 `EPISODE_COUNT=20/100`，避免 SR 曲线不可比。

4. 同步文档:
   - 更新 `code_review_issues.md` 中关于 `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW` 的描述。
   - 将当前策略、实验假设、待验证指标写清楚。

5. 清理提交边界:
   - 代码修改、文档记录、实验产物分别处理。
   - 不把大批历史日志删除和未跟踪运行目录混进同一个代码修复提交。

## 本次未覆盖

- 未运行完整 VLN eval。
- 未连接本地 Qwen/VLM 服务做真实 STOP verifier 行为回放。
- 未对最新 20 episode run 做逐 episode 复盘。

后续如果要验证修复效果，建议固定同一 episode 子集，至少比较:

- `visual_stop_allowed`
- `visual_stop_rejected`
- `allow_blockers`
- `stop_rejected_fallback`
- `selector_empty_prediction_fallback`
- `termination_reason`
- final `distance_to_goal`

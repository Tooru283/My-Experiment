# Open-Nav SR 提升方案 - 2026-06-23

## 当前判断

最近完整 ep20 日志:

```text
/root/wjj/Open-Nav/logs/navigation_records/ep20/20260623/ep20_series_qwen_siglip_local_20260623_183532_train_navigation_20260623_183557.jsonl
```

当前结果:

- 20/20 完成。
- SR: 15%。
- OSR: 35%。
- SPL: 0.1306。
- nDTW: 0.4987。
- 16 个 episode 因 `step_length_limit` 结束，4 个因 `stop_requested` 结束。
- 成功 episode: `226`, `602`, `166`。
- 其中只有 `166` 是主动 STOP 成功，`226/602` 是 step limit 结束时仍在 3m 内。

核心问题不是单一模块，而是三段链路叠加:

1. 到达目标附近后不会稳定停住，导致 OSR 不能转 SR。
2. completion / phase 对“动作是否完成”的判断不稳，会让 U1/U2 进入错误阶段。
3. selector 在 recover/late 阶段仍会被历史 landmark 和方向词带偏，继续移动离开目标。

优先级应按“能把 OSR 转成 SR”排序，因为 ep20 已经有 4 个 oracle-success 失败样本:

- `1092`: min 2.97m -> final 3.43m。
- `513`: min 0.51m -> final 3.74m。
- `265`: min 0.82m -> final 3.43m。
- `810`: min 0.04m -> final 5.18m。

这类样本只要停住，就是 SR 直接增加。

## 原则

- 不先盲目加大 step limit。`226/602` 当前靠 step limit 成功，延长步数可能会走出 3m。
- 不把 recover 阶段视觉 allow 放开。旧日志中 `748/11/804/140` 的 recover-stage current-view allow 有明显误停风险。
- 不回到 9B。当前慢的主要来源是视觉证据和多次 LLM 调用；9B 会更慢，且最近证据没有显示 9B 带来 SR 增益。
- 每个提升点必须能在 jsonl 中被离线复盘，避免只凭单次 SR 波动判断。

## 阶段 0: 固化评估与分析脚本

目标: 每轮 ep20/ep100 后自动回答“为什么 SR 没升”。

新增或整理一个分析脚本:

```text
scripts/analyze_navigation_jsonl.py
```

输出:

- SR / OSR / SPL / nDTW / mean final distance。
- `oracle_success=1 but success=0` 列表。
- 每个 episode 的 `min_distance -> final_distance`。
- STOP 成功、STOP 失败、STOP 被 V2/U2 拦截。
- `visual_evidence` 中高置信 `final_target_visible + arrival_evidence` 的 step。
- `selector_raw` 空预测次数。
- fallback / recovery 是否实际改动作。
- phase 分布和 phase 修正前后的差异。

验收:

- ep20 分析能稳定复现当前结论。
- 每次改配置/代码后先跑 ep20，输出同一张对照表。

收益:

- 不直接提升 SR，但能避免继续盲调。

## 阶段 1: 先把 OSR 转成 SR

### 1.1 current-view verify auto-stop

问题:

- 当前 U2 rescue 只在 selector 明确输出 STOP 时救。
- 但 `810` 这类样本在离目标 0.04m 时 selector 没输出 STOP，系统继续走远。

方案:

在 completion gate 之后、selector move 之前增加一个极窄的 current-view auto-stop 分支。

允许条件:

- `phase_evidence.phase == "verify"`。
- `current_step >= 3`。
- `stop_current_view_evidence` 已跑，且只看 `__current_view__` contact sheet。
- `VisualTargetVerifier` 对 current-view verdict 为 `allow`。
- `selected_candidate_verdict.verdict == "allow"`。
- `final_target_visible == true`。
- `arrival_evidence == true`。
- confidence >= 0.95。
- 无 contradictions。
- 无 hard allow blockers。
- 无 `missing_instruction_terms`。
- `uncorroborated_final_target:*` 只作为 soft warning。
- 不允许 recover/search/unknown 阶段触发。

日志:

- 新增事件 `visual_current_view_auto_stop_candidate`。
- 决策生效时记录 `visual_current_view_auto_stop`。
- 记录 phase、confidence、allow_blockers、warnings、contradictions、recent gains。

建议流程:

1. 先 `LOG_ONLY` 跑 ep20，看候选触发在哪些 episode/step。
2. 如果只命中 `513/810` 这类 near-goal 样本，再打开 decision-effect。
3. 如果命中 `748/11/804/140` 的 recover 旧误停类型，说明 phase 或条件还不够窄，不能打开。

预期收益:

- 主要提高 `oracle_success -> success` 转化。
- ep20 理想提升: `513/810` 至少一个转成功，SR +5% 到 +10%。

风险:

- 视觉模型把“看到目标”误判为“站在目标旁边”。

风险控制:

- 只允许 current-view contact sheet。
- 只允许 verify phase。
- confidence 维持 0.95。
- arrival_evidence 必须为 true。

### 1.2 weak target 的延迟 STOP 策略

问题:

- `doorway / room / hallway / area` 这类弱目标容易误停。
- 但完全保守又会错过 `226/602` 这类 doorway 成功。

方案:

保留当前 completion gate 的 weak target 最小步数限制，但增加“弱目标二次确认”:

- 第一次 weak target allow 只记录，不停。
- 如果连续 2 个 step 都在 current-view 看到同一 final target 且 arrival=true，再允许 STOP。
- 或者 selector 已连续 2 次提出 STOP，并且 current-view allow，则允许。

预期:

- 减少早停，同时让 doorway/room 类目标有可停路径。

### 1.3 near-goal hold，不盲目延长 step limit

问题:

- 当前 success 不要求显式 STOP，只看结束时距离。
- 继续走可能把已经成功的 episode 走失败。

方案:

不要全局把 10/12 改成 12/14。改成条件式:

- 如果最近 2 步距离收益为正，且 final target 没 current-view allow，可以额外给 1-2 步。
- 如果 current-view allow 且 phase verify，优先 STOP，不延长。
- 如果最近 2 步负收益，不延长，避免漂移。

实现方式:

- 加 `NAVIGATION_RUNTIME.CONDITIONAL_EXTRA_STEPS`。
- 默认为 0，先 log-only 计算哪些 episode 会获得额外步。

预期:

- 帮 `330/140/748` 这类最后仍在接近目标的样本。

风险:

- 帮一部分近失，也可能让 `226/602` 走出 3m。

## 阶段 2: 修 completion 和 phase 可靠性

### 2.1 已完成: phase 不再把 numbered None 当完成

已修:

```text
vlnce_baselines/common/opennav_ext/phase_evidence.py
```

效果:

- `1. None / 2. None / 3. None` 不再计数。
- `has not / not completed / incomplete` 等否定上下文不再计数。
- ep20 离线重算有 12 个 step phase 被纠正。

### 2.2 completion 输出 schema 化

问题:

- completion LLM 现在输出文本，解析靠字符串规则。
- `1. None` 这种格式天然容易误解析。

方案:

新增 completion JSON schema:

```json
{
  "executed_actions": [
    {"index": 1, "action": "...", "completed": true, "evidence": "..."}
  ],
  "final_action_completed": false,
  "all_actions_completed": false
}
```

保留旧文本 parser 作为 fallback。

验收:

- parse error rate < 5%。
- `None` / negative marker 不再影响 phase。

预期收益:

- 降低错误 verify 和错误 STOP。
- 对 SR 是间接提升，但对稳定性很关键。

### 2.3 phase 使用 stop gate metadata

问题:

- 当前 phase_evidence 调用时传入的 stop gate metadata 是空 `{}`。
- 这让 phase 和真正 stop gate 判断可能不一致。

方案:

- 在 completion gate 运行 `navigator.should_stop` 后，再更新一次 phase，或把 stop gate metadata 回填到 phase。
- 至少在 STOP rescue 前使用带 stop gate metadata 的 phase。

预期:

- U2 判断更贴近真正 stop gate。
- 减少“phase verify 但 stop gate 不支持”的冲突。

## 阶段 3: 提升 selector 找路能力

### 3.1 V4 context 从 log-only 变成受控 decision-effect

现状:

- V4 multimodal selector context 生成了视觉摘要，但 `LOG_ONLY=true`。
- U1 只在 search/approach 阶段使用部分 context。

方案:

先做 A/B:

- A: 当前 U1-only context。
- B: search/approach 阶段启用 V4 context，verify/recover 仍禁用。
- C: 只在 selector 原始预测为空时启用 V4 context。

约束:

- 不在 recover 阶段注入 arrival assertion。
- 不在 verify 阶段直接用候选图像诱导 STOP。

预期:

- 提升早期找目标能力，减少 `selector_empty_prediction_fallback`。

风险:

- 视觉 evidence 的 false positive 会直接带偏 selector。

已采用的第一版实现:

- V4 不直接设置 `LOG_ONLY=false`。
- V4 输出 `raw_summaries` 用于日志，同时输出 `selector_safe_summaries` 用于 selector。
- `selector_safe_summaries` 默认抑制 `final_target_visible / arrival_evidence / spatial_notes`。
- U1 优先读取 `selector_safe_summaries`。
- 只有 `search/approach` phase 会应用 safe V4 context。
- `verify/recover` 不直接注入 V4 movement summary。

### 3.2 fallback ranking 引入 phase-aware scoring

问题:

- selector 空预测时 fallback 当前按视觉目标/arrival 排序。
- 对 recover 阶段，arrival assertion 容易误导。

方案:

- search/approach: final_target_visible、matched terms 权重高。
- verify: current-view STOP 优先，movement fallback 降权。
- recover: 降低 `arrival_evidence` 权重，提高“未访问方向 / matched subgoal / positive historical gain”权重。

预期:

- 减少 recover 阶段继续被假目标吸走。

### 3.3 recovery 从“只替换无效候选”升级为“负收益重选”

现状:

- U3 只有当前 fallback candidate invalid/blocked 时才 reselect。
- 但很多错误候选是合法的，只是会走远。

方案:

谨慎扩展:

- 触发条件: 最近 2 步 gain <= 0，且 fallback rank top1 不是当前候选。
- 仅在 `failure_type=progress_drift/empty_fallback_bad`。
- 每 episode 最多 1 次。
- 不在 current-view STOP allow 时触发 movement recovery。

预期:

- 帮助 `721/643/804` 这类 drift。

风险:

- 可能用错误 rank 覆盖正确 selector。

先 log-only，再开 decision-effect。

### 3.4 selector prompt 处理 final STOP

问题:

- selector 很少主动输出 STOP。
- 即使 current environment 已接近终点，也倾向继续选方向。

方案:

修改 NAVIGATOR prompt:

- 在 `phase=verify` 时明确: 如果当前位置已满足 final stop/wait requirement，必须输出 STOP。
- 如果目标只是 visible but not beside，不要 STOP。
- 把 current subgoal 和 final target 独立放入 prompt，减少被历史 landmark 干扰。

配套:

- 必须依赖 V2/U2 校验，不能让 prompt 自由 STOP。

预期:

- 增加 selector STOP 候选，使 U2 rescue 有入口。

## 阶段 4: 视觉证据质量提升

### 4.1 current-view evidence prompt 加反例约束

问题:

- VL 模型容易把“目标可见”判断成“arrival=true”。

方案:

在 current-view prompt 中加入更强约束:

- 如果目标在远处、隔着 doorway、需要继续走，arrival=false。
- 只有当前位置就在目标旁、门口、桌旁、沙发间，arrival=true。
- 对 room/doorway/area 这类弱目标必须说明空间关系。

验收:

- 旧日志中 `748/11/804/140` recover false allow 降低。
- `513` 这类真实 near-goal allow 保留。

### 4.2 视觉 evidence 增加 relation fields

新增字段:

```json
{
  "target_distance_bin": "at|near|visible_far|not_visible",
  "relation_satisfied": true,
  "relation_notes": "..."
}
```

用于 STOP:

- `arrival_evidence=true` 且 `target_distance_bin in {at, near}`。
- relation-required 指令必须 `relation_satisfied=true`。

预期:

- 比单个 boolean 更可控。

### 4.3 候选采样策略

现状:

- `MAX_CANDIDATES=5`。
- 视觉 evidence 平均 17.3s，不能简单加到 8/12。

方案:

- search/approach 阶段最多 5。
- verify 阶段不加 movement candidate，改跑 current-view。
- selector empty 或 repeated negative gain 时，下一步临时扩大到 6 或 8。
- 只在需要 recovery 的 step 扩大视觉预算。

预期:

- 控制 runtime，同时提高关键步的候选覆盖。

## 阶段 5: 运行效率优化

问题:

- ep20 平均耗时主要来自:
  - visual_evidence: avg 17.3s。
  - completion_estimation: avg 6.95s。
  - navigator_move_to_next_vp: avg 8.39s。
  - stop_current_view_evidence: avg 5.84s。

方案:

1. 视觉证据缓存:
   - 同 episode 同 candidate image hash 重复出现时复用结果。
2. completion 低频化:
   - 前 2 步或 history 无变化时跳过 completion。
   - 或只在 movement 后 gain 较大/进入 late step 时重算。
3. current-view evidence 懒加载:
   - 只在 stop_proposal、phase verify、或 near-stop candidate 时跑。
4. prompt token 再压缩:
   - history 只保留最近 4 步 + landmarks seen summary。

收益:

- 先降运行时间，才能承受更多 ablation。

风险:

- completion 低频化可能影响 phase，需要 log-only 对照。

## 阶段 6: 实验矩阵

每次只改一个变量，先 ep20，再 ep100。

推荐顺序:

1. `Baseline-FixedPhase`
   - 当前 4B + U1 phase 修复 + 现有 U2 修复。
   - 目标: 确认 SR 是否自然超过 15%。

2. `AutoStop-LogOnly`
   - current-view verify auto-stop 只记录候选。
   - 目标: 看命中 episode 是否集中在 `oracle_success=1` 失败样本。

3. `AutoStop-Decision`
   - 打开 current-view verify auto-stop。
   - 目标: 提高 `stop_requested` 成功数。

4. `Completion-JSON`
   - completion schema 化。
   - 目标: 降低错误 verify / recover。

5. `Fallback-PhaseAware-LogOnly`
   - phase-aware fallback rank 只记录。
   - 目标: 看负收益 step 是否会选出更好候选。

6. `Recovery-NegGain-Decision`
   - 开启负收益重选，每 episode 最多 1 次。
   - 目标: 降低 drift。

7. `ConditionalExtraSteps`
   - 条件式额外 1-2 步。
   - 目标: 帮近失，不伤 `226/602`。

## 成功标准

ep20 短跑:

- SR 从 15% 提升到 >= 20% 才算有效候选。
- `stop_requested` 成功数从 1 增加到 >= 2。
- STOP false positive 不增加，尤其不能新增 final distance > 5m 的 stop_requested。
- OSR 不下降超过 5 个百分点。

ep100 长跑:

- SR 目标先定为超过原文 19%，最低应稳定 >= 22%。
- OSR 和 nDTW 不明显下降。
- 平均 episode runtime 不超过当前 1.2 倍，除非 SR 明显提升。

## 最推荐的下一步

立即跑:

```bash
cd /root/wjj/Open-Nav
EPISODE_COUNT=20 ./run_OpenNav.bash
```

跑完后先看:

- `oracle_success=1 but success=0` 是否减少。
- `stop_requested` 成功是否超过 1。
- `810/721/804/643` 的 phase 是否不再被 `None` 污染。
- U2 rescue 是否只命中 verify 阶段。

如果 ep20 仍然 SR=15% 左右，则优先实现:

1. current-view verify auto-stop log-only。
2. current-view verify auto-stop decision-effect。
3. completion JSON schema。

这三项是当前最有机会提升 SR、同时可控误停风险的路径。

# 2026-06-12 导航记录分析与修复

## 记录分析

分析目录:

```text
logs/navigation_records
```

共检查 5 组导航记录，重点查看最新 10 episode 记录:

```text
logs/navigation_records/a1_harness_qwen_siglip_local_20260612_113326_train_navigation_20260612_113357.jsonl
logs/navigation_records/a1_harness_qwen_siglip_local_20260612_113326_train_navigation_20260612_113357.log
```

主要现象:

- 最新 10 个 episode 中只有 1 个 success，且全部以 `step_length_limit` 结束。
- 成功 episode 也没有主动 STOP，而是在达到步数上限后结束。
- `completion_estimation` 已经多次判断最终 stop/wait 相关动作完成，但 selector 仍继续输出 viewpoint。
- 出现 2 次 `KeyError('error_next_vp')`，原因是异常哨兵值进入了 `radius_dict[next_vp]` / `distance_dict[next_vp]` 查表。
- 最新 10 个 episode 共 60 次移动，其中 20 次 `distance_gain_selected <= 0`，说明存在较多远离目标的动作。

结论:

- 当前主要问题不是单一视觉或 LLM 识别失败，而是 STOP 控制链路缺失。
- Prompt 明确要求完成后仍预测下一个 viewpoint，和导航任务的停止条件冲突。
- `test_decisions()` 的异常返回值不应进入环境动作执行。

## 代码修改

修改文件:

- `vlnce_baselines/common/navigator/spatialNavigator.py`
- `vlnce_baselines/common/navigator/prompts.py`
- `vlnce_baselines/common/base_il_trainer_llm.py`

关键改动:

- 新增 `STOP_CANDIDATE = "STOP"`，允许 navigator 输出 STOP。
- 修改 `NAVIGATOR` 和 `DECISION_TEST` prompt，完成指令后输出 STOP，而不是继续强制选择 viewpoint。
- 新增 `should_stop()`，当 completion estimator 判断动作完成，且当前观察能看到最终 landmark 时，直接触发 STOP。
- `move_to_next_vp()` 现在只保留合法候选或 STOP，过滤无效预测。
- `test_decisions()` 不再返回 `error_next_vp`，异常时回退到合法 fused candidate 或 observed candidate。
- trainer 中新增 `stop_decision` 记录事件。
- STOP 分支执行 Habitat STOP action:

```python
{"action": {"action": 0, "action_args": None}}
```

- 普通 waypoint 仍执行原有 action 4:

```python
{"action": {"action": 4, "action_args": {"angle": ..., "distance": ...}}}
```

- 在执行 action 前增加最后一道候选合法性检查，避免非法 `next_vp` 查表导致异常。
- STOP 后使用 `stop_requested` 作为 termination reason，并保留 `action_pre_step` / `action_post_step` 记录。
- 指标统计中补充 `collisions_` 默认值，避免早停或缺少 `current_path` 时统计阶段引用未定义变量。

## 验证

已执行:

```bash
python -m py_compile \
  vlnce_baselines/common/navigator/spatialNavigator.py \
  vlnce_baselines/common/navigator/prompts.py \
  vlnce_baselines/common/base_il_trainer_llm.py
```

结果:

```text
通过，无语法错误。
```

后续需要在 `opennav` 环境中跑 1 episode 或 10 episode 验证:

- 是否出现 `stop_decision`
- `termination_reason` 是否从 `step_length_limit` 转为 `stop_requested`
- 是否不再出现 `KeyError('error_next_vp')`
- success / SPL / nDTW 是否改善

## Thought Fusion 成本评估与调整

基于现有 navigation records 统计:

- 最新 10 episode 共 60 步，`thought_fusion` 平均耗时约 `9.22s/step`，总耗时约 `553s`。
- `thought_fusion` 约占整步耗时 `6.6%`，约占决策阶段耗时 `10.8%`。
- 但 raw predictions 几乎没有分歧: 最新记录中 58 个有效 step 的唯一候选数均为 1。
- `selector_final` 没有因为 fusion 改变动作: `final != raw majority` 为 `0/58`，`final != first raw` 也为 `0/58`。
- 主要原因是 LLM 请求固定 `temperature=0`，原先 `num_output=3` 会串行得到高度重复甚至完全相同的结果。

调整:

- `move_to_next_vp()` 从 `num_output=3` 降为单次输出。
- 移除外层 retry 循环，预测无效时交给已有合法候选 fallback 处理。
- `thought_fusion()` 增加短路逻辑: 当唯一候选数小于等于 1 时，直接返回原 thought，不再调用 LLM。
- 只有多个候选发生分歧时才执行 LLM fusion。

预期收益:

- 在当前日志分布下，每步可减少约 8 到 9 秒 fusion 时间。
- 单次预测还会额外减少原先重复生成 3 个相同 prediction 的时间。
- 行为风险较低，因为历史记录中 fusion 没有改变最终动作。

## STOP 审查后的加固

复审发现两个早停风险:

- STOP 解析原先使用子串匹配，只要预测文本中包含 `STOP` 就会接受，可能误接收 `do not STOP yet` 这类输出。
- Navigator 直接输出 STOP 时，trainer 会直接执行 STOP，没有复用 `should_stop()` 的 completion + landmark 硬门控。

修复:

- `_parse_prediction()` 改为只接受归一化后完整等于 `STOP` 的预测文本。
- 当 `test_decisions()` 返回 STOP 时，trainer 会再次调用 `should_stop()`。
- 如果 STOP gate 不通过，记录 `stop_rejected`，并从 fused candidates 中移除 STOP 后重新选择移动候选。
- 如果移除 STOP 后没有候选，继续使用已有 observed candidate fallback，避免空候选导致异常。

## STOP 拒绝路径二次保护与 landmark 判定收紧

最后一次代码审查后又补充了两处保护:

- `_final_landmark_visible()` 不再使用任意 landmark 命中即放行的 `any()` 逻辑。
- landmark 列表会先去重；如果只有 1 个有效 landmark，则要求命中 1 个；如果有多个有效 landmark，则至少命中 2 个。
- 当 STOP gate 拒绝 STOP 后，会移除 STOP 并再次调用 `test_decisions()` 选择移动候选。
- 如果第二次 fallback 仍返回 `STOP`，trainer 会强制使用 `observe_dict` 中第一个 observed movement candidate，并记录 `stop_rejected_fallback`。

涉及文件:

- `vlnce_baselines/common/navigator/spatialNavigator.py`
- `vlnce_baselines/common/base_il_trainer_llm.py`

目的:

- 降低因为普通 landmark 误命中导致的提前 STOP 风险。
- 防止 STOP gate 已拒绝 STOP 后，fallback 路径再次把 STOP 带回执行链路。

后续验证重点:

- 检查 navigation record 中 `stop_rejected_fallback` 是否出现；正常情况下应很少出现。
- 对比 `stop_rejected` 与 `stop_decision` 分布，确认 STOP gate 没有过度拒绝合理停止。
- 继续观察 `termination_reason=stop_requested` 的 episode 是否同时保持较高 success。

## V0: Qwen3.5-4B 多模态服务能力验证

目的:

- 确认当前本地 `23333` 端口是否为 Qwen OpenAI-compatible 服务。
- 验证 `/root/models/Qwen3.5-4B` 不只是配置层面支持视觉输入，实际服务接口也能接收 OpenAI-style `image_url` 多模态 payload。
- 为后续 V1 visual evidence logging 和 V2 visual TargetVerifier 提供前置依据。

检查结果:

- `http://127.0.0.1:23333/v1/models` 有服务响应，但返回模型列表为空。
- 文本 chat completion 健康检查通过。
- 图片 chat completion 健康检查通过。

文本测试:

```text
model=/root/models/Qwen3.5-4B
prompt=Reply with exactly: ok
response=ok
```

图片测试:

```text
image=/root/wjj/Open-Nav/SpatialBot/bunny/serve/examples/example_1.png
payload=OpenAI-style content list with text + image_url data URL
```

返回:

```json
{
  "contains_person": true,
  "main_objects": [
    "astronaut",
    "moon",
    "Earth",
    "beer bottle",
    "cooler",
    "space suit",
    "ladder",
    "stars"
  ]
}
```

结论:

- 当前 Qwen 服务已经具备实际可调用的图文输入能力。
- Open-Nav 当前 `llmClient.gpt_infer()` 仍只发送纯文本 prompt，尚未利用该能力。
- 下一步可以进入 V1: 在不影响导航决策的前提下，对候选视角图片做 visual evidence extraction，并写入 trace / navigation record。

## V1: Visual Evidence Logging 接入

目标:

- 在不影响导航动作选择、不影响 STOP gate 的前提下，把候选视角图像交给本地 Qwen3.5-4B 多模态服务做视觉证据抽取。
- 将视觉观察结果写入 navigation JSONL 与 harness trace，用于后续分析纯文本 observation 与真实图像证据之间的偏差。
- 为 V2 visual TargetVerifier 与 V3 VisualEvidenceMemory 做数据基础。

实现:

- 新增 `vlnce_baselines/common/opennav_ext/visual_evidence.py`。
- 新增 `VisualEvidenceLogger`，通过 OpenAI-compatible `/v1/chat/completions` 向 `http://127.0.0.1:23333/v1` 发送候选图像。
- 输入包含 instruction、decomposed actions、landmarks、candidate metadata、文本 observation 与候选 RGB 图像。
- 输出要求为 JSON，字段包括 `visible_landmarks`、`matched_instruction_terms`、`missing_instruction_terms`、`final_target_visible`、`arrival_evidence`、`spatial_notes`、`confidence`。
- 在 `base_il_trainer_llm.py` 中接入到 observation 之后、memory diagnostic 之前；仅记录 `visual_evidence` event，不参与 selector、STOP、fallback 或 env action。
- 在 `default.py` 增加 `OPENNAV_HARNESS.VISUAL_EVIDENCE` 配置，默认 `ENABLED=False`、`LOG_ONLY=True`。
- 在 `harness_config.py` 中把 `VISUAL_EVIDENCE` 纳入 A1 log-only 校验，防止误开决策影响。
- `run_OpenNav.bash` 末尾透传 `"$@"`，便于用命令行临时打开 V1/V2 配置，默认行为不变。

验证:

- `python -m py_compile` 覆盖新增模块、配置、trainer 集成文件，通过。
- 单图 smoke test 使用 `/root/wjj/Open-Nav/SpatialBot/bunny/serve/examples/example_1.png` 调用本地 23333 服务，通过。
- 在 `max_tokens=384` 下返回可解析 JSON，`parse_error=null`。
- 集成 smoke 使用 `EPISODE_COUNT=1`、`MAX_CANDIDATES=1` 启动真实导航循环；为避免长时间完整评估，在验证到第 3 个 `visual_evidence` event 后手动中断。
- 集成 smoke JSONL: `logs/navigation_records/v1_visual_evidence_smoke_20260612_182411_train_navigation_20260612_182437.jsonl`。
- 集成 smoke 逐行解析正常，共 27 条 record，其中 3 条 `visual_evidence`，steps 为 `[1, 2, 3]`，`parse_error` 全部为 `None`。

smoke test 摘要:

```json
{
  "requested_candidate_ids": ["0"],
  "parse_error": null,
  "parsed": {
    "candidates": [
      {
        "candidate_id": "0",
        "visible_landmarks": ["astronaut", "cooler"],
        "matched_instruction_terms": ["astronaut", "cooler"],
        "missing_instruction_terms": [],
        "final_target_visible": true,
        "arrival_evidence": true,
        "confidence": 0.95
      }
    ]
  }
}
```

注意:

- `max_tokens=160` 的压力测试会截断 JSON，产生 `JSONDecodeError`；当前默认保持 `384`，后续如果扩大候选数量需要同步提高 token 上限或缩短输出 schema。
- V1 仍是日志能力，不改变当前 A1 的导航行为。
- 本次 smoke 中仍观察到旧问题: `selector_raw.predictions=[]` 后 fallback 到第一个 observed candidate；这与 V1 无直接因果关系，但后续仍应纳入 selector/fallback 诊断。

下一步:

- 跑 1 到 2 个完整 episode 的 V1 harness 日志，检查每步是否稳定生成 `visual_evidence` event，并记录完整 episode metrics。
- 对比 `visual_evidence.parsed.candidates[].final_target_visible/arrival_evidence` 与现有 `stop_decision`，验证提前 STOP 是否能被视觉证据识别出来。
- 进入 V2: 将视觉证据封装为独立 TargetVerifier，但仍先以 log-only 形式跑对照。

## V2/V3: Visual TargetVerifier 与 VisualEvidenceMemory 初版

目标:

- V2: 消费 V1 `visual_evidence.parsed`，对 STOP proposal 做视觉验证，记录 `allow/reject/uncertain/not_applicable` verdict。
- V3: 将每步视觉证据维护为 episode 内记忆资产，记录目标可见、到达证据、缺失 landmark、best evidence 等状态。
- 两者均保持 log-only，不改变 selector、STOP gate、fallback 或 env action。

实现:

- 新增 `vlnce_baselines/common/opennav_ext/visual_target_verifier.py`。
- 新增 `VisualTargetVerifier`，在 `completion_gate` 与 `selector_stop_gate` 的 `should_stop()` 后记录 `visual_target_verifier` event。
- 新增 `vlnce_baselines/common/opennav_ext/visual_evidence_memory.py`。
- 新增 `VisualEvidenceMemory`，在每步 `visual_evidence` 后、`memory_diagnostic` 前记录 `visual_evidence_memory` event，并在 episode 切换时 reset。
- `default.py` 新增 `OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER` 与 `OPENNAV_HARNESS.VISUAL_EVIDENCE_MEMORY`，默认均为 `ENABLED=False`、`LOG_ONLY=True`。
- `harness_config.py` 将两个新模块纳入 A1 log-only 校验。

V2 输出重点:

```text
source
stop_proposal
verdict
final_target_visible
arrival_evidence
matched_final_landmarks
missing_final_landmarks
contradictions
best_candidate
confidence
```

V3 输出重点:

```text
records_added
history_size
target_seen_count
arrival_evidence_count
first_target_visible_step
latest_target_visible_step
best_target_evidence
seen_landmarks
matched_instruction_terms
recent_missing_instruction_terms
candidates_this_step
```

验证:

- `python -m py_compile` 覆盖 V2/V3 新模块、配置、trainer 集成文件，通过。
- 单元级 smoke 使用手工构造 V1 visual evidence:
  - candidate 只看到 `doorway/hallway`。
  - matched `door/hallway`。
  - missing `bedroom`。
  - `final_target_visible=false`。
  - `arrival_evidence=false`。
- 对 `landmarks=door, bedroom` 且 `stop_proposal=true`，V2 输出 `verdict=reject`，并记录 missing final landmark `bedroom`。
- V3 输出 `records_added=1`、`history_size=1`、`target_seen_count=0`、`recent_missing_instruction_terms=["bedroom"]`。
- 集成 smoke 使用真实导航循环打开 V1/V2/V3；验证到事件写入后手动中断，不作为完整 episode 指标。
- 集成 smoke JSONL: `logs/navigation_records/v123_visual_logonly_smoke_20260612_184828_train_navigation_20260612_184854.jsonl`。
- 集成 smoke 逐行解析正常，共 29 条 record:
  - `visual_evidence`: 2 条，`parse_error` 全部为 `None`。
  - `visual_evidence_memory`: 2 条，`history_size` 从 1 增加到 2，`target_seen_count=0`。
  - `visual_target_verifier`: 2 条，均为 `source=completion_gate`、`verdict=not_applicable`，但保留了 `missing_final_landmarks=["bedroom"]`。

下一步:

- 运行完整 episode，打开 V1/V2/V3:

```bash
EPISODE_COUNT=1 \
EXP_NAME=v123_visual_logonly_smoke \
conda run -n opennav bash run_OpenNav.bash \
  OPENNAV_HARNESS.VISUAL_EVIDENCE.ENABLED True \
  OPENNAV_HARNESS.VISUAL_EVIDENCE.MAX_CANDIDATES 1 \
  OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.ENABLED True \
  OPENNAV_HARNESS.VISUAL_EVIDENCE_MEMORY.ENABLED True
```

- 分析 `visual_target_verifier.verdict` 与 `stop_decision` 的一致性。
- 分析 `visual_evidence_memory.target_seen_count/arrival_evidence_count` 是否能解释提前 STOP 与 selector fallback。

## V 系列已完成部分审查修正

本次针对 V0-V3 已完成部分做代码审查后，修正了后续实验可能误归因的几个问题。

修正内容:

- V1 `VisualEvidenceLogger` 现在记录 `total_candidate_ids`、`requested_candidate_ids`、`sampled_all`、`sampled_candidate_count`、`total_candidate_count`。
- `run_harness_tool()` 在工具异常且 fallback 为 dict 时，会把 `skipped=true`、`reason=tool_failure`、`tool_name`、`error_type`、`error` 写入返回 payload，避免 navigation JSONL 中出现难以解释的空证据。
- V2 `VisualTargetVerifier` 改为 per-candidate verdict，不再跨 candidate 合并 `final_target_visible`、`arrival_evidence` 和 final landmark match。
- V2 只有同一个 candidate 同时满足 final target、arrival evidence、final landmarks 和 confidence 时才允许整体 `allow`。
- V2 在 `sampled_all=false` 且没有 supporting candidate 时返回 `uncertain`，不再对未覆盖全部候选的抽样证据强行 `reject`。
- V2 输出新增 `candidate_verdicts`、`supporting_candidate_id`、`visual_evidence_sample`。
- V3 `VisualEvidenceMemory` 区分 raw target 与 verified target:
  - `raw_target_seen_count`
  - `verified_target_seen_count`
  - `raw_arrival_evidence_count`
  - `verified_arrival_evidence_count`
- V3 `best_target_evidence` 只在 verified target 存在时返回；普通最高视觉证据改放入 `best_visual_evidence`。
- V3 更新时传入 final landmarks，用于计算 `matched_final_landmarks`、`missing_final_landmarks`、`verified_final_target_visible`、`verified_arrival_evidence`。
- `harness_config.py` 增加依赖校验: 启用 `VISUAL_TARGET_VERIFIER` 或 `VISUAL_EVIDENCE_MEMORY` 时，必须同时启用 `VISUAL_EVIDENCE`。

验证:

- `python -m py_compile` 覆盖 V1/V2/V3、trainer、harness config，通过。
- `git diff --check` 通过。
- 手工 smoke 1: 一个 candidate 只看到 `bedroom`，另一个 candidate 只有 `door + arrival_evidence`，V2 输出 `verdict=reject`、`supporting_candidate_id=null`，不再错误 allow。
- 手工 smoke 2: `sampled_all=false` 且抽样候选不支持 STOP 时，V2 输出 `verdict=uncertain`。
- 手工 smoke 3: V3 在 raw target 可见但 final landmark 不完整时，输出 `raw_target_seen_count=1`、`target_seen_count=0`、`best_target_evidence=null`、`best_visual_evidence` 保留普通视觉证据。
- 配置 smoke: 单独启用 `VISUAL_TARGET_VERIFIER` 且未启用 `VISUAL_EVIDENCE` 时，抛出 `ValueError`。

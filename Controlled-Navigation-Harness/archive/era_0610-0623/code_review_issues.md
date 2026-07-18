# 代码审查问题汇总

## 审查范围

该文件现在记录已审查实验记录中发现的所有问题。

来源记录:

```text
/root/wjj/Open-Nav/Controlled-Navigation-Harness/docs/experiment_record_20260612.md
/root/wjj/Open-Nav/Controlled-Navigation-Harness/docs/experiment_record_20260614.md
```

主要追踪实现文件:

- `vlnce_baselines/common/navigator/spatialNavigator.py`
- `vlnce_baselines/common/base_il_trainer_llm.py`
- `vlnce_baselines/common/opennav_ext/visual_evidence_schema.py`
- `vlnce_baselines/common/opennav_ext/visual_evidence.py`
- `vlnce_baselines/common/opennav_ext/visual_target_verifier.py`
- `vlnce_baselines/common/opennav_ext/visual_evidence_memory.py`
- `vlnce_baselines/common/opennav_ext/multimodal_selector_context.py`
- `vlnce_baselines/common/opennav_ext/visual_fallback.py`
- `vlnce_baselines/common/opennav_ext/oracle_metrics.py`
- `vlnce_baselines/config/default.py`
- `run_OpenNav.yaml`

## 重叠关系

12 号和 14 号的问题有主题重叠，但大多不是同一处代码:

- `STOP 是否应该被放行` 是共同主题。12 号主要在旧的 `spatialNavigator.should_stop()`、selector fallback 与文本完成判断；14 号主要在新的 visual target verifier / visual evidence 路径。
- `final landmark 是否可靠命中` 是共同主题。12 号问题是 landmarks 为空直接放行、弱词命中；14 号问题是 `_term_present()` 双向子串导致 `door/doorway`、`room/bedroom` 等误匹配。
- `fallback 决策是否会产生坏动作` 在两天都有表现。12 号是 STOP 被拒后或空预测后取第一个候选；14 号是视觉 verifier 的 allow/reject 结果可能影响是否触发 fallback。

因此修复时建议按功能链路合并处理，但问题记录保留到具体代码点，避免漏修。

## 当前代码复核状态

已按 2026-06-14 当前代码复核。结论:

- 问题 1、2、3、4、5、6、7、8、10 在当前源码中有直接证据。
- 问题 9 的原始表述需要收窄: 当前代码已经会写 `selector_empty_prediction_fallback` 事件；真实问题是没有 visual fallback ranker 时该事件可能只有空结果，随后 `test_decisions()` 内部 fallback 仍缺少结构化原因和排序依据。
- 问题 5 和 10 属于规则/配置风险，代码状态真实存在，但是否造成实际误拒需要后续样本统计。

## 本轮解决方案与执行状态 2026-06-15

本轮目标不是扩大实验矩阵，而是先封住会直接污染 decision-effect 的逻辑风险，并补足下一轮小样本所需的可观测字段。修改顺序为: 先写方案，再改代码，再做定向 smoke / py_compile 验证。

执行状态: 已按本节完成代码修改；仍需下一轮 10 到 12 episode 小样本验证 `visual_stop_allowed`、`stop_rejected_fallback`、`selector_empty_prediction_fallback` 和 schema warning 的实际分布。

### 2026-06-15 最新小样本复核后的修正

`v24_series_qwen_siglip_local_20260615_204351` 暴露出一个新的关键事实: `candidate_id=0` 不是稳定的 current-view，它只是正前方方向扇区的 waypoint candidate。该轮 66 个 step 中只有 34 个 step 的 waypoint candidate 集合包含 `0`，因此固定用 `selected_candidate="0"` 会产生大量伪 `selected_candidate_not_sampled:0`。

因此本轮将 `candidate_id=0` 方案降级为过渡方案，正式改为 STOP 专用 current-view evidence:

- movement selector / fallback 继续使用普通 waypoint candidate visual evidence。
- STOP verifier 在出现 STOP proposal 时单独生成 `stop_current_view_evidence`，候选 ID 固定为虚拟值 `__current_view__`。
- `__current_view__` 由当前 12 向全景 RGB contact sheet 构成，不依赖 waypoint predictor 是否产生正前方候选。
- STOP verifier 调用使用 `selected_candidate="__current_view__"` 和 `stop_evidence_mode="current_pano"`。
- `__current_view__` 不进入 selector / fallback 的 movement candidate 集合，避免虚拟候选被用于移动。
- V2 verifier 的 final landmark 条件同步改为只取最终 landmark group；中间路径地标只用于诊断，不再作为 STOP allow 必要条件或误触发条件。

新的复核标准:

- `selected_candidate_not_sampled:0` 应消失。
- STOP verifier 日志中 `stop_relevant_candidate_id="__current_view__"`。
- 每次 STOP proposal 都应有 `stop_current_view_evidence` 事件。
- `visual_stop_allowed` 必须来自 `candidate_alignment="aligned"` 的 `__current_view__`。
- `multimodal_selector_context`、`selector_empty_prediction_fallback`、`stop_rejected_fallback` 中不应把 `__current_view__` 当作可移动候选。

### 2026-06-15 `__current_view__` 复跑后的新增收紧

`v24_series_qwen_siglip_local_20260615_221911` 证明 `__current_view__` 流程已生效: 8 次 STOP proposal 均生成 `stop_current_view_evidence`，`selected_candidate_not_sampled:0` 消失，且 `__current_view__` 没有进入 movement fallback。但 episode `748` 暴露出新的误放行: 指令要求 `dining room table`，current-view 文本只稳定描述了 `table/chairs/couch/TV/staircase` 和 `living room/room`，VLM 却把普通 table 升格为 `dining room table`，导致 `visual_stop_allowed` 后仍以 `distance_to_goal=9.949` 失败。

新增方案:

- 对 `stop_evidence_mode="current_pano"` 的 STOP allow 增加 current-view 文本佐证。
- VLM 返回的 `visible_landmarks` / `matched_instruction_terms` 不能单独证明多词 final target。
- 对多词 final target，尤其包含地点/房间修饰词的目标，要求 `current_view_observation` 中同时出现修饰词和核心物体词；例如 `dining room table` 至少要有 `dining` 与 `table`，只有 `table` 不允许 STOP。
- 佐证失败时把 selected candidate 的 allow 降为 `uncertain`，记录 blocker `uncorroborated_final_target:<term>`。
- 该规则只作用于 STOP current pano allow，不影响 waypoint candidate ranking / fallback。

新增复核标准:

- episode `748` 这类 `dining room table` 但 observation 只有普通 `table` 的 STOP proposal 应变为 `visual_stop_rejected` 或 `uncertain`。
- `visual_stop_rejected.allow_blockers` 应包含 `uncorroborated_final_target:dining room table`。
- 正常包含完整 final phrase 或修饰词+核心物体词的 current-view STOP allow 不应被误挡。

### 设计原则

- STOP allow 必须使用当前位置相关证据。STOP 专用 current-view evidence 使用虚拟 `candidate_id=__current_view__`；普通 waypoint candidate 的 allow 只记录为 future-candidate evidence，不直接放行 STOP。
- STOP current-view allow 还必须被 current-view 文本观察佐证，防止 VLM 把普通物体升格为带房间/位置修饰的最终目标。
- V1 visual evidence 采样继续优先包含 `candidate_id=0`，但这只用于 waypoint evidence 覆盖率，不再作为 STOP current-view 的唯一依据。
- landmark 匹配规则在 V2 verifier 与 V3 memory 中共享，禁止继续使用双向子串。默认使用 token / phrase 边界匹配，只保留少量显式等价词，避免 `door -> doorway`、`room -> bedroom`、`hall -> hallway` 这类短词误命中。
- 旧 `should_stop()` 只消费结构化 `Executed Actions`。缺少 marker、空 landmarks、或 completion 文本中带否定上下文时，不允许 STOP。
- fallback 不能再无条件取第一个 dict key。STOP rejected fallback 和 empty-prediction fallback 优先复用 `VisualEvidenceFallbackRanker` 的排序结果；如果没有视觉证据，也要写出 `fallback_reason`、`available_candidates`、`ranked_candidates` 和最终选择依据。
- schema 修复必须区分 JSON parse 成功但 schema 不可用的情况。`parse_error=null` 不再等于 V1 可消费，日志必须包含 `schema_warnings`、raw / valid / invalid candidate count 等字段。

### 问题对应修改

1. STOP verifier 任意候选放行:
   - `completion_gate` 与 `selector_stop_gate` 调用 verifier 时传入 `selected_candidate="__current_view__"`，并增加 `stop_evidence_mode="current_pano"`。
   - 只有 STOP proposal 出现时才额外生成 `stop_current_view_evidence`，避免每步增加不必要的 VLM 成本。
   - `VisualTargetVerifier._verdict()` 在 STOP 场景下只根据 selected candidate 判定 `allow`；如果 selected candidate 未采样或不是 allow，则返回 `uncertain/reject`，不再扫描任意 allow candidate。
   - `current_pano` allow 必须通过 current-view observation 文本佐证；多词 final target 不能只依赖 VLM 的 matched terms。
   - 日志保留 `future_supporting_candidate_id` / `supporting_candidate_id`，用于分析“前方可见但当前位置不应 STOP”的样本。

2. 旧 STOP gate landmark 判定:
   - `spatialNavigator._final_landmark_visible()` 改为返回结构化 gate detail。
   - landmarks 为空时返回 false，reason=`empty_landmarks`。
   - 只取 landmarks 列表中的最终 landmark group 作为 STOP gate 的视觉条件；中间 landmark 只记录到 `all_landmark_terms`，不再单独触发 STOP allow。
   - 用共享 phrase matcher 判断 observation 是否命中 final landmark，记录 `required_landmark_terms`、`matched_landmark_terms`、`landmark_gate_reason`。

3. final landmark 匹配过宽:
   - 新增共享工具模块，提供 `term_present()` / `missing_terms()` / `matched_terms()`。
   - `visual_target_verifier.py` 和 `visual_evidence_memory.py` 统一改用该工具。
   - 默认不把 `door` 视为 `doorway`、不把 `room` 视为 `bedroom`、不把 `hall` 视为 `hallway`。

4. action completion 自由文本误判:
   - `estimate_completion()` 缺少 `Executed Actions` marker 时返回空 executed section，并记录 marker 缺失。
   - `should_stop()` 将 executed section 解析为 action 条目 / 编号集合，精确匹配 decomposed action，不再在完整自由文本上做子串判断。
   - 对 `not executed`、`not completed`、`not done` 等否定短语命中的 action 不计入 completed。

5. generic STOP blocker 过宽:
   - 本轮确认该 blocker 过宽，当前 `run_OpenNav.yaml` 与默认配置均保持 `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW=False`。
   - generic final terms 继续保留为可选诊断/ablation 开关，不作为当前默认 STOP hard blocker。

6. schema 合法 JSON 漂移:
   - `visual_evidence_schema.py` 增加 schema stats / warnings。
   - `VisualEvidenceLogger` 输出 `schema_warnings`、`schema_error`、`raw_candidate_count`、`valid_candidate_count`、`invalid_candidate_count`、`normalized_from_root_type`。
   - 下游 verifier / fallback / memory 继续通过统一 schema 工具读取候选。

7. STOP 被拒后的 fallback:
   - `base_il_trainer_llm.py` 中 STOP rejected fallback 改用视觉 ranker 对非 STOP candidates 重新排序。
   - 如果 ranker 不可用，则调用 navigator 内部 fallback，但内部 fallback 必须返回 metadata；任何强制移动都写 `stop_rejected_fallback` 结构化事件。

8. `test_decisions()` 错误计数:
   - exception 分支返回递增后的 `error_number`。
   - `test_decisions()` 增加可选 metadata 返回或缓存字段，记录 fallback source / reason / candidates。

9. 空预测 fallback 结构化记录:
   - `selector_empty_prediction_fallback` 无论 ranker 是否可用，都写出 reason、source_stage、available_candidates、selected_candidate、ranked_candidates。
   - navigator 内部 `_fallback_candidate()` 也返回 `fallback_metadata`，避免 fallback event 为空。

10. 默认配置一致性:
   - 默认 `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW=False`，与当前 V2 decision-effect 实验一致。
   - 文档继续提醒如果后续打开该配置，必须统计 generic blocker 的 near-goal case，避免有效 STOP 漏拒。

## P0 / P1 问题

### 1. STOP verifier 使用任意候选 allow 放行 STOP

来源: 2026-06-14

当前代码状态: 确认存在。

位置:

- `base_il_trainer_llm.py:1202`
- `base_il_trainer_llm.py:1433`
- `visual_target_verifier.py:463`
- `visual_target_verifier.py:541`

现象:

- `completion_gate` 调用 `record_visual_target_verifier()` 时没有传 `selected_candidate`。
- `selector_stop_gate` 调用 verifier 时传 `selected_candidate=None`。
- `VisualTargetVerifier` 在没有 selected candidate 时，只要任意候选 `verdict=allow`，整体 STOP verifier 就可能返回 `allow`。

风险:

- STOP 应该验证当前位置是否已经到达目标，而不是验证某个可移动候选方向是否看到了目标。
- 当前逻辑可能出现“目标在前方候选视角中可见，因此原地 STOP 被允许”的误停。
- 该问题会直接影响 `visual_stop_allowed`，属于 decision-effect 路径风险。

建议:

- STOP allow 只基于当前位置/当前视角证据，例如固定使用 `candidate_id=0` 或新增专门的 current-view visual evidence。
- 非当前位置候选证据最多用于 `reject` 或 `uncertain`，不要直接用于 allow STOP。
- 如果保留候选级 allow，需要在日志中区分 `current_view_allow` 与 `future_candidate_allow`。

验证项:

- 构造 `candidate_id=1` allow、`candidate_id=0` reject 的 STOP proposal，期望 verifier 不返回 `allow`。
- 检查后续日志中 `visual_stop_allowed.supporting_candidate_id` 是否只来自当前位置证据。

### 2. 旧 STOP gate 的 landmark 判定过脆

来源: 2026-06-12

当前代码状态: 确认存在。

位置:

- `spatialNavigator.py:110`
- `spatialNavigator.py:118`
- `spatialNavigator.py:120`
- `spatialNavigator.py:128`

现象:

- `_final_landmark_visible()` 在提取不到 landmark words 时直接返回 `True`。
- 多 landmark 时只要求命中两个词，且基于普通单词集合命中，不理解目标短语或最终目标。
- 对 “door before bedroom” 这类指令，可能把中间地标和最终目标混在一起处理。

风险:

- landmark 提取失败时，STOP gate 等于失去最后一层视觉/文本约束。
- 正常提取时，又可能要求看到不该作为当前位置目标的词，造成漏停。
- 该问题与 14 号的 final landmark 匹配过宽是同类问题，但发生在旧 navigator gate。

建议:

- landmarks 为空时不要直接允许 STOP，应返回不确定或不允许。
- 区分最终目标短语、中间路径地标和普通修饰词。
- 日志中记录 `matched_landmark_words`、`required_landmark_words` 和 `landmark_gate_reason`。

验证项:

- 空 landmarks + STOP proposal 时，`should_stop()` 不应返回 True。
- `door before bedroom` 只观察到 `door` 时，不应被当成已经到达 `bedroom`。
- landmark 命中结果需要能从 JSONL 日志中复盘。

### 3. final landmark 匹配过宽，`door` 会匹配 `doorway`

来源: 2026-06-14

当前代码状态: 确认存在。

位置:

- `visual_target_verifier.py:66`
- `visual_target_verifier.py:292`
- `visual_evidence_memory.py:50`
- `visual_evidence_memory.py:142`

现象:

`_term_present()` 使用双向子串:

```text
term_norm in value_norm or value_norm in term_norm
```

因此:

- `door` 会命中 `doorway`
- `room` 会命中 `bedroom`
- `hall` 会命中 `hallway`

风险:

- V2 verifier 会低估 `missing_final_landmarks`，从而更容易形成 allow candidate。
- V3 memory 会把并不完整的候选记录为 `verified_final_target_visible`，污染 episode memory。
- 已在日志中观察到候选 `missing_instruction_terms` 包含 `doorway`，但 `matched_final_landmarks` 仍包含 `doorway` 的情况。

建议:

- 改为 token/phrase 边界匹配。
- 对常见同义或形态变化建立显式映射，而不是双向子串。
- 至少避免短词反向命中长词，如 `door` 不应自动匹配 `doorway`。

验证项:

- `door` 不匹配 `doorway`。
- `room` 不匹配 `bedroom`。
- `hall` 不匹配 `hallway`，除非显式同义词表允许。
- V2/V3 使用同一套匹配函数，避免两边结果漂移。

### 4. action completion 使用自由文本子串匹配，可能误判已完成

来源: 2026-06-12

当前代码状态: 确认存在。

位置:

- `spatialNavigator.py:77`
- `spatialNavigator.py:82`
- `spatialNavigator.py:139`
- `spatialNavigator.py:144`

现象:

- `estimate_completion()` 如果没有找到 `Executed Actions` 标记，会返回完整 LLM response。
- `should_stop()` 将估计文本整体 normalize 后，直接用 `action in normalized_estimation` 判断动作是否完成。
- Thought、解释句、否定句里提到某个 action，也可能被当成该 action 已执行。

风险:

- “not executed / not completed” 附近的 action 仍可能被子串命中为已完成。
- LLM 输出格式轻微漂移时，STOP gate 容易把解释文本当结构化结果消费。
- 已在 2026-06-12 日志中观察到估计文本说明后续动作未执行，但仍触发 stop decision 的样本。

建议:

- 只解析 `Executed Actions` section；缺少 marker 时不要允许 STOP，或返回 uncertain。
- 将 executed actions 解析为列表后做精确规范化匹配，不要在完整 response 上做子串搜索。
- 对 action 附近的否定标记做局部检测。

验证项:

- Thought 中提到但明确未执行的 action，不应计入 completed actions。
- 缺少 `Executed Actions` 标记时，`should_stop()` 不应返回 True。
- `Executed Actions: 1` 不能让 action 2/3 被判为已完成。

## P1 / P2 问题

### 5. generic STOP blocker 过宽，可能拒绝有效 STOP

来源: 2026-06-14

当前代码状态: 规则真实存在，实际误拒率需要日志量化。

位置:

- `visual_target_verifier.py:83`
- `visual_target_verifier.py:472`
- `visual_target_verifier.py:507`
- `run_OpenNav.yaml:78`

现象:

默认泛化词包含:

```text
door, doorway, room, stairs, hallway, floor, archway ...
```

当前运行配置中:

```yaml
BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW: False
REJECT_ON_UNCERTAIN: True
```

该 blocker 当前默认关闭；如果后续再次打开，当 final landmarks 全部落入 generic terms 时，allow 会被降级为 `uncertain`，随后被 `REJECT_ON_UNCERTAIN=True` 拦截。

风险:

- 该规则能挡住 `floor / archway` 类误停，但也可能挡住大量有效目标，例如 `doorway`、`stairs`、`room`。
- 如果目标本身就是 VLN 中常见但有效的终点词，可能造成漏停。

建议:

- 收窄 generic terms，优先保留 `floor`、`area`、`hall`、`hallway` 这类低特异词。
- 对组合词做规则区分，例如 `stairs + doorway` 不应无条件视为 generic。
- 日志中单独统计被 `generic_final_terms` blocker 拦截后的 oracle distance，判断误拒率。

验证项:

- `floor, archway` 仍应进入 blocker。
- `stairs, doorway` 不应默认进入 blocker，除非缺少 arrival/current-view 证据。
- 后续小样本统计 `allow_blockers` 中 `generic_final_terms` 的成功/失败分布。

### 6. schema 修复仍会静默吞掉下一类合法 JSON 漂移

来源: 2026-06-14

当前代码状态: 确认存在。

位置:

- `visual_evidence_schema.py:4`
- `visual_evidence_schema.py:17`
- `visual_evidence.py:181`

现象:

当前 schema 工具兼容了:

- `{"candidates": [...]}`
- `[...]`

但如果模型返回合法 JSON 且不符合这两种形状，例如:

- `{"candidate": [...]}`
- `{"results": [...]}`
- `{"candidates": {"0": {...}}}`
- list 中元素不是 dict

当前会得到空候选列表，但 `parse_error` 仍可能为 `None`。

风险:

- 会再次出现“JSON 解析成功，但下游 `candidate_evidence_count=0`”的隐性失败。
- 单看 `parse_error=null` 容易误判 V1 输出正常。

建议:

- 增加 `schema_error` 或 `schema_warnings`。
- 记录 `normalized_from_root_type`、`raw_candidate_count`、`valid_candidate_count`、`invalid_candidate_count`。
- 如果 `parse_error is None` 但有效候选为空，应明确写日志标记。

验证项:

- 合法但非预期 schema 返回时，日志中必须有 schema warning。
- `candidate_evidence_count=0` 时能区分 no image、parse error、schema error、empty model output。

### 7. STOP 被拒后的 fallback 选择第一个候选，顺序依赖过强

来源: 2026-06-12

当前代码状态: 确认存在。

位置:

- `base_il_trainer_llm.py:1471`
- `base_il_trainer_llm.py:1477`
- `base_il_trainer_llm.py:1497`
- `base_il_trainer_llm.py:1508`

现象:

- selector 选中 STOP 后，如果 stop gate 拒绝，会重新对非 STOP candidate 做 `test_decisions()`。
- 如果 fallback 仍返回 STOP，代码直接使用 `next(iter(selector_observe_dict.keys()))` 强制移动。
- 该候选只依赖 dict 顺序，不依赖视觉证据、几何收益或历史状态。

风险:

- STOP 被正确拒绝后，系统可能立刻执行一个无根据的移动动作。
- 2026-06-12 日志中已观察到 fallback candidate 带来负 distance gain 的样本。

建议:

- STOP rejection fallback 使用视觉 fallback ranker、selector 非 STOP 排名或 oracle-free 的几何/novelty 打分。
- 如果只能强制移动，应记录 `fallback_reason`、候选排序依据和候选分数。
- 不应把第一个 dict key 当成默认安全动作。

验证项:

- STOP-only 或 fallback 返回 STOP 时，应选择有明确排序理由的非 STOP candidate。
- JSONL 中应能看到 `stop_rejected_fallback.reason`、`ranked_candidates` 和最终选择依据。

### 8. `test_decisions()` 错误计数被重置

来源: 2026-06-12

当前代码状态: 确认存在。

位置:

- `spatialNavigator.py:314`
- `spatialNavigator.py:316`
- `spatialNavigator.py:318`
- `spatialNavigator.py:319`

现象:

- exception 分支里 `error_number += 1`。
- 但返回时写成 `return next_vp, thought, 0`。

风险:

- 上层永远拿不到累计错误数。
- 空预测、无效预测、fallback 频率会被低估。
- 后续基于 `error_number` 的熔断或诊断无法生效。

建议:

- 返回递增后的 `error_number`。
- 将 fallback 类型和错误原因写入 navigation JSONL。

验证项:

- 连续两次 invalid decision 后，上层 `error_number` 应累计为 2。
- JSONL 中能区分 empty fused thought、invalid candidate、LLM parse failure。

### 9. 空预测 fallback 的结构化记录不完整

来源: 2026-06-12

当前代码状态: 部分确认。当前代码会写 `selector_empty_prediction_fallback` 事件，但内部 fallback 的原因、排序依据和最终选择来源仍不足。

位置:

- `base_il_trainer_llm.py:1336`
- `base_il_trainer_llm.py:1368`
- `spatialNavigator.py:283`
- `spatialNavigator.py:318`

现象:

- selector raw prediction 为空或 fusion 后为空时，`test_decisions()` 内部会进入 exception fallback。
- 当前上层会写 `selector_empty_prediction_fallback` 事件；如果 `visual_fallback_ranker` 存在且成功返回，会有一定结构化结果。
- 但如果 `visual_fallback_ranker` 不存在或没有选出候选，`fallback_results` 可能是空 dict，后续仍会进入 `test_decisions()` 的内部 fallback。
- 内部 `_fallback_candidate()` 可能选择第一个 observed candidate，但返回值没有携带 `fallback_reason`、排序依据或候选列表。

风险:

- 误动作难以归因到 selector 空输出、fusion 过滤、decision test 失败还是 fallback 排序。
- 小样本 case study 会低估 selector/fusion 阶段的问题。

建议:

- `test_decisions()` 返回 fallback metadata，或上层写 `selector_empty_prediction_fallback`。
- 记录 `fallback_reason`、`available_candidates`、`selected_candidate`、`source_stage`。

验证项:

- fusion 结果为空时，navigation JSONL 必须有结构化 fallback event，且不能只有空 payload。
- fallback event 能区分 visual fallback ranker 与 navigator 内部 `_fallback_candidate()`。
- navigator 内部 fallback 返回的候选需要记录 `fallback_reason` 和选择依据。

## 配置一致性问题

### 10. 默认配置与实验文档行为不一致

来源: 2026-06-14

当前代码状态: 配置差异确认存在。

位置:

- `vlnce_baselines/config/default.py:93`
- `run_OpenNav.yaml:87`

现象:

- 默认配置中 `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW=False`。
- 当前实验运行配置中该项也为 `False`。
- 该项已经从当前默认 hard blocker 降级为可选 ablation 开关。

风险:

- 后续如果重新打开该项，可能从“挡早停”变成“漏正确停”，需要配套统计 near-goal case。

建议:

- 明确保留默认关闭时，在文档中写清楚“该修复依赖实验配置打开”。
- 或将默认值改为 `True`，并在配置注释中说明该规则的副作用和适用场景。

## 建议修复顺序

1. 先修 STOP allow 证据来源，只允许当前位置/当前视角证据直接放行 STOP。
2. 统一旧 `should_stop()` 与新 V2/V3 verifier 的 final landmark 匹配策略。
3. 修 `estimate_completion()` / `should_stop()` 的结构化 parsing，禁止在完整自由文本上做 action 子串完成判断。
4. 重构 STOP rejected / empty prediction fallback，替换第一个 dict key 选择逻辑。
5. 缩窄或重构 `generic_final_terms` blocker。
6. 增加 visual evidence schema warning 字段和异常合法 JSON 的 smoke/regression。
7. 修复 `test_decisions()` 的 `error_number` 返回值，并补充 fallback observability。
8. 补充配置默认值说明或调整默认值。

## 最小回归测试清单

- 裸数组 V1 parsed 可规范化为 `{"candidates": [...]}`。
- `candidate_evidence_count` 对 dict/list 两种 V1 输出均大于 0。
- 合法 JSON 但 schema 不符合预期时，日志出现 `schema_error` 或 `schema_warnings`。
- `candidate_id=1 allow` 但当前位置不 allow 时，STOP verifier 不返回 `allow`。
- 空 landmarks + STOP proposal 时，`should_stop()` 不返回 True。
- `door` 不匹配 `doorway`。
- `room` 不匹配 `bedroom`。
- `hall` 不匹配 `hallway`，除非显式同义词表允许。
- Thought 中出现 “not executed action X” 时，action X 不计入 completed actions。
- 缺少 `Executed Actions` marker 时，`should_stop()` 不允许 STOP。
- STOP 被拒后 fallback 不使用第一个 dict key 作为无条件默认动作。
- 连续 invalid decision 后，`error_number` 能正确累计。
- fusion 结果为空时，JSONL 有结构化 fallback event。
- `floor, archway` 的 generic blocker 仍生效。
- `stairs, doorway` 不因 generic blocker 被无条件拒绝。

## 后续小样本需要关注

下一轮 10 到 12 episode 小样本建议额外统计:

```text
visual_stop_allowed.supporting_candidate_id
visual_stop_allowed.stop_relevant_candidate_id
visual_target_verifier.allow_blockers
visual_target_verifier.matched_final_landmarks
visual_target_verifier.missing_final_landmarks
visual_evidence_memory.verified_target_seen_count
visual_evidence.schema_error / schema_warnings
should_stop.landmark_gate_reason
should_stop.completed_actions
selector_empty_prediction_fallback.reason
stop_rejected_fallback.reason
test_decisions.error_number
```

尤其需要 case study:

- `completion_gate` 来源的 STOP allow；
- `selector_stop_gate` 来源的 STOP allow；
- 被 `generic_final_terms` blocker 拦截但最终距离很近的样本；
- `door/doorway`、`room/bedroom`、`hall/hallway` 相关样本；
- STOP 被拒后 fallback candidate 的 distance gain；
- action completion 文本里出现否定句但仍触发 stop decision 的样本。

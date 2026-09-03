# 2026-06-23 Open-Nav 4B 回切与 STOP 修复日报

## 背景

今日将本地 Qwen 服务与 Open-Nav 配置从 9B 回切到 4B，并分析最新 20 episode 小样本运行。

记录文件:

```text
/root/wjj/Open-Nav/logs/navigation_records/ep20/20260623/ep20_series_qwen_siglip_local_20260623_183532_train_navigation_20260623_183557.jsonl
```

服务状态:

- `23333` 已确认返回 `/root/models/Qwen3.5-4B@main`。
- `run_OpenNav.bash` 已加入 LLM preflight，模型不一致时会在进入 Habitat 前失败。

## ep20 运行结果

总体指标:

- episode: 20/20
- wall-clock: 18:36:38 到 20:54:57，约 2h18m
- SR: 3/20 = 15%
- OSR: 7/20 = 35%
- SPL: 0.1306
- nDTW: 0.4987
- mean distance_to_goal: 5.29m

成功情况:

- `166` 是真正 `stop_requested` 成功。
- `226`、`602` 是 step limit 时仍在 3m 内，被指标算 success，但策略没有主动 STOP。

## 问题定位

### 1. U2 visual rescue 被收得过死

现象:

- `stop_verification`: 222 次。
- `allow_rescue`: 0 次。

近点漏停:

- `513`: 最小距离 0.51m，最终 3.74m。
- `265`: 最小距离 0.82m，最终 3.43m。
- `810`: 最小距离 0.04m，最终 5.18m。
- `1092`: 最小距离 2.97m，最终 3.43m。

典型案例 `513 step 3`:

- phase: `verify`
- current-view visual verdict: `allow`
- `final_target_visible=True`
- `arrival_evidence=True`
- confidence: 0.95
- missing instruction terms: []
- 被旧 U2 阻断原因:
  - `rescue_confidence_below_threshold:0.95<0.99`
  - `allow_warnings_present`
  - `before_rescue_min_step:3<8`

判断:

- 旧策略能防误停，但几乎关死了真正有价值的视觉 STOP rescue。

### 2. completion gate 对通用目标词误停

4 次真正 STOP 中，只有 `166` 成功。

失败 STOP:

- `371`: final term 为 `door`，最终距离 3.08m。
- `116`: final term 为 `sink`，最终距离 5.40m。
- `1087`: final term 为 `door`，最终距离 5.40m。

判断:

- `door`、`sink` 这类通用词只表示“看到了物体”，不足以证明到达了空间关系目标。
- completion auto-stop 需要比 selector rescue 更保守。

### 3. recover 阶段视觉 arrival 有远距离误判

离线回放显示，如果只把 rescue 阈值从 0.99 放到 0.95，会新增 5 次 rescue:

- `513 step 3`: 合理，verify 阶段，近点。
- `513 step 9`: recover 阶段，已经漂移。
- `748 step 6/9`: recover 阶段，距离仍远。
- `804 step 10`: recover 阶段，距离仍远。

判断:

- 视觉模型会把远处可见的 `dining room table` 误报为 arrival。
- rescue 必须限定在 `verify` 阶段，不能在 recover 阶段直接 STOP。

## 今日修复

### 1. 放开 verify 阶段的当前视角视觉 STOP rescue

修改文件:

```text
vlnce_baselines/common/opennav_ext/stop_evidence_verifier.py
vlnce_baselines/common/base_il_trainer_llm.py
run_OpenNav.yaml
vlnce_baselines/config/default.py
```

策略:

- `RESCUE_CONFIDENCE_THRESHOLD`: 0.99 -> 0.95
- `RESCUE_MIN_STEP`: 8 -> 3
- `RESCUE_ALLOW_PHASE_VERIFY`: false -> true
- 仅允许 `phase == verify` 的 rescue。
- current-view `__current_view__` 必须:
  - visual verdict = allow
  - selected candidate verdict = allow
  - `final_target_visible=True`
  - `arrival_evidence=True`
  - no hard blockers
  - no contradictions
  - no missing instruction terms

soft warning 处理:

- `uncorroborated_final_target:*` 不再单独阻断 rescue。
- 其他 warning 仍阻断 rescue。

离线回放结果:

- 新增 rescue: 1 次。
- 目标: `513 step 3`。
- `748/804/140/11` 等 recover 阶段高置信误判被 `phase_not_verify` 阻断。

### 2. 收紧 completion auto-stop 的通用目标词

修改文件:

```text
vlnce_baselines/common/opennav_ext/visual_target_verifier.py
run_OpenNav.yaml
vlnce_baselines/config/default.py
```

策略:

- `BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW`: false -> true
- 该阻断只作用于 `completion_gate`。
- selector STOP rescue 不使用该通用词硬阻断。
- 通用目标词集中保留结构/弱目标词，如:
  - `door`
  - `doorway`
  - `room`
  - `hallway`
  - `sink`
  - `stairs`
- 移除 `lamp/table/bed/chair/couch`，避免误伤 `166` 这类具体物体目标。

预期效果:

- 阻断 `door/sink` completion false positive。
- 保留 `lamp` 等较具体物体的正常 STOP 可能性。

### 3. 保留 4B 运行与 preflight

修改文件:

```text
run_OpenNav.bash
run_OpenNav.yaml
vlnce_baselines/common/navigator/api.py
```

当前模型:

- `OPENNAV_LLM_MODEL=/root/models/Qwen3.5-4B`
- `--llm Qwen/Qwen3.5-4B`
- `VISUAL_EVIDENCE.MODEL=/root/models/Qwen3.5-4B`

preflight:

- 启动前请求 `/v1/chat/completions`。
- 如果服务端仍 pinned 到 9B，会立即报错，不再进入 Habitat 后失败。

## 验证

已执行:

```text
python3 -m py_compile stop_evidence_verifier.py visual_target_verifier.py base_il_trainer_llm.py default.py
bash -n run_OpenNav.bash
```

结果:

- 语法检查通过。
- 运行脚本语法检查通过。

逻辑检查:

- verify 阶段、current-view、confidence=0.95、soft warning 的样例允许 rescue。
- recover 阶段同样高置信样例被 `phase_not_verify` 阻断。
- completion gate 的 `door` 泛目标被 `generic_final_terms` 阻断。
- selector gate 的同类视觉 rescue 不被 generic final term 阻断。

## 下一步

建议先跑:

```bash
cd /root/wjj/Open-Nav
EPISODE_COUNT=20 ./run_OpenNav.bash
```

观察重点:

- `allow_rescue` 是否从 0 增加到少量高质量样例。
- `visual_stop_allowed` 的 false positive 是否下降。
- `stop_requested` 成功数是否超过 1。
- SR 是否从 15%-17% 区间上移。

如果 ep20 结果稳定，再跑 ep100。

## 21:31 完整链路复查与 U1 phase 修复

触发问题:

- 复查 ep20 完整链路时发现 U1 `phase_evidence` 会把 completion 输出中的编号 `None` 行误计为已完成动作。
- 典型错误样例:
  - `1. None`
  - `2. None`
  - `3. None`
- 该错误会让 episode 过早进入 `verify`，影响 selector 上下文、U2 STOP rescue 条件和 recovery 判断。

修复文件:

```text
vlnce_baselines/common/opennav_ext/phase_evidence.py
```

修复内容:

- 编号 completion 行优先按行解析。
- `None / no action / not executed / not completed / not done / not yet / has not / have not / incomplete / cannot be considered / cannot be done` 不再计入已完成动作。
- 非编号文本匹配原动作时，也检查动作附近的否定上下文，避免因为动作词出现而误计。

离线复盘影响:

- 用 6/23 ep20 日志重算 phase 后，有 12 个 step 的 phase 被纠正。
- 主要影响 episode:
  - `810`: 多个 `1. None / 2. None / 3. None` step 从 `verify` 改为 `search/unknown`。
  - `721`: `None` 和未完成最终动作导致的 `verify` 被改为 `search/unknown/approach`。
  - `804`, `643`: 早期 `None` 不再误判 verify。

验证:

```text
/root/anaconda3/envs/opennav/bin/python -m py_compile vlnce_baselines/common/opennav_ext/phase_evidence.py
```

结果:

- 语法检查通过。
- 本地样例验证通过:
  - `1. None / 2. None / 3. None` -> completed_count=0
  - `1. not executed / 2. not completed` -> completed_count=0
  - 完整正向动作列表 -> completed_count 正常计数

预期效果:

- 减少过早 `verify`。
- 降低错误 STOP 上下文和错误 U2 rescue 触发概率。
- 对 `810/721/804/643` 这类被 completion `None` 污染 phase 的样本，下一轮应更少发生“还没完成却按终点验证”的偏移。

## 22:05 V4 selector-safe 重新接入方案落地

背景:

- 直接打开 `MULTIMODAL_SELECTOR_CONTEXT.LOG_ONLY=false` 风险较高。
- 旧 ep20 中 recover 阶段存在多个视觉 `target/arrival` false positive，不能把这类字段直接喂给 movement selector。

实现策略:

- V4 继续保持 `LOG_ONLY=true`，不独立替换 selector 输入。
- V4 同时产出:
  - `raw_summaries`: 保留原始 `target/arrival/conf/note`，只用于日志诊断。
  - `selector_safe_summaries`: 去掉 `final_target_visible/arrival_evidence/spatial_notes`，用于 selector。
- U1 `PhaseAwareEvidenceScaffolder` 优先读取 `selector_safe_summaries`。
- U1 只在 `search/approach` 阶段把 safe V4 摘要拼入 selector 输入。
- `verify/recover` 阶段不直接注入 V4 movement summary；STOP 仍由 V2/U2 current-view verifier 负责。

修改文件:

```text
vlnce_baselines/common/opennav_ext/multimodal_selector_context.py
vlnce_baselines/common/opennav_ext/evidence_scaffolder.py
vlnce_baselines/common/base_il_trainer_llm.py
vlnce_baselines/config/default.py
run_OpenNav.yaml
```

新增配置:

```yaml
MULTIMODAL_SELECTOR_CONTEXT:
  DECISION_MODE: phase_gated_u1
  APPLY_PHASES: [search, approach]
  SUPPRESS_TARGET_ARRIVAL_FOR_SELECTOR: true
  MIN_CONFIDENCE_FOR_TARGET_HINT: 0.9
```

预期效果:

- 在 search/approach 阶段增强候选方向的 landmark match/missing 信息。
- 避免 `arrival=1` 直接影响 movement selector。
- 保留 raw V4 诊断能力，便于后续分析 V4 是否帮到 selector。

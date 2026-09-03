# 2026-06-10 Open-Nav 实验记录

## 目标

今天的目标是把 Open-Nav baseline 从外部 OpenAI API 切换到本机 Qwen 模型，并完成最小 1 episode smoke test。同时补齐导航过程记录，便于后续分析每一步观察、LLM 判断、动作选择和最终指标。

## 环境

- 项目路径: `/root/wjj/Open-Nav`
- Open-Nav 运行环境: `opennav`
- Python: `3.8.20`
- Qwen 服务环境: `qwen35-serve`
- Qwen API: `http://127.0.0.1:23333/v1`
- Qwen 模型路径: `/root/models/Qwen3.5-4B`
- SigLIP 模型路径: `/root/models/google-siglip-so400m-patch14-384`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Habitat 渲染: `EGL_DEVICE_ID=0`

## 前序阻塞

最开始执行 A0 baseline smoke test 时，配置为:

```bash
OPENNAV_HARNESS.ENABLED=False
OPENNAV_HARNESS.ENABLE_HARNESS_LOGGING=False
EVAL.EPISODE_COUNT=1
EVAL.SPLIT=val_unseen
```

失败点出现在 Habitat-Sim 子进程初始化 EGL/OpenGL context:

```text
GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue
ConnectionResetError: [Errno 104] Connection reset by peer
```

结论是 baseline 已进入 simulator 初始化阶段，但失败属于本机 Habitat-Sim / EGL / OpenGL 图形栈问题，不是 Harness 代码路径导致。后续图形栈恢复后，baseline 能继续推进到观察和导航循环。

## 本地 Qwen API 映射

修改文件:

- `vlnce_baselines/common/navigator/api.py`

关键行为:

- 当 `--llm` 使用 `Qwen/...` 时，不再走外部 OpenAI API。
- 默认 base URL 固定为 `http://127.0.0.1:23333/v1`。
- 默认模型固定为 `/root/models/Qwen3.5-4B`。
- 支持环境变量覆盖:
  - `OPENNAV_LLM_BASE_URL`
  - `OPENNAV_LLM_MODEL`

当前映射:

```text
DEFAULT_LOCAL_QWEN_BASE_URL = "http://127.0.0.1:23333/v1"
DEFAULT_LOCAL_QWEN_MODEL = "/root/models/Qwen3.5-4B"
```

服务启动方式:

```bash
conda run -n qwen35-serve transformers serve /root/models/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 23333 \
  --device cuda:0 \
  --dtype bfloat16 \
  --reasoning off \
  --log-level info
```

当前服务状态:

```text
PID: 66170
URL: http://127.0.0.1:23333/v1
```

已验证 Open-Nav 侧初始化:

```text
Initialized LLM client with model: /root/models/Qwen3.5-4B
```

运行日志中多次出现:

```text
POST http://127.0.0.1:23333/v1/chat/completions "HTTP/1.1 200 OK"
```

## SigLIP 本地模型映射

卡在 `Get Observation` 的主要原因是 SpatialBot3B 尝试在线下载:

```text
google/siglip-so400m-patch14-384
```

使用 Hugging Face 镜像站下载到本地:

```text
https://hf-mirror.com/google/siglip-so400m-patch14-384
```

本地目录:

```text
/root/models/google-siglip-so400m-patch14-384
```

权重文件已验证:

```text
model.safetensors
size: 3511950624 bytes
```

修改文件:

- `SpatialBot3B/modeling_bunny_phi.py`

关键行为:

- `google/siglip-so400m-patch14-384` 自动映射到本地目录。
- 支持环境变量覆盖:
  - `OPENNAV_SIGLIP_PATH`

当前映射:

```text
DEFAULT_LOCAL_SIGLIP_PATH = "/root/models/google-siglip-so400m-patch14-384"
```

1 episode 测试时启用了离线模式:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
OPENNAV_SIGLIP_PATH=/root/models/google-siglip-so400m-patch14-384
```

## 导航记录日志

修改文件:

- `vlnce_baselines/common/base_il_trainer_llm.py`

新增记录:

- episode start
- step start
- observation
- history review
- completion estimation
- selector raw
- selector fused
- selector final
- action pre step
- action post step
- history saved
- episode end
- error

日志命名包含日期时间:

```text
logs/navigation_records/*_navigation_YYYYMMDD_HHMMSS.log
logs/navigation_records/*_navigation_YYYYMMDD_HHMMSS.jsonl
```

本次有效导航记录:

```text
logs/navigation_records/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022_train_navigation_20260610_231040.log
logs/navigation_records/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022_train_navigation_20260610_231040.jsonl
```

## 1 Episode Smoke Test

实验名:

```text
a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022
```

运行命令:

```bash
MAGNUM_LOG=verbose \
HABITAT_SIM_LOG=verbose \
EGL_DEVICE_ID=0 \
CUDA_VISIBLE_DEVICES=0 \
OPENNAV_LLM_BASE_URL=http://127.0.0.1:23333/v1 \
OPENNAV_LLM_MODEL=/root/models/Qwen3.5-4B \
OPENNAV_SIGLIP_PATH=/root/models/google-siglip-so400m-patch14-384 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
conda run -n opennav python run.py \
  --exp_name a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022 \
  --exp-config run_OpenNav.yaml \
  --llm Qwen/Qwen3.5-4B \
  --api_key not-needed \
  EVAL.EPISODE_COUNT 1 \
  EVAL.SPLIT val_unseen \
  OPENNAV_HARNESS.ENABLED False \
  OPENNAV_HARNESS.ENABLE_HARNESS_LOGGING False
```

运行状态:

- episode id: `7`
- 总耗时: 约 `8m48s`
- 进度条: `1/1`
- `Get Observation` 阶段已通过
- 本地 Qwen API 正常返回
- 未出现 Python traceback
- 未出现 Habitat-Sim EGL/OpenGL 初始化失败

最终指标:

```json
{
  "steps_taken": 6.0,
  "distance_to_goal": 6.286280632019043,
  "success": 0.0,
  "oracle_success": 0.0,
  "path_length": 8.403072198596833,
  "collisions": 0.08823529411764706,
  "spl": 0.0,
  "ndtw": 0.7507694566078098
}
```

结果文件:

```text
logs/eval_results/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022/stats_ckpt_val_unseen.json
logs/eval_results/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022/stats_ep_ckpt_val_unseen_r0_w1.json
```

运行日志:

```text
logs/running_log/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022_console.log
logs/running_log/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022_train.log
```

## 结果判断

这次测试的意义是 smoke test 通过，不是导航任务成功。

已验证:

- Open-Nav 能在本机启动到 Habitat-Sim。
- 本地 SigLIP 能离线加载，不再卡在在线下载。
- RAM / SpatialBot3B 能生成观察描述。
- Open-Nav 能调用本地 Qwen OpenAI-compatible API。
- 1 episode 能完整跑到 `episode_end` 并写出 metrics。
- 导航记录能完整落盘到 `.log` 和 `.jsonl`。

未通过:

- 导航成功率为 `0.0`。
- SPL 为 `0.0`。
- 第 6 步后距离目标仍为 `6.286280632019043`。

因此，当前阻塞已经从环境和依赖问题转移到策略质量和决策逻辑问题。

## 日志清理

测试完成后清理了无用日志:

- 空日志文件
- 已结束进程的 `.pid`
- 只有初始化信息、没有 episode 结果的旧 smoke train log
- 空的 `logs/checkpoints/*` 目录
- 空的 `logs/eval_results/*` 目录
- 旧的无效 Qwen server pid/log

保留:

```text
logs/qwen_server/qwen35_transformers_serve.pid
logs/qwen_server/qwen35_transformers_serve.log
logs/running_log/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022_console.log
logs/running_log/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022_train.log
logs/navigation_records/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022_train_navigation_20260610_231040.log
logs/navigation_records/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022_train_navigation_20260610_231040.jsonl
logs/eval_results/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022/stats_ckpt_val_unseen.json
logs/eval_results/a0_baseline_smoke_qwen_siglip_local_1ep_20260610_231022/stats_ep_ckpt_val_unseen_r0_w1.json
logs/downloads/*.log
```

清理后 `logs/` 大小约 `348K`。

## 后续建议

1. 固定当前 smoke 命令作为 A0 本地 Qwen 基线复现实验。
2. 再跑 3 到 5 个 episode，观察失败是否集中在完成度估计、候选点选择或 SpatialBot 视觉描述偏差。
3. 基于 JSONL 记录分析 `distance_gain_selected`，优先定位导致距离变差的动作。
4. 检查 Qwen 输出是否过度依赖文本 landmark，特别是把 pool、bar、counter、chair 的语义关系反复重置。
5. 后续 A1 Harness 对照应在当前本地模型和本地 SigLIP 都稳定后再打开。

## A1 Harness 设计与接入

同日还完成了 A1 `instrumented / harnessed baseline` 的代码接口设计与初步接入。

核心约束:

```text
enable_harness_logging = true
enable_decision_effect = false
```

A1 只允许增加旁路状态和 trace，不允许改变:

- waypoint 候选生成与排序；
- LLM prompt；
- `observe_dict` / `observation` / `history_traj`；
- `next_vp`；
- `env_actions`；
- stop / fallback 逻辑；
- 模型、数据集、随机种子、评测脚本。

新增目录:

```text
/root/wjj/Open-Nav/vlnce_baselines/common/opennav_ext/
```

新增模块:

```text
__init__.py
agent_state.py
context_builder.py
geometry_query.py
grounder_diagnostic.py
harness_config.py
metrics_logger.py
oracle_metrics.py
visual_graph_memory.py
```

模块职责:

| 文件 | 职责 | A1 行为 |
|------|------|---------|
| `agent_state.py` | `AgentState` / `CandidateState` / candidate record 构造 | 只建结构，不影响决策 |
| `metrics_logger.py` | jsonl trace 写入、episode start/end、tool failure | logging-only |
| `harness_config.py` | Harness 总开关、子模块开关、A1 配置校验 | 禁止 decision effect |
| `geometry_query.py` | waypoint depth / approximate world point 诊断 | logging-only |
| `grounder_diagnostic.py` | landmarks 与 observation 文本的轻量匹配 | logging-only |
| `visual_graph_memory.py` | episode 内 pose revisit / novelty 诊断 | logging-only |
| `context_builder.py` | diagnostic context skeleton | 不注入 prompt |
| `oracle_metrics.py` | selected distance gain / step output summary | logging-only |

## A1 配置开关

修改:

```text
/root/wjj/Open-Nav/vlnce_baselines/config/default.py
```

新增配置:

```yaml
OPENNAV_HARNESS:
  ENABLED: false
  ENABLE_HARNESS_LOGGING: false
  ENABLE_DECISION_EFFECT: false
  TRACE_DIR: logs/harness_traces
  TRACE_FORMAT: jsonl
  LOG_IMAGES: false
  FAIL_OPEN: true

  GEOMETRY_QUERY:
    ENABLED: true
    LOG_ONLY: true

  GROUNDER_DIAGNOSTIC:
    ENABLED: true
    LOG_ONLY: true

  MEMORY_DIAGNOSTIC:
    ENABLED: true
    LOG_ONLY: true

  CONTEXT_BUILDER:
    ENABLED: true
    LOG_ONLY: true

  ORACLE_METRICS:
    ENABLED: true
    LOG_ONLY: true
```

保护规则:

- `OPENNAV_HARNESS.ENABLE_DECISION_EFFECT=true` 时直接报错；
- 任一 A1 子模块 `LOG_ONLY=false` 时直接报错；
- 子模块可独立关闭。

## A1 主循环接入

修改:

```text
/root/wjj/Open-Nav/vlnce_baselines/common/base_il_trainer_llm.py
```

新增 trace event:

| 位置 | 新增 trace event |
|------|------------------|
| episode 开始 | `episode_start` |
| 每步 pose / heading 后 | `step_start` |
| waypoint predictor 后 | `waypoint_candidates` |
| observation 后 | `observation` |
| GeometryQuery 后 | `geometry_query` |
| GrounderDiagnostic 后 | `grounder_diagnostic` |
| MemoryDiagnostic 后 | `memory_diagnostic` |
| ContextBuilder skeleton 后 | `diagnostic_context` |
| LLM raw prediction 后 | `selector_raw` |
| thought fusion 后 | `selector_fused` |
| final decision 后 | `selector_final` / `diagnostic_context_final` |
| `envs.step()` 前 | `action_pre_step` |
| `envs.step()` 后 | `action_post_step` |
| episode 结束 | `oracle_metrics` / `episode_end` |

重要说明:

- diagnostic 结果不写回 `observe_dict`；
- diagnostic 结果不写回 `observation`；
- diagnostic 结果不写回 `history_traj`；
- diagnostic 结果不改变 `next_vp`；
- diagnostic 结果不改变 `env_actions`。

## A1 Trace 输出

`MetricsLogger` 输出路径格式:

```text
{TRACE_DIR}/{run_id}/{split}/rank_{rank}/{episode_id}.jsonl
```

行为:

- 每个 `run_id` 独立目录；
- 每个 episode 一个 jsonl 文件；
- episode start 时清空同名 episode trace，避免重复运行追加污染；
- 写入失败时，如果 `FAIL_OPEN=true`，禁用 Harness 日志，不中断导航。

## A1 审查修复

已修复:

1. `FAIL_OPEN` 覆盖不足
   - 修复位置: `metrics_logger.py`
   - 现在目录创建、trace truncate、event 写入失败都会 fail-open。

2. trace 追加污染
   - 修复位置: `metrics_logger.py`
   - `run_id` 纳入路径；episode start 清空同名 jsonl。

3. `episode_end` step 固定为 0
   - 修复位置: `metrics_logger.py` 与 `base_il_trainer_llm.py`
   - `end_episode(..., step_id=episode_end_step)` 使用真实结束步。

4. Memory diagnostic 内部 step 与 rollout step 不一致
   - 修复位置: `visual_graph_memory.py` 与 `base_il_trainer_llm.py`
   - `VisualGraphMemoryDiagnostic.update()` 现在接收主循环 `current_step`。

## A1 Smoke Test

实验名:

```text
a1_harness_qwen_siglip_local_1ep_20260610_233812
```

执行配置要点:

```text
OPENNAV_HARNESS.ENABLED=True
OPENNAV_HARNESS.ENABLE_HARNESS_LOGGING=True
OPENNAV_HARNESS.ENABLE_DECISION_EFFECT=False
OPENNAV_HARNESS.TRACE_DIR=logs/harness_traces/a1
EVAL.EPISODE_COUNT=1
EVAL.SPLIT=val_unseen
```

结果文件:

```text
/root/wjj/Open-Nav/logs/eval_results/a1_harness_qwen_siglip_local_1ep_20260610_233812/stats_ckpt_val_unseen.json
```

结果:

```json
{
  "steps_taken": 6.0,
  "distance_to_goal": 6.286280632019043,
  "success": 0.0,
  "oracle_success": 0.0,
  "path_length": 8.403072198596833,
  "collisions": 0.08823529411764706,
  "spl": 0.0,
  "ndtw": 0.7507694566078098
}
```

trace 路径:

```text
/root/wjj/Open-Nav/logs/harness_traces/a1/a1_harness_qwen_siglip_local_1ep_20260610_233812_train_val_unseen_seed0_r0_w1_20260610_233829/val_unseen/rank_0/7.jsonl
```

导航记录:

```text
/root/wjj/Open-Nav/logs/navigation_records/a1_harness_qwen_siglip_local_1ep_20260610_233812_train_navigation_20260610_233829.log
/root/wjj/Open-Nav/logs/navigation_records/a1_harness_qwen_siglip_local_1ep_20260610_233812_train_navigation_20260610_233829.jsonl
```

trace 已确认包含:

- `episode_start`
- `episode_metadata`
- `step_start`
- `waypoint_candidates`
- `observation`
- `geometry_query`
- `grounder_diagnostic`
- `memory_diagnostic`
- `diagnostic_context`
- `selector_raw`
- `selector_fused`
- `diagnostic_context_final`
- `selector_final`
- `action_pre_step`
- `action_post_step`
- `oracle_metrics`
- `episode_end`

结论:

- A1 smoke 已完成 1 episode；
- A1 指标与 A0 baseline 完全一致；
- 当前 A1 为 logging-only / diagnostic-only，没有启用决策影响；
- `action_post_step.done=false` 后出现 `episode_end` 是因为主循环按 `step_length=6` 截断 episode，不是环境提前 `done`。

## A1 验证

已执行:

```bash
conda run -n opennav python -m py_compile \
  vlnce_baselines/common/base_il_trainer_llm.py \
  vlnce_baselines/config/default.py \
  vlnce_baselines/common/opennav_ext/*.py

bash -n run_OpenNav.bash

git diff --check -- \
  vlnce_baselines/common/base_il_trainer_llm.py \
  vlnce_baselines/common/opennav_ext/harness_config.py \
  vlnce_baselines/common/opennav_ext/metrics_logger.py \
  vlnce_baselines/common/opennav_ext/context_builder.py \
  run_OpenNav.bash
```

结果:

```text
全部通过
```

## 继续修改入口

如果后续要继续改:

- 加字段: 优先改 `agent_state.py` 和对应 logger payload；
- 改 trace schema: 优先改 `metrics_logger.py`；
- 改模块开关: 优先改 `default.py` 和 `harness_config.py`；
- 改主循环插桩: 只改 `base_il_trainer_llm.py`，并保持 logging-only；
- 增加新 diagnostic 工具: 放到 `vlnce_baselines/common/opennav_ext/`，并新增独立 `ENABLED` / `LOG_ONLY` 开关。

必须保持:

```text
A1 不影响导航行为。
```

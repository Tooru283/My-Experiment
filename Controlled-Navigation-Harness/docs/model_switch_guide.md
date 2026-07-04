---
title: 本地 LLM 切换操作指南
created: 2026-07-01
tags:
  - ops
  - lmdeploy
  - model-switch
---

# 本地 LLM 切换操作指南

本文档说明如何在 Open-Nav 实验中切换本地 Qwen 模型（4B ↔ 9B），以及对应 GPU 的显存要求。

## 概述

系统由两部分组成，切换模型时**两处都要改**：

| 组件 | 文件/命令 | 说明 |
|---|---|---|
| **lmdeploy 服务** | 启动命令 | 独立进程，提供 OpenAI-compatible API |
| **run 配置** | `run_OpenNav.bash` + `run_OpenNav.yaml` | 告诉 Open-Nav 用哪个模型 |

---

## 一、模型规格参考

| 模型 | 路径 | 推荐 GPU | VRAM 占用（bfloat16） |
|---|---|---|---|
| Qwen3.5-4B | `/root/models/Qwen3.5-4B` | RTX 3090 (24GB) | ~10–12 GB |
| Qwen3.5-9B | `/root/models/Qwen3.5-9B` | RTX 4090 (46GB) | ~22–24 GB |

> Habitat 评测进程本身还需要约 3–5 GB VRAM（WaypointBert + SpatialBot3B），务必留余量。

---

## 二、步骤一：停止当前 lmdeploy 服务

```bash
# 查找当前服务进程
ps aux | grep "transformers serve" | grep -v grep

# 按 PID 停止（替换 <PID> 为实际进程号）
kill <PID>

# 确认已停止
ps aux | grep "transformers serve" | grep -v grep
```

---

## 三、步骤二：启动新模型的 lmdeploy 服务

切换到 `qwen35-serve` conda 环境，后台启动：

```bash
# 切换到 9B（当前，RTX 4090）
conda run -n qwen35-serve \
  transformers serve /root/models/Qwen3.5-9B \
  --host 0.0.0.0 --port 23333 \
  --device cuda:0 --dtype bfloat16 \
  --reasoning off --log-level info &

# 切换到 4B（RTX 3090 / 低显存场景）
# conda run -n qwen35-serve \
#   transformers serve /root/models/Qwen3.5-4B \
#   --host 0.0.0.0 --port 23333 \
#   --device cuda:0 --dtype bfloat16 \
#   --reasoning off --log-level info &
```

等待服务就绪（约 30–60 秒），验证：

```bash
curl -s http://127.0.0.1:23333/v1/models | python3 -m json.tool | grep '"id"'
```

---

## 四、步骤三：修改 run_OpenNav.bash

文件位置：`/root/wjj/Open-Nav/run_OpenNav.bash`

找到以下区块（约第 23–27 行），切换注释：

```bash
# 4B (RTX 3090 / ~12GB):
# export OPENNAV_LLM_MODEL="${OPENNAV_LLM_MODEL:-/root/models/Qwen3.5-4B}"
# 9B (RTX 4090 / ~24GB):
export OPENNAV_LLM_MODEL="${OPENNAV_LLM_MODEL:-/root/models/Qwen3.5-9B}"
```

找到 LLM_NAME 区块（约第 87–90 行），切换注释：

```bash
# 4B (RTX 3090 / ~12GB):  LLM_NAME="Qwen/Qwen3.5-4B"
# 9B (RTX 4090 / ~24GB):  LLM_NAME="Qwen/Qwen3.5-9B"
LLM_NAME="Qwen/Qwen3.5-9B"   # ← 取消/注释此行
```

---

## 五、步骤四：修改 run_OpenNav.yaml

文件位置：`/root/wjj/Open-Nav/run_OpenNav.yaml`

找到 `VISUAL_EVIDENCE` 区块下的 `MODEL` 字段（约第 67–69 行），切换注释：

```yaml
BASE_URL: http://127.0.0.1:23333/v1
# MODEL: /root/models/Qwen3.5-4B  # 4B (RTX 3090 / ~12GB VRAM)
MODEL: /root/models/Qwen3.5-9B    # 9B (RTX 4090 / ~24GB VRAM)
```

---

## 六、步骤五：验证 preflight

直接运行 preflight 脚本，不启动正式评测：

```bash
cd /root/wjj/Open-Nav
conda run -n opennav bash -c "
  export OPENNAV_LLM_MODEL=/root/models/Qwen3.5-9B
  export OPENNAV_LLM_BASE_URL=http://127.0.0.1:23333/v1
  python - <<'PY'
import json, os, urllib.request
base_url = os.environ['OPENNAV_LLM_BASE_URL'].rstrip('/')
model = os.environ['OPENNAV_LLM_MODEL']
payload = {'model': model, 'messages': [{'role':'user','content':'ping'}], 'temperature':0, 'max_tokens':1}
req = urllib.request.Request(f'{base_url}/chat/completions',
    data=json.dumps(payload).encode(), method='POST',
    headers={'Authorization':'Bearer not-needed','Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=30) as r: r.read()
print(f'Preflight OK: {model}')
PY
"
```

输出 `Preflight OK: /root/models/Qwen3.5-9B` 则配置正确。

---

## 六b、步骤五b：切换 LLM_RUNTIME token 参数（模型相关，切换时必改）

**token 上限是模型相关的**，切换模型时和 MODEL 一起改。文件：`run_OpenNav.yaml` 的 `OPENNAV_HARNESS.LLM_RUNTIME` 区块。

背景：9B 的 Chain-of-Thought 比 4B 长得多。在 4B 调优的 token 上限下，9B 常在输出 `Prediction: X` 之前就被截断，导致 selector 空预测 fallback 率 **22.4%**（4B 仅 6.9%），性能被废（零调优 9B ep100：SR 20% / OSR 23%，全面低于 4B 的 24%/26%）。

| 参数 | 4B | 9B | 说明 |
|---|---|---|---|
| `NAVIGATOR_MAX_TOKENS` | 1024 | **2048** | 主因修复：给 9B 足够空间输出完整候选预测 |
| `COMPLETION_MAX_TOKENS` | 512 | **768** | 9B completion 进度估计 CoT 更长 |
| `THOUGHT_FUSION_MAX_TOKENS` | 256 | 256 | 不变 |
| `DECISION_MAX_TOKENS` | 64 | 64 | 不变 |

yaml 里已用注释形式保留两套值，切换时切注释即可：

```yaml
  LLM_RUNTIME:
    # 4B:  COMPLETION_MAX_TOKENS: 512
    COMPLETION_MAX_TOKENS: 768   # 9B
    # 4B:  NAVIGATOR_MAX_TOKENS: 1024
    NAVIGATOR_MAX_TOKENS: 2048    # 9B (fixes 22.4% empty-pred fallback from truncation)
    THOUGHT_FUSION_MAX_TOKENS: 256
    DECISION_MAX_TOKENS: 64
```

> 代价：`NAVIGATOR_MAX_TOKENS`→2048 增加 9B 每步耗时（已 50–77s/步），ep100 约 ~20h。
> 未解决：9B 假停率 80%（4B 68%）是过度停止，非 token 问题，需 harness 逻辑单独处理，不在本参数集内。

---

## 七、步骤六：正式启动评测

```bash
cd /root/wjj/Open-Nav
conda run -n opennav bash run_OpenNav.bash
```

---

## 八、快速参考：当前配置状态

| 项目 | 当前值（2026-07-01） |
|---|---|
| 活动模型 | Qwen3.5-9B |
| GPU | RTX 4090 (46GB) |
| lmdeploy 端口 | 23333 |
| VISUAL_EVIDENCE MODEL | `/root/models/Qwen3.5-9B` |
| LLM_NAME | `Qwen/Qwen3.5-9B` |
| NAVIGATOR_MAX_TOKENS | 2048（9B 调优） |
| COMPLETION_MAX_TOKENS | 768（9B 调优） |
| TRACE_DIR | `logs/harness_traces/p0_geometry_9b` |
| 生效实验 | P0 几何注入 + 9B 调参 |

---

## 九、回切到 4B 的完整命令序列

```bash
# 1. 停止 9B 服务
kill $(pgrep -f "transformers serve.*Qwen3.5-9B")

# 2. 启动 4B 服务
conda run -n qwen35-serve \
  transformers serve /root/models/Qwen3.5-4B \
  --host 0.0.0.0 --port 23333 \
  --device cuda:0 --dtype bfloat16 \
  --reasoning off --log-level info &

# 3. 修改 run_OpenNav.bash 中的 OPENNAV_LLM_MODEL 和 LLM_NAME → 4B
# 4. 修改 run_OpenNav.yaml 中的 VISUAL_EVIDENCE.MODEL → 4B
# 5. 修改 run_OpenNav.yaml 中的 LLM_RUNTIME → 4B（NAVIGATOR 1024 / COMPLETION 512，见 六b）
# 6. 验证 preflight，再跑评测
```

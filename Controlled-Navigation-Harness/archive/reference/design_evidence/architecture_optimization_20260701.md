---
date: 2026-07-01
tags:
  - architecture
  - optimization
  - OSR
  - frontier-comparison
status: active
related:
  - "[[current_task]]"
  - "[[总方案-可靠终止决策-执行版-20260629]]"
---

# 架构优化方案：对标前沿，从"可靠终止"转向"攻 OSR"

本文档把我们当前 Open-Nav 架构逐层拆开，对照 2025-26 前沿论文（SmartWay / Fast-SmartWay / Spatial-VLN / AgenticNav / DV-VLN / VLN-Zero），定位结构性瓶颈并给出优化路线。P0 附带落到代码的具体实现设计。

结论先行：**过去所有工作押在第④层（可靠终止），转化率已 92.3% 封顶；真正的 SR 天花板是 OSR（到达能力），瓶颈在第①②③⑤层。堆模型无效（9B OSR=23% 反低于 4B 26%），必须改架构。**

---

## 1. 当前架构（五层）与代码位置

| 层 | 模块 | 代码位置 | 现状 |
|---|---|---|---|
| ① 航点生成 | WaypointBert（冻结 ResNet-50） | `policy.net` → `construct_image_dicts` (base_il_trainer_llm.py:236) | 旧预测器，输出候选方向+距离+角度 |
| ② 感知 | SpatialBot3B + RAM | `observe_view` (navigator/api.py:153) | 图像→文本，**距离靠 VLM 幻觉估计** |
| ③ 导航决策 | NAVIGATOR（纯文本） | `move_to_next_vp` (spatialNavigator.py:545) | 单次文本选候选 ID |
| ④ 终止决策 | V2/U2/E3 | `should_stop` + selector_stop_gate | 重型，转化 92.3%（**已封顶**）|
| ⑤ 记忆/恢复 | U3 重选 + nav_history | `review_history` (spatialNavigator.py:84) | 无显式地图，无回溯 |

**关键数据流缺口**：`construct_image_dicts` (base_il_trainer_llm.py:1304) 已算出每候选的**真实几何** `distance_dict`（WaypointBert 预测距离）+ `radius_dict`（角度弧度），但这些**只在动作执行时用**（:3080-3081），**从未注入 navigator 的观测文本**。navigator 决策时读的是 `observe_view` 里 SpatialBot 对 prompt "how far are these objects... in meter" 的**幻觉回答**（日志实证："Mirror: 20 meters, Staircase: 10 meters" 均为编造）。

---

## 2. 逐层对标前沿

| 层 | 我们 | 前沿怎么做 |
|---|---|---|
| ① 航点 | 冻结旧 ResNet-50 | **4篇全攻**：SmartWay 换 DINOv2+occupancy 重训；Fast-SmartWay 端到端消除（MLLM 直出角度+距离）；AgenticNav VLM 选像素绕过；Spatial-VLN value-based 采样 |
| ② 感知 | 图像→文本+幻觉距离 | SmartWay/Fast-SmartWay MLLM 原生看图+RAM；**Fast-SmartWay 注入 depth 方向分桶真实距离**；Spatial-VLN 显式门状态/区域边界 |
| ③ 决策 | single-shot 文本选择 | DV-VLN generate-then-verify；**Fast-SmartWay 不确定性感知（困惑→重扫+FPBR）+8pp**；Spatial-VLN 多专家+冲突探索 |
| ④ 终止 | V2/U2/E3 重型 | 都很轻——他们不需要，OSR 本就高 |
| ⑤ 记忆 | U3 重选 | SmartWay 回溯；VLN-Zero 场景图+缓存；AgenticNav 记忆工具 |

**范式迁移**：Open-Nav（2024，我们）= 两阶段 + 图像转文本 + 纯文本决策 → 前沿（2025-26）= 端到端 MLLM + 结构化空间接地 + 决策级验证/再观测。

---

## 3. 优化路线（按 ROI，守住 4B + training-free + 四周窗口）

| 优先级 | 方案 | 预期 | 工作量 | 重训 | 攻击目标 |
|---|---|---|---|---|---|
| **P0** | 结构化空间感知注入 | OSR/SR +2~5 | **低** | 否 | ②图像转文本丢空间 → doorway/multi-room |
| P1 | 导航级再观测（校准驱动） | +5~8（外部已验证） | 中 | 否 | ③单次决策脆弱 → 迷路36集 |
| P2 | 回溯机制 | +5~12（SmartWay真机） | 中 | 否 | ⑤无恢复 → 死胡同 |
| P3 | 航点瓶颈修/绕/消除 | +10~15（最高天花板） | 高 | 视路线 | ①旧预测器 |
| P4 | 冻结④，停止投入 | — | 0 | — | 已封顶 |

---

## 4. P0 具体实现设计：真实几何注入

### 4.1 核心思想
用**已有的** WaypointBert 真实几何（`distance_dict`/`radius_dict`）替换/补充 navigator 观测里 SpatialBot 的幻觉距离，并显式给出每个候选的**朝向角 + 距离**。这是 Fast-SmartWay "Spatial-Semantic Textual Description" 的最小实现，且我们的数据**已经算好了**，只差注入。

### 4.2 改动点

**改动1（核心，必做）：把真实几何注入候选观测串**

- 位置：`base_il_trainer_llm.py:1330` 调 `observe_environment` 处，把 `radius_dict`/`distance_dict` 一并传入
- 在 `spatialNavigator.observe_environment` (spatialNavigator.py:53) 或 `api.observe_view` (api.py:153) 组装 observation 串时追加几何字段：

```python
# api.py observe_view，在 view_observation 后追加
angle_deg = round(np.rad2deg(radius_dict[direction_idx]))
wp_dist   = round(float(distance_dict[direction_idx]), 1)
geo = f"[Geometry: heading {angle_deg} deg, waypoint {wp_dist} m ahead] "
observe_result = f"Direction {direction_idx} ... " + geo + view_observation
```

- 效果：navigator 决策时看到 "Direction 3 [Geometry: heading 75 deg, waypoint 2.4 m ahead] Scene Description: ..." —— 有了**可信的空间锚点**，不再被 SpatialBot 的 "20 meters" 幻觉误导。

**改动2（可选，Fast-SmartWay 完整版）：depth 方向分桶障碍距离**

- 从 `image_dict[dir]['depth']` 算地面投影，输出方向分桶 "left30: obstacle 2.5m / forward: clear"
- 比改动1 多一步 depth 几何计算，但给出的是**障碍距离**（可通行性），与改动1的**航点距离**互补
- 建议先做改动1，看增益再决定是否加改动2

**改动3（配合）：completion/navigator prompt 里显式指示优先信任 [Geometry] 字段**
- 在 COMPLETION_ESTIMATION / navigator system prompt 加一句 "Trust the [Geometry] distances (sensor-measured) over any distances mentioned in Scene Description (which may be estimated)."

### 4.3 为什么这是最高 ROI 起手
- **数据已存在**：`distance_dict`/`radius_dict` 已在 :1304 算好，零额外模型调用
- **改动小**：主要是字符串拼接 + 一处传参
- **直击实证问题**：日志确认 SpatialBot 距离是幻觉；navigator 现在盲选
- **training-free**：不碰模型，4B 主线直接用
- **可消融**：加/不加 [Geometry] 字段跑对照，干净

### 4.4 风险与验证
- 风险：WaypointBert 的距离本身也有误差（%Open 仅 82%），但仍远好于 VLM 幻觉；且这是"补充信息"非"替换决策"，navigator 仍综合语义判断
- 验证：先 ep1 跑通看 observation 串格式，再 ep100 对比 e3_carry_forward。成功判据：OSR 上升（说明到达能力改善），不只 SR
- trace 目录：`logs/harness_traces/p0_geometry`

---

## 5. 下一步
1. 实现 P0 改动1，ep1 验证 observation 格式
2. ep100 对比 e3_carry_forward（4B），看 OSR 是否上升
3. OSR 有改善则接 P1（导航级再观测，复用 E3 infra）
4. 每轮更新 [[current_task]]

## 5b. WaypointBert 诊断记录（2026-07-01）

针对"是否/如何改进 WaypointBert（P3）"，做了两项分析，结论：**ResNet 替换 / 占用过滤这条路可以关闭；瓶颈在选择层不在候选生成层。**

### 5b.1 结构事实（TRM_net.py:22-54）
`BinaryDistPredictor_TRM`：`visual_fc_rgb = Linear(2048×7×7 → hidden)` 死绑 ResNet-50 特征 → `visual_merge` → 2层 `waypoint_TRM` → `vis_classifier` → 120角×12距 heatmap。
- **ResNet 不可 drop-in 替换**：预测头在 ResNet 特征分布上训练，换 DINOv2（384维 patch token）直接 shape 报错 + 分布漂移。SmartWay 的 DINOv2 是**连着重训**做的。"替换 ResNet" = 必须重训预测头（需 MP3D 数据 + 训练流程），非轻量。

### 5b.2 可达性测量（4B e3_carry_forward trace，用 step_start.positions vs 预测距离）
| 指标 | 值 |
|---|---|
| 实际位移/预测距离 | 均值 **0.971**，中位 **1.000**（n=678步）|
| 明显被挡步（比值<0.5） | **1.9%** |
| 疑似被挡步（比值<0.8） | 4.4% |
| 每步候选数 | 均值 **3.8**（12方向中，分布 2/3/4/5=1%/30%/53%/16%）|

**核心发现：agent ~98% 精确到达预测航点。我们是 waypoint-teleport setup，无低层碰撞截断。**

### 5b.3 结论
1. **占用过滤（depth 后过滤"航点进墙"）在我们这里无效**——该失败模式仅占 2-4%。SmartWay 的 %Open 收益来自真机/低层控制撞墙，**不迁移到 teleport setup**。
2. **替换 ResNet 不值得**——它改善的正是可达性/%Open，而我们已 98%；为不存在的失败模式重训是浪费。
3. WaypointBert 唯一残余可疑点是**候选覆盖**（每步 ~4/12 方向，可能漏掉目标方向），但：(a) 从现有 trace 测不了——`distance_to_goal` 是**测地距离**，无法三角定位目标以计算未选候选的目标距离；(b) 相对选择层是次要因素。

### 5b.4 未决问题（需诊断性重跑才能测）
- **gen-vs-selection 分离**：失败步里，好的朝目标候选是"不在候选集"（生成问题）还是"在但没选"（选择问题）？需给每候选加"测地距离到目标"的 logging（sim 内可查），随 P0 ep100 顺带采集。
- **候选覆盖是否限制**：~4/12 方向够不够？可试训练-free 调低 heatmap 阈值增候选，但更多候选=更难选，选择层已是瓶颈时可能反伤，双刃。

### 5b.5 P3 路线判定（更新）
- ~~路线A 重训 occupancy 预测器~~ → **暂关**（可达性已好，无收益靶点）
- ~~占用后过滤~~ → **关闭**（失败模式不存在）
- **仅保留**：若 5b.4 诊断显示候选覆盖漏目标方向 → 才考虑增覆盖或换生成方式；否则 WaypointBert 冻结不动。

---

## 6. 参考
SmartWay(2503.10069) / Fast-SmartWay(2511.00933) / Spatial-VLN(2601.12766) / AgenticNav(2606.10577) / DV-VLN(2601.18492) / VLN-Zero(2509.18592)。VLN-MME（Oracle 0→52% 证选择层瓶颈）/ NavBench（61% Incorrect Plan）由用户提供。详见 [[current_task]] 第九节。

---
date: 2026-07-01
tags:
  - 日报
  - 9B
  - 战略转向
  - OSR
  - P0几何注入
  - E5_rescue守卫
  - WaypointBert诊断
status: done
related:
  - "[[current_task]]"
  - "[[architecture_optimization_20260701]]"
  - "[[model_switch_guide]]"
  - "[[project_e3_20260630]]"
---

# 实验记录 2026-07-01

## 零、当日主线

从 e3_carry_forward（4B，SR24）出发，做了三件大事：(1) 9B 模型切换与 ep100 评测；(2) 文献对比引发**战略转向——从"可靠终止"转向"攻 OSR"**；(3) 实现 P0 几何注入 + E5 rescue 守卫并做 ep1 验证。核心结论：**转化率已封顶（92.3%），SR 天花板是 OSR；堆模型（9B）无效，须改架构。**

---

## 一、9B 模型评测（切换 + ep100）

### 1.1 切换与调参
- RTX 4090 上切到本地 `Qwen3.5-9B`，配置改动用注释保留 4B（见 [[model_switch_guide]]）。
- ep1 测试通过（SR=1），据此启动 ep100。

### 1.2 9B ep100 结果（100/100 完成）

| 指标 | 4B e3_carry_forward | 9B 零调优 | 差异 |
|---|---|---|---|
| SR | 24% | **20%** | −4 |
| OSR | 26% | **23%** | −3 |
| SPL | 0.187 | 0.149 | −0.038 |
| nDTW | — | 0.434 | — |
| OSR→SR | 92.3% | 87.0% | −5.3 |
| 假停率 | 68% (38/56) | **80% (52/65)** | +12 |
| 空预测 fallback | 6.9% | **22.4%** | +15.5 |

**9B 全面劣于 4B。根因两条**：
1. **过度停止**：9B 长推理链更易得出"任务完成"，假停从 38 增到 51，砸了 OSR。
2. **输出截断**：`NAVIGATOR_MAX_TOKENS=1024` 对 9B 太短，22.4% 的步在输出 `Prediction:X` 前被截断 → 回落 fallback。

**关键判断**：9B 的 OSR（23%）也低于 4B（26%）——**堆模型解决不了 OSR，瓶颈是架构不是模型容量。**

---

## 二、失败分析：OSR 是天花板

### 2.1 4B e3_carry_forward 失败分类（100集）

| 类型 | 数量 | 说明 |
|---|---|---|
| SR=1 ✓ | 24 | 成功 |
| OSR=0 **假停** | **38** | 从未接近目标（avg 8m）就停 |
| OSR=0 **迷路** | **36** | 迷路耗尽步数 |
| OSR=1 SR=0 | 2 | 接近过但步数不够（ep1084/1106）|

OSR 天花板=26，SR 天花板=26。**先前对 ep1084/1106 的"模型视觉盲区"归因有误**——两者都是 step_limit 且 oracle=1（途中进过 3m），是步数问题不是看不见。

### 2.2 距离增益诊断（选择层 vs 生成层）

| 集类型 | 正增益步 | 负增益步 | 净均值 |
|---|---|---|---|
| 成功 | 79% | 18% | +0.83 m/步 |
| 假停(38) | 55% | 42% | +0.22 m/步 |
| 迷路(36) | 52% | 48% | +0.11 m/步 |

失败集里 agent **~45% 的步往目标反方向走**（随机游走）——失败在候选层，且更像**选择/推理层**问题。

---

## 三、文献对比与战略转向

### 3.1 对标（R2R-CE val_unseen, zero-shot）

| 方法 | 模型 | SR | OSR | OSR→SR | 核心机制 |
|---|---|---|---|---|---|
| SmartWay(2503) | GPT-4o | 29 | 51 | 57% | occupancy航点+回溯 |
| Spatial-VLN(2601) | DeepSeek-v3 | 33 | 高 | — | 显式空间感知+冲突探索 |
| VLN-Zero(2509)* | GPT-4.1/5 | 42.4 | 51.6 | 82% | 两阶段探索建图 |
| Fast-SmartWay(2511) | GPT-4o | 27.75 | — | — | 端到端消除航点+不确定性再观测 |
| **我们** | Qwen 4B | 24 | 26 | **92.3%** | 可靠终止 V2/U2/E3 |

\* 两阶段，任务设定不同。

### 3.2 三条关键读数
1. **我们的 OSR→SR 转化 92.3% 是全场最高**（超 SmartWay 57%）——转化机制已过度优化。
2. **OSR 才是天花板**：我们 26% vs SmartWay 51%。若保持 92.3% 转化、OSR→51%，SR≈47%。
3. **航点预测器是四方一致点名的瓶颈**（SmartWay 重训/AgenticNav 像素绕过/Spatial-VLN 采样/Fast-SmartWay 消除）。

### 3.3 战略转向
过去押"可靠终止"（[[总方案-可靠终止决策-执行版-20260629]]），但转化已封顶、边际归零。**转向攻 OSR**：把已有的 E3"弃权+再观测"原语从"终止决策"扩到"导航决策"（详见 [[architecture_optimization_20260701]] 与 [[current_task]] 第四节）。撞车评估：DV-VLN 不撞（训练式+离散）；Spatial-VLN 最大撞车风险；Fast-SmartWay 部分撞车但校准触发差异化。

---

## 四、WaypointBert 诊断（关闭 ResNet 路线）

针对"是否改进 WaypointBert"，用现有 trace 测可达性：

| 指标 | 值 |
|---|---|
| 实际位移/预测距离 | 均值 **0.971**，中位 **1.000**（n=678）|
| 明显被挡步（<0.5） | 1.9% |
| 每步候选数 | 均值 3.8（12方向中）|

**结论**：agent ~98% 精确到达预测航点（teleport setup 无碰撞截断）。
- **占用过滤 / 替换 ResNet 均无靶点**——它们改善的是可达性/%Open，我们已 98%。SmartWay 的 %Open 收益来自真机撞墙，不迁移。
- `TRM_net.py:22` `visual_fc_rgb=Linear(2048×7×7)` 死绑 ResNet，换 DINOv2 非 drop-in 需重训。
- 残余唯一可疑=候选覆盖（~4/12 方向），但 `distance_to_goal` 是测地距离，trace 测不了，需诊断重跑。
- **净结论：航点路线关闭，火力集中选择层。** 详见 [[architecture_optimization_20260701]] §5b。

---

## 五、P0 实现与 9B 调参

### 5.1 P0 = 真实几何注入
**关键发现**：`construct_image_dicts`（base_il_trainer_llm.py:1304）已算出每候选真实几何 `distance_dict`（WaypointBert 距离），但只在动作执行用，**从未注入 navigator**。navigator 决策时读 SpatialBot **幻觉距离**（api.py:155，实证"镜子20米"）。

改动（3处代码 + prompt）：
- `spatialNavigator.py`：`observe_environment` 加 `distance_dict` 参数 + `_inject_waypoint_distance`，注入 `[Waypoint distance: X m]`（只注入距离，角度已由 direction id 编码，避免冗余）。
- `base_il_trainer_llm.py:1330`：传入 distance_dict。
- `prompts.py`：NAVIGATOR 加"信任 measured 距离胜过 Scene Description 估计"。

**定位（诚实）**：P0 是补课非创新——Fast-SmartWay/Spatial-VLN 已用结构化空间文本；我们本来更差（喂幻觉距离），P0 只是填平。唯一可能 novelty 是 P1 的"校准驱动触发"。

### 5.2 9B 调参（记入 [[model_switch_guide]] 六b）
- `NAVIGATOR_MAX_TOKENS`: 1024→**2048**（修 22.4% fallback）
- `COMPLETION_MAX_TOKENS`: 512→**768**

---

## 六、P0 ep1 冒烟 → 发现 rescue bug → E5

### 6.1 P0+调参 ep1（ep244）
- ✅ P0 格式生效（`[Waypoint distance: 1.2 m]`，同视角 SpatialBot 幻觉说"2/3/4米"）
- ✅ token 截断修复（fallback **0%**，Prediction 完整）
- ⚠️ **回归**：ep244 从零调优 SR=1（10步）→ **SR=0（第3步 6.2m 假停）**

### 6.2 根因：非 P0，是 U2 rescue 无距离守卫
- navigator 停止推理 **0 次**引用几何——靠地标/子目标语义推理停的，**P0 不是元凶**。
- 真凶：`base_il_trainer_llm.py:2681` rescue override 分支，9B 自信 STOP + 泛化"doorway"视觉匹配 → **"rescue override granted despite gain/phase conditions"** 在 6.2m 提交停止，**无距离守卫**。
- **连带发现**：这正是 38 集假停机制之一。

### 6.3 E5 rescue 距离守卫 + P0 开关
- `RESCUE_MAX_GOAL_DIST=4.0`：守卫 2661/2681 两个 rescue 分支，dist>4m 时拦截 → fall-through 到 movement_fallback。
- `GEOMETRY_INJECTION` 开关：`_inject_waypoint_distance` 在 None 时返回原串，关闭=传 None，供干净消融。

### 6.4 E5 ep1 验证（ep244）

| 版本 | 停止点 | 步数 | 结果 |
|---|---|---|---|
| 零调优9B | 2.33m | 10 | SR=1 |
| P0+调参（无E5） | 6.2m/step3 | 3 | SR=0 假停 |
| **P0+调参+E5** | 3.68m/step7 | 7 | SR=0，大幅改善 |

- E5 拦截 2 次（6.23m、4.74m），agent 从 6.2m 逼近到 3.68m（近 2.5m）✓
- 距离轨迹：6.05→6.23→4.74→4.08→5.10→3.68→停
- **残余**：卡最后 0.68m（stop@3.68m，oracle=0 从未进 3m 圈）——性质从"远距假停"变成"最后一米 OSR 问题"，留给 P1。

---

## 七、当日交付物
- 代码：P0 几何注入（3文件）、E5 rescue 守卫、GEOMETRY_INJECTION 开关。
- 配置：9B 调参、RESCUE_MAX_GOAL_DIST=4.0。
- 文档：[[architecture_optimization_20260701]]（新建，含 §5b WaypointBert 诊断）、[[current_task]]（滚动更新）、[[model_switch_guide]] 六b。
- 记忆：project_arch_pivot_20260701。

---

## 八、下一步（2026-07-02）
1. **ep100 消融**（都在 9B+调参+E5 上）：
   - Run1 基线：`GEOMETRY_INJECTION=False`，TRACE_DIR=`p0_off_9b`（已切好，待跑）
   - Run2 P0：`GEOMETRY_INJECTION=True`，TRACE_DIR=`p0_geometry_9b`
   - Run2 − Run1 = **P0 净效果**
   - 注意：`EPISODE_COUNT=100` 显式指定（bash 默认 1）
2. Run1 顺带回答：调参+E5 有没有把 9B 从 SR20 救回来。
3. 待定：`RESCUE_MAX_GOAL_DIST` 是否从 4.0 收紧到 3.0（[3,4]m band 停止必失败），用 ep100 数据决定，勿基于 n=1。
4. 若 P0 有效 → 接 P1（导航级再观测，攻最后一米/迷路）。

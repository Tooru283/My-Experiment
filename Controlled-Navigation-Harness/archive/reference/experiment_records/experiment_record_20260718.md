---
date: 2026-07-18
tags:
  - 日报
  - 9B
  - depth_veto
  - 俯仰视角
  - oracle审计
  - 转化率
status: done
related:
  - "[[current_task]]"
  - "[[实验框架-路线状态外置-20260713]]"
  - "[[sr_roadmap_20260706]]"
  - "[[project_oracle_leak_20260704]]"
---

# 实验记录 2026-07-18

## 零、当日主线

项目自 07-05 起停摆 13 天（GPU 空转、depth veto smoke 死在环境 bug 后未重试）。今日做了四件事：(1) 修复两处潜伏的 staged 代码/脚本缺陷，把 depth veto smoke 跑通；(2) **一次 ep100 白跑 9 小时**，根因是开关未进运行时且全链路无生效回显，已补横幅；(3) 从 smoke 单集反推出**深度否决的机制单向性**——抬 OSR 不抬转化；(4) 发现 12 相机机位**俯仰恒为 0**，据此实现「终局俯视视角」机制。

**核心结论：队列里现有的所有开关（veto / 回溯）都只作用于 OSR 侧，没有一个治转化，而 SR = OSR × 转化率。今日新增的俯仰机制是第一个落在转化侧的杠杆。**

---

## 一、解除阻塞：两处潜伏缺陷

### 1.1 `AttributeError: SIMULATOR`（base_il_trainer_llm.py:871）
07-05 staged 的 depth veto 代码读 `config.SIMULATOR`，但该作用域下 `config` 是顶层配置，`SIMULATOR` 挂在 `config.TASK_CONFIG` 下。此 bug 自 07-05 起潜伏——当时的 smoke 在 import 阶段就死了，从未执行到这一行。

### 1.2 smoke 脚本缺运行时环境
`smoke_depth_veto.sh` / `smoke_backtrack.sh` 均未导出 `run_OpenNav.bash` 的环境块。关键是 `OPENNAV_LLM_MODEL`：`transformers serve` **是模型钉死的**，会 400 掉任何 `model` 字段不匹配的请求。缺这一行导致客户端默认标 4B、被 9B 后端拒绝。

> 过程记录：我最初判断 4B 标签"只是装饰性"，被随后的运行证伪。教训——`transformers serve` 的 model 字段是真校验，不是标签。

---

## 二、depth veto smoke：通过，但暴露机制单向性

ep377（典型假停集，基线 conf-1.00 @ 5.42m）：

- 19 条 `depth_stop_veto` 事件，`depth_reading_m` **全部非空**，`fail_open_unavailable` = 0 → 深度传感器确实被读（预注册 caveat 2 满足）
- 10 次停止被真实几何拦下；1.09/2.51/2.69/2.91m → 放行，3.17/4.01/4.02/7.70m → 拦截，双向行为正确

| ep377 | 基线 | veto 开 |
|---|---|---|
| SR | 0 | 0 |
| OSR | 0 | **1** |
| d2g | 5.42m | 4.04m |
| steps | 3 | 8 |

**读法**：veto 把"3 步就放弃"变成"8 步探索并真的走进了成功半径"——机制是通的。但最终仍停在 4.04m，SR 未变。

### 2.1 为什么阈值 3.0m 却停在 4.04m
`_depth_stop_ok` 取的是**正前方中心区深度中位数**，不是到目标距离。任何 3m 内的墙面/家具都满足条件。所以它只能表达"别在空旷处远远地停"，**无法表达"该在这儿停"**。

### 2.2 误杀风险（预注册红线：真到达误杀 = 0）
基线 16 集成功中，约 6–8 集的停止目标是**开口类**：

```
371  Wait by the door at the top of the stairs
513  ...next to the entrance to the dining room
259  Exit the bathroom, wait at the white rug in the hallway
531  Walk out open bedroom door, and wait at top of stair landing
526  Wait at open white door
824  Stop once you enter the next room
```

站在开口处平视，中心深度天然 >3m → veto 会系统性误杀这一族真到达。

⚠ **证据强度声明**：以上为**文本启发式**，非深度实测。基线跑时 veto 关闭、`_depth_stop_ok` 直接 return，未记录任何深度读数，故此风险的实测值尚不存在。

---

## 三、事故：ep100 白跑 9 小时

`logs/eval_results/ep100/20260718/ep100_series_m420260718_113750/` 逐集与 20260705 clean_baseline_v1 **逐字节相同**（100/100 集 d2g 差 <1e-6，SR 翻转 0，SR/OSR/SPL 三数全等）。

**根因链**：`run_OpenNav.bash` 未带 YACS 尾参数启动 → 开关保持 yaml 默认 False → 跑出的就是基线。而**全链路无任何生效回显**，9 小时内没有信号可判断开关状态。exp_name 为 `ep100_series_m4…`（`run_OpenNav.bash:10` 的默认值）也印证了未传参。

**修复**：
1. `base_il_trainer_llm.py` 在 episode loop 前打印 `[HARNESS SWITCHES] …`，模型加载后数秒可见，已验证输出正确
2. 开关直接写进 `run_OpenNav.yaml`，不再依赖命令行透传 → 结构上消除该失败面

**附带收获**：贪心 + 固定 seed 下管线**逐字节可复现**。这对配对 McNemar 是硬需求——运行间噪声为零，差异可全部归因到开关。

---

## 四、oracle 审计：一处已清干净，一处有残留

### 4.1 终止侧已清干净 ✅
`latest_goal_dist` 全代码库 grep **0 命中**。`E3_ARRIVAL_OVERRIDE_DIST` / `RESCUE_MAX_GOAL_DIST` / `TRAJECTORY_BYPASS_DIST` 等看似 oracle 的距离阈值，代码里**只判 `> 0`，已退化为纯开/关标志**；真正的门换成了 `persistent_visual_confirm`（连续 ≥2 帧视觉确认）+ `depth_confirm`。见 `stop_evidence_verifier.py:218-240, 280-295`。

### 4.2 phase 侧有残留 ⚠ 新发现
`phase_evidence.py:148`：

```python
elif current_step >= self.late_step_threshold and non_positive_count >= 2:
    phase = "recover"
```

`non_positive_count` 数的是 `recent_distance_gains <= 0`，而该量是**模拟器测地目标距离的差分 = GT**。实测 clean_baseline_v1：

- `recent_distance_gains` 在 650 步中 **550 步非空**
- **51 次** phase 被判为 `recover`，理由为 "late step with repeated non-positive distance gain"
- `PHASE_EVIDENCE.LOG_ONLY: false` 且 U1 决策生效 → 该路径影响决策

07-04 审计在 yaml 注释里写的 "recent_distance_gains now only feeds logging" 对 `stop_evidence_verifier` 成立，**对 `phase_evidence` 不成立**。

**待办**：查 `decision_audit` 判定这 51 次是否*决定性*（07-04 审计中多个 oracle 门测得 0 决定性）。**不影响开跑，但影响 SR=16 能否写进论文。**

---

## 五、新机制：终局俯视视角（ENDGAME_TILT_VIEW）

### 5.1 起因：机位俯仰恒为 0
`vlnce_baselines/utils.py:155`：

```python
orient_dict[str(base_angle_deg*k)] = [0.0, base_angle_rad*k, 0.0]
                                      ^^^ pitch 硬编码 0.0
```

12 路相机**只有 yaw 在变**，垂直方向零覆盖。`observe_view` 里的 `Elevation: Eye Level` 是字面事实。动作空间 `POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, HIGHTOLOW]` 也**没有 LOOK_UP/LOOK_DOWN** → agent 物理上无法低头。

**几何后果**：VFOV 90°、相机高约 1.25m（habitat 默认，yaml 未覆盖，*此值待实测确认*）→ 正前方地面进入画面的最近距离约 **1.25m**。R2R 目标大量是地面/矮家具（rug / bed side / stairs / chair / pool table）。**所以"近距欠检测 46%"里有一部分不是模型看错，是目标压根不在任何一张图里。**

### 5.2 为什么这条比队列里现有的值钱
veto 和回溯全在 OSR 侧。**俯仰视角作用在停止判据 = 转化侧**，正是 SR = OSR × 转化率 里没人管的那个乘数。

### 5.3 触发器设计（全部 oracle-free）

**武装（判定"进入指令最后阶段"，三条任一）：**

| 条件 | 来源 | 干净性 |
|---|---|---|
| `completed_action_count >= action_count - 1` | `_completed_action_count(estimation, actions)`，纯文本 | ✅ |
| `recent_ftv_window[-3:]` 有 True | 终点目标近期可见，纯感知 | ✅ |
| `current_step >= 0.6 × step_length` | 步数兜底 | ✅ |

**刻意不用 `phase == "verify"`** —— 见 §4.2，phase 带 oracle 残留，任何 key 在它上面的机制都会继承污染。

**击发**：武装状态下有 STOP 被提出（navigator 提停或 PSG 触发）。选它而不是"距离小于某值"，因为**距离就是 oracle**。

**限流**：`stop_verification` 实测 1008 次 / 650 步 ≈ 1.55 次每步，无门击发会让 VLM 成本翻倍。故加：位姿去重（<0.5m 且 <15° 复用）、每集上限 3 次。加武装门后实际落到每集约 1–3 次。

**俯仰角 −30°**：地面覆盖 0.33–4.66m，正好补上当前全盲的 0.33–1.25m，同时保留远景。0° 是现状，−45° 会丢远景。首选 −30°，但应在探针里同时试 −20/−30/−45 用数据选。

### 5.4 实现要点

| 文件 | 改动 |
|---|---|
| `config/default.py` | 新增 `OPENNAV_HARNESS.ENDGAME_TILT_VIEW` 配置块，`ENABLED: False` / `LOG_ONLY: True` |
| `base_il_trainer_llm.py:4104` | 仅在启用时追加 `TILT_RGB`/`TILT_DEPTH` 传感器 → 关闭时基线逐字节不变 |
| `base_il_trainer_llm.py:generate_input` | **跳过 `tilt_` 前缀键** |
| `base_il_trainer_llm.py:891+` | 配置解析、每集状态、`_tilt_endgame_armed` / `_tilt_pose_is_new` / `_tilt_observe` |
| `base_il_trainer_llm.py:_depth_stop_ok` | 在**早返回之前**调用 `_tilt_observe`，与 veto 开关解耦 |
| `run_OpenNav.yaml` | 配置块 + 横幅新增 `endgame_tilt=` 字段 |

⚠ **最大的坑（已避开）**：`generate_input` 按 `observations.keys()` 顺序给 RGB **位置编号**，`construct_image_dicts` 再把 1–12 映射到朝向。若俯视相机进入该扫描，**整个方向映射会被静默旋转**，导航链路全毁且不报错。故 `tilt_` 前缀隔离是承重设计，不是命名习惯。

### 5.5 状态
代码已合入，`py_compile` 通过，自由变量作用域已审计。**尚未运行验证** —— 需要一次 smoke 确认：(a) 横幅显示 `endgame_tilt=True`；(b) `endgame_tilt_view` 事件产出且 `tilt_depth_center_m` 非空；(c) 关闭时轨迹与基线逐字节一致。

---

## 六、当日结论与次日计划

### 结论
1. **SR = OSR × 转化率。队列里所有现成开关都只推 OSR，转化侧空无一物** —— 这是 SR 长期不动的结构性原因。
2. depth veto 机制成立但**单向**：能说"别在这儿停"，不能说"该在这儿停"。
3. 俯仰视角是第一个转化侧杠杆，且成因是**几何**而非模型能力 —— 这类问题修起来确定性最高。
4. clean_baseline_v1 的 oracle 清除**不完整**，phase 侧尚有 51 次 GT 依赖。

### 次日计划
1. **离线探针**（~30 min，无 LLM 调用）：回放基线 100 集停止位姿，一趟测三件事 —— 平视深度分布（定 veto 阈值 + 实测误杀数）、三档俯仰视图（定俯仰角 + 测能救回多少集）、锚点 ≤2m 阈值的标定依据。
2. **tilt smoke** 验证 §5.5 三条。
3. 查 `decision_audit` 判定 phase oracle 残留是否决定性。
4. ep100 待手动启动（当前 yaml：veto=True / backtrack=True / tilt=False）。

### 预期管理
本轮 ep100 若启动，**预期 OSR +5~10 / SR −3~+3 / SPL 下跌**。上行来自 56 集假停的大池子，下行来自 16 集成功的误杀，量级接近故方向不定。**应按测量跑，不按性能跑。**


- 从 4B 升到 7B/8B，很可能明显改善格式稳定性、候选选择和基础空间关系。
- 从 8B 升到更大的模型，收益更可能集中在复杂指令、相似目标区分和多图联合推理。
- 如果主要失败来自 waypoint 候选根本没有覆盖正确方向、Depth 不准确或目标实例跟踪错误，换大模型不会从根本上解决。
- 如果日志中大量出现“候选包含正确方向，但 Qwen 选错”或“有完整视觉关系，但模型把普通 doorway 当成指定 doorway”，换大模型才最值得。

#### 第 0 步：任务进入系统

Habitat 首先给系统一条导航指令和智能体的初始观测：

```text
Instruction:
Go through the door and turn left. Go to the left of the stairs.
Stop in the doorway to the left of the white double doors.

Initial observation:
12 RGB images
12 aligned Depth images
agent pose
episode_id
```

12 个 Direction ID 为 `0～11`，相邻中心方向间隔 `30°`，合计覆盖 `360°`。每个 RGB 都与同方向的 Depth 对齐。RGB 用来识别场景和物体，Depth 用来估计该方向附近表面的距离。

#### 第 1 步：指令只在开始时拆解一次

本地 Qwen3.5-4B 把原始指令拆成动作和地标：

```text
actions:
Go through the door
Turn left
Go to the left of the stairs
Stop in the doorway to the left of the white double doors

landmarks:
door
stairs
white double doors
doorway
```

Anchor Chain v2 再把它转换为当前代码实际使用的 JSON 对象：

```json
{
  "schema_version": "opennav.anchor_chain.v2",
  "anchors": [
    {
      "idx": 0,
      "landmark": "door",
      "room": null,
      "action": "go",
      "raw": "Go through the door",
      "terminal": false,
      "kind": "object",
      "key": "door",
      "parse_status": "parsed",
      "verification": {
        "predicate": "object_nearby",
        "parameters": {
          "max_waypoint_distance_m": 3.0,
          "min_consecutive_hits": 2
        },
        "on_unverifiable": "abstain"
      },
      "terminal_target": false
    },
    {
      "idx": 1,
      "landmark": null,
      "room": null,
      "action": "turn",
      "raw": "Turn left",
      "terminal": false,
      "kind": "direction",
      "key": "left",
      "parse_status": "parsed",
      "verification": {
        "predicate": "odometry_motion",
        "parameters": {
          "min_consecutive_hits": 1
        },
        "on_unverifiable": "abstain"
      },
      "terminal_target": false
    },
    {
      "idx": 2,
      "landmark": "stairs",
      "room": null,
      "action": "go",
      "raw": "Go to the left of the stairs",
      "terminal": true,
      "kind": "object",
      "key": "stairs",
      "parse_status": "parsed",
      "verification": {
        "predicate": "object_nearby",
        "parameters": {
          "max_waypoint_distance_m": 3.0,
          "min_consecutive_hits": 2
        },
        "on_unverifiable": "abstain"
      },
      "terminal_target": false,
      "terminal_predecessor": true
    }
  ],
  "n_anchors": 3,
  "n_landmarks": 4,
  "n_landmarks_aligned": 3,
  "alignment_coverage": 0.6666666666666666,
  "kind_counts": {
    "location": 0,
    "object": 2,
    "direction": 1,
    "unknown": 0
  },
  "degenerate": false,
  "terminal_policy": {
    "present": true,
    "directives": [
      "Stop in the doorway to the left of the white double doors"
    ],
    "target": "doorway",
    "predicate": "coordinated_stop",
    "requires": [
      "chain_complete",
      "all_previous_verified",
      "final_target_evidence",
      "stop_coordinator_allow"
    ],
    "in_progress_queue": false
  }
}
```

`anchors[0:3]` 是 ACN 按顺序检查的途中路线；STOP 指令不会进入该数组，而是完整保存在 `terminal_policy.directives` 中，最终目标为 `terminal_policy.target="doorway"`。其中 `terminal=true` 只表示 `stairs` 是路线队列的最后一项，不表示它是最终停止目标。

Anchor Chain 是静态计划，运行进度保存在另一个状态对象中。episode 初始化时的核心状态为：

```json
{
  "current_anchor_index": 0,
  "route_progress_complete": false,
  "terminal_target_confirmed": false,
  "goal_complete": false
}
```

#### 第 2 步：从 12 路观测中找出当前可走方向

Waypoint Predictor 根据当前 12 路 RGB-D 特征预测一组局部可通行 waypoint，由相对转角和移动距离表示的几何可行点：

```text
waypoint = angle_rad + distance_m
```

预测器先输出一张 `120 个角度格 × 12 个距离格` 的极坐标热图。角度分辨率为 `3°`，距离分辨率为 `0.25 m`；随后通过 NMS 抑制相邻重复点，最多保留 5 个峰值。多个精细角度如果落入同一个 30° Direction，还会合并或覆盖，所以最终候选可能少于 5 个。

例如当前得到 4 个候选：

```text
candidate_id=0,  direction_id=0,  angle_rad=6.283185, distance_m=2.00
candidate_id=2,  direction_id=2,  angle_rad=1.047198, distance_m=2.00
candidate_id=3,  direction_id=3,  angle_rad=1.570796, distance_m=1.50
candidate_id=11, direction_id=11, angle_rad=5.759587, distance_m=2.25
```

其中 `angle_rad` 的计算为 `2π - angle_idx / 120 × 2π`，`distance_m=(distance_idx+1)×0.25`。结合智能体当前世界位置和朝向，这两个值可以换算成世界坐标下的 waypoint。系统再把它关联到最接近的 Direction RGB-D 图像，让导航模型理解朝该点移动时当前能看到什么；图像只是 waypoint 的视觉说明，不是 waypoint 本身。

#### 第 3 步：把候选图像变成可理解的观测

RAM 和 spatial bot 读取每个候选对应的 RGB 图像，生成场景描述和物体标签，同时保留该方向的 Depth 与 waypoint 距离：

```text
Step ID: 1
Direction 2 Direction Viewpoint ID: 1
Elevation: Eye Level
Scene Description: An open doorway leads into a hallway.
Scene Objects: doorway | hall | wall | floor
Waypoint Distance: 2.0 meters
```

这些数据组成当前 step 的观测合同，随后同时提供给 ACN、终点证据检查和导航选择器。

#### 第 4 步：ACN 更新路线进度

ACN L1 比较当前观测、历史状态和 Anchor Chain，但只按顺序推进，不会因为看见后面的 `stairs` 就跳过前面的 `door`：

```text
ProgressUpdate:
current_anchor_index = 0
observed_anchor = door
anchor_hit_streak = 1
route_progress_complete = false
```

如果下一步再次得到一致且满足条件的 `door` 证据，标准实例连续命中达到 2 次后，ACN 才把第一个 Anchor 标记完成，并推进到 `left`。

#### 第 5 步：检查最终目标，但此时不允许误停

同一个 step 最多调用一次 Visual Target Verifier。它检查最终目标是否可见、在哪个 Direction、置信度是多少，并把该 Direction 和同方向 Depth 绑定：

```text
TerminalEvidence:
target = doorway
visible = true
confidence = 0.86
target_direction_id = 2
local_depth_m = 2.70
terminal_target_confirmed = false
reason = confidence_below_threshold
```

当前强证据置信度阈值为 `0.90`，最早允许确认的 step 为 `3`，目标方向局部 Depth 上限为 `3.0 m`。这个例子即使看到了 doorway，也因为置信度不足且路线尚未完成而不能停止。

#### 第 6 步：本地 Qwen 选择下一方向

导航选择器收到原始指令、路线进度、历史动作以及当前所有合法候选，返回一个候选 ID：

```text
Navigator:
Thought: The instruction first requires passing through the door. Direction 2 shows the open doorway.
Prediction: 1
```

`Prediction: 1` 指的是 `candidate_id=1`，该候选对应 `direction_id=2`。它不是 Habitat 的最终动作编号。

#### 第 7 步：把候选编译成环境动作

ActionCompiler 查回候选的真实角度和距离，并生成唯一可提交给 Habitat 的动作：

```json
{
  "action": {
    "action": 4,
    "action_args": {
      "angle": 1.047198,
      "distance": 2.0
    }
  }
}
```

`action=4` 表示导航移动；`1.047198 rad` 约为左转 `60°`，然后沿该 waypoint 移动 `2.0 m`。只有 ActionCompiler 可以生成最终环境动作。

#### 第 8 步：Habitat 执行并返回下一帧

系统调用 `env.step(action)`。Habitat 更新智能体位置和朝向，然后返回：

```text
collision
episode_over
new agent pose
12 new RGB images
12 new Depth images
```

系统用动作前后的 pose 计算真实位移。在弱通用目标的路线成熟度统计中，只有无碰撞且实际位移大于 `0.05 m` 的前进才计数；模型命令的 `2.0 m` 不等于机器人一定真实移动了 `2.0 m`。

#### 第 9 步：用新观测继续循环

随后系统再次执行同一条链：

```text
12-direction RGB-D observation
        ↓
navigable waypoint candidates
        ↓
scene descriptions and object labels
        ↓
ACN ProgressUpdate
        ↓
TerminalEvidence
        ↓
Navigator prediction
        ↓
StopCoordinator
        ↓
ActionCompiler
        ↓
Habitat env.step()
        ↓
next observation
```

例如经过数步后，ACN 状态可能依次变为：

```text
Step 2: door completed, current_anchor_index=1
Step 3: left completed, current_anchor_index=2
Step 5: stairs completed, route_progress_complete=true
```

`route_progress_complete=true` 只表示途中路线走完，不会自动变成 STOP。

#### 第 10 步：路线完成后确认最终目标

假设 Step 6 的 Direction 3 同时出现目标文字、视觉判断、到达迹象和近距离 Depth：

```text
TerminalEvidence:
target = doorway
visible = true
confidence = 0.94
target_direction_id = 3
local_depth_m = 1.32
instance_key = 513:terminal-instance:4
same_instance_hit_streak = 2
terminal_target_confirmed = true

ProgressUpdate:
route_progress_complete = true
terminal_target_confirmed = true
goal_complete = true
```

`same_instance_hit_streak=2` 表示连续两次证据属于同一个目标实例。路线完成与终点确认同时成立后，才得到 `goal_complete=true`。

#### 第 11 步：STOP 仲裁与正式停止

completion 链提出 STOP，但请求仍需经过 StopCoordinator：

```text
StopProposal:
source = completion
reason = goal_complete

StopDecision:
outcome = commit_goal_stop
```

StopCoordinator 批准后，ActionCompiler 输出：

```json
{
  "action": {
    "action": 0,
    "action_args": null
  }
}
```

Habitat 接收到 `action=0` 后结束 episode，并在评测阶段计算 `Success`、`SPL`、`nDTW` 和最终距离。这些评测真值只用于运行后的统计，不参与在线导航或 STOP 判定。

整个过程可以压缩为一句话：

```text
Instruction + 12 RGB-D pairs
→ Anchor Chain
→ waypoint candidates
→ visual descriptions
→ ACN progress
→ terminal evidence
→ local Qwen selection
→ StopCoordinator
→ ActionCompiler
→ Habitat action
→ next 12 RGB-D pairs
→ repeat until action 0 or step limit
```

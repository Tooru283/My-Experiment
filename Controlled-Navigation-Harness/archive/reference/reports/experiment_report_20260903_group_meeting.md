---
date: 2026-09-03
tags: [组会报告, Pipeline重构, ACN, STOP, TerminalTrack, EP100]
status: ready
related:
  - "[[主Pipeline重构设计-20260901]]"
  - "[[EP10问题审查与优化记录-20260902]]"
  - "[[STOP链条全面审查与重构建议-20260902]]"
  - "[[项目日报-20260902]]"
---

# Open-Nav / Controlled Navigation Harness 本周实验进展

**汇报时间：** 2026-09-03  
**统计范围：** 2026-08-31 至 2026-09-03  
**汇报目的：** 主 Pipeline 重构、ACN/STOP 状态链落地、本周 EP1/EP10/EP100 结果，以及大样本暴露的新问题和解决方案。

## 一、本周实验数据

### 1.1 EP10 测试

| 实验 | Steps | SR | OSR | SPL | nDTW | 最终距离 |
|---|---:|---:|---:|---:|---:|---:|
| goal_complete_split_ep10 | 9.70 | 0.30 | 0.60 | 0.1359 | 0.4644 | 5.303m |
| goal_complete_split_ep10_re | 9.90 | 0.30 | 0.60 | 0.1643 | 0.4663 | 4.906m |
| goal_complete_ep10 | 10.20 | 0.30 | 0.70 | 0.1353 | 0.4485 | 4.639m |
| goal_complete | 9.00 | 0.30 | 0.60 | 0.1722 | 0.4662 | 5.412m |
| stop_fix | 9.90 | 0.30 | 0.70 | 0.1672 | 0.4633 | 4.736m |

EP10 的 SR 始终为 0.30。其价值是逐轮发现远距离误停、proactive 重复 proposal、MOVE_BACK 崩溃，以及安全性收紧后 goal_complete=0 的活性问题。

### 1.2 EP100 大样本结果

**实验：** `terminal_track_v3_ep100_20260902_204313`

| 指标 | 数值 |
|---|---:|
| Episodes / steps | 100 / 970 |
| 平均步数 | 9.70 |
| SR | 0.20 |
| Oracle Success | 0.38 |
| OSR→SR | 52.6% |
| OSR-SR gap | 18 episodes |
| SPL | 0.109785 |
| nDTW | 0.341558 |
| 平均最终距离 | 7.684m |
| 平均路径长度 | 14.880m |
| 平均碰撞率 | 0.062 |
| 出现碰撞的 episode | 56 |
| Traceback / navigation_error | 0 / 0 |

与可信旧基线 SR20/OSR25 相比，本轮 SR 仍为 20，但是 OSR 提升为 38。

## 二、已解决的问题

### 2.1 非正式视觉字段能够绕过正式终点判断

旧逻辑中，只要视觉模型连续两帧都认为“当前画面里能看到最终目标”（日志字段 `final_target_visible`），并且认为“机器人已经处于可以到达或停止的位置”（日志字段 `arrival_evidence`），系统就可能进一步判定“终点已经确认”（`terminal_target_confirmed`）。即使 V2 正式检查认为证据不成立，或认为当前场景不适合判断，这两项较宽松的原始判断仍可能参与最终完成判定。

~~~text
V2 正式检查认为：当前证据不足，不能确认终点
视觉模型却认为：画面中存在最终目标，而且机器人已经到达
路线模块同时认为：所有途中步骤已经完成
→ 旧逻辑仍可能把整个导航任务判定为完成
~~~

**解决方法：** 建立唯一的 `TerminalEvidence` 判定入口。终点确认必须同时满足 V2 允许、目标可见、存在到达证据、连续两帧确认和目标方向 RGB-D 距离通过。同一步只生成一份证据，由 Goal Join、STOP proposal 和 StopCoordinator 共同复用。

**当前状态：已修复。**

### 2.2 StopCoordinator 的硬约束不完整

机器人执行正式 STOP 前，会由 StopCoordinator 汇总各模块的意见。旧策略只把旧版综合检查当作必须满足的条件；新版视觉检查虽然已经发现“目标身份不确定”或“距离证据不足”，其反对意见却可能只被记录在日志中，没有真正阻止停止。

~~~text
旧版综合检查认为：可以停止
新版视觉检查认为：当前目标或距离证据不足
路线模块认为：任务可能已经完成
→ StopCoordinator 仍可能采用旧版结果并提交 STOP
~~~

这意味着新版检查虽然参与了分析，却没有实际否决权。系统表面上存在多层门控，最终仍可能由较宽松的旧结论决定是否停止。

**解决方法：** StopCoordinator 现在必须同时获得旧链路基础检查、新版视觉终点检查和路线完成状态的支持。只要新版视觉检查明确反对，或因证据不足无法作出判断，正式 STOP 就会被拒绝；如果路线尚未完成或最终目标尚未确认，同样不得停止。三个检查项在日志中分别记录为 `legacy_combined`、`v2` 和 `progress`。

**当前状态：已修复。**

### 2.3 proactive STOP 过早且重复检查

首轮 EP10 测试 只有 97 个决策步，却产生了 33 次“可能已经到达终点”的主动停止提案，绝大部分随后又被正式协调器拒绝。同一个 step 内还可能重复调用视觉模型；由于没有增加新的图像或位移，两次回答的差异只是模型输出波动，而不是获得了新证据。

~~~text
第一次正式检查：当前证据不足，不能确认终点
同一步再次进行 proactive 检查：模型改口认为可能已经到达
系统产生额外 STOP proposal
→ StopCoordinator 随后又把该提案拒绝
~~~

这种重复检查既增加了推理耗时，也可能使同一步出现相互矛盾的终点结论。

**解决方法：** proactive 路径现在只记录“当前画面可能接近终点”的诊断提示（`proactive_terminal_hint`），便于之后分析，但该提示没有改变机器人动作的权限（`decision_effect=false`），不能直接生成正式 STOP proposal，也不能绕过统一的终点证据和 StopCoordinator。

**当前状态：已修复。**

### 2.4 MOVE_BACK 与普通候选动作的数据格式不一致

普通导航动作来自候选 waypoint 列表，每个候选都带有对应的方向、RGB 观察和距离等信息。`MOVE_BACK` 则是系统在需要脱困或回退时临时生成的合成动作，它并不对应候选列表中的某个 waypoint。旧代码虽然能够成功执行后退，却仍按照普通候选的格式保存历史，因此会查找一个实际上不存在的候选记录。

~~~text
机器人成功执行 MOVE_BACK
历史模块尝试在普通候选列表中查找 MOVE_BACK
候选列表中不存在这个合成动作
→ episode 在保存历史时出现 KeyError 或 ValueError
~~~

问题不在后退动作本身，而在执行模块和历史模块对其数据格式理解不一致：前者知道它是特殊动作，后者却把它当成普通 waypoint。

**解决方法：**`MOVE_BACK` 现在以独立的 synthetic action 格式写入历史，只记录后退原因、执行结果及必要的位姿变化，不再要求它提供普通候选才具有的观察数据。真正的普通候选如果缺少必要数据，系统仍会明确报错，避免静默掩盖数据损坏。

**当前状态：已修复；EP100 中未再出现由 MOVE_BACK 引起的 navigation error。**

### 2.5 终点限定语被错误拆成路线 Anchor

自然语言指令经常使用逗号补充终点位置。逗号后的内容可能是在描述同一个终点与参考物的关系，而不是要求机器人再执行一个新的途中步骤。旧解析器按逗号拆分时，可能把终点限定语错误放入路线进度队列。

~~~text
原始指令：
Stop between the sofas, next to the dining room.

正确含义：
停在两张沙发之间，并且这个位置位于餐厅旁边

旧解析结果：
途中目标 1：到达 sofas
途中目标 2：寻找 dining room
→ “next to the dining room” 被误当成新的路线步骤
~~~

这样不仅会让 ACN 多等待一个并不存在的途中步骤，还会使终点检查丢失“与 dining room 相邻”这一用于区分目标实例的重要条件。

**解决方法：**解析器现在先识别完整的 STOP 子句，再将逗号后的空间关系短语与终点主体合并。上面的指令会被保留为“目标位于两张沙发之间，参考物为 dining room，关系为 next to”，这些信息进入 terminal policy，而不是 progress queue。

**当前状态：该类逗号限定语已经修复并补充测试。像 `turn right past the stairs` 这种同时包含转向和地标关系的复合途中动作，仍需单独设计。**

## 三、EP100 暴露出的核心问题

本轮 EP100 最值得讨论的，不是某一个判断条件应该改成多少，而是系统怎样从“看见一个相似物体”走到“确认语言所指的具体目标”，以及怎样判断一条途中路线已经真正完成。下面将问题归纳为五类。

### 3.1 看见正确类别，不等于识别出正确目标

17 次正式 goal STOP 中只有 5 次成功，另外 12 次停在错误位置：

~~~text
goal STOP precision = 5 / 17 = 29.4%
~~~

其中两个最明显的错误位置距离真实目标 17.105m 和 22.348m。这不是成功半径边缘的小误差，而是系统把另一个对象或区域当成了指令目标。

例如，指令是：

~~~text
Walk out through the door.
Follow the red carpet to the right of the table.
Stop in the next doorway.
~~~

场景中可能同时存在入口 A、走廊门洞 B 和路线末端门洞 C。模型在中途看到 B 时，确实可以正确识别出 doorway，但指令要求的是经过前述路线后出现的 next doorway，即 C。

~~~text
识别结果：这里有一个 doorway       → 可能正确
任务判断：这里就是 next doorway     → 仍可能错误
~~~

类似情况还包括：

- `the couch at the end of the hallway`，而不是任意 couch；
- `the sink in the other bathroom`，而不是当前 bathroom 的 sink；
- `the doorway to the left of the white double doors`，而不是任意 doorway。

**核心问题：**

> 在存在多个同类物体的室内环境中，怎样从“识别物体类别”进一步达到“识别语言所指的具体实例”？

连续几帧看到同一个 doorway 只能说明观察稳定，不能证明它就是指令所指的 doorway。可靠判断还需要参考物、空间关系和路线顺序。

### 3.2 多种证据同时成立，为什么仍然会判断错误？

当前系统综合视觉、方向、距离、连续观察和路线进度判断终点。直觉上，多种证据应比单一判断可靠，但它们可能共同继承同一个错误来源。

例如：

~~~text
1. Qwen 把普通 doorway B 认成最终 doorway C
2. Qwen 同时给出 target_direction_id=3
3. Direction 3 近处恰好有门框或墙面
4. 系统得到 local surface depth=1.2m
5. 后续两帧仍然看到 doorway B
6. 路线进度此时也被判定完成
7. 所有条件共同支持 STOP
~~~

表面上系统获得了五种证据：

~~~text
目标可见
+ 目标方向
+ 近距离
+ 连续观察
+ 路线完成
~~~

但前三项都建立在最初的视觉识别上；距离也只是该方向附近表面的距离，不一定属于目标本身；连续观察则可能是在稳定跟踪同一个错误对象。

**核心问题：**

> 多证据系统怎样避免多个判断共同继承同一个上游错误？

真正独立的证据应该能够互相反驳。例如：

- 视觉检测负责给出目标和参考物的位置；
- Depth 只读取目标区域内的像素；
- 几何规则单独验证 `left of`、`under`、`between`；
- 路线历史单独验证 `next`、`after`、`first`；
- 跨帧跟踪验证是否为同一个空间实例。

这样即使一个模块认错，其他证据仍有机会拒绝，而不是共同放大第一次错误。

### 3.3 自然语言关系怎样变成可检查条件？

错误 episode 中大量出现关系词和顺序词：

~~~text
next doorway
under the mirror
in front of the animal head
left of the white double doors
between the two white sofas
first room on the right
~~~

小模型比较容易识别 doorway、mirror、rug、sofa 和 room 等主体名词，却难以稳定确认“哪一个”“相对位置是什么”和“在路线的哪个阶段出现”。

以这条指令为例：

~~~text
Stop in the doorway to the left of the white double doors.
~~~

系统不能只保存：

~~~json
{"target": "doorway"}
~~~

更完整的目标应表达为：

~~~json
{
  "target": "doorway",
  "reference_object": "white double doors",
  "relation": "left_of"
}
~~~

随后把语言条件转换为可观察问题：

~~~text
是否检测到 doorway？
是否检测到 white double doors？
二者是否出现在同一场景？
doorway 的图像位置是否在 double doors 左侧？
两者的 Depth 是否符合相邻关系？
路线是否已经走到允许寻找终点的阶段？
~~~

对于 next、after、first room 等顺序关系，单张图像无法独立回答，还必须结合机器人已经走过的路线。

**核心问题：**

> 如何把开放式导航语言中的空间关系和路线顺序，转化为能够由图像、深度和运动历史共同验证的条件？

普通目标检测只能解决“物体在哪里”，不能单独解决“是不是下一个”“是不是经过某地之后的那个”。

### 3.4 模型自报的置信度是否可信？

下面的表不是 17 次正式 STOP 的分布，而是 EP100 中 970 次 current-view 视觉证据调用的原始 confidence 分布：

| Confidence | 次数 |
|---:|---:|
| 0 | 16 |
| 0.80 | 13 |
| 0.85 | 9 |
| 0.90 | 356 |
| 0.95 | 557 |
| 1.00 | 18 |

其中 0.90 和 0.95 合计 913 次，占 94.1%。而在最终进入正式停止的样本中，正确和错误判断又几乎都给出 0.95，包括距离真实目标十几米甚至二十多米的错误样本。

这表明 Qwen3.5-4B 的 confidence 更像“模型表达确定程度时偏好的几个数字”，不是经过真实标签校准的正确概率。

例如：

~~~text
模型输出 confidence=0.95
不等于
在所有同类样本中有 95% 的判断正确
~~~

正式 STOP 又要求 confidence≥0.90，因此进入停止链的样本本来就被筛选成高分样本。这解释了为什么正式停止中 0.95 特别多，但无法解释或消除其中的错误。

**核心问题：**

> 当视觉语言模型的自报置信度与实际正确率不一致时，系统应该怎样表示和使用不确定性？

暂定方案不是简单把阈值从 0.90 调到 0.95，而是：

1. 引入 Grounding DINO 或 YOLO-World，获得目标和参考物的实际位置；
2. 使用目标框或 mask 内的 Depth；
3. 将关系、距离和跨帧一致性分别验证；
4. 把 Qwen confidence 降为辅助信号；
5. 用带正确/错误标签的回归样本检查不同分数对应的真实正确率。

如果 confidence 与真实正确率没有单调关系，就不应让它参与正式 STOP 放行。

### 3.5 ACN 应在什么证据强度下宣布途中指令完成？

ACN 在 970 个 step 中推进 115 次、放弃判断 760 次，约 78.4% 的更新为 abstain；100 个 episode 中只有 40 个曾确认路线完成。

保守判断能够避免系统凭空宣布路线完成，但也可能让机器人实际已经完成动作，ACN 却仍停留在旧步骤。

当前 object Anchor 的典型条件是：

~~~text
某个候选方向的图像中出现目标
+
该方向关联的 waypoint distance ≤ 3.0m
+
连续命中至少 2 次
~~~

这里的 waypoint distance 不是目标物体的真实距离，也不一定来自最终被选中和执行的候选。它只表示：看到目标的方向上存在一个不超过 3m 的候选移动点。

因此可能出现两种相反错误。

#### 情况 A：过早推进

~~~text
机器人在远处看见 door
该方向存在一个 2m waypoint
连续两帧仍能看见 door
→ ACN 宣布 Go through the door 已完成
~~~

但机器人可能还没有穿过门。系统随后开始执行 Turn left，导致后续路线整体提前。

#### 情况 B：延迟推进

~~~text
Step 2：门在正前方，只形成第一次有效命中
Step 3：机器人真正穿过门，门移动到侧后方
Step 4：当前候选图像中不再出现门
~~~

ACN 没有获得连续两次命中，仍保持：

~~~json
{
  "current_anchor_index": 0,
  "transition": "abstain"
}
~~~

此时机器人实际已经穿过门，但导航模型仍收到：

~~~text
current_target = Go through the door
~~~

它可能回头重新找门，或者一直无法进入 Turn left 和 stairs 阶段。

**核心问题：**

> 路线进度估计应如何平衡错误推进和延迟推进？

这不能通过统一调整一个 3m 阈值解决，因为不同动作的完成含义不同：

| 指令类型 | 更合理的完成证据 |
|---|---|
| Go toward the fireplace | 与目标的距离持续下降并进入近距离 |
| Go past the couch | 先接近，再发生相对方位变化 |
| Go through the door | 门从前方变到侧后方，同时发生有效前进 |
| Enter the bedroom | 穿过入口后，房间语义持续成为主要观察 |
| Turn left | 实际累计朝向变化达到要求 |

后续实验应分别测量：

~~~text
False Advance：动作未完成但 ACN 推进
False Hold：动作已完成但 ACN 不推进
推进延迟：真实完成后多少 step 才被确认
~~~

只有这样才能判断 ACN 是太宽松还是太保守，而不能仅凭 78.4% 的 abstain 比例下结论。

## 四、运行时间

本轮运行约 14 小时 38 分钟，平均 54.3 秒/step。

| 操作 | 次数 | 平均 | 总耗时 | 墙钟占比约 |
|---|---:|---:|---:|---:|
| `visual_evidence` | 970 | 22.063s | 21400.7s | 40.6% |
| `stop_current_view_evidence` | 970 | 7.154s | 6939.1s | 13.2% |
| `navigator_move_to_next_vp` | 953 | 11.583s | 11038.9s | 20.9% |
| 三项合计 | — | 40.8s/step | 39378.7s | 74.7% |

其余约 13.5 秒/step 主要来自逐候选 RAM、SpatialBot3B、Qwen observation/thought summary、Waypoint Predictor 和 Habitat。ACN、实例跟踪、StopCoordinator、ActionCompiler 均为毫秒级。

## 五、`visual_evidence` 的重新定位

当前 visual_evidence 同时负责开放词汇地标、指令匹配、终点可见性、arrival、target direction 和 selector fallback。它每步调用、耗时最高，且 EP100 未表现出可靠 STOP 精度。

~~~text
Terminal parser
→ target / reference object / relation / order
→ open-vocabulary detector or grounder
→ target bbox / reference bbox
→ bbox/mask-aligned Depth
→ deterministic relation checker
→ VLM only for complex relations or fallback
→ StopCoordinator
~~~

普通 YOLO 适合固定类别，但不足以覆盖 doorway、archway、next doorway、between two sofas 等开放指令。优先评估 YOLO-World、Grounding DINO 或 OWL-ViT 一类开放词汇 grounder；VLM 改为路线末段按需调用。

## 六、结论的可信边界

### 6.1 已验证

- 主链稳定运行 100 episodes；
- ACN L1 是唯一 ProgressProvider；
- proactive 不直接提出 STOP；
- StopCoordinator 与 ActionCompiler 单一出口成立；
- MOVE_BACK 不再造成运行错误；
- 三种完成状态拆分生效；
- target direction 与 Depth 已绑定；
- EP100 存在显著误停、漏停和 selector empty；
- 模型调用是主要耗时。

### 6.2 尚未验证为性能收益

- Anchor Chain v2 对 SR/OSR 的独立贡献；
- TerminalTrack 对 STOP precision 的净收益；
- 弱通用路线成熟度的总体收益；
- local Qwen4B 相比旧模型的净收益；
- backtrack、candidate prior、短时证据缓存的独立贡献；
- 开放词汇 detector 相比当前 VLM 的收益。

## 七、下一轮实验计划

### R0：建立 35 个 episode 的定向回归集

- 12 个错误 goal STOP；
- 18 个 Oracle Success=1, Success=0；
- 5 个正确 goal STOP。

先观察错误类型转移，不立即跑新 EP100。

### R1：关系型终点结构化

~~~json
{
  "target": "doorway",
  "reference_objects": ["white double doors"],
  "relations": [
    {"type": "left_of", "subject": "doorway", "object": "white double doors"}
  ],
  "route_order_constraint": "after stairs"
}
~~~

原指令有 reference/relation 时，主体词单独命中不得提交 goal STOP。

### R2：目标区域 Depth

~~~text
query → bbox/mask → aligned Depth → target_region_depth_m
~~~

实现开放词汇 TargetGrounder；没有 bbox/mask 时，Direction local surface depth 降为辅助证据。

### R3：visual_evidence A/B

| 组 | 策略 |
|---|---|
| A | 当前每 step 完整调用 |
| B | 仅最后 Anchor、route complete、selector STOP 或有效 memory 时调用 |
| C | 禁用终点 VLM，只用 grounder/RAM/规则 |

先用 EP20 比较 SR、OSR、goal STOP precision、漏停和 step latency。推荐 B，而不是删除全部视觉终点能力。

### R4：缩短模型调用链

- selector 只输出短 JSON candidate ID；
- 删除或降频 observation/thought summary；
- 为 candidate perception、history summary、env step 增加计时；
- 评估 RAM/SpatialBot3B 批处理和缓存；
- 将平均 step 时间先降到 30 秒以内。

### R5：再决定步数预算

在关系终点和 selector 稳定性改善前，不先将 10/12 步提高到 15/17；待漏停回放证明主要受预算限制后，再做单变量 C2 A/B。

引用文件与数据

- [主 Pipeline 设计](../../../docs/主Pipeline重构设计-20260901.md)
- [当前全链路图解](../../../docs/当前全链路输入输出图解-20260902.md)
- [EP10/EP100 问题审查](../../../docs/EP10问题审查与优化记录-20260902.md)
- [STOP 链条审查](../../../docs/STOP链条全面审查与重构建议-20260902.md)
- [项目日报](../../../docs/项目日报-20260902.md)
- [EP100 主日志](../../../../logs/navigation_records/ep100/20260902/terminal_track_v3_ep100_20260902_204313_train_navigation_20260902_204331.log)
- [EP100 JSONL](../../../../logs/navigation_records/ep100/20260902/terminal_track_v3_ep100_20260902_204313_train_navigation_20260902_204331.jsonl)
- [EP100 汇总](../../../../logs/eval_results/ep100/20260902/terminal_track_v3_ep100_20260902_204313/stats_ckpt_val_unseen.json)

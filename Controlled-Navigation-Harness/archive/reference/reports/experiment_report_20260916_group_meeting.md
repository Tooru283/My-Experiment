---
date: 2026-09-16
tags: [组会报告, 两阶段解析, ACN, 动作完成验证, STOP, EP100]
status: ready
related:
  - "[[experiment_report_20260903_group_meeting]]"
  - "[[动作完成验证-问题与优化方案-20260908]]"
  - "[[指令两步解析修复与验证-20260908]]"
  - "[[EP10关系与动作目标绑定修复-20260909]]"
  - "[[EP100运行分析-parser_v22_prior_logonly-20260910]]"
---

# Open-Nav / Controlled Navigation Harness 近两周实验进展

**汇报时间：** 2026-09-16  
**统计范围：** 2026-09-03 至 2026-09-16  

一、实验数据 

### 1.1 固定 EP10

固定集合均为 `244、218、226、371、602、1092、748、116、513、11`。各轮代码与配置不完全相同，因此下表用于展示问题演进，不作为严格单变量消融。

| 实验 | Steps | SR | OSR | SPL | nDTW | 最终距离 | 墙钟时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACN action completion v3，09-07 | 10.10 | 0.30 | 0.70 | 0.1200 | 0.4630 | 4.727m | 1:31:24 |
| LLM roles v1，联合解析| 9.90 | 0.20 | 0.30 | 0.0832 | 0.3014 | 6.438m | 1:10:07 |
| two-stage 132650| 10.00 | 0.20 | 0.40 | 0.0937 | 0.2803 | 7.464m | 1:14:18 |
| two-stage 205014| 10.00 | **0.30** | 0.40 | **0.1350** | 0.2943 | 7.087m | 1:21:13 |


### 1.2 EP100 大样本结果

| 运行 | Episodes | SR | OSR | SPL | nDTW | 最终距离 | Steps | Path length | Collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| terminal_track v3，09-02，参考基线 | 100 | 0.20 | **0.38** | 0.1098 | 0.3416 | 7.684m | 9.70 | 14.880m | 0.0620 |
| target_binding v3.3 retry| 100 | 0.14 | 0.31 | 0.0871 | 0.3361 | 7.865m | 7.74 | 11.764m | **0.0426** |
| parser v2.2 + prior log-only| 100 | **0.22** | 0.30 | **0.1244** | **0.3651** | **7.483m** | 10.04 | 13.244m | 0.0690 |

`target_binding_v3.3_retry` 虽然完成 100 条，但有 24 条在导航开始前因 计划校验失败而局部终止 。只看 76 个有效计划，结果为 `SR=0.1842、OSR=0.4079`。

parser v2.2 先进行了不启动 Habitat 的 EP100 实时预检：100/100 计划有效；action 阶段 100 条首次通过，landmark 阶段 87 条首次通过、13 条一次纠错后通过，共 213 次逻辑请求。随后完整 EP100 也实现 100/100 计划解析成功。

与 09-02 参考基线相比，最新运行 SR 增加 2 个百分点，但 OSR 减少 8 个百分点。固定十项的 SR/OSR 仍为 `0.20/0.40`，低于 09-02 的 `0.40/0.70`。目前最稳妥的结论是：解析覆盖率已经恢复，整体 SR 略高于参考基线，但进入过成功半径的能力没有恢复。

## 二、已解决的问题

### 2.1 动作完成从名词可见改为状态变化验证

旧链路容易把“看到门、楼梯或房间”理解成动作已经完成。当前 ACN 根据动作类型使用不同证据：转向读取真实 heading delta，普通移动读取实际位移，进入或离开房间比较动作前后房间状态，并把无法支持的 `cross/pass/traverse` 明确标成 unsupported。

同时修复了以下确定性错误：

- turn heading 符号与实际执行统一；88 次真实转向回放的最大角度误差约 `4×10⁻⁶` 度；
- 动作起点在动作开始时记录，不再晚一步采集；
- 候选视图按世界朝向比较，同一相机编号不再自动视为同一观察方向；
- route complete 成为稳定终态，不再访问不存在的下一项而触发 IndexError；
- `Leave the bathroom and closet` 拆成独立离开动作；
- `Go toward the house and through the doors` 拆成接近与穿越动作；
- `Walk across the floor` 使用 motion 位移证据，不再把 floor 当成需要跨越的实体。

最后三项已在固定 EP10 中看到对应在线结构生效；
但是`cross/pass/traverse` 的完成事件本身仍未实现，需要后续改进

### 2.2 LLM 解析指令仍需要两阶段解析

模型继续负责开放语言理解、候选视觉解释和每步选路，路线完成和正式 STOP 权限由可检查的程序状态决定。

~~~text
原指令
→ 阶段一：actions
→ 阶段二：landmarks + terminal_target
→ 程序验证完整 STOP、主终点与参照物
→ ACN 路线状态与正式 STOP 证据链
~~~

联合解析因 EP513 把一个 STOP 拆成两个而废弃。新的两阶段流程保留完整停止原文，并把 `area between the two white sofas` 作为主停止位置，把 dining-room entrance 作为关系参照物。缓存指纹包含两个提示词、模型、地址与 schema 版本，旧提示词缓存不会静默复用。

解析失败最多纠错一次；持续失败时记录 `instruction_plan_failed`。在 v3.3 retry 中，这类失败已经局部隔离，不再让整轮 EP100 崩溃；在 v2.2 中，已知的 `there`、`once/when`、句首位置短语和中间停顿等句式实现 100/100 parse-only 与 100/100 在线解析成功。

## 三、近期的核心问题

### 3.1 指令虽然解析成功，后续模块却可能在追不同的终点

这里的“解析成功”只表示 Qwen 输出的 actions、landmarks 和终点字段格式正确，并通过了当时的程序检查。它不保证导航过程中每个模块最终使用的是同一个终点。

问题出在解析结果向后传递的过程中：解析器已经判断出终点，但这个结果没有作为一个固定字段一直传下去。系统只保留了一个地标列表，后面的路线进度模块和 STOP 验证模块丢失了目标。

新写的代码中 terminal_target 是后来加入解析器的，而下游仍保留早期的 actions + landmarks 接口，代码没有完全修改统一。

例如：

~~~text
原指令：Go alongside the pool towards the bar

人的理解：沿着泳池走，最终前往 bar
解析器识别的终点：bar
路线进度模块后来选择的终点：pool
STOP 验证模块后来选择的终点：bar
~~~

这样会产生直接冲突：路线进度模块可能认为“到达 pool 就完成路线”，STOP 模块却在检查“是否到达 bar”。即使两个模块各自都正常运行，它们确认的也不是同一个地方，最终可能过早完成、一直不完成，或者在错误位置提交 STOP。


修复方法是让解析器输出唯一的 `resolved_terminal_target`，并把它原样传给 AnchorChain、ProgressProvider 和 VisualTargetVerifier。后续模块只能读取这个字段，不能重新猜终点；如果启动前发现三个模块的终点不相同，就直接报告契约错误，不进入导航。

### 3.2 SR 与 指令完成不是同一个概念 - 让最终位置更靠近正确终点，而不是主动stop，影响到后续的改进

最新 EP100 的 22 个 SR 中，21 个在步数上限结束，只有 1 个通过正式 STOP。整轮共有 60 次 STOP 提案：55 次 selector 提案全部拒绝；5 次 progress-completion 提案中 3 次提交、2 次拒绝。最终状态为：

~~~text
route_progress_complete = 13
terminal_target_confirmed = 6
goal_complete = 3
正式 STOP 成功 = 1
~~~

EP1117 在 `distance_satisfied=false`、TerminalEvidenceMemory 未确认、`terminal_target_confirmed` 为空时仍得到 `goal_complete=true` 并在 7.137m 处停止。直接原因是隐式终点策略允许 route complete 单独推导 goal complete。这违反三段状态的预期语义。


### 3.3 方向局部表面深度不能证明目标实例距离 - 

v3.3 的 4 次正式 STOP 只有 1 次成功；最新运行的 3 次正式 STOP 也只有 1 次成功。两个错误样本分别停在 10.948m 和 7.137m。最新运行全部 1004 个 `terminal_spatial_evidence` 仍使用：

~~~text
direction_local_surface_not_instance_mask
~~~

当前做法只能说明目标方向附近某个表面较近，不能说明被语言指代的 lamp、doorway、sofa 区域或其他目标实例较近。目标实例 mask 深度仍是 STOP 的 P0。

### 3.4 `cross/pass/traverse` 缺少动作事件验证

最新 EP100 最终有 30 个 episode 卡在未实现动作事件：`pass=13`、`cross=9`、`traverse=8`。这些 unknown 已经阻止 ACN 推进，直接影响正式进度。

可靠验证需要动作前后事件，而不是单帧名词可见性：

| 动作 | 所需证据 |
|---|---|
| cross / go through | 开口或边界实例在动作前方，发生有效位移，随后出现在侧后方或房间语义改变 |
| pass | 先接近同一实例，再发生相对方位翻转并继续前进 |
| traverse / across region | 区域内有效位移、入口与出口或区域语义的时序变化 |

waypoint distance 只是候选移动长度，不能当作目标距离或穿越完成证据。

## 四、后续规划

1、加入受语言动作约束的未来物理空间推演

  计算每个 waypoint 会如何改变目标相对位置，再指导选路。

公式大概：

$$ \hat r_{t+1}^{(i)} = R(-\Delta\psi_i) (r_t-\Delta p_i) $$

其中：

\(r_t\)：当前目标相对 agent 的位置；
\(\Delta p_i\)：候选动作计划让 agent 平移多少；
\(\Delta\psi_i\)：候选动作计划让 agent 转多少；
\(\hat r_{t+1}^{(i)}\)：假设候选被完整执行后，目标相对新 agent 的预计位置。

2、对于判断“是不是同一个 sofa”的问题，可以截取 图片 → Qwen的内部特征来解决，但是精确几何深度的问题仍然没解决

  可以借鉴的一种方法：不是让 VLM“先把自己看到的世界翻译成 JSON”，再让规划器读取 JSON；而是让规划器直接读取 VLM 已经形成的内部神经表征。






















引用文件与数据：

- [动作完成验证：问题与优化方案](动作完成验证-问题与优化方案-20260908.md)
- [LLM 调用与提示词审查](LLM调用与提示词审查-20260908.md)
- [两阶段解析修复与验证](指令两步解析修复与验证-20260908.md)
- [固定 EP10 205014 分析](EP10最新运行分析-205014-20260908.md)
- [关系与动作目标绑定修复](EP10关系与动作目标绑定修复-20260909.md)
- [target_binding v3.3 EP100 分析](EP100运行分析-target_binding_v33-20260909.md)
- [parser v2.2 EP100 分析](EP100运行分析-parser_v22_prior_logonly-20260910.md)
- [最新 EP100 汇总](../../logs/eval_results/ep100/20260909/parser_v22_prior_logonly_ep100_20260909_234512/stats_ckpt_val_unseen.json)
- [最新 EP100 逐集统计](../../logs/eval_results/ep100/20260909/parser_v22_prior_logonly_ep100_20260909_234512/stats_ep_ckpt_val_unseen_r0_w1.json)
- [最新 EP100 导航记录](../../logs/navigation_records/ep100/20260909/parser_v22_prior_logonly_ep100_20260909_234512_train_navigation_20260909_234530.jsonl)

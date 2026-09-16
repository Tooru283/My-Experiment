---
date: 2026-09-16
updated: 2026-09-16
tags: [Qwen-Drive, 方法借鉴, training-free, 空间证据, 事件状态, 航点指导]
status: method-adaptation-proposal
related:
  - "[[物理事件导航构想与当前实验对照评估-20260916]]"
  - "[[EP100运行分析-parser_v22_prior_logonly-20260910]]"
---

# 借鉴 Qwen-Drive 方法改进 Open-Nav：training-free 空间、事件与选路方案

> 本次修订依据用户明确的目标：仿照论文的方法改进现有实验，并保持 training-free。本文将论文事实、当前实现和拟议改动分开说明；以下新模块尚未实现，收益需要实验验证。

## 1. 核心结论

**最值得借鉴的是：让空间理解产生可检查的几何输出，让规划消费这些信息，并用实际执行结果检查决策是否兑现。** 在我们的项目中，可以落实为：

```text
当前 Qwen3.5-4B + 冻结的实例感知模型
                   ↓
统一的目标实例、相对几何与短时运动证据
                   ↓
Physical Event State + 候选执行后的相对状态预测
                   ↓
现有 waypoint selector → 实际执行 → 事件证据更新
```

这会给当前实验增加三个具体能力：

1. 将“目标方向附近有一个近表面”升级为“所指实例在哪里、哪个区域的深度可用”；
2. 将“当前还在执行 PASS”细化为“同一 sofa 已接近，接下来应形成经过关系”；
3. 让候选选择看到每个动作对事件的预期影响，避免一直朝目标物体靠近。

**最小落地顺序：统一终点契约 → 同一实例上的几何证据 → PASS 事件验证 → 候选物理转移指导。** 多帧 VLM 输入作为可独立验证的辅助项；KV、特征注入和训练新 head 不进入本轮方案。

原文档把根 VLM 替换 A/B 放在前面，偏离了本次需求。本版将实验变量改为证据表示、时序观察和事件指导，固定现有 backbone。

## 2. 论文里真正需要借鉴的机制

### 2.1 论文事实及其适用范围

Qwen-Drive 保留 VLM 主干结构，增加显式三维感知和规划模块；感知监督会更新共享表征，随后训练规划模块。它不是仅靠提示词或冻结模型拼装得到的系统。论文也承认，规划轨迹可能不符合生成的理由，三维监督对规划增益的因果归属尚未建立。[论文 §2、§3.4、§5](https://arxiv.org/html/2609.00111v1)

据此，本项目应借鉴它的**结构分工与证据接口**，由现有 RGB-D 和确定性计算承担几何部分。我们提出的 PASS/CROSS 验证、权威 EventState 和 ACN 推进是室内导航适配设计，不是论文已有模块的复刻。

### 2.2 六项方法迁移

| 借鉴点 | 对当前项目的启发 | training-free 实现方式 | 验证什么 |
|---|---|---|---|
| 显式空间输出 | 文本说“近、经过、在左边”还需要可检查的对象和坐标 | 冻结目标定位/分割 + 原始 RGB-D + 标定，输出实例几何 | 实例正确率、距离误差、关系判断 |
| 几何与语义互补 | 类别和指令角色由语义提供，米制位置由传感器约束 | 语义身份与几何字段分别记录、相互校验 | 是否减少错误实例上的自洽证据 |
| 有标识的多视图、多时刻输入 | 明确每张图来自何时、哪个方向，避免混合观察 | 目标相关方向的两帧输入 + 时间/位姿标签 | 时序辨别是否优于同预算单帧输入 |
| 共享信息服务不同任务 | 选路、完成验证和 STOP 应引用一致证据 | 同一步共用不可变 EvidenceBundle，模块只读 | 目标、帧、实例是否一致；冗余调用是否减少 |
| 专用动作输出接口 | 规划应对具体执行量负责 | 为每个现有 waypoint 输出预测关系、事件进展和有效性 | 预测是否兑现、选路是否改善 |
| 分层检查能力与行为 | 空间输出正确，不自动等于整段导航成功 | 空间评测 → 事件回放 → 候选预测评测 → 在线导航 | 每层改善能否传到 SR/OSR/SPL |

这张表是本项目的迁移判断，不是论文对室内导航的实验结论。具体源机制见下面的官方接口核对。

### 2.3 三个容易误读的地方

**第一，共享表征不等于共享 JSON。** 官方规划模块直接读取 VLM 内部 K/V，并在数值轨迹空间中输出动作；不是先将 BEV 检测结果写成文字，再让聊天模型重新规划。我们第一版共享的是有来源的结构化证据，尚不具备训练后的共享潜在表征，应准确描述这一差别。[官方 model.md](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/model.md)

**第二，外接模块不等于没有训练。** 官方感知路径融合视觉和 VLM 特征，包含三维检测、占用和地图分支。将这些任务改成冻结感知加几何计算，是我们为 training-free 约束做出的设计选择，不能宣称获得了原感知头同等能力。[官方 perception.md](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/perception.md)

**第三，多帧标签只规定输入组织。** 官方数据接口明确相机、帧次序与 ego 坐标；标签本身不保证小模型会可靠跟踪实例。我们需要实际的跨帧关联和坐标变换，再检验标签的增量作用。[官方 data.md](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/data.md)

## 3. 当前实验为什么适合这样改

当前完整 EP100 是 `parser_v22_prior_logonly_ep100_20260909_234512`：SR=22%、OSR=30%。1004 个终点空间证据均使用方向局部表面深度；30 个 episode 最后卡在未实现的 PASS/CROSS/TRAVERSE。现有代码已有 ACN、ActionReceipt 和只读进度反馈，因此可以保留主流程，逐步补齐空间与动作证据。[当前实验分析](EP100运行分析-parser_v22_prior_logonly-20260910.md)

| 当前接口/行为 | 缺口 | 借鉴后应新增的能力 |
|---|---|---|
| `visual_evidence` 的可见性、文字关系与自报 confidence | 缺乏实例区域与米制依据 | 把语义声明绑定到具体 mask/区域、视图和证据来源 |
| `terminal_spatial_evidence` 的方向中心/近表面统计 | 近处表面未必是目标 | 对已关联实例的有效深度做估计并保留不确定性 |
| `TerminalInstanceTracker` 的现有近似关联 | 内部一致仍可能跟错对象 | 跨帧外观、几何、目标修饰语的联合关联 |
| `ProgressLocator` 对 PASS 等返回 unsupported | 无法可靠推进当前动作 | 以执行回执和新观察验证时间事件 |
| `format_navigation_feedback` 提供当前动作和缺失项 | 没有明确的期望物理状态与候选影响 | 传入 phase、expected transition、候选预测摘要 |
| selector 返回候选 ID | 没有可直接加权的语义概率 | 先用有界结构化提示指导；显式重排另做实验 |

终点契约修复仍是共同前置条件。它确保各模块讨论同一目标，属于工程正确性；后续方法组都使用同一修复，避免把 bug fix 计成新结构的收益。

## 4. 第一项改进：建立任务相关的空间证据接口

### 4.1 从局部空间开始

第一版只表示当前 anchor 的目标、必要参照物、附近可观察障碍/可通过区域。用稀疏实例点集、边界片段和局部自由空间就能启动 PASS；没有必要先构建全屋稠密 BEV。

建议流程：

```text
解析后的目标与参照物
→ 冻结的开放词汇定位/分割
→ 对齐原始 RGB 与 depth
→ 像素反投影和相机到 agent 坐标变换
→ 同类实例之间的关联/消歧
→ TaskSpatialEvidence
```

若 depth 为相机光轴方向的米制深度，像素反投影为：

\[
p_{cam}=d(u,v)K^{-1}[u,v,1]^\top,\qquad
p_{ego}=T_{ego\leftarrow cam}p_{cam}.
\]

实际接入前要核对 depth 是光轴深度还是射线距离、是否归一化、RGB/depth 分辨率和内外参。不能把供模型观看的归一化深度 JPEG 当米制值。

### 4.2 输出不是一个“目标距离”标量

```text
TaskSpatialEvidence:
  episode_id, frame_id, timestamp, coordinate_frame, pose_source
  target_spec_id, target_role, reference_spec_ids
  instance_hypotheses:
    instance_key, bbox_or_mask_ref, view_id
    position_or_boundary, distance_interval, bearing_interval
    association_status, valid_depth_ratio, evidence_refs
  relations:
    target_instance, reference_instance, relation_type
    reference_frame, observed_value, status
  local_free_space:
    observed_region, occupied_region, unknown_region
  source_kind, valid_until, missing_fields
```

冻结预训练感知模型允许使用；不训练新 head，也不拟合实例关联权重。可以采用固定、可解释的关联规则，但必须在独立开发集确定容差，保留匹配歧义，并测试规则在新场景中的稳定性。

四项约束必须明确：

- 找到 sofa mask 不等于找到指令所指 sofa；仍需修饰语、上下文及跨帧关联。
- 物体表面、门洞、房间区域的几何表示不同；不能用门框最近点代替“穿过开口”。
- `left_of` 等关系要明确观察者/对象/指令参考系；不能只看像素左右。
- 目标表面的欧氏距离不等于 benchmark 的目标点测地距离；内部到位判据和官方 SR 分开评测。

这个接口同时为终点确认和当前动作服务，解决两者反复各自猜目标、各自读取深度的问题。

## 5. 第二项改进：多视图时序观察与统一证据快照

### 5.1 两帧、少方向，有实际运动记录

建议先选当前目标方向及最多两个相关方向，保存最近两个时刻；每次最多约 6 张图作为初始预算。这是待验证的工程配置，不是论文给出的最优值。

示例格式：

```text
<VIEW camera_0>
  frame:t-1  timestamp:...  yaw:...  image:...
  frame:t    timestamp:...  yaw:...  image:...
<VIEW camera_1>
  frame:t-1  timestamp:...  yaw:...  image:...
  frame:t    timestamp:...  yaw:...  image:...

Actual motion between frames:
  receipt_id, measured_translation, measured_yaw_change, collision
Current anchor:
  PASS(target_instance), confirmed_subevents, missing_requirements
```

view tag 是相机/方向标签，不是稳定地点 ID。机器人转头以后，同一个 camera_0 观察的是不同世界方向；实例关联必须结合实际运动和相机外参，不能仅把同编号图像当成同一对象。

当前 candidate 图来自当前位置的不同朝向。历史帧只有在真实执行之后才能标为 `t-1`；不能把候选方向图当成执行前后帧。帧间隔记录实际时间/执行段，不假设所有 step 等时长。

对于较长 waypoint，只有前后两帧可能漏掉 BESIDE 或门洞边界。可保存事件关键帧或已有执行过程中的观测；新增观测的传感与计算开销单独统计。

### 5.2 Qwen 在时序链里的权限

Qwen 可以提出目标身份、遮挡解释或关系假设；程序检查 mask、深度和运动证据，并由 ACN 更新正式进度。VLM 输出“已经过”只是一条待核验声明。

原本用于单时刻全景的 prompt 必须与 temporal prompt 分开：只有实际包含多个时刻和匹配回执的请求才允许询问变化；缺帧时保持 unknown，避免混淆输入权限。

### 5.3 复用同一份证据，而不是反复询问相同事实

每步建立不可变的 `EvidenceBundle`，键至少包含 episode、frame、target-spec 版本；目标身份或位姿变化后检查失效条件。EventVerifier、selector 和 STOP 链引用同一证据 ID。

这可以逐步减少重复编码或相互矛盾的识别，但合并调用不是本方案已证实的加速结果。应先保持输入覆盖一致做接口迁移，再单独测调用复用。

三种缓存应区分：原始图像/特征缓存、已解释的结构化证据、模型计算缓存。第一版只需要前两者；不要求开放服务器内部 KV，也不把 KV 当作权威事件记忆。

## 6. 第三项改进：把规划接口改成“候选动作会带来什么”

### 6.1 保留现有候选，显式预测动作效果

室内实验第一版可以直接复用现有 waypoint 集合，不必生成新的连续轨迹。对候选 `w_i` 使用实际 angle/distance，预测执行后同一目标的相对位置与事件变化。

令 `r_t` 为目标在当前 agent 坐标系的位置，`Δp_i` 为该坐标系下的候选平移，`Δψ_i` 为预计转角：

\[
\hat r_{t+1}^{(i)}=R(-\Delta\psi_i)(r_t-\Delta p_i).
\]

第一版假设目标静止；动态目标缺少可信运动估计时不使用这个静态预测。转角符号、坐标轴、转向和平移的执行顺序需与 ActionCompiler 一致。候选运动是指令值，遇到碰撞或截断不会完整兑现，因此这只是预测。

输出建议为：

```text
CandidateEffect:
  state_version, candidate_id, command_angle, command_distance
  predicted_target_range, predicted_target_bearing
  expected_event_transition, transition_support
  observed_path_clearance, unknown_path_fraction
  validity, uncertainty_reasons, supporting_evidence_ids
```

未观测区域保持 unknown，不能把“深度没看到障碍”解释为全程可通行。第一版只做一步候选效果估计，执行后重新观察与规划，避免在不完整地图上累计多步预测误差。

### 6.2 PASS 的例子

假设当前实例关联可靠，状态为 `PASS / APPROACHING`，下一个期望关系为 `BESIDE`。下面是解释性示例，不是已有实验结果：

| 候选 | 当前语义描述可能看到什么 | 新接口额外提供什么 |
|---|---|---|
| 向 sofa 正面继续靠近 | sofa 更明显，语义匹配较高 | 距离减少但没有形成经过关系，继续逼近可能受阻 |
| 沿 sofa 旁边的可观察通道前进 | sofa 未必位于画面中央 | 在保持必要间距时，更接近侧方/越过关系 |

当前 4B selector 可继续负责语言约束与多目标权衡，但它不再需要仅靠历史文字推断上述几何差异。

事件判定必须排除原地转身造成的方位翻转，也要考虑物体范围和指令要求的通过侧。`front → rear` 不是单独充分条件。若目标丢失或实例歧义过大，暂停该事件的候选指导，保留原有选择能力。

### 6.3 第一版用结构化提示，第二版再评估显式重排

现有 selector 输出候选 ID，不能假定已有 `S_semantic` 数值概率。因此建议分两步：

**G1：向现有 selector 注入 CandidateEffect。** 每个候选最多几项有效字段，明确预测性质，不追加长篇反思或独立仲裁调用。输出仍通过现有候选白名单和 ActionCompiler。

**G2：单独测试确定性重排。** 如果 G1 有用，再研究用事件势函数比较候选：

\[
S_{event}(w_i)=\Phi_E(\hat S_{t+1}^{(i)})-\Phi_E(S_t)
-\lambda_r C_{risk}(w_i)-\lambda_u C_{uncertainty}(w_i).
\]

`Φ_E` 是预先定义的事件进展度量，不是从测试集拟合出来的成功预测器；风险和不确定性必须来自可观察证据。不同动作的量纲要统一，候选无效/不确定时显式 abstain。权重只在开发协议中固定，做敏感性分析。

不能直接把旧 candidate prior 的 softmax gap 当作上述分数，也不能因为所有候选都没有单步到 BESIDE 就全拒绝。一个候选能改善到期望关系的几何条件，就可能有价值。

### 6.4 预测、观察和正式推进分开

```text
S_t：已确认事件状态
  ↓ 只读
预测每个候选的效果
  ↓
选择并提交执行
  ↓
ActionReceipt + 新 ObservationFrame
  ↓
关联同一实例，检查实际平移与几何变化
  ↓
ACN reducer 更新 S_(t+1)
```

预测“执行后应到 BESIDE”不能直接把状态写成 BESIDE。文本解释也不能确认动作完成。正式事件证据需包含执行回执、前后帧、实例关联和使用的坐标系。

`unknown` 不生成完成事实；必要条件仍需满足。单帧丢失不删除历史子事件，但当前空间关系可以因回退而变化。STOP 继续由统一终点契约及 StopCoordinator 决定，不能仅因候选预测达到 AFTER 就停止。

## 7. 拟接入当前代码的位置

以下文件名中标注“新增”的是建议，不表示已经存在。

| 接入点 | 拟议变更 | 接口边界 |
|---|---|---|
| [pipeline_contracts.py](../../vlnce_baselines/common/opennav_ext/pipeline_contracts.py) | 扩展带 frame/target/version 的空间证据与候选效果契约 | 检查 GT 目标距离等字段不进入 runtime 契约 |
| `spatial_evidence.py`（新增） | 冻结 grounder 输出适配、RGB-D 反投影、实例几何与关联 | 输出观测或假设，不推进 anchor |
| `temporal_observation_buffer.py`（新增） | 保存少量原始帧、位姿、实例引用及回执 | 按 episode 清空；明确帧龄与失效条件 |
| [visual_evidence.py](../../vlnce_baselines/common/opennav_ext/visual_evidence.py) | 新增明确的 temporal 输入构造，复用证据引用 | 单帧与多帧 prompt 分开；VLM声明不自动变成物理事实 |
| [progress_locator.py](../../vlnce_baselines/common/opennav_ext/progress_locator.py) | 将 PASS 从 unsupported 接到可测试事件 reducer | ACN 保持唯一进度写入口 |
| `candidate_event_prediction.py`（新增） | 对现有候选预测相对几何和预期转移 | 只写预测日志，不更新真实事件状态 |
| [navigation_guidance.py](../../vlnce_baselines/common/opennav_ext/navigation_guidance.py) | 生成紧凑 phase/expected/candidate-effect 摘要 | 只读 EventState 与证据快照 |
| [spatialNavigator.py](../../vlnce_baselines/common/navigator/spatialNavigator.py) | 将摘要加入现有候选选择输入 | 输出仍为合法候选 ID；不重写 completion |
| [base_il_trainer_llm.py](../../vlnce_baselines/common/base_il_trainer_llm.py) | 组装同一步证据、调度预测、执行后核验 | 保留单一执行出口，记录预测与真实差异 |

建议新开关相互独立：

```text
SPATIAL_EVIDENCE.ENABLED / LOG_ONLY
TEMPORAL_OBSERVATION.ENABLED
EVENT_VERIFIER.PASS_ENABLED
EVENT_GUIDANCE.MODE = off / prompt / rerank
```

这些是新接口草案。关闭新增模块时应回到同一修复基线；grounding 暂缺、几何无效或多帧不足时返回明确状态，不能静默降成“已完成”。

## 8. 实验应怎样设计，才能证明借鉴有效

### 8.1 四个问题，四层证据

| 问题 | 首要实验 | 主要指标 |
|---|---|---|
| 实例空间输出是否更可信？ | 冻结影像、标定与目标指令的离线对照 | 实例匹配、有效深度覆盖率、距离/方位误差、错误实例 FPR |
| 事件是否真的被识别？ | 独立标注的 PASS 片段与 hard negatives | Completion F1、过早确认、漏确认、边界延迟、unknown |
| 候选预测是否准确且会改变动作？ | 预测与所执行候选的真实后果对账 | 有效预测率、关系误差、转移一致率、候选变化率 |
| 路线是否因此更好？ | 同代码、同预算、同 episode 的在线配对 | SR/OSR/SPL/nDTW、正式 STOP、碰撞、耗时 |

只有实际执行的候选可以直接对账。要比较未执行候选的真实收益，需要隔离的仿真分支评测；其 GT 结果不进入在线 selector。独立事件标签也不能由待评测 reducer 的同一组规则自动生成。

### 8.2 建议消融组

| 组别 | 配置 | 回答的问题 |
|---|---|---|
| B | 终点契约修复后的现有系统，prior log-only | 共同基线 |
| B+S | B + 单帧实例空间接口 | 几何证据本身有何作用 |
| B+S+V | B+S + PASS 验证，使用几何/运动历史，暂不增加多帧 VLM | 正式事件验证及进度推进有何作用 |
| B+S+V+G1 | 再给 selector 提供期望转移与候选效果摘要 | 上游物理指导的增量 |
| B+S+V+T | V 组额外启用两帧 VLM 辅助身份/变化判断 | 多帧语义辅助的独立增量 |
| B+S+V+G1+T | 同时启用候选指导与多帧辅助 | 两种机制是否互补 |

这样不会把“多看了一帧”“有了实例深度”“能推进 anchor”和“选路得到物理预测”混成一个收益。G2 重排、PROBE、调用合并另外做实验。

S 组对所有依赖它的分支使用同一感知模型与参数；T 对比固定图像预算，并增加重复当前帧/不提供有效历史的对照，检验提升究竟来自真实时序还是更多图像 token。历史时间标签必须与实际输入一致，不能把构造对照冒充真实历史。

当前 ACN 进度已进入 selector，所以 V 组本身就可能改变路线；不能将 V 简称为“完全只改裁判、不影响行为”。G1 的作用应以相对 V 的增量来解释。

### 8.3 验收与停止扩展条件

1. **契约回归**：保留原固定 EP10，加入 EP1117/176/568/677；有终点目标时不得出现 goal complete 却无终点确认。
2. **空间/事件回放**：至少覆盖靠近未经过、错误同类实例、纯旋转、遮挡、跨步漏采、经过后退回。报告独立标签质量和未能判定比例。
3. **指导验证**：如果事件 F1 改善但候选预测无效或动作不改变，先修接口与预测；如果动作改变而 SR 下降，检查指导是否过强、几何是否错绑、候选是否可达。
4. **配对在线**：固定采样配置、动作预算、STOP 与 recovery；先 EP10 回归，再 EP100 诊断和独立保留集。报告配对差异及不确定性，不能仅给一次总 SR。
5. **资源成本**：报告每步视觉请求数、输入图像数、时延和峰值显存；不预设“共享证据必然更快”。模型权重冻结也不代表推理成本不增加。

现有 EP100 已被反复分析，应承担诊断角色。方法有效性的最终判断需要未参与规则选择的保留数据。阈值敏感性、遮挡和里程计噪声是这套 training-free 规则能否泛化的重要检查。

## 9. training-free 与数据权限的精确定义

本轮采用以下约束：

- Qwen 和接入的预训练感知模型权重冻结，不训练新的事件/规划 head，不做 LoRA、SFT 或 RL。
- RGB-D、相机标定、真实执行回执和协议允许的位姿信息作为运行输入。
- Habitat 目标坐标、目标测地距离、GT 实例和参考轨迹仅用于隔离的离线标注、诊断与评测。
- 固定规则和容差需公开；不对测试集拟合权重，也不将 learned candidate prior 偷换成“无需训练的物理模块”。

现有 `candidate_prior.py` 明确说明其权重来自历史离线监督，因此本轮继续 log-only，不作为新 training-free 控制模块的一部分。现有 waypoint predictor 和感知 backbone 是预训练模型；论文表述应为“所提模块无需额外训练”，并披露系统沿用的预训练组件。

当前日志中的位姿来源是 `simulator_agent_pose`。若沿用，应明确 benchmark 的定位输入假设；若声称无真值位姿部署，则需另行提供里程计并测试误差。仅排除 goal GT 并不能自动证明所有定位输入都符合每个比较协议。

## 10. 下一步最小任务

建议将第一次实现收敛为 **PASS 的单实例闭环**，交付：

1. 固定一套目标短语、实例关联、mask/区域与米制几何的证据记录；
2. 一个消费真实位移和前后关系的 PASS reducer；
3. 一份每步候选执行后的相对位置与事件效果表；
4. 将该表接入当前 Qwen selector，并保留预测/观察对账；
5. 在同一感知条件下比较 Verification-only 与 Verification+Guidance。

若这个闭环成立，再用相同证据契约扩展门洞边界和 CROSS。此时 Qwen-Drive 的借鉴已经变成可检验的实验结构：空间输出可核查、决策有动作后果、执行有反馈；是否改善 SR 则由我们的室内实验回答。

## 11. 阅读与代码依据

本次重新核对了论文的方法、训练流程、消融和局限，以及官方模型、数据、感知接口；本地检查了视觉输入、深度取值、ACN 反馈、候选选择和执行回执。本文不运行模型训练或在线导航，不改动导航代码/配置。

- [Qwen-Drive 技术报告](https://arxiv.org/html/2609.00111v1)：§2 方法与训练，§3.4 消融，§5 局限。
- [官方模型接口](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/model.md)：规划与 VLM 信息连接方式。
- [官方数据接口](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/data.md)：视图/时刻输入、坐标与运动数据。
- [官方感知接口](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/perception.md)：可检查的空间输出。
- [当前完整 EP100 分析](EP100运行分析-parser_v22_prior_logonly-20260910.md)：当前问题与性能口径。
- [物理事件方案对照评估](物理事件导航构想与当前实验对照评估-20260916.md)：事件语义、评测和状态边界。
- [candidate_prior.py](../../vlnce_baselines/common/opennav_ext/candidate_prior.py)：已有 prior 的离线训练来源与局限。
- [run_OpenNav.yaml](../../run_OpenNav.yaml)：当前主配置；正式复现实验以各轮 resolved config 为准。

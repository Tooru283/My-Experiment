# Backtracking (SmartWay 回溯) 设计 — 20260703

## 为什么现在做
- P1 触发式再观测的 SR 期望被两个数压低:(1) dispersion 闸门 FAIL(AUROC 0.542,干净集 0.530);
  (2) 天花板分析 —— 即便触发器完美,30% 预算才覆盖 ~70% 走错步,再观测的纠正率还未知。
  → P1 顺利也大概率只有低个位数 pp。
- Backtracking **不依赖任何触发信号校准**:死胡同/负进展是行为层信号(近 ground-truth),
  不是诱导/自报信念。外部证据最硬(SmartWay 真机 +12pp)。攻的是 72 迷路集里 P1 覆盖不到的
  "已经走错、当前视点候选全在错误分支"子集 —— 这正是 RecoveryPolicy(U3)治不了的。
- 写代码不占 GPU,与 confusion replay 并行推进。

## 关键架构发现(决定实现形态)
- **视点是相对极坐标移动,不是绝对图节点。** 选中候选被执行为
  (`base_il_trainer_llm.py:3130`)::

      {'action': 4, 'action_args': {'angle': radius_dict[vp], 'distance': distance_dict[vp]}}

  → **无需 teleport/graph API**。回溯 = 合成一个反向移动 `{'angle':180, 'distance': 上一步distance}`
  (转身,原距离退回)。这是本设计能低成本落地的根因。
- **⚠ 触发信号不能用 `recent_distance_gains` —— 它是 oracle。** `selected_distance_gain =
  distances[-2]-distances[-1]`,`distances = info["position"]["distance"]` = 模拟器**到目标测地距离**
  (SR/NE 同源 GT)。推理时读它 = oracle 泄漏,零样本论文审稿一票否决、SR 作废。
  **替代 = 纯可观测 dead-reckoning**:只用 agent 自己发出的 `(angle,distance)` 动作序列航位推算,
  最近窗口"路程大但净位移小"= 物理打转/死胡同(`displacement_stalled`);外加 waypoint 生成器
  前向候选枯竭 `dead_end` 标志。两者都不碰目标距离,行为层零校准。
  **另需你核**:`recent_distance_gains` 也被传进 `failure_diagnostic`/`stop_evidence_verifier`——
  若那些在推理时 branch 于它,则现有管线已有 oracle 泄漏(RecoveryPolicy 已核=仅log不branch)。
- **动作空间注入点**:`spatialNavigator.move_to_next_vp:590`
  `candidate_list = candidate_ids + [STOP_CANDIDATE]` —— MOVE_BACK 与 STOP 同法注入。
- **解析**:`_parse_prediction:503`(STOP 特判 `:506`)—— MOVE_BACK 加同款特判。
- **已有可复用件**:`RecoveryPolicy`(U3,当前视点重选 + `blocked_candidates` + 每集预算)是模板;
  回溯与它**互补**(它横向换候选,回溯纵向退一步)。

## 落地件(本次已交付,additive,未接线,不影响在跑系统)
`vlnce_baselines/common/opennav_ext/backtrack_policy.py`:
- `progress_stalled(recent_distance_gains, window=3, tol=0.0)` — 行为触发。
- `reverse_action_args(last_move)` → `{'angle':180,'distance':d}`(零距离/无前步→None)。
- `BacktrackPolicy.available(...)` — 是否把 MOVE_BACK 摆上菜单(预算>0 ∧ 有可逆前步 ∧ 停滞 ∧ 非刚回溯)。
- `BacktrackPolicy.apply(...)` — navigator 选了 MOVE_BACK 时出反向动作 + 扣预算 + 记被弃 heading。
- 纯函数、无框架依赖、自带 off-GPU 自测(已过)。契约对齐 RecoveryPolicy(返回 log dict 供 ablation)。

## 待接线(4 处,评审后再改 trainer;每处都要 log_u_event 以便消融)
1. `move_to_next_vp`(spatialNavigator.py:590):当 `BacktrackPolicy.available().offered` 时
   `candidate_list += [MOVE_BACK_CANDIDATE]`;`available` 所需的 last_move / recent_gains / budget /
   just_backtracked 由 trainer 透传进来(改签名或走 self 状态)。
2. `_parse_prediction`(:503):加 `if normalized == MOVE_BACK_CANDIDATE: return MOVE_BACK_CANDIDATE`。
3. NAVIGATOR user prompt(`navigator/prompts.py`):动作选项里加一行 MOVE_BACK 说明
   —— "MOVE_BACK: 若当前路径像死胡同/走错,退回上一位置换分支"。**措辞需防偏置**(同 confusion 的 STOP 教训:
   别把它写成鼓励项,否则模型滥用;冒烟看使用率)。
4. 执行(base_il_trainer_llm.py:3129 else 分支):`next_vp == MOVE_BACK` 时
   env_action = `{'action':4,'action_args': BacktrackPolicy.apply(...).action_args}`;
   set `just_backtracked=True`,把被弃 heading 塞进下一步 blocked,扣预算。

## 护栏 / 消融
- 预算 `max_backtracks_per_episode=2`(同 RecoveryPolicy 默认);反震荡:不连续两步回溯、
  到达后一步不回溯。
- 消融臂:off / on;可与 U3 RecoveryPolicy 正交叠加(横向×纵向)测边际。
- **风险**:相对反向移动受障碍物影响不保证精确回到原点(habitat 会 collide-slide);冒烟看实际
  回退位置误差。若误差大,退化方案 = 回溯后强制重扫(等价触发 Disambiguation 全景),吃这个成本。
- 迷路子集(72 集)优先验:先在这批上看 MOVE_BACK 使用率 + SR 增益,再决定全量。

## 与 P1 的关系
正交。P1 = 决策层"慢下来再想/再看";Backtracking = 记忆/恢复层"走错了退回来"。
即便 P1 缩水,二者作用于不同失败模式(P1 攻假停+近目标;回溯攻迷路+死胡同),可同时进论文。

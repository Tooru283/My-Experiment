---
name: project-selector-bottleneck-20260719
description: ep100 20260719 三个决定性否定结果——深度否决AUROC0.531无效、回溯1.9%接受率空转、选择器语义贡献仅1点
metadata: 
  node_type: memory
  type: project
  originSessionId: 492084f4-1f0f-4063-8a92-2e46db3bcfff
  modified: 2026-07-19T04:22:46.619Z
---

2026-07-19 ep100 (SR20/OSR25/SPL0.155, vs clean_baseline SR16/OSR21) 的归因分析,
McNemar p=0.388 不显著,8涨4跌,99/100 集轨迹改变。

**三个否定结果(都用离线 oracle 复算,合法):**

1. **深度否决是死路**:提交停止时刻的前向深度对成败 AUROC=0.531(成功中位2.10m/
   失败2.37m,分布重合)。阈值扫描 2.0-8.0m **没有任何净正操作点**(T=3.0 拦掉5个
   成功换16个失败)。ep100 的 +4 SR 是轨迹扰动噪声,不是过滤生效。
2. **回溯空转**:offered 156 次、applied 仅 3 次(接受率1.9%,ep715/171/765),
   3 集都不在涨跌名单 → 对 SR 贡献为 0。选择层几乎从不采纳 MOVE_BACK。
   注意 verify_smoke.py 的 backtrack 模式会报**假 FAIL**(2次offer在2%接受率下
   观测不到 apply,应判 INCONCLUSIVE)。
3. **选择器只比随机强 9 点、比最好的平凡启发式强 1 点**:
   命中最优候选 36.7% | 随机 27.7% | 步长最长 35.7% | 转向角最小 32.9% |
   raw_rank最小 30.0% | 步长最短 17.2%。→ 语义 grounding 的增量贡献 ≈ 1 点。
   成功集命中 48.4%/regret 0.06m vs 失败集 34.0%/0.66m(选择质量确实关联成败)。

**方法学(可复用)**:目标位置在
`data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz`
离线可得,**不需要重跑**。候选世界坐标可从 `step_start` 的 positions/headings 加
`waypoint_candidates` 的 angle_rad/distance 精确重建,角度约定是
`θ = -heading - angle_rad`,`x+=d·sin(θ), z-=d·cos(θ)` —— 用"预测选中候选位置 vs
下一步实际位置"标定,误差中位 **0.000m**(其余三种约定 1.3-1.8m)。
欧氏 vs 测地:相关 0.854、中位低估 0.81m,排序统计受影响小。

另:有一条停止路径绕过深度否决(base_il_trainer_llm.py:3218
`if old_stop_flag and visual_target_verifier_allows_stop(...)` 无 depth 合取项),
21/65 集在四门全否决时仍停止。**不要堵** —— 堵上会杀死 5 个成功(占全部20个的25%)。

见 [[project-endgame-tilt]] [[feedback-no-autonomous-runs]]

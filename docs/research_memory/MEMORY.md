# Memory Index

- [SOTA差距与改造路径 20260727](project_sota_gap_20260727.md) — **谈"要不要改架构/发不发得了"先读它**;SR16=Open-Nav忠实复现但赛道已到53.3;差距=持久空间记忆+分层规划;M1的null适用范围被高估

- [全链条架构梳理(活文档)](reference_full_chain_doc.md) — **问瓶颈/模块作用/某条路试过没,先读它**;含已排除的7条路

- [A0 Baseline Performance](project_a0_baseline.md) — SR=16%, OSR=25%, conversion=64%, T1 walk-through 77.8%, confirmed 20260628 ep100 val_unseen
- [max_config ep100 20260628](project_max_config_20260628.md) — SR=21%, OSR→SR=80.8%, 5 near-goal failures: ep11/377/824/1084/1106
- [M3 proactive_stop_gate](project_m3_harness.md) — ep100 SR=20% (↓1); E3 abstain implemented 20260630: COMMIT_DIST=2m, arrival_override=2.5m, abstain=2m
- [E3 carry_forward ep100 20260630](project_e3_20260630.md) — SR=24%, OSR→SR=92.3%, gap=2; ep377/ep824 fixed; ep1084/ep1106 Group B blindspot; OSR瓶颈=38 false stops + 36 step-limit
- [每步调用构成与产出信息量](project_vlm_call_count.md) — 20260719更正:选择器是纯文本非VLM;真实VLM每步5~6次(observe_view每候选一次);completion_estimation自证伪(r=0.12/75次倒退)、observe_view距离是语言先验(r=0.175);43%耗时产出经不起检验
- [SigLIP role in architecture](project_siglip_role.md) — SigLIP是SpatialBot3B的视觉编码器，每步主动调用；obs20记录"not active"是错误的
- [P1 设计定稿 20260702](project_p1_design_20260702.md) — 校准vote-dispersion触发(非启发式);Path B耐用管线(记录真实prompt+共享采样器);预注册AUROC≥0.60闸门;决胜=预算配平4臂消融;re-observe≠re-sample
- [架构战略转向 20260701](project_arch_pivot_20260701.md) — 转化92.3%封顶，SR天花板是OSR；waypoint路线20260702正式关闭(72迷路集测得候选84%够用、失败集随机游走48%走远=选择层瓶颈)；Run1(9B调参+E5)=SR24/OSR28/SPL0.200反超4B、假停52→20；Run2(P0on)前34集P0净负待定论；火力压P1
- [Oracle泄漏火警 20260704](project_oracle_leak_20260704.md) — 整个E3/E5/PSG终止侧keyed于单一oracle变量latest_goal_dist(=测地目标距离GT)；E5 far-block决定性168/39、E3 arrival_override14/7+abstain24/7+traj_bypass34/13、增益门594但0决定性、ArrivalGate仅日志；SR=24带毒待干净重跑；Step0审计完成
- [别自己跑评测](feedback_no_autonomous_runs.md) — 占GPU的长任务一律给命令、由用户手动启动；只读分析不受限
- [选择层瓶颈定量 20260719](project_selector_bottleneck_20260719.md) — 深度否决AUROC0.531死路/回溯1.9%空转/选择器仅比随机强9点、比平凡启发式强1点；候选位置离线精确重建法(θ=-heading-angle，误差0.000m)
- [Arm C 候选先验 20260719](project_arm_c_20260719.md) — **20260720已结算:命中率+7点未转化为SR(净+1,p=1.000,SPL持平),因果链断在第二环**；先验只能靠覆盖生效；nmatch是唯一有用语义特征；仲裁上界57.5%
- [评测轮次掉电耐受性](project_run_durability.md) — 宿主KVM会无预警掉电；checkpoint曾因rename-without-fsync清零(已修)；navigation_records逐集追加是唯一救命记录；无断点续跑
- [M1多模态选路上界 20260727](project_m1_20260727.md) — **结算FAIL**:图像臂41%<50%闸门且分辨率已饱和;错配图对照证明模型真在读像素(18%<随机)=非伪null;**20260727c规模曲线39-43%平台(9B/vl-plus/omni-plus)→「换更大模型」经验性关闭**;TXT−GEO仅2-4点=文本感知通道近乎无贡献→落"收口成最小可发表版"
- [G1隐藏态探针 20260723](project_g1_hidden_state_20260723.md) — **20260724结算=决定性FAIL**:隐藏态线性读≈随机(regret1.16劣于基线0.995/AUROC0.52/命中25%),方案A打分头判死→退backbone快筛/最小可发表版;连同选择层近随机+先验不转化=选择层三方证据同向封死

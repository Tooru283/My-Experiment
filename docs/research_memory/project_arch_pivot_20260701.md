---
name: project-arch-pivot-20260701
description: 战略转向：SR天花板是OSR非转化，转化92.3%已封顶；攻OSR=航点+感知+导航级再观测；P0=注入真实几何
metadata: 
  node_type: memory
  type: project
  originSessionId: d7e660c1-ead5-4844-8398-9823f61067f3
---

2026-07-01 文献对比 + 9B ep100 结果后的重大战略转向。

**核心判断**：过去所有工作押在"可靠终止决策"（[[project-e3-20260630]]），把 OSR→SR 转化从 64% 推到 **92.3%（全场最高，超 SmartWay 57%）**，但转化已封顶、边际归零。**真正的 SR 天花板是 OSR（到达能力）**：我们 OSR=26%，SmartWay 51%、Spatial-VLN 更高。

**9B 反证模型不是瓶颈**：9B ep100 完成 = SR 20% / OSR 23% / SPL 0.149 / 转化 87%，**全面低于 4B**（SR24/OSR26）。假停率80%（vs 4B 68%）、空预测fallback 22.4%（vs 6.9%，NAVIGATOR_MAX_TOKENS=1024截断9B长输出）。关键：**9B 的 OSR(23%) 也低于 4B(26%)——堆模型解决不了 OSR，瓶颈是架构**。

**前沿共识（6篇）**：OSR 靠三招——①修/绕航点预测器（SmartWay重训/Fast-SmartWay消除/AgenticNav像素/Spatial-VLN采样，四方一致 WaypointBert 是瓶颈）②给决策者结构化空间感知③导航级验证/再观测（Fast-SmartWay +8pp最大杠杆）。我们过度投④、欠投①②③⑤。

**优化路线**（training-free优先）：P0 结构化空间感知注入 → P1 导航级再观测（校准驱动，护城河=区别于Fast-SmartWay的"问MLLM困惑吗"）→ P2 回溯 → P3 航点。详见 `Controlled-Navigation-Harness/docs/architecture_optimization_20260701.md` 和 `docs/current_task.md`。

**P0 关键代码发现**：`base_il_trainer_llm.py:1304` construct_image_dicts 已算出每候选真实几何 `distance_dict`(WaypointBert距离)+`radius_dict`(角度)，但只在动作执行(:3080)用，**从未注入 navigator 观测**。navigator 现读 SpatialBot 幻觉距离（api.py:155 observe_view，日志实证"镜子20米"）。P0=把真实几何拼进 observation 串，零额外模型调用。

**总方案已裁决（20260702）**：用户明确"先做性能，暂不考虑论文"。旧总方案（可靠终止决策-执行版-20260629）标记 superseded；新建 `总方案-性能优先-执行版-20260702.md`，以攻 OSR/max-SR 为中心。V2/U2/E3 终止机制冻结（已到极限），E5=唯一补丁（rescue 距离守卫）。优化路线 E5(done)→P0(消融中)→P1(导航级再观测,核心)→P2(回溯)；P3航点暂关。

**WaypointBert 诊断（20260701，重要修正）**：前沿点名旧 ResNet-50 航点预测器是 OSR 瓶颈，但那基于真机撞墙场景。我们自测（4B trace，step_start.positions vs 预测距离）：**航点 ~98% 精确到达（位移/预测=0.971，中位1.0），"被挡"仅2-4%**——我们是 waypoint-teleport setup，可达性没问题。**故占用过滤/替换ResNet/重训预测器全部关闭**（改善靶点是可达性/%Open，我们已98%，无靶点）。TRM_net.py:22 visual_fc_rgb=Linear(2048×7×7) 死绑ResNet，换DINOv2非drop-in需重训。用户提供 VLN-MME(Oracle 0→52%证选择层瓶颈)/NavBench(61% Incorrect Plan)佐证。详见 docs/architecture_optimization_20260701.md §5b。

**覆盖率诊断——waypoint 路线正式关闭（20260702，决定性反证）**：曾留"候选覆盖"作唯一待测疑点，现用 Run1 trace(P0off 9B 完整100集)+数据集真实目标坐标(`OpenNav_R2R-CE_100_bertidx.json.gz` goals[0].position)在 72 迷路集(=100−OSR28)640决策点上诊断。**测地 ground-truth(`post_action_progress.distance_gain_selected`,无代理偏差)：所选候选 48% 走远目标、48% 走近、净增益均值+0.07m/中位+0.03m=随机游走**。欧氏覆盖交叉印证：前向候选 ~84% 步都存在、真死胡同仅11–16%、selector 只31%步选靠近候选。裁决：**覆盖不是瓶颈,选择/路由是**。SmartWay repo(github.com/sxyxs/SmartWay-Code,实为原版非Fast-SmartWay)强化版预测器=同 `BinaryDistPredictor_TRM` 类但 RGB 换 DINOv2-small(384d,我们ResNet 2048×7×7)+ID_CrossAttention+occupancy loss,**非drop-in需换整个RGB前端且攻的是可达性(我们已98%),SR上限低,路线关闭**。P3 从"无靶点暂关"升级为"有直接反证"。火力压 P1(选择层再观测)。

**Run1/Run2 结果（20260702）**：Run1(9B+调参NAV_TOK2048/COMP768+E5,P0off) ep100=**SR24/OSR28/SPL0.200/转化85.7%**,OSR与SPL反超4B(26/0.187),假停52→20砍半,主导失败转为56集从未靠近(=P1靶心)。9B跑速7.0min/集(满100约11.7h)。Run2(P0on)**满100集定论**=SR18/OSR21/SPL0.155/nDTW0.413 vs Run1 SR24/OSR28/SPL0.200,边际P0净−6/−7。**但配对McNemar(100集全配对)拆穿边际数字**:SR失配16(伤11/帮5,精确p=0.210),OSR失配19(伤13/帮6,p=0.167)——**方向负但不显著**。故论文措辞收紧:❌不能说"P0有害/掉1/4"(11vs5小样本);✅只能说"naive几何注入无显著增益(McNemar p=0.21),点估计为负"——仍足以反驳"信息量是瓶颈",降级为no-benefit旁证(排除平凡解释),非harmful证据。翻负集全是路由被带偏终点飘远6.7–7.8m,疑navigator误读`[Waypoint distance]`步长为到目标距离。P0.1=修正注入语义(可选)。与选择层脆弱一致:塞信息治不了,治本靠P1。**教训:边际差会骗人,进论文前配对检验**。

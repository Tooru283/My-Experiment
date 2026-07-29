---
name: project-p1-design-20260702
description: P1定稿=校准vote-dispersion触发(非启发式);Path B耐用管线(记录真实prompt+共享采样器);预注册AUROC闸门;决胜=预算配平4臂消融
metadata: 
  node_type: memory
  type: project
  originSessionId: d7e660c1-ead5-4844-8398-9823f61067f3
---

P1（导航级选择性再观测）设计定稿，2026-07-02。详见 `Controlled-Navigation-Harness/docs/p1_design_20260702.md`（§8 启发式版旧、§9 Path B 主力）+ 脚本 `scratchpad/p1_dispersion_calib.py`（就绪，dep-free 统计已单测）。承接 [[project-arch-pivot-20260701]]（waypoint 路线已关，火力压选择层）。

**论文创新点排序（用户 2026-07-02 校准）**：#1 核心=**校准回答"何时再观测"**(触发信号来源,唯一护城河,但目前只是声明未证);#2 最可防守=**小VLM导航信念校准刻画**(E1 ECE0.32+近距欠检测43%+覆盖/选择诊断84%/31%,method-agnostic,论文地板,P1崩也在,建议升格为独立贡献);#3 统一命题=同一机制覆盖路由+终止(框架层新,但完全依赖#1,退化成启发式则只剩修辞)。一句话:创新不是"做了再观测",是"用校准回答何时再观测"+首份小VLM导航信念校准分析。转化92.3%/跨backbone 放支撑层。

**核心机制**：selector 决策处，触发时多采样 K 次唤醒**已建好但休眠**的 `thought_fusion`(spatialNavigator.py:612)+`test_decisions`(:637)。落点 `base_il_trainer_llm.py:2419`。

**触发信号=校准后的 vote-dispersion**（诱导信念，self-consistency/semantic-entropy 血统 Wang/Kuhn/Farquhar）。启发式 A∪B(进度停滞∪phase=recover)**降为降级对照臂**——因 progress-stall 用 oracle 距离=推理时泄漏，且非校准。novelty 精确表述="首次将校准后的诱导信念用作具身导航再观测触发"，非发明 dispersion。

**Path B 耐用管线（否决了从 trace 反向重建 prompt——与模板/observation/history 格式紧耦合、改代码即失效）**：①navigator 加共享 `sample_predictions`+`vote_dispersion`；②决策点记录**真实拼好的 prompt 字符串**(`navigator_prompt` 事件)→离线 replay 零重建、活过重构；③在线 P1 与离线校准走**同一采样函数**。前提：`api.py:83 temperature=0` 须参数化(默认0向后兼容)。

**dispersion 指标**：相邻候选按角度≤45°合并成方向簇再算熵(semantic-entropy 先聚类精髓，否则拥挤候选场景系统高估)；归一化簇熵(除logK跨步可比)+1−top簇占比对照。

**离线校准（go/no-go 闸门，预注册防目测）**：收集用 **4B 全100集**(P1关只记prompt;天然含72迷路集富集判闸门+28到达集拟合带)。保真自检:temp0复现贪心应≈100%。标签=`distance_gain_selected<0`(测地,有绕行噪声故AUROC别期望0.9)。**PASS=AUROC≥0.60且boot95%下界>0.55;≤0.55=FAIL推倒省一轮;中间=GRAY上聚类/re-perceive**。辅判Spearmanρ≥0.3、每桶Wilson CI。**离线只是kill-switch,创新点靠决胜消融**。

**离线预算修正(20260702,用户纠正:我错按串行8s/call估②③)**:离线replay是纯吞吐任务(850 prompt全在日志、无步间依赖)。两开关:①**并发**(lmdeploy continuous batching,校准脚本改asyncio/线程池并发8-16)→②11h→1-2h,③2h→15min,白捡5-8×;②**prefix caching**(同点K次采样共享前缀,K采样≈增量成本)→K=5不降、子采样不做。修正总账≈12(收集)+2(离线)+32(在线2臂)≈46h,②③从过夜变**当天出闸门**(日程价值>省的小时数)。**待办:收集跑结束前把calib脚本改并发**(现dry-run版串行,够跑但慢)。

**④消融臂决策点(用户提醒,20260702)**:④原只排2臂(校准+confusion)是决胜对比,但缺**配平预算随机触发臂**→挡不住"随便什么触发的再观测都涨"审稿反驳。**决策:先跑2臂,校准vs confusion差距小→随机臂必需(+13-16h,账上有余量);差距悬殊→降级为limitation一句话**。别一开始就砍随机臂。

**决胜消融=预算配平 risk-coverage 四臂**(横轴=额外navigator调用数compute):校准dispersion带 / confusion-prompt(Fast-SmartWay代理) / A∪B启发式 / 匹配预算随机(下界)。**关键:四臂 risk-coverage 全部离线可算(零GPU)**——四信号值在收集数据都能提取(dispersion replay/confusion补问/A∪B用trace的gains+phase/随机平凡),证明"哪信号最会识别坏决策"。**在线仅验证工作点(离线证不了的:再观测是否真提升端到端SR): baseline(=收集跑)+校准-P1+confusion-P1 ~2-3跑**,否决4臂×3=12跑(排不下7/27)。诚实边界:离线=识别质量,在线=治得了,互补。=E1待补risk-coverage图一图两用。

**re-observe≠re-sample(方法节要写,第二个对Fast-SmartWay区分点)**:同分布重采样降不了认知不确定;K次采样是探测器,响应=re-perceive(新视角/新证据:看全景没surface的方向12选~4、或不同prompt重跑SpatialBot攻近距欠检测),fusion只降推理噪声(aleatoric)。温度一致性硬约束:离线校准TEMP必须==在线部署TEMP。校准须随{模型,prompt,温度,流程}任一变更重算。

**代码已落地(20260702晚,§9.1+§9.2)**:api.py gpt_infer加`temperature=0`形参(默认贪心不变);spatialNavigator.py move_to_next_vp加`k/temperature/return_prompt`(k=1&temp=0逐字节等价)+新增`vote_dispersion`静态方法(归一化投票熵,已单测)+顶部import math/Counter;base_il_trainer_llm.py:2419取回prompt并记`navigator_prompt`事件。**关键坑(已修)**:navigator_prompt必须用`log_fallback_event`(:996)写而非`write_navigation_record`——后者只进nav_jsonl(`event`/`step` schema),而校准脚本读harness trace(`event_type`/`step_id`);log_fallback_event两套都写。已核实脚本依赖的selector_final/post_action_progress(带selected_candidate+distance_gain_selected)/waypoint_candidates(带angle_deg)全在harness trace。yaml收集配置:GEOMETRY_INJECTION=False(护栏:校准分布须绑P1部署配置,不带P0污染),TRACE_DIR=logs/harness_traces/p1_collect_9b。**未上**:§8 Diff1/2/3在线触发(过闸门后才加,避免过早改动)。

**headline配置定为9B(20260702晚,用户拍板,推翻§9.6原定4B)**:理由=Run1(9B+调参+E5)是当前最优(OSR28/SPL0.200均超4B)、E5已在其上验证、失败集中在56迷路集"从未靠近"=P1靶心;4B那边E5未验证+假停38集P1治不了。**三条连带决定(已接受)**:①校准/闸门/触发带/四臂消融全套在9B做,校准不跨模型,4B降为跨backbone验证点(之后单独小规模重算校准);②**E1终止校准(ECE0.32那套)现为4B trace,须在Run1(9B)trace上重算配套headline,离线便宜,待办**;③下游全继承9B推理成本。**预算总账(真实9B navigator延迟median6.5/mean10.8/p90 26s,~850决策点/收集)**:收集~12h + 离线K采样(850×(1+K5)=5100call×8s)~11h(K3/子采样可降7h)+ confusion replay~2h + 在线验证2×ep100~24-32h = **~57h≈2.4天串行**,距7/27(25天)按3×留余量~7天排得下,硬约束不binding。服务`transformers serve Qwen3.5-9B`(PID8776)本就在9B,无需重启。启动=`bash run_OpenNav.bash`(默认9B+LLM预检)。

**20260703 执行记录(推理后端大转向+校准跑起)**:
- **收集跑 ep100(20260702_230800)= Run1 逐位复现**(SR24/OSR28/SPL0.2002594558414838/nDTW 16位相同)→ §9.1/9.2 代码重构零行为改变;产出 **804 navigator_prompt 决策点**(角度100%覆盖、标签基率43.3%走远)。失败=OSR天花板(72集never reach)+转化缺口仅4集(244/150/338/705)。
- **serving 后端连环踩坑**:①`transformers serve`(收集跑用的)**只贪心不采样**(temp0.7=temp1.5=换seed 输出逐字节同)→K采样恒同dispersion≡0,硬阻断。②vllm 0.21(qwen35-serve已装)是**CUDA13编译,本机驱动12.4→vllm-flash-attn "driver insufficient",engine core崩**,升驱动不可行。③`flash-linear-attention` pip装上后**破坏 transformers5.9 的 Qwen3_5 加载**(is_flash_linear_attention_available 崩/Qwen3_5ForCausalLM导入失败);卸载残留`site-packages/fla/`目录致`version.parse('N/A')`崩,须手删该目录恢复。**结论:放弃HTTP服务,校准直接 transformers `model.generate(do_sample=True)`**(torch慢速GDN回退,~31s/点,但可用)。
- **关键修正**:保真参照必须用 **`selector_raw.predictions[0]`(navigator原始贪心)**,不是`selector_final.selected_candidate`(经U系列/rescue覆写)——用错致假失配66.7%;修正后**保真9/9=100%**(同库贪心逐点复现→无跨后端漂移,§9.3灰区/重跑分支用不上)。
- **校准配置钉死(.meta.json)**:transformers-direct-generate + enable_thinking=False(匹配serve --reasoning off)+ K=5 + temp=0.7 + **top_p=0.9**(非1.0:1.0致40%样本1024token不决策截断)+ max_new=2048(截断降至3/10、每点仅31s近似1024) + seed_base=1234 + 方向聚类45°。全量804点K采样20260703午后启动,ETA~6.9h,增量落盘+种子乱序(可读部分AUROC早信号)+可续跑。
- **遗留(过闸门后才需解决)**:在线P1部署仍缺可采样HTTP服务(vllm阻断、transformers serve不采样);直生成仅离线可用。在线rollout前需另起可采样服务或修驱动/vllm。

**落地顺序**(等Run2今晚~22:20跑完释放GPU):应用§9.1/9.2代码(全向后兼容)→**收集跑=4B+E5+全100(P1关只记prompt),一跑三用**(收prompt+论文4B+E5数字[E5首上4B攻38假停]+E5延长episode利于闸门,此跑即在线baseline;校准绑4B+E5配置,部署须同配置,9B另算)→跑校准脚本→预注册AUROC闸门→过则离线四臂risk-coverage定工作点+在线验证~2-3跑。代码全additive默认关,不影响运行中进程。

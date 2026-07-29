---
name: project-g1-hidden-state-20260723
description: G1隐藏态探针=方案A生死闸门,20260724跑完=决定性FAIL(隐藏态线性读≈随机),方案A打分头判死
metadata: 
  node_type: memory
  type: project
  originSessionId: c8a3d56a-4edd-4a28-913f-5fc4d9a302f0
  modified: 2026-07-24T05:09:45.951Z
---

**【20260724 结算：G1 = 决定性 FAIL → 方案A(打分头训练线)判死】**
全量 dump(683决策点/100集)保真 639/683=93.6% GRAY(smoke的85%因撞stop簇偏低)。
`g1_hidden.py` 双口径一致 FAIL：best=layer20/mean，折外 **regret 1.16**(劣于LLM基线0.995)、
**AUROC(选对) 0.52**(≈随机)、**top1命中25-28%**(每点~4候选=随机1/4)。三指标自洽三角验证=
**冻结9B候选token隐藏态无可线性解码的regret排序信息**，非实现伪影。Δregret=−0.17m/步≈−1.1m/集
(地板+0.12/上线+0.5，实测为负)。→ 退 backbone快筛/最小可发表版；G1本身作为干净的表征探针负结果入论文。
边界诚实：线性≠穷尽，但AUROC≈0.5说明连一阶信号都无，且需大非线性头则廉价性前提失效=等价微调，超本闸门范畴。
**稳健性(20260724 PCA容量扫描)**：PCA 64/128/256/512(8×容量)best AUROC=0.521/0.512/0.474/0.537,始终≈随机、regret稳~1.16→FAIL非PCA/容量伪影。
这也基本堵死"常识/training-free相似度打分头"(候选隐藏态vs子目标嵌入)——它是探针搜索空间的固定方向特例,监督探针是其上界且8×容量下仍随机。
用户问"常识打分头"三形态收口:①纯先验=Arm C已死(不转化);②training-free相似度=被G1上界+PCA512否掉;③训练划分学通用常识=未绝对证死但已不便宜/不zero-shot且信号存疑。
连同[[project-selector-bottleneck-20260719]]近随机、[[project-arm-c-20260719]]先验不转化=选择层三方独立证据同向，"改进选择层"路在本setup封死。

---
**（以下为20260723实现记录，已落地执行）**

**G1 正式版（隐藏态线性探针）= 训练线唯一生死闸门**。20260723 把实现从"在线 hook + 12h collect"
改为**离线 re-forward**，脚本已写完并 CPU 验通。详见 current_task §二.3 的 20260723 落地块。
产物：`paper_analysis/scoring_head/hidden_states.npy`(683记录，每点mean+last池化×层[16,20,24,28,32])。

**为什么能离线**（三条查明的事实）：
1. `navigator_prompt` 事件（selector 完整 prompt + candidate_ids）**100/100 集已记录**在 20260719/clean_baseline_v1
   trace（`base_il_trainer_llm.py:3100`，p1_design §9.2 早已实装）→ 无需改 harness、无需重跑 collect。
2. 9B 选择器是 **lmdeploy HTTP 服务**（`api.py`→OpenAI client→`127.0.0.1:23333`），端点只回文本、拿不到隐藏态。
3. 贪心可复现 → 离线用 HF transformers 重跑 prompt 拿隐藏态即可。

**关键坑**：`/root/models/Qwen3.5-9B` 其实是 **VLM**（`Qwen3_5ForConditionalGeneration`，混合 linear+full attention，
mrope），dims 在 `config.text_config`（32层/hidden4096）**不在顶层**。离线前向须用 **qwen35-serve conda 环境**
（`/root/anaconda3/envs/qwen35-serve/bin/python`，transformers 5.9，与服务端 chat 模板一致）+
`AutoModelForImageTextToText` 加载；纯文本前向跳过视觉塔。opennav 环境的 tf 4.44 不保证支持 Qwen3.5。

**脚本**（`paper_analysis/scoring_head/`）：
- `collect_hidden_states.py`：离线 dump，`--dry`(CPU验管线) / `--smoke N` / `--fidelity-only`。抽候选"Direction {id}"块
  token 隐藏态，**mean + last 两种池化**、默认层 [16,20,24,28,32]。
- `g1_hidden.py`：复用 `g1_lowerbound.py` 的 regret label 管线（euclid、场景级 GroupKFold-5，n=656/10场景），
  X 换隐藏态，train-fold PCA(numpy SVD)→listwise 线性探针，扫 layer×pooling。

**运行顺序（用户手动起）**：①`--smoke 40 --fidelity-only` 过**保真闸门**（HF复现 lmdeploy 决策
≥95% PASS / 85–95% GRAY留匹配子集 / <85% RERUN退在线hook）→ ②全量 `--out hidden_states.npy` →
③`g1_hidden.py --hs hidden_states.npy`。**G1 闸门：折外 regret ≤0.85 且 AUROC(选对)>0.60**；
地板=标量下界 +0.12m/集（须 5× 到 ~0.5 才值得训）。不过 → 方案A死，退 backbone 快筛/最小可发表版。

见 [[reference-full-chain-doc]] [[project-selector-bottleneck-20260719]] [[project-arm-c-20260719]] [[feedback-no-autonomous-runs]]

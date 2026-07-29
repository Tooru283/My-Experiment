---
name: reference-full-chain-doc
description: 全链条架构梳理文档路径——感知/认知/决策/执行/反馈各环节实现与瓶颈判定
metadata: 
  node_type: memory
  type: reference
  originSessionId: 492084f4-1f0f-4063-8a92-2e46db3bcfff
  modified: 2026-07-20T04:10:09.721Z
---

`/root/wjj/Open-Nav/Controlled-Navigation-Harness/docs/全链条梳理-感知认知决策执行反馈-20260720.md`

2026-07-20 建立的活文档（`status: living`），下次问"瓶颈在哪/某模块干嘛的/这条路试过没"
先读它,别重新翻代码。含：

- 五环节各自的实现、模型栈、实测数字、瓶颈判定（附"是否瓶颈"汇总表）
- **§6.1 已排除的 7 条路**（深度否决 / 回溯 / 重排候选顺序 / 几何注入 /
  手工特征仲裁器 / waypoint 路线 / 纯提升每步命中率）—— 提新方案前先对照这张表
- §6.2 当前优先级；§七 诚实限定

模型栈：habitat v0.1.7 + MP3D → `BinaryDistPredictor_TRM`(路点) → RAM(标签) +
SpatialBot3B/SigLIP(场景描述,每候选一次) → Qwen3.5-9B(纯文本:指令解析/完成度/选择器)。
**选择器全程不看图像**,只读 SpatialBot 生成的文本。

死代码：`visual_graph_memory.py`、`landmark_matching.py` 主循环 0 引用。

见 [[project-arm-c-20260719]] [[project-selector-bottleneck-20260719]]
[[project-vlm-call-count]] [[project-run-durability]]

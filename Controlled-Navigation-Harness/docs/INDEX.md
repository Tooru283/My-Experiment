---
date: 2026-07-06
tags:
  - index
  - 归档
updated: 2026-07-06
---

# 文档索引与归档说明（2026-07-06 整理）

> 整理原则：**只移动不删除**。三层分类：现行有效（驱动行动）/ 长期参考（不驱动行动但有据证价值）/ 归档（superseded 或过时代际）。
> 背景：2026-07-06 决策放弃 AAAI-27 DDL，项目重定义为"性能+创新点"，训练解禁、backbone 可换。

## 一、现行有效（读这些就够了）

| 文档 | 价值 |
|---|---|
| [[项目总控]] | 项目入口（**待更新**至新战略） |
| [[current_task]] | 滚动任务状态（**待更新**至新战略） |
| [[experiment_report_20260706_group_meeting]] | 最新周期总结：冲顶→证伪→审计→重置，当前事实的权威快照 |
| [[实验框架-路线状态外置-20260713]] | **当前实验框架 v1.1**（两轨 A/B、预算表、闸门状态） |
| [[backtracking_design_20260703]] | 已 staged 的回溯机制设计（默认关，待 A/B） |
| [[model_switch_guide]] / [[qwen35_4b_local_deployment]] | 运维手册——backbone 扫描马上要用 |
| `../../paper_analysis/` | 论文草稿（tex+中文版）、核数脚本 `verify_paper_numbers.py`、`episode_metrics.json`、`e1_clean9b.txt`、图表——**全部数字经 trace 复核的唯一真源** |

## 二、长期参考（有据证价值，不驱动行动）

| 文档 | 价值 |
|---|---|
| experiment_record_20260610 ~ 20260719（11 篇） | 实验台账，链式不可变记录；论文 provenance 与复盘依据。**20260719 = veto/回溯双双证伪 + 选择层瓶颈定量，读它之前先读 [[current_task]] §一ter** |
| experiment_report_20260624 / 20260629 | 历史组会报告，展示指标演进链 |
| [[p1_design_20260702]] | P1 路线已关闭，但 §9 的**离线校准流水线**（共享采样器/真实 prompt 日志/预注册闸门/预算配平消融）是可复用方法论——训练线的标签管道会直接用到 |
| [[architecture_optimization_20260701]] | P0/P3 路线的关闭证据（waypoint 覆盖诊断、几何注入设计）——论文 §4.3/§4.5 的出处 |
| e1_calibration / e1_calibration_20260630 / e1_calibration_20260702 | 三代校准图表（A0/max 代、4B 代、9B 干净代各不相同，勿去重）；最新数据以 paper_analysis/e1_clean9b.txt 为准 |

## 三、归档（archive/，停止引用）

| 位置 | 内容 | 归档原因 |
|---|---|---|
| `archive/superseded_方案/` | 总方案四代（异构pipeline→可靠终止→AAAI上升版→性能优先）、aaai_execution_plan、近目标终止两篇、M2 方案、对话交接-20260626 | 战略代际更替：终止叙事被转化率封顶否定 → 性能优先被 oracle 审计重置 → AAAI 路线被 7/6 弃 DDL 决策终结。各文件头部有 superseded 标注与指针 |
| `archive/era_0610-0623/` | code_1、code_review 两篇、sr_/v_series_ 五篇、latest_run_comparison | V/U 系列早期代际的工作文档，已被后续代际完全消化 |
| `archive/duplicates/` | 文件名带 `1` 后缀的两份 | 同步产生的重复副本（一份逐位相同、一份为缺 superseded 头的旧版） |

## 四、整理时发现的待办

1. [[项目总控]] 与 [[current_task]] 尚未更新至 7/6 新战略（弃 DDL、性能+创新、训练解禁、锚点机制入队）——下一次会话第一件事。
2. 知识的时间线速查：**当前有效数字** = clean_baseline_v1（SR 16/OSR 21）；24% 及其之前的所有 SR 均带 oracle（见 20260706 组会报告 §6），引用历史数字时必须带此脚注。

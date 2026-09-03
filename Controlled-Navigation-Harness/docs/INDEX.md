---
date: 2026-07-06
updated: 2026-09-03
tags: [index, 归档]
---

# 文档索引与归档说明

> 整理原则：只移动、不删除。`docs/` 只保留会驱动当前工作或运行所需的文档；历史记录和证据保存在与 `docs/` 平级的 `archive/`。
>
> 事实状态、设计提案和历史证据分开管理。归档不表示内容错误，只表示它不再直接定义当前实施动作。

## 一、当前执行入口

建议按以下顺序阅读：

| 顺序 | 文档 | 作用 |
|---:|---|---|
| 0 | [[导航系统通俗全链路说明-20260903]] | 面向非技术读者：从自然语言指令、RGB-D 观察到移动、STOP、运行命令和日志 |
| 1 | [[空间状态工作任务-20260901]] | 当前任务、证据口径、实验队列和 Go/No-Go 条件 |
| 1.5 | [[当前全链路输入输出图解-20260902]] | 当前代码的详细模块输入输出、TerminalTrack 和统一 STOP 数据流 |
| 2 | [[全链路状态与模块输入输出-20260901]] | 当前源码和配置实际运行链，不等于目标架构 |
| 3 | [[主Pipeline重构设计-20260901]] | RouteState 驱动的新主链目标、迁移阶段和验收契约 |
| 4 | [[current_task]] | 历史指标与滚动状态；引用数字前需核对更新时间 |

## 二、当前框架与运维

| 文档 | 作用 |
|---|---|
| [[实验框架-路线状态外置-20260713]] | 两轨 A/B、预算、配对和预注册框架 |
| [[backtracking_design_20260703]] | 回溯机制设计；在线实验中作为独立决策面固定或单独测试 |
| [[model_switch_guide]] | 模型切换操作说明 |
| [[qwen35_4b_local_deployment]] | Qwen 本地部署说明 |

## 三、长期参考归档

路径均相对于 `Controlled-Navigation-Harness/`。

| 位置 | 内容 | 使用规则 |
|---|---|---|
| `archive/reference/experiment_records/` | 2026-06 至 2026-07 的逐轮实验台账和汇总日志 | 保留原始 provenance；不能用旧配置覆盖新实验口径 |
| `archive/reference/reports/` | 历史组会、方向和风格审查报告 | 用于复盘，不直接定义当前任务 |
| `archive/reference/design_evidence/` | ACN/L1/C5 规格、全链路审查、OSR 与动作接口分析等 | 作为设计依据；结论以当前执行入口的勘误为准 |
| `archive/reference/calibration/` | 三代 E1 校准图表和说明 | 各代模型与数据口径不同，不合并或去重 |

主要历史判断的当前位置：

| 文档 | 当前位置 |
|---|---|
| [[状态串联项目-问题审查与下一步-20260831]] | `archive/reference/design_evidence/`；已由当前工作任务和主 Pipeline 设计承接 |
| [[全链条重审-状态维度-20260730]] | `archive/reference/design_evidence/` |
| [[ACN代码落地与首轮离线结果-20260805]] | `archive/reference/design_evidence/` |
| [[L1规格-约束队列进度定位-20260802]] | `archive/reference/design_evidence/` |
| [[C5规格-拓扑记忆接入决策-20260805]] | `archive/reference/design_evidence/` |

## 四、停止引用的旧代际

| 位置 | 内容 | 原因 |
|---|---|---|
| `archive/superseded_方案/` | 旧总方案、AAAI 执行计划、近目标终止和旧实验框架 | 已被后续战略、oracle 审计或新框架替代 |
| `archive/era_0610-0623/` | 早期代码审查、SR/V 系列方案和运行比较 | 已被后续实现和证据链消化 |
| `archive/duplicates/` | 同步产生的重复副本 | 只保留用于历史核对 |
| `archive/*-封存-*.md` 等根级文件 | 早期 V/U/L 系列与自由工具调用方案 | 明确封存，不再驱动行动 |

## 五、引用纪律

1. 代码存在、离线结果和在线收益必须分别表述。
2. 当前主链事实以 [[全链路状态与模块输入输出-20260901]] 和实际 resolved config 为准。
3. 目标架构以 [[主Pipeline重构设计-20260901]] 为准，但其 `status: proposed`，不能写成已实现。
4. 历史 SR/OSR 必须同时给出配置、episode 数和来源；SR 16/OSR 21 与 SR 20/OSR 25 不得混成同一基线。
5. Wiki 链接按文件 basename 解析；移动到 archive 后仍保持唯一文件名，避免建立重复副本。

## 六、目录边界

```text
Controlled-Navigation-Harness/
├─ docs/                         当前执行入口、框架和运维
└─ archive/
   ├─ reference/                 有效历史证据，不驱动当前行动
   │  ├─ experiment_records/
   │  ├─ reports/
   │  ├─ design_evidence/
   │  └─ calibration/
   ├─ superseded_方案/           已被替代方案
   ├─ era_0610-0623/             早期代际
   └─ duplicates/                重复副本
```

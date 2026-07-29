---
name: feedback-no-autonomous-runs
description: 用户要求：不要自己启动评测/长时 GPU 任务，准备好后通知用户手动跑
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 492084f4-1f0f-4063-8a92-2e46db3bcfff
  modified: 2026-07-18T15:26:19.196Z
---

不要自己启动 Open-Nav 的评测跑(ep100 / smoke / 任何占 GPU 的长任务)。改动准备好之后，把启动命令给用户，由用户手动执行。

**Why:** 用户在 20260718 明确说了两次("我手动启动就行了"、"记住别自己跑，要跑的时候通知我手动跑")。单卡 4090 串行，一次 ep100 约 9–12 小时，GPU 是独占资源；用户要自己掌握什么时候占卡、占多久。第一次说的时候我没记住，第二次才写下来。

**How to apply:** 代码/配置改完 → 自检(语法、开关生效横幅)→ **把命令贴出来，停下**。不要用 Bash 后台跑 run.py，也不要"先跑一集验证一下"。只读的分析(读 trace、算指标、grep 代码)不受此限，照做。相关：[[project_oracle_leak_20260704]]

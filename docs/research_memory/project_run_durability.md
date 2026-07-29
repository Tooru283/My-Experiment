---
name: project-run-durability
description: 评测轮次的掉电耐受性——checkpoint曾因rename-without-fsync清零，navigation_records是真正的救命记录
metadata: 
  node_type: memory
  type: project
  originSessionId: 492084f4-1f0f-4063-8a92-2e46db3bcfff
  modified: 2026-07-20T04:02:14.876Z
---

**宿主是 KVM 虚机（`ens3` / `192.168.122.7`），会无预警硬掉电。**
20260720 run 004318 在第 83/100 集被掉电打断（journal 末行 09:37:10 + 关机序列 0 条 +
37 分钟后重启；训练进程与 lmdeploy 后端同刻停止、后端日志断在半个请求上）。

**诊断顺序（复用这条）**：无 traceback → 查两个进程是否同刻死 → `journalctl -b -1`
末行 + `grep -c "Reached target Shutdown"` → 排除 OOM(`dmesg | grep oom-kill`)、
排除 SIGHUP(比对 session 号与 CPU 用时)、排除 NVRM Xid。**同刻死 + 零关机记录 = 宿主事件，
不是代码 bug。**

**`checkpoint_stats_episodes.json` 曾落盘 0 字节**——不是"只在最后写"（它每集都写、
且已用 `os.replace`），而是 **`os.replace` 前缺 `fsync`**：rename 元数据落盘时数据块
还在页缓存。已修（`base_il_trainer_llm.py:4087` 加 `flush()+os.fsync(fileno())`，
`os.replace` 后再 fsync 目录 fd）。

**真正救命的是 `logs/navigation_records/*.jsonl`** —— 逐集追加写，含完整官方指标
（success/oracle_success/spl/ndtw/path_length），82 集结果全靠它恢复。
harness_traces 里的 `episode_termination` **不含**指标，只有 reason/step_length。
**逐集追加 > 原子替换的汇总文件。**

**没有断点续跑**：`base_il_trainer_llm.py:473` `stats_episodes={}` 每次从空开始。
4181 行有 `if next_episodes[i].episode_id in stats_episodes: pause` 的跳过逻辑，但它在
episode 结束后的 pause 分支里、不在 reset 之后，**光灌字典不足以跳过首批**，要另加显式跳过。
温度 0 下重跑会逐字节复现已跑过的集 → 补跑少数集的代价是整轮时长。

见 [[project-arm-c-20260719]] [[feedback-no-autonomous-runs]]

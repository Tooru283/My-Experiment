---
name: project-a0-baseline
description: "A0 baseline confirmed result (20260628, ep100 val_unseen) — SR/OSR/conversion anchor for near-goal termination research"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5d2f1a43-eb8f-42ea-8246-03f8a4eac31d
---

A0 baseline (clean, ENABLE_DECISION_EFFECT=False, U_SERIES.ENABLED=False, all V-series LOG_ONLY=True) confirmed on 20260628, ep100 val_unseen.

Run: `logs/eval_results/ep100/20260628/ep100_series_qwen_siglip_local_20260628_125248/`
Config: `run_OpenNav.yaml`, TRACE_DIR: `logs/harness_traces/a0_baseline`

| Metric | Value |
|---|---|
| SR | 16.0% (16/100) |
| OSR | 25.0% (25/100) |
| OSR→SR conversion | 64.0% (16/25) |
| OSR-SR gap | 9 episodes |
| SPL | 0.1015 |
| nDTW | 0.4458 |
| avg steps | 8.22 |

Near-goal failure taxonomy (OSR=1, SR=0, 9 episodes):
- walk-through (T1 漏停): 7/9 (77.8%) — episodes 377, 513, 516, 531, 568, 586, 824
- off-goal-stop (T2/T4): 2/9 (22.2%) — episodes 602, 1084

Worst T1 cases by drift: 824 (4.58m), 516 (4.52m), 513 (0.51m min_dist — nearly reached goal)

STOP behavior:
- selector_stop_requests: 220
- final_stop_actions: 44 (many gated out, 176 stop_rejected_fallback)
- final_stop_success: 9/44 = 20.5%

Visual evidence (V-series log-only):
- current_view_samples: 220 (one per stop request)
- high_conf(≥0.95) + target + arrival: 72/220 (32.7%) — potential M2 trigger signal

**Why:** This is the anchor baseline for the 总方案 plan. All subsequent ablations (M1 arrival_gate, M2 T1 auto-stop, M3-M5) must be compared against SR=16%, OSR=25%, conversion=64%.

**How to apply:** When evaluating any new run, compare SR/OSR/conversion against these numbers. A run is only an improvement if conversion rate rises without OSR degrading significantly. The T1 walk-through rate (77.8% of near-goal failures) sets the theoretical upper bound for M2.

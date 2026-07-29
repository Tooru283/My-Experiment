---
name: project-max-config-20260628
description: "max_config ep100 first evaluation — SR=21%, OSR→SR=80.8%, 5 near-goal failures identified"
metadata: 
  node_type: memory
  type: project
  originSessionId: d7e660c1-ead5-4844-8398-9823f61067f3
---

max_config (U1+U2+U3 full decision-effect) first 100-episode evaluation on 2026-06-28.

**Why:** Baseline after A0 (SR=16%). max_config added harness stop gates with full decision effect.

**Results:**
- SR=21% (+5pp vs A0), OSR=26%, OSR→SR=80.8% (+16.8pp), SPL=0.1424
- OSR-SR gap reduced from 9→5 episodes
- Failure mode shift: walk-through (7→2), stop-blocked (0→3)

**5 near-goal failures (OSR=1, SR=0):**
| ep | min_dist | failure | root cause |
|---|---|---|---|
| 11 | 0.40m | walk-through | selector doesn't STOP at near-goal steps; step 2 STOP blocked by before_min_steps |
| 377 | 1.78m | walk-through | visual_allow=False (reasons=[]) |
| 824 | 2.58m | stop-blocked | final_target_not_visible + relation_contradiction |
| 1084 | 1.32m | walk-through | final_target_not_visible + selector never outputs STOP |
| 1106 | 2.68m | stop-blocked | final_target_not_visible + relation_contradiction |

**Harness traces:** `logs/harness_traces/ep100/20260628/max_config/`
**Stats:** `logs/eval_results/ep100/20260628/ep100_series_qwen_siglip_local_20260628_231537/`

**How to apply:** Use as baseline for comparing M-series fixes. M3 fixed ep11; RC3 targets ep824/1084/1106.

# SR decline solution - 2026-06-16

## Problem

Run:

`logs/navigation_records/ep100_series_qwen_siglip_local_20260615_232107_train_navigation_20260615_232129.jsonl`

Final metrics from 100 episodes:

- SR: 0.16
- OSR: 0.24
- `step_length_limit`: 74 episodes, 10 successes
- `stop_requested`: 26 episodes, 6 successes
- Oracle-only near misses: 8 episodes
- Movement quality: 714 moves, 294 with `distance_gain_selected <= 0`

The apparent monotonic SR drop is mainly cumulative averaging. Window SR is not monotonic:

`0.30, 0.20, 0.10, 0.20, 0.10, 0.00, 0.20, 0.30, 0.10, 0.10`

Cumulative SR drops from 0.30 after 10 episodes to 0.16 after 100 because later failures dominate the average.

## Root Causes

1. Step budget is too tight.
   Short action sequences were hard-coded to 8 steps. Most failed episodes hit `step_length_limit`, and several oracle-success episodes were cut off near the goal.

2. Completion parsing is fragile.
   Most completion-estimator calls missed the `Executed Actions` marker, often because verbose reasoning consumed the token cap before the structured section. `should_stop()` then rejected STOP due to no structured executed actions.

3. Selector STOP was too permissive.
   When the navigator predicted STOP, visual evidence could override the completion gate. This allowed STOP on generic landmarks such as door, sink, bed, couch, doorway, or table even when the route order was not completed.

4. Later steps often lose progress.
   Average distance gain after step 4 is near zero, so extra exploration without better STOP gating can cause loops or drift. The fix must combine more step budget with stricter STOP acceptance.

## Implemented Fixes

1. Completion estimator now emits `Executed Actions:` first.
   This keeps the machine-readable section before verbose reasoning, reducing truncation failures.

2. Completion parsing now extracts only the `Executed Actions` section.
   It also recovers likely executed actions from older unstructured responses when the marker is missing.

3. Selector STOP no longer bypasses completion.
   Visual verifier can still reject unsupported STOP, but it cannot directly allow selector STOP unless the completion gate already passed.

4. Visual STOP allow is more conservative for selector STOP.
   Generic final terms remain configurable diagnostics, but the default fix does not block completion-confirmed STOP on terms such as `door`, `sink`, or `bed`. The main guard is that selector STOP cannot bypass completion progress.

5. Step limit is configurable.
   `OPENNAV_HARNESS.NAVIGATION_RUNTIME` replaces the 8/10 hard-code. Current run config uses 10 steps for short action chains and 12 for longer chains.

## Validation Plan

Run a small controlled evaluation first, then the 100-episode run:

- Check `completion_estimation` non-empty rate.
- Check `stop_requested` success ratio.
- Check `step_length_limit` count.
- Track SR/OSR/SPL plus `distance_gain_selected <= 0`.
- Compare window SR and cumulative SR, not cumulative SR alone.

Expected direction:

- Fewer false `stop_requested` episodes.
- Fewer near-goal `step_length_limit` misses.
- Higher OSR-to-SR conversion.
- Slightly longer runtime because completion output and step budget are less constrained.

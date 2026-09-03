#!/usr/bin/env python
"""A/B smoke two-hop verifier for the staged increments (tag increments_v2_staged_20260705).

Reads harness_traces jsonl emitted after a smoke launch and asserts the ONE caveat that
cannot be checked offline for each switch:

  depth   : DEPTH_STOP_VETO actually reads the depth sensor -> `depth_stop_veto` events
            exist AND at least one carries a non-null `depth_reading_m`.
            (all-null == helper never resolved a depth key -> mechanism inert.)

  backtrack: BACKTRACK fires end-to-end -> a `backtrack_offer` with offered==True is
            followed by a `backtrack_apply` with applied==True. The apply event firing at
            all proves MOVE_BACK survived test_decisions and reached the single action-4
            emitter (the apply block only runs when next_vp == MOVE_BACK_CANDIDATE).

Usage:
  python verify_smoke.py --mode depth     --since <epoch> [--traces-root logs/harness_traces]
  python verify_smoke.py --mode backtrack  --since <epoch> [--traces-root logs/harness_traces]

Exit code 0 == PASS, 2 == FAIL, 3 == INCONCLUSIVE (mechanism never exercised; widen episodes).
Never reads or reports SR / success / distance metrics.
"""
import argparse
import glob
import json
import os
import sys


def _load_events(traces_root, since):
    """Yield (episode_id, event_type, payload) for jsonl files modified after `since`."""
    for path in glob.glob(os.path.join(traces_root, "**", "*.jsonl"), recursive=True):
        try:
            if os.path.getmtime(path) < since:
                continue
        except OSError:
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield (
                    str(obj.get("episode_id")),
                    obj.get("event_type"),
                    obj.get("payload") or {},
                )


def verify_depth(events):
    total = non_null = vetoed = fail_open = 0
    per_ep = {}
    for ep, et, pl in events:
        if et != "depth_stop_veto":
            continue
        total += 1
        reading = pl.get("depth_reading_m")
        if reading is not None:
            non_null += 1
        if pl.get("vetoed"):
            vetoed += 1
        if pl.get("fail_open_unavailable"):
            fail_open += 1
        per_ep.setdefault(ep, []).append((reading, pl.get("vetoed"), pl.get("view_id")))

    print("=== DEPTH_STOP_VETO smoke ===")
    print(f"  depth_stop_veto events        : {total}")
    print(f"  with non-null depth_reading_m : {non_null}")
    print(f"  vetoed (reading > threshold)  : {vetoed}")
    print(f"  fail_open_unavailable (null)  : {fail_open}")
    for ep, rows in sorted(per_ep.items()):
        for reading, v, view in rows:
            rd = f"{reading:.2f}m" if isinstance(reading, (int, float)) else "null"
            print(f"    ep{ep}: reading={rd} vetoed={v} view_id={view}")

    if total == 0:
        print("VERDICT: INCONCLUSIVE -- no stop was evaluated in the smoke episodes.")
        print("  -> the veto never got a chance to fire. Re-run with OPENNAV_EPISODE_IDS")
        print("     including a known stop case (ep377 = conf-1.00@5.42m), or more episodes.")
        return 3
    if non_null == 0:
        print("VERDICT: FAIL -- events fired but every depth_reading_m is null.")
        print("  -> _current_view_depth_array never resolved a depth key; mechanism is inert.")
        print("     Fix key resolution before trusting any A/B result.")
        return 2
    print("VERDICT: PASS -- depth sensor is being read (non-null readings present).")
    if vetoed:
        print(f"  -> {vetoed} stop(s) vetoed on real geometry; offline hit/false-veto channel live.")
    return 0


def verify_backtrack(events):
    offered = applied = not_applied = offer_seen = 0
    apply_reasons = {}
    for ep, et, pl in events:
        if et == "backtrack_offer":
            offer_seen += 1
            if pl.get("offered"):
                offered += 1
        elif et in ("backtrack_apply", "backtrack_apply_failed"):
            if pl.get("applied"):
                applied += 1
            else:
                not_applied += 1
                r = pl.get("reason", "unknown")
                apply_reasons[r] = apply_reasons.get(r, 0) + 1

    print("=== BACKTRACK smoke ===")
    print(f"  backtrack_offer events        : {offer_seen}")
    print(f"  offers with offered==True     : {offered}")
    print(f"  backtrack_apply applied==True : {applied}")
    print(f"  apply not-applied             : {not_applied} {dict(apply_reasons) if apply_reasons else ''}")

    if offered == 0:
        print("VERDICT: INCONCLUSIVE -- backtrack was never offered (no ego-stall / dead_end hit).")
        print("  -> the trigger conditions never occurred in the smoke set. Re-run with")
        print("     OPENNAV_EPISODE_IDS covering wandering / step-limit episodes (e.g. 1084, 1106).")
        return 3
    if applied == 0:
        print("VERDICT: FAIL -- offered but never applied: MOVE_BACK is not reaching the emitter.")
        print("  -> check test_decisions offer_move_back gating and next_vp==MOVE_BACK_CANDIDATE path.")
        return 2
    print("VERDICT: PASS -- offered -> applied end-to-end; MOVE_BACK reaches the action-4 emitter.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["depth", "backtrack"])
    ap.add_argument("--since", required=True, type=float,
                    help="epoch seconds; only jsonl modified after this are scanned")
    ap.add_argument("--traces-root", default="logs/harness_traces")
    args = ap.parse_args()

    events = list(_load_events(args.traces_root, args.since))
    if not events:
        print(f"No harness_traces jsonl modified after {args.since} under {args.traces_root}.")
        print("Did the smoke run write traces? (harness logging enabled? correct traces-root?)")
        sys.exit(3)

    if args.mode == "depth":
        sys.exit(verify_depth(events))
    else:
        sys.exit(verify_backtrack(events))


if __name__ == "__main__":
    main()

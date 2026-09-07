"""Read-only audit of the September 2 EP100; writes derived review artifacts only.

Goal distances are offline diagnostic labels, never policy inputs. At decision
step t > 1, use the previous action_post_step distance: STOP can retain a stale
distance_gain_selected, so do not reconstruct each step from distance + gain.
"""

import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TRACE = ROOT / "logs/navigation_records/ep100/20260902/terminal_track_v3_ep100_20260902_204313_train_navigation_20260902_204331.jsonl"
STATS = ROOT / "logs/eval_results/ep100/20260902/terminal_track_v3_ep100_20260902_204313/stats_ckpt_val_unseen.json"
EVENTS = {
    "action_post_step", "goal_progress_update", "visual_target_verifier",
    "pipeline_stop_decision", "terminal_evidence_memory",
}


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    episodes = collections.defaultdict(lambda: {"steps": collections.defaultdict(dict)})
    counts = collections.Counter()
    with TRACE.open() as stream:
        for line in stream:
            event = json.loads(line)
            kind, payload = event["event"], event["payload"]
            counts[kind] += 1
            if event["episode_id"] is None:
                continue
            episode = episodes[str(event["episode_id"])]
            if kind == "episode_start":
                episode["instruction"] = payload["instruction"]
            elif kind == "episode_end":
                episode.update(payload)
            elif kind in EVENTS:
                episode["steps"][int(event["step"])][kind] = payload

    rows, near_rows = [], []
    active, completed = collections.Counter(), collections.Counter()
    for eid, episode in episodes.items():
        if "metrics" not in episode:
            continue
        metrics = episode["metrics"]
        previous_distance = None
        route_already_complete = False
        stop_steps, raw_stop_verdicts, near = [], [], []
        for step, events in sorted(episode["steps"].items()):
            action = events.get("action_post_step", {}).get("step_outputs", [{}])[0]
            current_distance = action.get("distance_to_goal")
            # No audited episode proposes STOP at step 1. Gain is used only to
            # recover the initial distance; later steps use the prior endpoint.
            if previous_distance is None and current_distance is not None:
                gain = action.get("distance_gain_selected")
                previous_distance = current_distance + gain if gain is not None else None
            progress = events.get("goal_progress_update", {})
            (completed if route_already_complete else active)[progress.get("transition")] += 1
            route_already_complete |= progress.get("route_progress_complete") is True
            if previous_distance is not None and previous_distance < 3.0:
                near.append({
                    "episode_id": eid, "step": step,
                    "distance_before_action_m": previous_distance,
                    "success": int(metrics["success"]),
                    "route_complete": progress.get("route_progress_complete"),
                    "terminal_confirmed": progress.get("terminal_target_confirmed"),
                    "goal_complete": progress.get("goal_complete"),
                    "current_target": progress.get("current_target"),
                    "raw_v2_verdict": events.get("visual_target_verifier", {}).get("verdict"),
                })
            decision = events.get("pipeline_stop_decision", {})
            if decision.get("outcome") == "commit_goal_stop":
                stop_steps.append(step)
                raw_stop_verdicts.append(events.get("visual_target_verifier", {}).get("verdict"))
            if current_distance is not None:
                previous_distance = current_distance
        near_rows.extend(near)
        rows.append({
            "episode_id": eid, "success": int(metrics["success"]),
            "oracle_success": int(metrics["oracle_success"]),
            "final_distance_m": metrics["distance_to_goal"],
            "termination_reason": episode["termination_reason"],
            "goal_stop": bool(stop_steps), "goal_stop_steps": json.dumps(stop_steps),
            "raw_stop_verdicts": json.dumps(raw_stop_verdicts),
            "near_decision_steps": len(near),
            "route_complete_at_any_near_step": any(s["route_complete"] is True for s in near),
            "terminal_confirmed_at_any_near_step": any(s["terminal_confirmed"] is True for s in near),
            "goal_complete_at_any_near_step": any(s["goal_complete"] is True for s in near),
            "instruction": episode["instruction"],
        })
    sets = {
        "false_goal_stop": {r["episode_id"] for r in rows if r["goal_stop"] and not r["success"]},
        "missed_success": {r["episode_id"] for r in rows if r["oracle_success"] and not r["success"]},
        "true_goal_stop": {r["episode_id"] for r in rows if r["goal_stop"] and r["success"]},
        "step_limit_success": {r["episode_id"] for r in rows if r["termination_reason"] == "step_length_limit" and r["success"]},
    }
    missed = [r for r in rows if r["episode_id"] in sets["missed_success"]]
    groups = collections.defaultdict(list)
    for row in missed:
        key = "route_{}_terminal_{}".format(
            int(row["route_complete_at_any_near_step"]),
            int(row["terminal_confirmed_at_any_near_step"]),
        )
        groups[key].append(row["episode_id"])

    anchor = load_module("review_anchor", "vlnce_baselines/common/opennav_ext/anchor_chain.py")
    locator = load_module("review_locator", "vlnce_baselines/common/opennav_ext/progress_locator.py")
    chain = anchor.build_anchor_chain("Go through the door\nGo past the sofa", "door\nsofa")
    queue = locator.ConstraintQueueLocator()
    queue.reset_episode(chain["anchors"])
    probe = []
    # Observe the SECOND landmark first, without executing any motion.
    for step, tag in enumerate(["sofa", "sofa", "door", "door"]):
        state = queue.update(step, [0, 0, 0], 0.0, {"0": [tag]}, {"0": {"distance": 2.0}})
        probe.append({"step": step, "tag": tag, "j": state["j"], "complete": state["complete"]})

    stats = json.loads(STATS.read_text())
    assert len(rows) == 100 and counts["action_post_step"] == 970
    assert sum(r["success"] for r in rows) / len(rows) == stats["success"]
    assert sum(r["oracle_success"] for r in rows) / len(rows) == stats["oracle_success"]
    assert all(r["near_decision_steps"] for r in missed)
    assert counts["navigation_error"] == 0
    summary = {
        "trace": str(TRACE), "trace_sha256": hashlib.sha256(TRACE.read_bytes()).hexdigest(),
        "stats": stats, "episodes": len(rows), "decision_steps": counts["action_post_step"],
        "event_counts": dict(counts),
        "sets": {k: sorted(v, key=int) for k, v in sets.items()},
        "false_stop_missed_overlap": sorted(sets["false_goal_stop"] & sets["missed_success"], key=int),
        "report_regression_union_size": len(sets["false_goal_stop"] | sets["missed_success"] | sets["true_goal_stop"]),
        "protected_regression_union_size": len(set().union(*sets.values())),
        "near_goal_blocker_groups": dict(groups),
        "active_route_transitions": dict(active), "already_complete_transitions": dict(completed),
        "stationary_reverse_order_probe": probe,
        "probe_meaning": "Synthetic code counterexample, not an online performance estimate.",
        "distance_meaning": "Offline simulator labels at decision boundaries; not evidence of instruction completion.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_csv("episode_review.csv", rows)
    write_csv("near_goal_steps.csv", near_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

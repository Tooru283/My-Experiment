#!/usr/bin/env python3
"""Replay ACN over recorded observations. Does not evaluate a new navigation policy."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "vlnce_baselines/common/opennav_ext" / (name + ".py")
    spec = importlib.util.spec_from_file_location("replay_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay(trace_dir):
    build = load("anchor_chain").build_anchor_chain
    locator_type = load("progress_locator").ConstraintQueueLocator
    episodes = []
    for path in sorted(Path(trace_dir).glob("*.jsonl")):
        with path.open() as stream:
            events = [json.loads(line) for line in stream if line.strip()]
        metadata = next((e["payload"] for e in events if e["event_type"] == "episode_metadata"), None)
        if metadata is None:
            continue
        plan = build(metadata["actions"], metadata["landmarks"])
        locator = locator_type(enable_location=True)
        locator.reset_episode(plan["anchors"])
        previous_receipt = None
        statuses, missing, old_failures = Counter(), Counter(), 0
        old_final = None
        state = {}
        updates = 0
        for event in events:
            payload = event["payload"]
            if event["event_type"] == "pipeline_action_receipt":
                previous_receipt = payload
            elif event["event_type"] == "progress_locator":
                if "j" in payload:
                    old_final = payload["j"]
                else:
                    old_failures += 1
            elif event["event_type"] == "pipeline_observation_frame":
                tags = {}
                for key, observation in payload.get("candidate_observations", {}).items():
                    match = re.search(r"scene objects:(.*)", observation, re.I | re.S)
                    if match:
                        tags[str(key)] = [t.strip().lower() for t in match[1].split("|") if t.strip()]
                state = locator.update(
                    event["step_id"], payload.get("position"), payload.get("heading"),
                    tags, payload.get("candidate_geometry", {}), previous_receipt,
                )
                updates += 1
                evaluation = state.get("last_evaluation", {})
                if state["complete"]:
                    statuses["route_complete"] += 1
                else:
                    statuses[evaluation.get("status", "not_evaluated")] += 1
                    missing.update(evaluation.get("missing_evidence", []))
                assert state["regressions"] == 0, (path, event["step_id"])
        completed = state.get("completion_events", [])
        episodes.append({
            "episode_id": metadata.get("episode_id", path.stem),
            "updates": updates,
            "recorded_final_index": old_final,
            "replayed_final_index": state.get("j"),
            "total": plan["n_anchors"],
            "current_action": state.get("current_raw"),
            "recorded_tool_failures": old_failures,
            "statuses": dict(statuses),
            "missing_evidence_counts": dict(missing),
            "completion_events": completed,
        })
    if not episodes:
        raise ValueError("No episode traces with metadata found in " + str(trace_dir))
    return {
        "scope": "fixed_trajectory_progress_replay_not_navigation_evaluation",
        "episodes": episodes,
        "episode_count": len(episodes),
        "updates": sum(e["updates"] for e in episodes),
        "completed_anchors": sum(e["replayed_final_index"] for e in episodes),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.trace_dir), ensure_ascii=False, indent=2))

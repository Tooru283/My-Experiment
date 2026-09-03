#!/usr/bin/env python3
"""Validate oracle-free RouteState and terminal-gate trace contracts offline."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


FORBIDDEN_ROUTE_STATE_KEYS = frozenset(
    {
        "latest_goal_dist",
        "distance_to_goal",
        "distance_gain_selected",
        "recent_distance_gains",
        "success",
        "oracle_success",
    }
)


def _events(paths: Iterable[Path]) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as trace_file:
            for line_number, line in enumerate(trace_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
                if not isinstance(event, dict):
                    raise ValueError(f"{path}:{line_number}: event is not an object")
                yield path, line_number, event


def _forbidden_paths(value: Any, prefix: str = "") -> List[str]:
    if isinstance(value, dict):
        paths = []
        for key, item in value.items():
            key = str(key)
            current = f"{prefix}.{key}" if prefix else key
            if key in FORBIDDEN_ROUTE_STATE_KEYS:
                paths.append(current)
            paths.extend(_forbidden_paths(item, current))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_forbidden_paths(item, f"{prefix}[{index}]"))
        return paths
    return []


def audit(trace_dir: Path) -> Tuple[List[str], Dict[str, int]]:
    trace_files = sorted(trace_dir.rglob("*.jsonl"))
    errors: List[str] = []
    counts = {
        "trace_files": len(trace_files),
        "route_state": 0,
        "run_metadata": 0,
        "terminal_gate_rejected": 0,
    }
    if not trace_files:
        return [f"No JSONL traces found below {trace_dir}"], counts

    rejected_stops: Set[Tuple[str, int]] = set()
    emitted_stops: Set[Tuple[str, int]] = set()
    for path, line_number, event in _events(trace_files):
        event_type = event.get("event_type")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        episode_id = str(event.get("episode_id", ""))
        step_id = event.get("step_id")
        step_key = (episode_id, int(step_id)) if isinstance(step_id, int) else None

        if event_type == "run_metadata":
            counts["run_metadata"] += 1
            for field in (
                "config_snapshot",
                "resolved_config_sha256",
                "exp_config_sha256",
            ):
                if not payload.get(field):
                    errors.append(f"{path}:{line_number}: run_metadata lacks {field}")

        if event_type == "route_state":
            counts["route_state"] += 1
            state = payload.get("state")
            if not isinstance(state, dict):
                errors.append(f"{path}:{line_number}: route_state lacks payload.state")
                continue
            forbidden = _forbidden_paths(state)
            if forbidden:
                errors.append(
                    f"{path}:{line_number}: oracle fields leaked into RouteState: {', '.join(forbidden)}"
                )
            if state.get("schema_version") != "route_state.v1":
                errors.append(f"{path}:{line_number}: unexpected RouteState schema")

        if event_type == "terminal_gate_rejected" and step_key is not None:
            counts["terminal_gate_rejected"] += 1
            rejected_stops.add(step_key)

        if event_type == "action_pre_step" and step_key is not None:
            action = payload.get("env_action")
            action = action.get("action") if isinstance(action, dict) else None
            action_code = action.get("action") if isinstance(action, dict) else None
            if action_code == 0:
                emitted_stops.add(step_key)

    conflicting = rejected_stops & emitted_stops
    for episode_id, step_id in sorted(conflicting):
        errors.append(
            f"episode {episode_id}, step {step_id}: terminal gate rejected STOP but STOP was emitted"
        )
    if counts["run_metadata"] == 0:
        errors.append("No run_metadata event found; config provenance is not auditable")
    if counts["route_state"] == 0:
        errors.append("No route_state event found; route-state coverage is not auditable")
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()

    errors, counts = audit(args.trace_dir)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: RouteState and terminal-gate trace contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

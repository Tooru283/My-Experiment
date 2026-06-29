#!/usr/bin/env python3
"""Analyze Open-Nav navigation records and harness traces.

The script accepts JSONL navigation records, JSONL harness traces, and Habitat
stats JSON files. It is intentionally schema-tolerant because the project has
both navigation-record rows with an ``event`` field and harness-trace rows with
an ``event_type`` field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Number = Optional[float]


@dataclass
class EpisodeSummary:
    episode_id: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    distances: List[Tuple[int, float]] = field(default_factory=list)
    gains: List[Tuple[int, float]] = field(default_factory=list)
    phases: List[Tuple[int, str]] = field(default_factory=list)
    stop_requests: List[int] = field(default_factory=list)
    final_stop_steps: List[int] = field(default_factory=list)
    stop_rejections: List[int] = field(default_factory=list)
    selector_empty_steps: List[int] = field(default_factory=list)
    fallback_events: List[Dict[str, Any]] = field(default_factory=list)
    recovery_events: List[Dict[str, Any]] = field(default_factory=list)
    unit_overrides: List[Dict[str, Any]] = field(default_factory=list)
    action_overrides: List[Dict[str, Any]] = field(default_factory=list)
    failure_types: Counter = field(default_factory=Counter)
    recover_reasons: Counter = field(default_factory=Counter)
    had_weak_target: bool = False
    abstain_steps: List[int] = field(default_factory=list)

    def final_distance(self) -> Number:
        value = self.metrics.get("distance_to_goal")
        if value is not None:
            return to_float(value)
        if self.distances:
            return self.distances[-1][1]
        return None

    def min_distance(self) -> Tuple[Number, Optional[int]]:
        candidates = [(dist, step) for step, dist in self.distances]
        if not candidates:
            return None, None
        dist, step = min(candidates, key=lambda item: item[0])
        return dist, step

    def success(self, radius: float) -> Optional[bool]:
        value = self.metrics.get("success")
        if value is not None:
            return bool(to_float(value))
        final = self.final_distance()
        if final is None:
            return None
        return final <= radius

    def oracle_success(self, radius: float) -> Optional[bool]:
        value = self.metrics.get("oracle_success")
        if value is not None:
            return bool(to_float(value))
        min_dist, _ = self.min_distance()
        if min_dist is None:
            return None
        return min_dist <= radius

    def last_phase(self) -> str:
        if not self.phases:
            return ""
        return self.phases[-1][1]


@dataclass
class AnalysisState:
    episodes: Dict[str, EpisodeSummary] = field(default_factory=dict)
    event_files: List[Path] = field(default_factory=list)
    stats_files: List[Path] = field(default_factory=list)
    aggregate_stats: Dict[str, Any] = field(default_factory=dict)
    event_counts: Counter = field(default_factory=Counter)
    phase_counts: Counter = field(default_factory=Counter)
    stop_sources: Counter = field(default_factory=Counter)
    stop_allowed_by_source: Counter = field(default_factory=Counter)
    stop_rejected_by_source: Counter = field(default_factory=Counter)
    stop_rescue_by_source: Counter = field(default_factory=Counter)
    stop_reject_reasons: Counter = field(default_factory=Counter)
    current_view_total: int = 0
    current_view_high_conf: int = 0
    current_view_high_conf_arrival: int = 0
    selector_empty_count: int = 0
    selector_empty_fallback_count: int = 0
    stop_rejected_fallback_count: int = 0
    fallback_changed_count: int = 0
    fallback_trusted_count: int = 0
    recovery_applied_count: int = 0
    action_override_count: int = 0
    abstain_count: int = 0
    abstain_weak_target_count: int = 0
    arrival_gate_total: int = 0
    arrival_gate_triggered: int = 0
    arrival_gate_triggered_episodes: set = field(default_factory=set)
    schema_errors: Counter = field(default_factory=Counter)
    schema_warnings: Counter = field(default_factory=Counter)
    parse_errors: Counter = field(default_factory=Counter)
    failure_types: Counter = field(default_factory=Counter)
    recover_reasons: Counter = field(default_factory=Counter)
    bad_json_lines: int = 0
    duplicate_events: int = 0
    seen_event_hashes: set = field(default_factory=set)

    def episode(self, episode_id: Any) -> EpisodeSummary:
        key = str(episode_id)
        if key not in self.episodes:
            self.episodes[key] = EpisodeSummary(episode_id=key)
        return self.episodes[key]


def to_float(value: Any) -> Number:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def event_name(row: Dict[str, Any]) -> str:
    return str(row.get("event_type") or row.get("event") or "")


def event_step(row: Dict[str, Any]) -> int:
    return to_int(row.get("step_id", row.get("step", 0)))


def event_episode(row: Dict[str, Any]) -> Optional[str]:
    episode_id = row.get("episode_id")
    if episode_id is None and isinstance(row.get("payload"), dict):
        episode_id = row["payload"].get("episode_id")
    if episode_id is None:
        return None
    return str(episode_id)


def stable_event_hash(row: Dict[str, Any]) -> str:
    payload = row.get("payload", {})
    key = {
        "episode_id": event_episode(row),
        "step": event_step(row),
        "event": event_name(row),
        "payload": payload,
    }
    blob = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path, state: AnalysisState) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                state.bad_json_lines += 1
                continue
            if isinstance(value, dict):
                yield value


def expand_paths(paths: Sequence[str], suffixes: Tuple[str, ...]) -> List[Path]:
    result: List[Path] = []
    for raw in paths:
        matches = sorted(Path().glob(raw)) if any(ch in raw for ch in "*?[]") else [Path(raw)]
        for path in matches:
            if path.is_dir():
                for suffix in suffixes:
                    result.extend(sorted(path.rglob(f"*{suffix}")))
            elif path.is_file() and path.suffix in suffixes:
                result.append(path)
    seen = set()
    unique: List[Path] = []
    for path in result:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def extract_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    parsed = payload.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
        return [item for item in parsed["candidates"] if isinstance(item, dict)]
    if isinstance(payload.get("candidates"), list):
        return [item for item in payload["candidates"] if isinstance(item, dict)]
    return []


def selected_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    value = payload.get("selected_candidate")
    if value is None:
        value = payload.get("final_action")
    if value is None:
        return None
    return str(value)


def predictions_include_stop(payload: Dict[str, Any]) -> bool:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        return False
    return any(str(item).upper() == "STOP" for item in predictions)


def action_is_stop(value: Any) -> bool:
    return str(value).upper() == "STOP"


def process_action_post_step(ep: EpisodeSummary, step: int, payload: Dict[str, Any]) -> None:
    outputs = payload.get("step_outputs")
    if not isinstance(outputs, list):
        outputs = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        distance = to_float(output.get("distance_to_goal"))
        if distance is not None:
            ep.distances.append((step, distance))
        gain = to_float(output.get("distance_gain_selected"))
        if gain is not None:
            ep.gains.append((step, gain))
    stop_reason = payload.get("stop_reason")
    if stop_reason:
        ep.final_stop_steps.append(step)


def process_stop_verification(
    state: AnalysisState,
    ep: EpisodeSummary,
    step: int,
    payload: Dict[str, Any],
) -> None:
    source = str(payload.get("source") or "unknown")
    state.stop_sources[source] += 1
    allow_stop = bool(payload.get("allow_stop"))
    allow_rescue = bool(payload.get("allow_rescue"))
    if allow_stop:
        state.stop_allowed_by_source[source] += 1
    else:
        state.stop_rejected_by_source[source] += 1
        if source == "selector_stop_gate":
            ep.stop_rejections.append(step)
    if allow_rescue:
        state.stop_rescue_by_source[source] += 1
    for reason in payload.get("reject_reasons") or []:
        state.stop_reject_reasons[str(reason)] += 1
    if payload.get("weak_target"):
        ep.had_weak_target = True
    if payload.get("abstain"):
        state.abstain_count += 1
        ep.abstain_steps.append(step)
        if payload.get("weak_target"):
            state.abstain_weak_target_count += 1


def process_current_view(
    state: AnalysisState,
    threshold: float,
    payload: Dict[str, Any],
) -> None:
    candidates = extract_candidates(payload)
    for candidate in candidates:
        if str(candidate.get("candidate_id")) != "__current_view__":
            continue
        state.current_view_total += 1
        confidence = to_float(candidate.get("confidence")) or 0.0
        final_target = bool(candidate.get("final_target_visible"))
        arrival = bool(candidate.get("arrival_evidence"))
        if confidence >= threshold:
            state.current_view_high_conf += 1
            if final_target and arrival:
                state.current_view_high_conf_arrival += 1


def process_fallback_event(
    state: AnalysisState,
    ep: EpisodeSummary,
    event: str,
    payload: Dict[str, Any],
) -> None:
    selected = selected_from_payload(payload)
    baseline = None
    metadata = payload.get("decision_test_metadata")
    if isinstance(metadata, dict):
        baseline = metadata.get("selected_candidate")
    if baseline is None:
        available = payload.get("available_candidates")
        if isinstance(available, list) and available:
            baseline = available[0]
    changed = selected is not None and baseline is not None and str(selected) != str(baseline)
    trusted = bool(payload.get("recovery_rank_trusted"))
    record = {
        "event": event,
        "selected": selected,
        "baseline": str(baseline) if baseline is not None else None,
        "changed": changed,
        "trusted": trusted,
        "reason": payload.get("fallback_reason") or payload.get("reason"),
    }
    ep.fallback_events.append(record)
    if event == "selector_empty_prediction_fallback":
        state.selector_empty_fallback_count += 1
    elif event == "stop_rejected_fallback":
        state.stop_rejected_fallback_count += 1
    if changed:
        state.fallback_changed_count += 1
    if trusted:
        state.fallback_trusted_count += 1


def process_event(
    row: Dict[str, Any],
    state: AnalysisState,
    current_view_threshold: float,
    dedupe_events: bool,
) -> None:
    if dedupe_events:
        fingerprint = stable_event_hash(row)
        if fingerprint in state.seen_event_hashes:
            state.duplicate_events += 1
            return
        state.seen_event_hashes.add(fingerprint)

    event = event_name(row)
    if not event:
        return
    state.event_counts[event] += 1
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    episode_id = event_episode(row)
    step = event_step(row)
    ep = state.episode(episode_id) if episode_id is not None else None

    if ep is not None and event == "episode_end":
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            ep.metrics.update(metrics)

    elif ep is not None and event == "action_post_step":
        process_action_post_step(ep, step, payload)

    elif ep is not None and event == "phase_evidence":
        phase = str(payload.get("phase") or "unknown")
        ep.phases.append((step, phase))
        state.phase_counts[phase] += 1
        if phase == "recover":
            last_failure = payload.get("last_failure") if isinstance(payload.get("last_failure"), dict) else {}
            reason = last_failure.get("failure_type") or last_failure.get("trigger_type") or payload.get("phase_reason")
            if reason:
                reason_text = str(reason)
                ep.recover_reasons[reason_text] += 1
                state.recover_reasons[reason_text] += 1

    elif ep is not None and event == "selector_raw":
        predictions = payload.get("predictions")
        if predictions == []:
            ep.selector_empty_steps.append(step)
            state.selector_empty_count += 1
        if predictions_include_stop(payload):
            ep.stop_requests.append(step)

    elif ep is not None and event == "selector_final":
        selected = payload.get("selected_candidate")
        if action_is_stop(selected):
            ep.stop_requests.append(step)

    elif ep is not None and event == "decision_audit":
        original = payload.get("original_action")
        final = payload.get("final_action")
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        if action_is_stop(final) or bool(extra.get("stop_flag")):
            ep.final_stop_steps.append(step)
        override = bool(payload.get("override")) or (
            original is not None and final is not None and str(original) != str(final)
        )
        if override:
            state.action_override_count += 1
            ep.action_overrides.append(payload)

    elif ep is not None and event == "unit_override":
        ep.unit_overrides.append(payload)
        if payload.get("action_affecting"):
            state.action_override_count += 1

    elif ep is not None and event == "arrival_gate":
        state.arrival_gate_total += 1
        if payload.get("triggered"):
            state.arrival_gate_triggered += 1
            if episode_id is not None:
                state.arrival_gate_triggered_episodes.add(str(episode_id))

    elif ep is not None and event == "stop_verification":
        process_stop_verification(state, ep, step, payload)

    elif event == "stop_current_view_evidence":
        process_current_view(state, current_view_threshold, payload)

    elif ep is not None and event in {"selector_empty_prediction_fallback", "stop_rejected_fallback"}:
        process_fallback_event(state, ep, event, payload)

    elif ep is not None and event == "failure_type_diagnostic":
        failure_type = payload.get("failure_type")
        trigger_type = payload.get("trigger_type")
        if failure_type:
            state.failure_types[str(failure_type)] += 1
            ep.failure_types[str(failure_type)] += 1
        if trigger_type:
            state.failure_types[f"trigger:{trigger_type}"] += 1

    elif ep is not None and event == "failure_recovery":
        ep.recovery_events.append(payload)
        if payload.get("applied") or payload.get("applied_to_action"):
            state.recovery_applied_count += 1

    if event in {"visual_evidence", "stop_current_view_evidence"}:
        schema_error = payload.get("schema_error")
        if schema_error:
            state.schema_errors[str(schema_error)] += 1
        for warning in payload.get("schema_warnings") or []:
            state.schema_warnings[str(warning)] += 1
        parse_error = payload.get("parse_error")
        if parse_error:
            state.parse_errors[str(parse_error)] += 1


def load_stats(path: Path, state: AnalysisState) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    state.stats_files.append(path)
    if data and all(isinstance(value, dict) for value in data.values()):
        for episode_id, metrics in data.items():
            state.episode(episode_id).metrics.update(metrics)
    else:
        state.aggregate_stats.update(data)


def classify_near_goal_failure(ep: EpisodeSummary, radius: float) -> str:
    """Classify the failure mode of an OSR=1, SR=0 episode.

    Categories (ordered by diagnostic priority):
      stop-blocked    — a STOP was requested and rejected near the closest approach
      walk-through    — episode ended at step limit without any explicit STOP
      premature-stop  — agent stopped before reaching its closest-to-goal point
      near-goal-drift — got within radius, then drifted >1.5 m before stopping
      off-goal-stop   — stopped outside radius with no other distinguishing signal
    """
    min_dist, min_step = ep.min_distance()
    near_window = 3
    if min_step is not None:
        near_rejections = [s for s in ep.stop_rejections if abs(s - min_step) <= near_window]
    else:
        near_rejections = list(ep.stop_rejections)
    if near_rejections:
        return "stop-blocked"
    if not ep.final_stop_steps:
        return "walk-through"
    final_dist = ep.final_distance()
    final_stop_step = min(ep.final_stop_steps)
    if min_step is not None and final_stop_step < min_step:
        return "premature-stop"
    drift = (final_dist - min_dist) if (min_dist is not None and final_dist is not None) else None
    if drift is not None and drift > 1.5:
        return "near-goal-drift"
    if ep.had_weak_target:
        return "weak-target-miss"
    return "off-goal-stop"


def mean(values: Iterable[Number]) -> Number:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def fmt_number(value: Number, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def metric_value(ep: EpisodeSummary, key: str) -> Number:
    return to_float(ep.metrics.get(key))


def build_summary(state: AnalysisState, radius: float) -> Dict[str, Any]:
    episodes = list(state.episodes.values())
    metric_eps = [ep for ep in episodes if ep.success(radius) is not None]
    success_count = sum(1 for ep in metric_eps if ep.success(radius))
    oracle_known = [ep for ep in episodes if ep.oracle_success(radius) is not None]
    oracle_success_count = sum(1 for ep in oracle_known if ep.oracle_success(radius))
    final_stops = [(ep, step) for ep in episodes for step in sorted(set(ep.final_stop_steps))]
    stop_success = sum(1 for ep, _ in final_stops if ep.success(radius))

    return {
        "episode_count": len(metric_eps),
        "event_file_count": len(state.event_files),
        "stats_file_count": len(state.stats_files),
        "success_count": success_count,
        "oracle_success_count": oracle_success_count,
        "sr": success_count / len(metric_eps) if metric_eps else None,
        "osr": oracle_success_count / len(oracle_known) if oracle_known else None,
        "osr_sr_conversion_rate": success_count / oracle_success_count if oracle_success_count else None,
        "spl": mean(metric_value(ep, "spl") for ep in metric_eps),
        "ndtw": mean(metric_value(ep, "ndtw") for ep in metric_eps),
        "final_distance": mean(ep.final_distance() for ep in metric_eps),
        "steps": mean(metric_value(ep, "steps_taken") for ep in metric_eps),
        "final_stop_count": len(final_stops),
        "final_stop_success_count": stop_success,
        "duplicate_events": state.duplicate_events,
        "bad_json_lines": state.bad_json_lines,
    }


def print_counter(title: str, counter: Counter, top: int) -> None:
    print(f"\n{title}")
    if not counter:
        print("  -")
        return
    for key, value in counter.most_common(top):
        print(f"  {key}: {value}")


def print_episode_table(title: str, rows: List[Tuple[Any, ...]], headers: Sequence[str], top: int) -> None:
    print(f"\n{title}")
    if not rows:
        print("  -")
        return
    widths = [len(header) for header in headers]
    shown = rows[:top]
    for row in shown:
        for idx, item in enumerate(row):
            widths[idx] = max(widths[idx], len(str(item)))
    header_line = "  " + "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    print(header_line)
    print("  " + "  ".join("-" * width for width in widths))
    for row in shown:
        print("  " + "  ".join(str(item).ljust(widths[idx]) for idx, item in enumerate(row)))
    if len(rows) > top:
        print(f"  ... {len(rows) - top} more")


def print_report(state: AnalysisState, radius: float, current_view_threshold: float, top: int) -> None:
    summary = build_summary(state, radius)
    episode_count = summary["episode_count"]
    print("Open-Nav Navigation Analysis")
    print(f"  event_files: {summary['event_file_count']}")
    print(f"  stats_files: {summary['stats_file_count']}")
    print(f"  episodes_with_metrics: {episode_count}")
    if summary["bad_json_lines"]:
        print(f"  bad_json_lines: {summary['bad_json_lines']}")
    if summary["duplicate_events"]:
        print(f"  duplicate_events_skipped: {summary['duplicate_events']}")

    print("\nMetrics")
    print(f"  SR: {summary['success_count']}/{episode_count} ({pct(summary['success_count'], episode_count)})")
    print(
        "  OSR: "
        f"{summary['oracle_success_count']}/{episode_count} "
        f"({pct(summary['oracle_success_count'], episode_count)})"
    )
    print(
        "  OSR-SR gap: "
        f"{summary['oracle_success_count'] - summary['success_count']} episodes"
    )
    print(
        "  OSR→SR conversion: "
        f"{pct(summary['success_count'], summary['oracle_success_count'])}"
    )
    print(f"  SPL: {fmt_number(summary['spl'])}")
    print(f"  nDTW: {fmt_number(summary['ndtw'])}")
    print(f"  final_distance: {fmt_number(summary['final_distance'])}")
    print(f"  steps: {fmt_number(summary['steps'], 2)}")

    if state.aggregate_stats:
        print("\nAggregate Stats File")
        for key in ["success", "oracle_success", "spl", "ndtw", "distance_to_goal", "steps_taken", "path_length"]:
            if key in state.aggregate_stats:
                print(f"  {key}: {state.aggregate_stats[key]}")

    osr_not_sr = []
    drift_rows = []
    taxonomy_counts: Counter = Counter()
    for ep in sorted(
        state.episodes.values(),
        key=lambda item: (0, int(item.episode_id)) if item.episode_id.isdigit() else (1, item.episode_id),
    ):
        success = ep.success(radius)
        oracle = ep.oracle_success(radius)
        min_dist, min_step = ep.min_distance()
        final_dist = ep.final_distance()
        drift = None if min_dist is None or final_dist is None else final_dist - min_dist
        near_goal_label = classify_near_goal_failure(ep, radius) if (oracle and not success) else None
        if near_goal_label:
            taxonomy_counts[near_goal_label] += 1
        row = (
            ep.episode_id,
            fmt_number(min_dist),
            "-" if min_step is None else min_step,
            fmt_number(final_dist),
            fmt_number(drift),
            ep.last_phase() or "-",
            near_goal_label or (",".join(key for key, _ in ep.failure_types.most_common(2)) or "-"),
        )
        if oracle and not success:
            osr_not_sr.append(row)
        if drift is not None:
            drift_rows.append((drift, row))

    print_episode_table(
        "\nOSR=1 but SR=0",
        osr_not_sr,
        ["episode", "min_dist", "min_step", "final_dist", "drift", "last_phase", "failure_type"],
        top,
    )
    drift_rows.sort(key=lambda item: item[0], reverse=True)
    print_episode_table(
        "Largest min->final distance drift",
        [row for _, row in drift_rows],
        ["episode", "min_dist", "min_step", "final_dist", "drift", "last_phase", "failure_type"],
        top,
    )

    osr_gap = summary["oracle_success_count"] - summary["success_count"]
    print(f"\nNear-Goal Failure Taxonomy  (OSR=1, SR=0 — {osr_gap} episodes)")
    if taxonomy_counts:
        for label, count in taxonomy_counts.most_common():
            print(f"  {label}: {count}/{osr_gap} ({pct(count, osr_gap)})")
    else:
        print("  - (no OSR=1, SR=0 episodes or no distance data)")

    print("\nArrival Gate")
    if state.arrival_gate_total > 0:
        trig_eps = len(state.arrival_gate_triggered_episodes)
        total_eps = summary["episode_count"]
        print(f"  total_gate_checks: {state.arrival_gate_total}")
        print(
            f"  triggered_steps: {state.arrival_gate_triggered}/{state.arrival_gate_total} "
            f"({pct(state.arrival_gate_triggered, state.arrival_gate_total)})"
        )
        print(f"  triggered_episodes: {trig_eps}/{total_eps} ({pct(trig_eps, total_eps)})")
    else:
        print("  (no arrival_gate events — gate disabled or run predates M1)")

    print("\nSTOP Summary")
    selector_stop_requests = sum(len(set(ep.stop_requests)) for ep in state.episodes.values())
    print(f"  selector_stop_requests: {selector_stop_requests}")
    print(f"  final_stop_actions: {summary['final_stop_count']}")
    print(
        "  final_stop_success: "
        f"{summary['final_stop_success_count']}/{summary['final_stop_count']} "
        f"({pct(summary['final_stop_success_count'], summary['final_stop_count'])})"
    )
    print(f"  stop_rejected_fallback_events: {state.stop_rejected_fallback_count}")
    print_counter("  stop_verification_by_source", state.stop_sources, top)
    print_counter("  stop_allowed_by_source", state.stop_allowed_by_source, top)
    print_counter("  stop_rejected_by_source", state.stop_rejected_by_source, top)
    print_counter("  stop_reject_reasons", state.stop_reject_reasons, top)
    print(f"  abstain_count: {state.abstain_count}")
    print(f"  abstain_weak_target_count: {state.abstain_weak_target_count}")

    print("\nCurrent-view Visual Evidence")
    print(f"  current_view_samples: {state.current_view_total}")
    print(
        f"  high_confidence >= {current_view_threshold}: "
        f"{state.current_view_high_conf}/{state.current_view_total} "
        f"({pct(state.current_view_high_conf, state.current_view_total)})"
    )
    print(
        "  high_confidence_with_target_and_arrival: "
        f"{state.current_view_high_conf_arrival}/{state.current_view_total} "
        f"({pct(state.current_view_high_conf_arrival, state.current_view_total)})"
    )

    print("\nFallback / Recovery")
    print(f"  selector_empty_raw_steps: {state.selector_empty_count}")
    print(f"  selector_empty_fallback_events: {state.selector_empty_fallback_count}")
    print(f"  fallback_changed_from_baseline: {state.fallback_changed_count}")
    print(f"  trusted_visual_rank_fallbacks: {state.fallback_trusted_count}")
    print(f"  recovery_applied_to_action: {state.recovery_applied_count}")
    print(f"  action_overrides: {state.action_override_count}")
    print_counter("  failure_types", state.failure_types, top)

    print_counter("\nPhase Distribution", state.phase_counts, top)
    print_counter("Recover trigger reasons", state.recover_reasons, top)

    print("\nSchema / Parse Health")
    print(f"  schema_errors: {sum(state.schema_errors.values())}")
    print(f"  schema_warnings: {sum(state.schema_warnings.values())}")
    print(f"  parse_errors: {sum(state.parse_errors.values())}")
    print_counter("  schema_error_types", state.schema_errors, top)
    print_counter("  schema_warning_types", state.schema_warnings, top)
    print_counter("  parse_error_types", state.parse_errors, top)


def write_json_summary(path: Path, state: AnalysisState, radius: float) -> None:
    summary = build_summary(state, radius)
    near_goal_taxonomy: Counter = Counter()
    for ep in state.episodes.values():
        if ep.oracle_success(radius) and not ep.success(radius):
            near_goal_taxonomy[classify_near_goal_failure(ep, radius)] += 1
    summary.update(
        {
            "event_counts": dict(state.event_counts),
            "phase_counts": dict(state.phase_counts),
            "stop_sources": dict(state.stop_sources),
            "stop_allowed_by_source": dict(state.stop_allowed_by_source),
            "stop_rejected_by_source": dict(state.stop_rejected_by_source),
            "stop_reject_reasons": dict(state.stop_reject_reasons),
            "failure_types": dict(state.failure_types),
            "near_goal_taxonomy": dict(near_goal_taxonomy),
            "abstain_count": state.abstain_count,
            "abstain_weak_target_count": state.abstain_weak_target_count,
            "recover_reasons": dict(state.recover_reasons),
            "schema_errors": dict(state.schema_errors),
            "schema_warnings": dict(state.schema_warnings),
            "parse_errors": dict(state.parse_errors),
        }
    )
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Open-Nav navigation JSONL records and stats files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="JSONL/JSON files, directories, or globs. Directories are scanned recursively.",
    )
    parser.add_argument(
        "--stats",
        action="append",
        default=[],
        help="Additional stats JSON file, directory, or glob.",
    )
    parser.add_argument(
        "--success-radius",
        type=float,
        default=3.0,
        help="Distance threshold used when success/oracle_success is missing.",
    )
    parser.add_argument(
        "--current-view-confidence",
        type=float,
        default=0.95,
        help="Threshold for high-confidence current-view evidence.",
    )
    parser.add_argument("--top", type=int, default=20, help="Maximum rows per printed table.")
    parser.add_argument("--json-out", type=Path, help="Optional path for machine-readable summary JSON.")
    parser.add_argument(
        "--dedupe-events",
        action="store_true",
        help="Skip repeated events with the same episode, step, event type, and payload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = AnalysisState()
    all_paths = list(args.paths)
    jsonl_files = expand_paths(all_paths, (".jsonl",))
    stats_files = expand_paths(all_paths + args.stats, (".json",))

    for path in stats_files:
        if path.name.startswith("stats") or "stats" in path.name:
            load_stats(path, state)

    for path in jsonl_files:
        state.event_files.append(path)
        for row in iter_jsonl(path, state):
            process_event(row, state, args.current_view_confidence, args.dedupe_events)

    print_report(state, args.success_radius, args.current_view_confidence, args.top)
    if args.json_out:
        write_json_summary(args.json_out, state, args.success_radius)
        print(f"\nWrote JSON summary: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

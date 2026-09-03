"""Serializable, oracle-free route state snapshots.

The navigation loop has several useful state modules, but they used to be logged as
independent events.  This module provides one stable snapshot shape for offline replay.
It deliberately excludes simulator goal distance and success labels; those belong in
the separate audit stream and must never be consumed as route state.
"""

from typing import Any, Dict, Iterable, Optional


ROUTE_STATE_SCHEMA_VERSION = "route_state.v1"
ORACLE_FIELDS_EXCLUDED = (
    "latest_goal_dist",
    "distance_to_goal",
    "distance_gain_selected",
    "recent_distance_gains",
    "success",
    "oracle_success",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _copy(value: Any) -> Any:
    """Copy nested trace data without importing the runtime's optional tensor stack."""
    if isinstance(value, dict):
        return {str(key): _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _copy(value.to_dict())
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value

def _without_oracle_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_oracle_fields(item)
            for key, item in value.items()
            if str(key) not in ORACLE_FIELDS_EXCLUDED
        }
    if isinstance(value, (list, tuple)):
        return [_without_oracle_fields(item) for item in value]
    return _copy(value)



def _ids(values: Optional[Iterable[Any]]) -> list:
    return [str(value) for value in (values or []) if value is not None]


def _provenance(
    source: str,
    confidence: Optional[float] = None,
    abstained: bool = False,
    decision_effect: bool = False,
) -> Dict[str, Any]:
    return {
        "source": source,
        "confidence": confidence,
        "abstained": bool(abstained),
        "decision_effect": bool(decision_effect),
    }


def build_route_state(
    *,
    episode_id: Any,
    step_id: int,
    stage: str,
    instruction: Any,
    actions: Any,
    landmarks: Any,
    anchors: Optional[Iterable[Dict[str, Any]]] = None,
    anchor_chain_result: Optional[Dict[str, Any]] = None,
    progress_locator_result: Optional[Dict[str, Any]] = None,
    progress_update_result: Optional[Dict[str, Any]] = None,
    progress_locator_decision_effect: bool = False,
    phase_evidence: Optional[Dict[str, Any]] = None,
    spatial_memory_result: Optional[Dict[str, Any]] = None,
    landmark_pool_result: Optional[Dict[str, Any]] = None,
    failure_status: Optional[Dict[str, Any]] = None,
    terminal_gate_result: Optional[Dict[str, Any]] = None,
    stop_decision_result: Optional[Dict[str, Any]] = None,
    stop_evidence_items: Optional[Iterable[Dict[str, Any]]] = None,
    candidate_ids: Optional[Iterable[Any]] = None,
    selected_candidate: Any = None,
    stop_requested: bool = False,
    stop_reason: Any = "",
    step_limit: Optional[int] = None,
    action_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one complete route-state snapshot from existing harness outputs.

    All fields are derived from instruction text, agent pose/geometry, perception,
    external evidence modules, and observable action outcomes.  Goal-distance values
    are intentionally not accepted as arguments, which makes accidental promotion of
    the simulator's metric into route state harder.
    """
    chain = _as_dict(anchor_chain_result)
    locator = _as_dict(progress_locator_result)
    progress_update = _as_dict(progress_update_result)
    phase = _as_dict(phase_evidence)
    spatial = _as_dict(spatial_memory_result)
    pool = _as_dict(landmark_pool_result)
    failure = _as_dict(failure_status)
    terminal = _as_dict(terminal_gate_result)
    coordinator = _as_dict(stop_decision_result)
    coordinator_evidence = list(stop_evidence_items or [])
    action = _as_dict(action_result)

    locator_degenerate = bool(locator.get("degenerate"))
    locator_available = bool(locator) and not locator_degenerate
    progress_source = (
        "constraint_queue"
        if progress_locator_decision_effect and locator_available
        else "acn_l1_abstain"
    )
    progress_j = locator.get("j") if locator_available else None
    progress_n = locator.get("n_anchors") if locator_available else None

    current_phase = phase.get("phase") or "unknown"
    phase_confidence = phase.get("confidence")
    terminal_allow = terminal.get("allow_stop")
    if terminal_allow is None:
        terminal_allow = True

    route_state = {
        "schema_version": ROUTE_STATE_SCHEMA_VERSION,
        "episode_id": str(episode_id),
        "step_id": int(step_id),
        "stage": str(stage),
        "plan": {
            "instruction": str(instruction or ""),
            "actions": str(actions or ""),
            "landmarks": str(landmarks or ""),
            "anchors": _copy(list(anchors or [])),
            "anchor_chain": _copy(chain),
        },
        "progress": {
            "current_index": progress_j,
            "total": progress_n,
            "completed_indices": _copy(locator.get("ever_satisfied") or []),
            "complete": bool(locator.get("complete")) if locator_available else False,
            "route_progress_complete": progress_update.get(
                "route_progress_complete",
                bool(locator.get("complete")) if locator_available else False,
            ),
            "terminal_target_confirmed": progress_update.get(
                "terminal_target_confirmed"
            ),
            "goal_complete": progress_update.get("goal_complete", False),
            "source": progress_source,
            "locator": _copy(locator),
            "phase": current_phase,
            "phase_confidence": phase_confidence,
        },
        "spatial_graph": {
            "node_id": spatial.get("node_id"),
            "node_visit_count": spatial.get("node_visit_count"),
            "node_count": spatial.get("node_count"),
            "floor": spatial.get("floor"),
            "floor_count": spatial.get("floor_count"),
            "offered_directions": _ids(spatial.get("offered_directions")),
            "taken_directions": _ids(spatial.get("taken_directions")),
            "unexplored_branch_count": spatial.get("unexplored_branch_count"),
            "candidate_ids": _ids(candidate_ids),
        },
        "landmark_memory": {
            "stage_id": pool.get("stage_id"),
            "pool_size": pool.get("pool_size"),
            "visible_count": pool.get("visible_count"),
            "arrived_count": pool.get("arrived_count"),
            "entries": _copy(pool.get("entries") or []),
        },
        "loop_status": {
            "loop_alert": bool(spatial.get("loop_alert")),
            "visit_count": spatial.get("visit_count"),
            "revisit_score": spatial.get("revisit_score"),
            "alert_applied": bool(spatial.get("applied")),
        },
        "budget": {
            "step": int(step_id),
            "step_limit": step_limit,
            "remaining": (
                max(0, int(step_limit) - int(step_id))
                if step_limit is not None
                else None
            ),
            "at_limit": bool(step_limit is not None and step_id >= step_limit),
        },
        "failure_status": _without_oracle_fields(failure),
        "stop_evidence": {
            "requested": bool(stop_requested),
            "committed": bool(stop_requested and action.get("stop_committed", True)),
            "reason": str(stop_reason or ""),
            "terminal_gate": _without_oracle_fields(terminal),
            "terminal_gate_allow_stop": bool(terminal_allow),
            "terminal_gate_action_effect": bool(
                terminal.get("decision_effect_enabled")
            ),
            "coordinator_decision": _without_oracle_fields(coordinator),
            "coordinator_evidence": _without_oracle_fields(
                coordinator_evidence
            ),
            "authoritative_source": (
                "stop_coordinator" if coordinator else "none"
            ),
        },
        "action": {
            "selected_candidate": (
                str(selected_candidate) if selected_candidate is not None else None
            ),
            "stop_requested": bool(stop_requested),
            "stop_reason": str(stop_reason or ""),
            "result": _without_oracle_fields({
                key: action.get(key)
                for key in ("done", "collision", "steps_taken", "stop_committed")
                if key in action
            }),
        },
        "provenance": {
            "plan": _provenance("instruction_and_anchor_chain"),
            "progress": _provenance(
                progress_source,
                confidence=phase_confidence,
                abstained=bool(locator.get("abstained_this_step"))
                if locator_available
                else True,
                decision_effect=progress_locator_decision_effect,
            ),
            "spatial_graph": _provenance(
                "agent_pose_candidate_geometry",
                decision_effect=bool(spatial.get("applied")),
            ),
            "landmark_memory": _provenance(
                "rgb_ram_and_waypoint_geometry",
                abstained=not bool(pool),
                decision_effect=False,
            ),
            "failure_status": _provenance(
                "observable_selector_or_topology_event",
                confidence=failure.get("confidence"),
                abstained=not bool(failure),
                decision_effect=False,
            ),
            "stop_evidence": _provenance(
                "stop_coordinator" if coordinator else "no_stop_decision",
                abstained=not bool(coordinator),
                decision_effect=bool(coordinator),
            ),
            "action": _provenance("selector_and_action_emitter"),
            "oracle_fields_excluded": list(ORACLE_FIELDS_EXCLUDED),
        },
    }
    return _without_oracle_fields(route_state)

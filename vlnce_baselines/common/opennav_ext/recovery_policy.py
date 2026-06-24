from typing import Any, Dict, List, Optional, Set


RECOVERABLE_FAILURES = {
    "empty_fallback_bad",
    "progress_drift",
    "stop_false_positive",
    "candidate_missing",
}


def _candidate_ids_from_ranked(ranked_candidates: Any) -> List[str]:
    if not isinstance(ranked_candidates, list):
        return []
    candidate_ids: List[str] = []
    for item in ranked_candidates:
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id")
        if candidate_id is None:
            continue
        candidate_ids.append(str(candidate_id))
    return candidate_ids


def _has_trusted_recovery_rank(fallback_results: Dict[str, Any]) -> bool:
    return bool(
        isinstance(fallback_results, dict)
        and fallback_results.get("recovery_rank_trusted")
        and fallback_results.get("fallback_strategy") == "visual_evidence_ranked"
    )


class RecoveryPolicy:
    """Selects a conservative alternate movement candidate for U3 ablations."""

    def __init__(self, max_recovery_per_episode: int = 2) -> None:
        self.max_recovery_per_episode = max(0, int(max_recovery_per_episode))

    def propose(
        self,
        trigger_type: str,
        failure_signal: Optional[Dict[str, Any]],
        fallback_results: Optional[Dict[str, Any]],
        observe_dict: Optional[Dict[str, Any]],
        current_candidate: Any,
        recovery_budget_remaining: int,
        recent_distance_gains: Optional[List[Any]] = None,
        blocked_candidates: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        failure_signal = failure_signal or {}
        fallback_results = fallback_results or {}
        observe_dict = observe_dict or {}
        blocked_candidates = set(blocked_candidates or set())
        current_candidate_id = (
            None if current_candidate is None else str(current_candidate)
        )
        failure_type = str(failure_signal.get("failure_type") or "unknown")
        ranked_candidate_ids = (
            _candidate_ids_from_ranked(fallback_results.get("ranked_candidates"))
            if _has_trusted_recovery_rank(fallback_results)
            else []
        )
        available_candidate_ids = [str(candidate_id) for candidate_id in observe_dict]
        candidate_order = ranked_candidate_ids or available_candidate_ids
        selected_candidate = None
        selection_reason = "no_valid_alternate_candidate"

        if int(recovery_budget_remaining) <= 0:
            return {
                "applied": False,
                "selected_candidate": current_candidate_id,
                "original_candidate": current_candidate_id,
                "reason": "recovery_budget_exhausted",
                "trigger_type": trigger_type,
                "failure_type": failure_type,
                "budget_before": recovery_budget_remaining,
                "budget_after": recovery_budget_remaining,
                "candidate_order": candidate_order,
            }
        if failure_type not in RECOVERABLE_FAILURES:
            return {
                "applied": False,
                "selected_candidate": current_candidate_id,
                "original_candidate": current_candidate_id,
                "reason": "failure_type_not_recoverable",
                "trigger_type": trigger_type,
                "failure_type": failure_type,
                "budget_before": recovery_budget_remaining,
                "budget_after": recovery_budget_remaining,
                "candidate_order": candidate_order,
            }
        if not ranked_candidate_ids:
            return {
                "applied": False,
                "selected_candidate": current_candidate_id,
                "original_candidate": current_candidate_id,
                "reason": "no_trusted_ranked_recovery_candidates",
                "trigger_type": trigger_type,
                "failure_type": failure_type,
                "budget_before": recovery_budget_remaining,
                "budget_after": recovery_budget_remaining,
                "candidate_order": candidate_order,
                "fallback_strategy": fallback_results.get("fallback_strategy"),
                "recovery_rank_trusted": bool(
                    fallback_results.get("recovery_rank_trusted")
                ),
            }

        current_candidate_valid = (
            current_candidate_id is not None
            and current_candidate_id in observe_dict
            and current_candidate_id not in blocked_candidates
        )
        if current_candidate_valid:
            return {
                "applied": False,
                "selected_candidate": current_candidate_id,
                "original_candidate": current_candidate_id,
                "reason": "current_candidate_not_blocked",
                "trigger_type": trigger_type,
                "failure_type": failure_type,
                "budget_before": recovery_budget_remaining,
                "budget_after": recovery_budget_remaining,
                "candidate_order": candidate_order,
                "blocked_candidates": sorted(blocked_candidates),
                "recent_distance_gains": list(recent_distance_gains or []),
            }

        for candidate_id in candidate_order:
            if candidate_id == current_candidate_id:
                continue
            if candidate_id in blocked_candidates:
                continue
            if candidate_id not in observe_dict:
                continue
            selected_candidate = candidate_id
            selection_reason = "alternate_ranked_candidate"
            break

        applied = selected_candidate is not None
        budget_after = (
            max(0, int(recovery_budget_remaining) - 1)
            if applied
            else int(recovery_budget_remaining)
        )
        return {
            "applied": applied,
            "selected_candidate": selected_candidate or current_candidate_id,
            "original_candidate": current_candidate_id,
            "reason": selection_reason,
            "trigger_type": trigger_type,
            "failure_type": failure_type,
            "budget_before": recovery_budget_remaining,
            "budget_after": budget_after,
            "candidate_order": candidate_order,
            "blocked_candidates": sorted(blocked_candidates),
            "recent_distance_gains": list(recent_distance_gains or []),
        }

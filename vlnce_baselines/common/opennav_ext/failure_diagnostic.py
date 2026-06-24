from typing import Any, Dict, List, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_ranked_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    ranked = payload.get("ranked_candidates") if isinstance(payload, dict) else None
    if isinstance(ranked, list) and ranked:
        first = ranked[0]
        return first if isinstance(first, dict) else {}
    return {}


class FailureDiagnostic:
    """Classifies failure triggers for U0 logging and U3 recovery."""

    def __init__(self, negative_gain_window: int = 2) -> None:
        self.negative_gain_window = max(1, int(negative_gain_window))

    def diagnose(
        self,
        trigger_type: str,
        current_step: int,
        phase_evidence: Optional[Dict[str, Any]] = None,
        recent_distance_gains: Optional[List[Any]] = None,
        fallback_results: Optional[Dict[str, Any]] = None,
        stop_verifier_results: Optional[Dict[str, Any]] = None,
        selector_outputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        phase_evidence = phase_evidence or {}
        fallback_results = fallback_results or {}
        stop_verifier_results = stop_verifier_results or {}
        selector_outputs = selector_outputs or {}
        gains = [
            value
            for value in (_safe_float(item) for item in (recent_distance_gains or []))
            if value is not None
        ]
        recent_window = gains[-self.negative_gain_window :]
        non_positive_count = sum(1 for value in recent_window if value <= 0.0)
        first_ranked = _first_ranked_candidate(fallback_results)
        selected_candidate = fallback_results.get("selected_candidate")
        selected_original_order = None
        ranked_candidates = fallback_results.get("ranked_candidates")
        if isinstance(ranked_candidates, list):
            for ranked in ranked_candidates:
                if (
                    isinstance(ranked, dict)
                    and str(ranked.get("candidate_id")) == str(selected_candidate)
                ):
                    selected_original_order = ranked.get("original_order")
                    break

        failure_type = "unknown"
        confidence = 0.35
        reasons = []
        if trigger_type == "selector_empty":
            failure_type = "empty_fallback_bad"
            confidence = 0.7
            reasons.append("selector returned no predictions")
            if selected_original_order == 0:
                confidence = 0.8
                reasons.append("fallback selected original first candidate")
        elif trigger_type == "stop_rejected":
            failure_type = "stop_false_positive"
            confidence = 0.72
            reasons.append("STOP proposal was rejected")
        elif trigger_type == "negative_gain":
            failure_type = "progress_drift"
            confidence = 0.68
            reasons.append("recent selected actions did not reduce distance")
        elif trigger_type == "loop":
            failure_type = "progress_drift"
            confidence = 0.62
            reasons.append("loop or repeated view signal")
        elif trigger_type == "candidate_missing":
            failure_type = "candidate_missing"
            confidence = 0.65
            reasons.append("no valid candidate available")

        if (
            failure_type == "unknown"
            and phase_evidence.get("phase") == "recover"
            and non_positive_count >= self.negative_gain_window
        ):
            failure_type = "progress_drift"
            confidence = 0.6
            reasons.append("phase evidence indicates recovery")

        return {
            "trigger_type": trigger_type,
            "failure_type": failure_type,
            "confidence": round(confidence, 3),
            "reasons": reasons,
            "current_step": current_step,
            "phase": phase_evidence.get("phase"),
            "current_subgoal": phase_evidence.get("current_subgoal"),
            "evidence": {
                "recent_distance_gains": gains,
                "recent_non_positive_gain_count": non_positive_count,
                "fallback_strategy": fallback_results.get("fallback_strategy"),
                "fallback_selected_candidate": selected_candidate,
                "fallback_selected_original_order": selected_original_order,
                "fallback_first_ranked_candidate": first_ranked.get("candidate_id"),
                "stop_verdict": stop_verifier_results.get("verdict"),
                "stop_reason": stop_verifier_results.get("reason"),
                "selector_predictions": selector_outputs.get("predictions"),
            },
        }

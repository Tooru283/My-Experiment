from typing import Any, Dict, List, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _truncate(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _phase_mode(phase: str) -> str:
    if phase == "search":
        return "route_overview"
    if phase == "approach":
        return "subgoal_candidate"
    if phase == "verify":
        return "stop_verify"
    if phase == "recover":
        return "recovery"
    return "conservative"


def _eligible_for_application(phase: str, mode: str, apply_phases: set) -> bool:
    return phase in apply_phases and mode in {"route_overview", "subgoal_candidate"}


class PhaseAwareEvidenceScaffolder:
    """Builds U1 phase-aware selector context from existing V4 evidence."""

    def __init__(
        self,
        max_context_chars: int = 220,
        apply_phases: Optional[List[str]] = None,
    ) -> None:
        self.max_context_chars = max(64, int(max_context_chars))
        self.apply_phases = {
            str(phase).strip().lower()
            for phase in (apply_phases or ["search", "approach"])
            if str(phase).strip()
        }

    def build(
        self,
        observe_dict: Dict[str, str],
        phase_evidence: Optional[Dict[str, Any]],
        unified_context_results: Optional[Dict[str, Any]] = None,
        latest_failure_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(observe_dict, dict):
            observe_dict = {}
        phase_evidence = _as_dict(phase_evidence)
        unified_context_results = _as_dict(unified_context_results)
        latest_failure_signal = _as_dict(latest_failure_signal)
        phase = str(phase_evidence.get("phase") or "unknown").strip().lower()
        mode = _phase_mode(phase)
        unified_summaries = _as_dict(
            unified_context_results.get("selector_safe_summaries")
            or unified_context_results.get("summaries")
        )
        current_subgoal = _truncate(phase_evidence.get("current_subgoal"), 90)
        phase_reason = _truncate(phase_evidence.get("phase_reason"), 90)
        visual_budget = str(phase_evidence.get("visual_budget") or "low")
        confidence = phase_evidence.get("confidence")
        recent_progress = _as_dict(phase_evidence.get("recent_progress"))
        augmented: Dict[str, str] = {}
        summaries: Dict[str, str] = {}
        suppressed_slots: List[str] = []

        if mode == "route_overview":
            suppressed_slots = ["target_arrival_assertion", "failure_history"]
        elif mode == "subgoal_candidate":
            suppressed_slots = ["failure_history"]
        elif mode == "stop_verify":
            suppressed_slots = ["selector_target_override"]
        elif mode == "recovery":
            suppressed_slots = ["target_arrival_assertion"]
        else:
            suppressed_slots = ["phase_specific_assertions"]

        for candidate_id, text_obs in observe_dict.items():
            cid = str(candidate_id)
            visual_summary = _truncate(
                unified_summaries.get(cid, ""),
                max(48, self.max_context_chars // 2),
            )
            slot_parts = [
                "phase={}".format(phase),
                "mode={}".format(mode),
                "budget={}".format(visual_budget),
            ]
            if confidence is not None:
                slot_parts.append("phase_conf={}".format(confidence))
            if mode in {"subgoal_candidate", "stop_verify"} and current_subgoal:
                slot_parts.append("subgoal={}".format(current_subgoal))
            if mode == "route_overview" and phase_reason:
                slot_parts.append("phase_reason={}".format(phase_reason))
            if mode == "recovery":
                failure_type = latest_failure_signal.get("failure_type")
                non_positive = recent_progress.get("non_positive_gain_count")
                if failure_type:
                    slot_parts.append("failure={}".format(failure_type))
                if non_positive is not None:
                    slot_parts.append("non_positive_gain={}".format(non_positive))
            if visual_summary and phase in self.apply_phases:
                slot_parts.append(visual_summary)
            summary = _truncate(
                " [U1: {}]".format("; ".join(slot_parts)),
                self.max_context_chars,
            )
            summaries[cid] = summary
            augmented[cid] = "{}{}".format(str(text_obs), summary)

        return {
            "phase": phase,
            "context_mode": mode,
            "eligible_for_application": _eligible_for_application(
                phase,
                mode,
                self.apply_phases,
            ),
            "visual_budget": visual_budget,
            "current_subgoal": current_subgoal,
            "phase_confidence": confidence,
            "augmented_observe_dict": augmented,
            "augmented_observation": list(augmented.values()),
            "summaries": summaries,
            "suppressed_slots": suppressed_slots,
            "candidate_count": len(observe_dict),
            "candidate_evidence_count": len(
                unified_context_results.get("candidates_with_evidence") or []
            ),
            "unified_context_applied": unified_context_results.get("applied"),
            "selector_context_source": "v4_selector_safe_summaries",
            "apply_phases": sorted(self.apply_phases),
            "raw_v4_summaries_available": bool(
                unified_context_results.get("raw_summaries")
            ),
        }

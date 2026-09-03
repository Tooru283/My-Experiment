import re
from typing import Any, Dict, List, Optional


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _split_actions(actions: Any) -> List[str]:
    text = str(actions or "")
    chunks = [
        re.sub(r"^\s*(?:[-*]|\d+[\.\)]|action\s*\d+\s*:)\s*", "", chunk, flags=re.I)
        .strip()
        for chunk in re.split(r"[\n;]+", text)
        if chunk.strip()
    ]
    if len(chunks) <= 1 and "," in text:
        chunks = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    return chunks


def _completed_action_count(estimation: Any, actions: List[str]) -> int:
    text = _normalize_text(estimation)
    if not text or text in {"none", "1. none", "no action", "no actions"}:
        return 0
    negative_markers = {
        "none",
        "no action",
        "no actions",
        "not executed",
        "not completed",
        "not done",
        "not yet",
        "has not",
        "have not",
        "incomplete",
        "cannot be considered",
        "cannot be done",
        "n/a",
        "na",
    }
    numbered = []
    numbered_seen = False
    for line in str(estimation or "").splitlines():
        match = re.match(r"^\s*\d+[\.\)]\s*(.+?)\s*$", line)
        if not match:
            continue
        numbered_seen = True
        item = _normalize_text(match.group(1))
        if not item:
            continue
        if item in negative_markers:
            continue
        if any(item.startswith("{} ".format(marker)) for marker in negative_markers):
            continue
        if any(
            marker in item
            for marker in {
                "not executed",
                "not completed",
                "not done",
                "not yet",
                "has not",
                "have not",
                "incomplete",
                "cannot be considered",
                "cannot be done",
            }
        ):
            continue
        numbered.append(item)
    if numbered_seen:
        return min(len(numbered), len(actions))

    count = 0
    for action in actions:
        norm_action = _normalize_text(action)
        if not norm_action:
            continue
        action_index = text.find(norm_action)
        if action_index < 0:
            continue
        window = text[
            max(0, action_index - 80) : action_index + len(norm_action) + 80
        ]
        if any(marker in window for marker in negative_markers):
            continue
        if norm_action in text:
            count += 1
    if count:
        return min(count, len(actions))
    return 0


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class PhaseEvidenceTracker:
    """Builds U0/U1 phase evidence from non-oracle inputs.

    ``recent_distance_gains`` is retained only for trace compatibility and audit
    comparison. It is simulator goal-distance information and must never choose a
    phase or alter selector context.
    """

    def __init__(
        self,
        late_step_threshold: int = 4,
        unknown_confidence_threshold: float = 0.4,
    ) -> None:
        self.late_step_threshold = max(1, int(late_step_threshold))
        self.unknown_confidence_threshold = float(unknown_confidence_threshold)

    def build(
        self,
        instruction: str,
        actions: str,
        landmarks: str,
        estimation: str,
        history: str,
        current_step: int,
        recent_distance_gains: List[Any],
        stop_gate_metadata: Optional[Dict[str, Any]] = None,
        last_failure_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        action_list = _split_actions(actions)
        completed_count = _completed_action_count(estimation, action_list)
        action_count = len(action_list)
        final_action_completed = bool(action_count and completed_count >= action_count)
        # Keep the historical field for offline comparison, but mark it explicitly as
        # audit-only. The phase decision below must not branch on this list.
        gains = [
            value
            for value in (_safe_float(item) for item in (recent_distance_gains or []))
            if value is not None
        ]
        non_positive_count = sum(1 for value in gains if value <= 0.0)
        stop_gate_metadata = stop_gate_metadata or {}
        last_failure_signal = last_failure_signal or {}
        failure_type = last_failure_signal.get("failure_type")
        trigger_type = last_failure_signal.get("trigger_type")

        phase = "unknown"
        confidence = self.unknown_confidence_threshold
        reason = "no reliable phase signal"
        if trigger_type in {"selector_empty", "stop_rejected", "loop"}:
            phase = "recover"
            confidence = 0.78
            reason = "recent failure trigger: {}".format(trigger_type)
        elif final_action_completed or stop_gate_metadata.get("final_action_completed"):
            phase = "verify"
            confidence = 0.75
            reason = "final action appears completed"
        elif current_step <= 2 and completed_count == 0:
            phase = "search"
            confidence = 0.65
            reason = "early step with no completed action"
        elif completed_count > 0:
            phase = "approach"
            confidence = 0.68
            reason = "some instruction actions completed"

        current_subgoal = ""
        if action_list:
            idx = min(completed_count, len(action_list) - 1)
            current_subgoal = action_list[idx]

        if phase == "search":
            visual_budget = "low"
        elif phase in {"approach", "verify"}:
            visual_budget = "medium"
        elif phase == "recover":
            visual_budget = "high"
        else:
            visual_budget = "low"

        return {
            "phase": phase,
            "current_subgoal": current_subgoal,
            "phase_reason": reason,
            "visual_budget": visual_budget,
            "confidence": round(confidence, 3),
            "action_count": action_count,
            "completed_action_count": completed_count,
            "final_action_completed": final_action_completed,
            "recent_progress": {
                "recent_distance_gains": gains,
                "non_positive_gain_count": non_positive_count,
                "source": "simulator_goal_distance_audit_only",
                "decision_effect": False,
                "abstained": True,
            },
            "last_failure": {
                "trigger_type": trigger_type,
                "failure_type": failure_type,
            },
            "context_digest": {
                "instruction_chars": len(str(instruction or "")),
                "actions_chars": len(str(actions or "")),
                "landmarks_chars": len(str(landmarks or "")),
                "history_chars": len(str(history or "")),
                "estimation_chars": len(str(estimation or "")),
                "current_step": current_step,
            },
        }

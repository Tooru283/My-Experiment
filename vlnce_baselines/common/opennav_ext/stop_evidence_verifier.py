import re
from typing import Any, Dict, List, Optional


WEAK_TARGET_TERMS = {
    "area",
    "room",
    "hall",
    "hallway",
    "space",
    "place",
    "spot",
    "location",
    "zone",
    "end",
    "way",
    "there",
}

ROOM_TERMS = {"room", "hall", "hallway", "corridor", "area", "space"}
RELATION_TERMS = {
    "left",
    "right",
    "near",
    "beside",
    "next",
    "between",
    "inside",
    "outside",
    "front",
    "behind",
    "facing",
    "across",
    "before",
    "after",
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _tokenize(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "allow", "visible"}:
            return True
        if normalized in {"false", "no", "0", "reject", "missing"}:
            return False
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_landmarks(landmarks: Any) -> List[str]:
    text = str(landmarks or "")
    chunks = [
        re.sub(r"^\s*(?:[-*]|\d+[\.\)]|landmark\s*\d+\s*:)\s*", "", chunk, flags=re.I)
        .strip()
        for chunk in re.split(r"[\n;]+", text)
        if chunk.strip()
    ]
    if len(chunks) <= 1 and "," in text:
        chunks = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    return chunks


def _last_landmark_terms(landmarks: Any) -> List[str]:
    chunks = _split_landmarks(landmarks)
    if not chunks:
        return []
    return _tokenize(chunks[-1])


def _has_relation_language(*values: Any) -> bool:
    tokens = set()
    for value in values:
        tokens.update(_tokenize(value))
    return bool(tokens & RELATION_TERMS)


class StopEvidenceVerifier:
    """Builds U0/U2 stop evidence from existing stop-gate and visual signals."""

    def __init__(
        self,
        enable_rescue: bool = False,
        enable_relation_check: bool = False,
        enable_weak_target_adjustment: bool = False,
        rescue_confidence_threshold: float = 0.99,
        rescue_min_step: int = 8,
        rescue_max_non_positive_gains: int = 1,
        rescue_require_positive_recent_gain: bool = True,
        rescue_allow_phase_verify: bool = False,
        trajectory_bypass_dist: float = 0.0,
    ) -> None:
        self.enable_rescue = bool(enable_rescue)
        self.enable_relation_check = bool(enable_relation_check)
        self.enable_weak_target_adjustment = bool(enable_weak_target_adjustment)
        self.rescue_confidence_threshold = float(rescue_confidence_threshold)
        self.rescue_min_step = max(0, int(rescue_min_step))
        self.rescue_max_non_positive_gains = max(
            0,
            int(rescue_max_non_positive_gains),
        )
        self.rescue_require_positive_recent_gain = bool(
            rescue_require_positive_recent_gain
        )
        self.rescue_allow_phase_verify = bool(rescue_allow_phase_verify)
        # When > 0, downgrade trajectory_support from "no" to "unknown" if the
        # latest goal distance is within this threshold. Prevents the agent from
        # being blocked by trajectory_incomplete when already inside the success
        # radius.
        self.trajectory_bypass_dist = max(0.0, float(trajectory_bypass_dist))

    def verify(
        self,
        source: str,
        visual_verifier_results: Optional[Dict[str, Any]],
        stop_gate_metadata: Optional[Dict[str, Any]],
        phase_evidence: Optional[Dict[str, Any]],
        instruction: str,
        actions: str,
        landmarks: str,
        estimation: str,
        current_step: int,
        latest_goal_dist: Optional[float] = None,
    ) -> Dict[str, Any]:
        visual = visual_verifier_results or {}
        stop_gate = stop_gate_metadata or {}
        phase = phase_evidence or {}
        selected_verdict = visual.get("selected_candidate_verdict") or {}
        allow_blockers = list(visual.get("allow_blockers") or [])
        allow_warnings = list(visual.get("allow_warnings") or [])
        contradictions = list(visual.get("contradictions") or [])
        selected_missing_instruction_terms = list(
            selected_verdict.get("missing_instruction_terms") or []
        )
        selected_confidence = _safe_float(selected_verdict.get("confidence"), 0.0)
        landmark_terms = _last_landmark_terms(landmarks)
        weak_target = bool(landmark_terms) and all(
            term in WEAK_TARGET_TERMS for term in landmark_terms
        )

        target_type = "specific_target"
        if weak_target:
            target_type = "weak_target"
        elif set(landmark_terms) & ROOM_TERMS:
            target_type = "room_or_area"
        elif not landmark_terms:
            target_type = "unknown"

        intrinsic_visible = _safe_bool(visual.get("final_target_visible"))
        if intrinsic_visible is None:
            intrinsic_visible = _safe_bool(selected_verdict.get("final_target_visible"))
        arrival_evidence = _safe_bool(visual.get("arrival_evidence"))
        if arrival_evidence is None:
            arrival_evidence = _safe_bool(selected_verdict.get("arrival_evidence"))
        intrinsic_support = "yes" if intrinsic_visible else "no"
        if intrinsic_visible is None:
            intrinsic_support = "unknown"

        relation_required = _has_relation_language(instruction, actions, landmarks)
        relation_support = "unknown"
        if self.enable_relation_check:
            if contradictions:
                relation_support = "no"
            elif arrival_evidence:
                relation_support = "yes"
            elif relation_required:
                relation_support = "unknown"
            else:
                relation_support = "not_required"

        final_action_completed = bool(
            stop_gate.get("final_action_completed")
            or stop_gate.get("all_actions_completed")
            or phase.get("final_action_completed")
            or phase.get("phase") == "verify"
        )
        if final_action_completed:
            trajectory_support = "yes"
        elif (
            stop_gate.get("all_actions_completed") is False
            or stop_gate.get("final_action_completed") is False
        ):
            trajectory_support = "no"
        else:
            trajectory_support = "unknown"

        # Distance bypass: when the agent is already within trajectory_bypass_dist
        # of the goal, downgrade trajectory_support from "no" to "unknown" so that
        # a clear visual signal can still allow the stop. This prevents the agent
        # from being trapped by trajectory_incomplete when it is physically inside
        # the success radius.
        trajectory_dist_bypassed = False
        if (
            trajectory_support == "no"
            and self.trajectory_bypass_dist > 0
            and latest_goal_dist is not None
            and latest_goal_dist < self.trajectory_bypass_dist
        ):
            trajectory_support = "unknown"
            trajectory_dist_bypassed = True

        weak_target_adjustment = "none"
        weak_target_blocked = False
        if weak_target and self.enable_weak_target_adjustment:
            if trajectory_support != "yes":
                weak_target_adjustment = "require_trajectory_support"
                weak_target_blocked = True
            elif intrinsic_support == "yes":
                weak_target_adjustment = "trajectory_intrinsic_supported"
            else:
                weak_target_adjustment = "require_intrinsic_support"
                weak_target_blocked = True

        hard_blockers = [
            blocker
            for blocker in allow_blockers
            if blocker != "selector_stop_without_completion_support"
        ]
        reject_reasons: List[str] = []
        if intrinsic_support == "no":
            reject_reasons.append("final_target_not_visible")
        if relation_support == "no":
            reject_reasons.append("relation_contradiction")
        if trajectory_support == "no":
            reject_reasons.append("trajectory_incomplete")
        if weak_target_blocked:
            reject_reasons.append(weak_target_adjustment)
        reject_reasons.extend(hard_blockers)

        visual_allow = visual.get("verdict") == "allow" or (
            selected_verdict.get("verdict") == "allow"
            and intrinsic_support == "yes"
            and bool(arrival_evidence)
        )
        allow_stop = bool(
            visual_allow
            and intrinsic_support == "yes"
            and relation_support != "no"
            and not reject_reasons
        )
        rescue_blockers = self._rescue_blockers(
            source=source,
            visual=visual,
            selected_verdict=selected_verdict,
            selected_confidence=selected_confidence,
            intrinsic_visible=intrinsic_visible,
            arrival_evidence=arrival_evidence,
            allow_warnings=allow_warnings,
            contradictions=contradictions,
            hard_blockers=hard_blockers,
            selected_missing_instruction_terms=selected_missing_instruction_terms,
            phase=phase,
            current_step=current_step,
        )
        allow_rescue = bool(
            self.enable_rescue
            and source == "selector_stop_gate"
            and allow_stop
            and trajectory_support == "yes"
            and not rescue_blockers
        )

        # Abstain: evidence is insufficient to decide either way.
        # Triggered when intrinsic visibility is unknown and no hard blockers
        # exist to clearly reject. In future S3 this will trigger re-observation;
        # for now it surfaces as a logged third state alongside allow/reject.
        abstain = bool(
            not allow_stop
            and not reject_reasons
            and not hard_blockers
            and intrinsic_support == "unknown"
        )
        abstain_reason = (
            "intrinsic_visibility_unknown_no_hard_blockers" if abstain else ""
        )

        confidence = 0.0
        if intrinsic_support == "yes":
            confidence += 0.35
        if bool(arrival_evidence):
            confidence += 0.25
        if trajectory_support == "yes":
            confidence += 0.2
        if relation_support in {"yes", "not_required", "unknown"}:
            confidence += 0.1
        if not hard_blockers and not contradictions:
            confidence += 0.1

        return {
            "source": source,
            "target_type": target_type,
            "weak_target": weak_target,
            "landmark_terms": landmark_terms,
            "intrinsic_support": intrinsic_support,
            "relation_required": relation_required,
            "relation_support": relation_support,
            "trajectory_support": trajectory_support,
            "trajectory_dist_bypassed": trajectory_dist_bypassed,
            "weak_target_adjustment": weak_target_adjustment,
            "visual_verdict": visual.get("verdict"),
            "selected_candidate_verdict": selected_verdict.get("verdict"),
            "final_target_visible": intrinsic_visible,
            "arrival_evidence": arrival_evidence,
            "allow_blockers": allow_blockers,
            "allow_warnings": allow_warnings,
            "contradictions": contradictions,
            "hard_blockers": hard_blockers,
            "allow_stop": allow_stop,
            "allow_rescue": allow_rescue,
            "abstain": abstain,
            "abstain_reason": abstain_reason,
            "reject_reasons": reject_reasons,
            "confidence": round(min(confidence, 1.0), 3),
            "config": {
                "enable_rescue": self.enable_rescue,
                "enable_relation_check": self.enable_relation_check,
                "enable_weak_target_adjustment": self.enable_weak_target_adjustment,
                "rescue_confidence_threshold": self.rescue_confidence_threshold,
                "rescue_min_step": self.rescue_min_step,
                "rescue_max_non_positive_gains": (
                    self.rescue_max_non_positive_gains
                ),
                "rescue_require_positive_recent_gain": (
                    self.rescue_require_positive_recent_gain
                ),
                "rescue_allow_phase_verify": self.rescue_allow_phase_verify,
            },
            "rescue_blockers": rescue_blockers,
            "rescue_policy": {
                "selected_confidence": selected_confidence,
                "selected_missing_instruction_terms": (
                    selected_missing_instruction_terms
                ),
                "recent_progress": phase.get("recent_progress"),
            },
            "context_digest": {
                "instruction_chars": len(str(instruction or "")),
                "actions_chars": len(str(actions or "")),
                "landmarks_chars": len(str(landmarks or "")),
                "estimation_chars": len(str(estimation or "")),
                "current_step": current_step,
                "phase": phase.get("phase"),
            },
        }

    def _rescue_blockers(
        self,
        source: str,
        visual: Dict[str, Any],
        selected_verdict: Dict[str, Any],
        selected_confidence: float,
        intrinsic_visible: Optional[bool],
        arrival_evidence: Optional[bool],
        allow_warnings: List[str],
        contradictions: List[str],
        hard_blockers: List[str],
        selected_missing_instruction_terms: List[str],
        phase: Dict[str, Any],
        current_step: int,
    ) -> List[str]:
        blockers: List[str] = []
        phase_name = str(phase.get("phase") or "")
        hard_allow_warnings = [
            warning
            for warning in allow_warnings
            if not str(warning).startswith("uncorroborated_final_target:")
        ]
        base_current_view_supported = (
            self.enable_rescue
            and source == "selector_stop_gate"
            and visual.get("stop_evidence_mode") == "current_pano"
            and str(visual.get("stop_relevant_candidate_id")) == "__current_view__"
            and visual.get("verdict") == "allow"
            and selected_verdict.get("verdict") == "allow"
            and intrinsic_visible is True
            and arrival_evidence is True
            and selected_confidence >= self.rescue_confidence_threshold
            and not hard_allow_warnings
            and not contradictions
            and not hard_blockers
            and not selected_missing_instruction_terms
            and phase_name == "verify"
        )
        if not self.enable_rescue:
            blockers.append("rescue_disabled")
        if source != "selector_stop_gate":
            blockers.append("not_selector_stop_gate")
        if phase_name != "verify":
            blockers.append("phase_not_verify")
        if visual.get("stop_evidence_mode") != "current_pano":
            blockers.append("not_current_pano")
        if str(visual.get("stop_relevant_candidate_id")) != "__current_view__":
            blockers.append("not_current_view_candidate")
        if visual.get("verdict") != "allow":
            blockers.append("visual_verdict_not_allow")
        if selected_verdict.get("verdict") != "allow":
            blockers.append("selected_candidate_not_allow")
        if intrinsic_visible is not True:
            blockers.append("final_target_not_visible")
        if arrival_evidence is not True:
            blockers.append("arrival_evidence_missing")
        if selected_confidence < self.rescue_confidence_threshold:
            blockers.append(
                "rescue_confidence_below_threshold:{:.2f}<{}".format(
                    selected_confidence,
                    self.rescue_confidence_threshold,
                )
            )
        if hard_allow_warnings:
            blockers.append("allow_warnings_present")
        if contradictions:
            blockers.append("visual_contradictions_present")
        if hard_blockers:
            blockers.append("hard_blockers_present")
        if selected_missing_instruction_terms:
            blockers.append("selected_missing_instruction_terms_present")

        phase_verify_override = (
            self.rescue_allow_phase_verify
            and phase_name == "verify"
            and selected_confidence >= self.rescue_confidence_threshold
        )
        if (
            current_step < self.rescue_min_step
            and not phase_verify_override
            and not base_current_view_supported
        ):
            blockers.append(
                "before_rescue_min_step:{}<{}".format(
                    current_step,
                    self.rescue_min_step,
                )
            )

        recent_progress = phase.get("recent_progress") or {}
        recent_gains = [
            _safe_float(gain)
            for gain in (recent_progress.get("recent_distance_gains") or [])
        ]
        if recent_progress.get("non_positive_gain_count") is None:
            non_positive_gain_count = sum(1 for gain in recent_gains if gain <= 0)
        else:
            non_positive_gain_count = int(
                recent_progress.get("non_positive_gain_count") or 0
            )
        if (
            non_positive_gain_count > self.rescue_max_non_positive_gains
            and not base_current_view_supported
        ):
            blockers.append(
                "too_many_non_positive_recent_gains:{}>{}".format(
                    non_positive_gain_count,
                    self.rescue_max_non_positive_gains,
                )
            )
        if self.rescue_require_positive_recent_gain and not base_current_view_supported:
            latest_gain = recent_gains[-1] if recent_gains else None
            if latest_gain is None or latest_gain <= 0:
                blockers.append("latest_recent_gain_not_positive")
        return blockers

from typing import Any, Dict, Iterable, List, Optional, Sequence

from vlnce_baselines.common.opennav_ext.landmark_matching import (
    final_landmark_terms,
    matched_terms as match_landmark_terms,
    missing_terms as missing_landmark_terms,
    normalize_text,
    split_landmark_terms,
    term_present,
    unique_normalized,
)
from vlnce_baselines.common.opennav_ext.visual_evidence_schema import (
    candidate_evidence_from_result,
)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _norm(text: Any) -> str:
    return normalize_text(text)


def _unique(values: Iterable[str]) -> List[str]:
    return unique_normalized(values)


def _landmark_terms(landmarks: str) -> List[str]:
    return split_landmark_terms(landmarks)


def _final_landmark_terms(landmarks: str) -> List[str]:
    return final_landmark_terms(landmarks)


def _term_present(term: str, values: Iterable[str]) -> bool:
    return term_present(term, values)


def _candidate_evidence(visual_evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    return candidate_evidence_from_result(visual_evidence)


_DEFAULT_GENERIC_FINAL_TERMS = (
    "area",
    "archway",
    "door",
    "doorway",
    "entry way",
    "entryway",
    "floor",
    "hall",
    "hallway",
    "room",
    "stair",
    "stairs",
    "staircase",
)

_STOP_FINAL_TARGET_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "beside",
    "by",
    "in",
    "inside",
    "near",
    "next",
    "of",
    "on",
    "the",
    "to",
}

_LOCATION_MODIFIER_TOKENS = {
    "bath",
    "bathroom",
    "bed",
    "bedroom",
    "dining",
    "foyer",
    "hall",
    "hallway",
    "kitchen",
    "living",
    "lounge",
    "office",
    "outdoor",
    "patio",
    "room",
}


class VisualTargetVerifier:
    """Verifier for STOP proposals using V1 visual evidence."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        require_arrival_evidence: bool = True,
        reject_on_missing_final_landmarks: bool = True,
        min_steps_before_allow: int = 0,
        require_full_coverage_for_allow: bool = False,
        block_generic_final_terms_for_allow: bool = False,
        require_completion_for_selector_stop: bool = True,
        require_current_view_text_corroboration_for_allow: bool = False,
        generic_final_terms: Optional[Iterable[str]] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.require_arrival_evidence = require_arrival_evidence
        self.reject_on_missing_final_landmarks = reject_on_missing_final_landmarks
        self.min_steps_before_allow = max(0, _as_int(min_steps_before_allow, 0))
        self.require_full_coverage_for_allow = require_full_coverage_for_allow
        self.block_generic_final_terms_for_allow = (
            block_generic_final_terms_for_allow
        )
        self.require_completion_for_selector_stop = (
            require_completion_for_selector_stop
        )
        self.require_current_view_text_corroboration_for_allow = (
            require_current_view_text_corroboration_for_allow
        )
        self.generic_final_terms = {
            _norm(term)
            for term in (generic_final_terms or _DEFAULT_GENERIC_FINAL_TERMS)
            if _norm(term)
        }

    def verify(
        self,
        source: str,
        instruction: str,
        actions: str,
        landmarks: str,
        estimation: str,
        history: str,
        observation: Any,
        visual_evidence: Dict[str, Any],
        stop_proposal: bool,
        stop_reason: str = "",
        selected_candidate: Optional[str] = None,
        step_id: Optional[int] = None,
        stop_evidence_mode: str = "selected_candidate",
    ) -> Dict[str, Any]:
        candidates = _candidate_evidence(visual_evidence)
        all_landmark_terms = _landmark_terms(landmarks)
        final_terms = _final_landmark_terms(landmarks)
        sample_info = self._sample_info(visual_evidence)
        parse_error = (
            visual_evidence.get("parse_error")
            if isinstance(visual_evidence, dict)
            else None
        )
        schema_error = (
            visual_evidence.get("schema_error")
            if isinstance(visual_evidence, dict)
            else None
        )
        schema_warnings = (
            _as_list(visual_evidence.get("schema_warnings"))
            if isinstance(visual_evidence, dict)
            else []
        )
        current_view_observation = (
            str(visual_evidence.get("current_view_observation", ""))
            if isinstance(visual_evidence, dict)
            else ""
        )

        candidate_verdicts = [
            self._candidate_verdict(candidate, final_terms)
            for candidate in candidates
        ]
        best_candidate = self._best_candidate(candidate_verdicts)
        supporting_candidate = self._supporting_candidate(candidate_verdicts)
        selected_candidate_verdict = self._selected_candidate_verdict(
            candidate_verdicts,
            selected_candidate,
        )
        candidate_alignment = self._candidate_alignment(
            selected_candidate,
            supporting_candidate,
            selected_candidate_verdict,
        )
        matched_terms = self._aggregate_terms(
            candidate_verdicts, "matched_final_landmarks"
        )
        missing_terms = self._aggregate_missing_terms(
            final_terms, candidate_verdicts
        )

        final_target_visible = any(
            candidate["final_target_visible"] for candidate in candidate_verdicts
        )
        arrival_evidence = any(
            candidate["arrival_evidence"] for candidate in candidate_verdicts
        )
        confidence = _as_float(
            best_candidate.get("confidence") if best_candidate else 0.0
        )

        contradictions = self._contradictions(
            estimation,
            stop_proposal,
            final_target_visible,
            arrival_evidence,
            missing_terms,
            parse_error,
            schema_error,
        )
        allow_policy = self._allow_policy_findings(
            source,
            stop_proposal,
            step_id,
            sample_info,
            final_terms,
            selected_candidate_verdict,
            candidate_verdicts,
            selected_candidate,
            stop_evidence_mode,
            current_view_observation,
            estimation,
        )
        allow_blockers = allow_policy["blockers"]
        allow_warnings = allow_policy["warnings"]
        verdict = self._verdict(
            stop_proposal,
            parse_error,
            candidate_verdicts,
            sample_info["sampled_all"],
            selected_candidate_verdict,
            allow_blockers,
        )
        stop_selected_candidate_required = (
            bool(stop_proposal) and selected_candidate is not None
        )

        return {
            "source": source,
            "step_id": step_id,
            "stop_proposal": bool(stop_proposal),
            "stop_reason": stop_reason,
            "verdict": verdict,
            "final_target_visible": final_target_visible,
            "arrival_evidence": arrival_evidence,
            "matched_final_landmarks": matched_terms,
            "missing_final_landmarks": missing_terms,
            "all_landmark_terms": all_landmark_terms,
            "required_landmark_terms": final_terms,
            "best_candidate": best_candidate,
            "supporting_candidate_id": supporting_candidate,
            "future_supporting_candidate_id": supporting_candidate
            if (
                selected_candidate is None
                or supporting_candidate is None
                or str(supporting_candidate) != str(selected_candidate)
            )
            else None,
            "stop_relevant_candidate_id": (
                str(selected_candidate) if selected_candidate is not None else None
            ),
            "stop_evidence_mode": stop_evidence_mode,
            "stop_selected_candidate_required": stop_selected_candidate_required,
            "selected_candidate_verdict": selected_candidate_verdict,
            "candidate_alignment": candidate_alignment,
            "candidate_verdicts": candidate_verdicts,
            "selected_candidate": selected_candidate,
            "visual_evidence_candidate_count": len(candidates),
            "visual_evidence_parse_error": parse_error,
            "visual_evidence_schema_error": schema_error,
            "visual_evidence_schema_warnings": schema_warnings,
            "current_view_corroboration": self._current_view_corroboration_report(
                stop_evidence_mode,
                final_terms,
                current_view_observation,
            ),
            "visual_evidence_sample": sample_info,
            "contradictions": contradictions,
            "allow_blockers": allow_blockers,
            "allow_warnings": allow_warnings,
            "confidence": confidence,
            "reason": self._reason(
                verdict,
                stop_proposal,
                final_target_visible,
                arrival_evidence,
                missing_terms,
                confidence,
                sample_info["sampled_all"],
                allow_blockers,
            ),
            "context_digest": {
                "instruction_chars": len(str(instruction or "")),
                "actions_chars": len(str(actions or "")),
                "history_chars": len(str(history or "")),
                "observation_count": len(observation)
                if isinstance(observation, list)
                else None,
            },
        }

    def _sample_info(self, visual_evidence: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(visual_evidence, dict):
            return {
                "requested_candidate_ids": [],
                "total_candidate_ids": [],
                "sampled_all": False,
                "sample_limited": True,
            }
        requested = [
            str(candidate_id)
            for candidate_id in visual_evidence.get("requested_candidate_ids", [])
        ]
        total = [
            str(candidate_id)
            for candidate_id in visual_evidence.get("total_candidate_ids", [])
        ]
        sampled_all = bool(
            visual_evidence.get(
                "sampled_all",
                bool(total) and set(requested) == set(total),
            )
        )
        return {
            "requested_candidate_ids": requested,
            "total_candidate_ids": total,
            "sampled_all": sampled_all,
            "sample_limited": not sampled_all,
            "sampled_candidate_count": len(requested),
            "total_candidate_count": len(total),
        }

    def _candidate_verdict(
        self,
        candidate: Dict[str, Any],
        final_terms: List[str],
    ) -> Dict[str, Any]:
        visible_terms = _as_list(candidate.get("visible_landmarks"))
        matched_instruction_terms = _as_list(candidate.get("matched_instruction_terms"))
        evidence_terms = visible_terms + matched_instruction_terms
        matched_final_terms = match_landmark_terms(final_terms, evidence_terms)
        missing_final_terms = missing_landmark_terms(
            final_terms, matched_final_terms
        )
        final_target_visible = _as_bool(candidate.get("final_target_visible"))
        arrival_evidence = _as_bool(candidate.get("arrival_evidence"))
        confidence = _as_float(candidate.get("confidence"))
        arrival_ok = arrival_evidence or not self.require_arrival_evidence
        if (
            final_target_visible
            and arrival_ok
            and not missing_final_terms
            and confidence >= self.confidence_threshold
        ):
            verdict = "allow"
        elif (
            self.reject_on_missing_final_landmarks
            and missing_final_terms
        ) or not final_target_visible or not arrival_ok:
            verdict = "reject"
        else:
            verdict = "uncertain"

        return {
            "candidate_id": str(candidate.get("candidate_id", "")),
            "verdict": verdict,
            "final_target_visible": final_target_visible,
            "arrival_evidence": arrival_evidence,
            "target_direction_id": (
                None if candidate.get("target_direction_id") is None
                else str(candidate.get("target_direction_id"))
            ),
            "confidence": confidence,
            "visible_landmarks": visible_terms,
            "matched_instruction_terms": matched_instruction_terms,
            "missing_instruction_terms": _as_list(
                candidate.get("missing_instruction_terms")
            ),
            "matched_final_landmarks": matched_final_terms,
            "missing_final_landmarks": missing_final_terms,
        }

    def _best_candidate(
        self, candidate_verdicts: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not candidate_verdicts:
            return None
        best = max(
            candidate_verdicts,
            key=lambda candidate: (
                candidate.get("verdict") == "allow",
                _as_bool(candidate.get("final_target_visible")),
                _as_bool(candidate.get("arrival_evidence")),
                _as_float(candidate.get("confidence")),
            ),
        )
        return {
            "candidate_id": str(best.get("candidate_id", "")),
            "verdict": str(best.get("verdict", "")),
            "final_target_visible": _as_bool(best.get("final_target_visible")),
            "arrival_evidence": _as_bool(best.get("arrival_evidence")),
            "target_direction_id": best.get("target_direction_id"),
            "confidence": _as_float(best.get("confidence")),
            "visible_landmarks": _as_list(best.get("visible_landmarks")),
            "matched_instruction_terms": _as_list(
                best.get("matched_instruction_terms")
            ),
            "missing_instruction_terms": _as_list(
                best.get("missing_instruction_terms")
            ),
            "matched_final_landmarks": _as_list(
                best.get("matched_final_landmarks")
            ),
            "missing_final_landmarks": _as_list(
                best.get("missing_final_landmarks")
            ),
        }

    def _supporting_candidate(
        self, candidate_verdicts: List[Dict[str, Any]]
    ) -> Optional[str]:
        for candidate in candidate_verdicts:
            if candidate.get("verdict") == "allow":
                return str(candidate.get("candidate_id", ""))
        return None

    def _selected_candidate_verdict(
        self,
        candidate_verdicts: List[Dict[str, Any]],
        selected_candidate: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if selected_candidate is None:
            return None
        selected = str(selected_candidate)
        for candidate in candidate_verdicts:
            if str(candidate.get("candidate_id", "")) == selected:
                return candidate
        return None

    def _candidate_alignment(
        self,
        selected_candidate: Optional[str],
        supporting_candidate: Optional[str],
        selected_candidate_verdict: Optional[Dict[str, Any]],
    ) -> str:
        if selected_candidate is None:
            return "unknown"
        if selected_candidate_verdict is None:
            return "not_sampled"
        if (
            supporting_candidate is not None
            and str(selected_candidate) == str(supporting_candidate)
        ):
            return "aligned"
        return "mismatch"

    def _aggregate_terms(
        self, candidate_verdicts: List[Dict[str, Any]], key: str
    ) -> List[str]:
        values = []
        for candidate in candidate_verdicts:
            values.extend(_as_list(candidate.get(key)))
        return _unique(values)

    def _aggregate_missing_terms(
        self,
        final_terms: List[str],
        candidate_verdicts: List[Dict[str, Any]],
    ) -> List[str]:
        matched_terms = self._aggregate_terms(
            candidate_verdicts,
            "matched_final_landmarks",
        )
        return missing_landmark_terms(final_terms, matched_terms)

    def _contradictions(
        self,
        estimation: str,
        stop_proposal: bool,
        final_target_visible: bool,
        arrival_evidence: bool,
        missing_terms: List[str],
        parse_error: Any,
        schema_error: Any,
    ) -> List[str]:
        contradictions = []
        if parse_error:
            contradictions.append("visual_evidence_parse_error")
        if schema_error:
            contradictions.append("visual_evidence_schema_error")
        if not stop_proposal:
            return contradictions
        if not final_target_visible:
            contradictions.append("stop_proposed_without_final_target_visible")
        if not arrival_evidence:
            contradictions.append("stop_proposed_without_arrival_evidence")
        if missing_terms:
            contradictions.append("stop_proposed_with_missing_final_landmarks")
        estimation_text = _norm(estimation)
        negative_markers = (
            "not executed",
            "not done",
            "has not",
            "have not",
            "not completed",
            "no evidence",
        )
        if any(marker in estimation_text for marker in negative_markers):
            contradictions.append("stop_proposed_while_estimation_is_negative")
        return contradictions

    def _has_allow_candidate(
        self,
        selected_candidate_verdict: Optional[Dict[str, Any]],
        candidate_verdicts: List[Dict[str, Any]],
    ) -> bool:
        if selected_candidate_verdict is not None:
            return selected_candidate_verdict.get("verdict") == "allow"
        return any(candidate["verdict"] == "allow" for candidate in candidate_verdicts)

    def _all_final_terms_generic(self, final_terms: List[str]) -> bool:
        if not final_terms:
            return False
        for term in final_terms:
            if _norm(term) not in self.generic_final_terms:
                return False
        return True

    def _meaningful_tokens(self, term: str) -> List[str]:
        tokens = []
        for token in _norm(term).split():
            if token in _STOP_FINAL_TARGET_STOPWORDS:
                continue
            if len(token) <= 1:
                continue
            tokens.append(token)
        return unique_normalized(tokens)

    def _requires_current_view_corroboration(self, term: str) -> bool:
        if _norm(term) in self.generic_final_terms:
            return True
        tokens = self._meaningful_tokens(term)
        if len(tokens) < 2:
            return False
        return any(token in _LOCATION_MODIFIER_TOKENS for token in tokens)

    def _current_view_term_corroborated(
        self,
        term: str,
        current_view_observation: str,
    ) -> bool:
        if not self._requires_current_view_corroboration(term):
            return True
        tokens = self._meaningful_tokens(term)
        if not tokens:
            return True
        if term_present(term, [current_view_observation]):
            return True
        return all(term_present(token, [current_view_observation]) for token in tokens)

    def _uncorroborated_current_view_terms(
        self,
        final_terms: Sequence[str],
        current_view_observation: str,
    ) -> List[str]:
        if not str(current_view_observation or "").strip():
            return [
                term
                for term in unique_normalized(final_terms)
                if self._requires_current_view_corroboration(term)
            ]
        return [
            term
            for term in unique_normalized(final_terms)
            if not self._current_view_term_corroborated(
                term,
                current_view_observation,
            )
        ]

    def _current_view_corroboration_report(
        self,
        stop_evidence_mode: str,
        final_terms: Sequence[str],
        current_view_observation: str,
    ) -> Dict[str, Any]:
        if stop_evidence_mode != "current_pano":
            return {
                "required": False,
                "uncorroborated_terms": [],
            }
        uncorroborated_terms = self._uncorroborated_current_view_terms(
            final_terms,
            current_view_observation,
        )
        return {
            "required": True,
            "uncorroborated_terms": uncorroborated_terms,
            "observation_chars": len(str(current_view_observation or "")),
        }

    def _allow_policy_findings(
        self,
        source: str,
        stop_proposal: bool,
        step_id: Optional[int],
        sample_info: Dict[str, Any],
        final_terms: List[str],
        selected_candidate_verdict: Optional[Dict[str, Any]],
        candidate_verdicts: List[Dict[str, Any]],
        selected_candidate: Optional[str],
        stop_evidence_mode: str,
        current_view_observation: str,
        estimation: str,
    ) -> Dict[str, List[str]]:
        if not stop_proposal or not self._has_allow_candidate(
            selected_candidate_verdict, candidate_verdicts
        ):
            return {"blockers": [], "warnings": []}

        blockers = []
        warnings = []
        if (
            source == "selector_stop_gate"
            and self.require_completion_for_selector_stop
            and not self._estimation_supports_stop(estimation)
        ):
            blockers.append("selector_stop_without_completion_support")
        if selected_candidate is not None and selected_candidate_verdict is None:
            blockers.append("selected_candidate_not_sampled:{}".format(selected_candidate))
        if (
            stop_evidence_mode == "current_pano"
            and selected_candidate_verdict is not None
            and selected_candidate_verdict.get("verdict") == "allow"
        ):
            for term in self._uncorroborated_current_view_terms(
                final_terms,
                current_view_observation,
            ):
                finding = "uncorroborated_final_target:{}".format(_norm(term))
                if self.require_current_view_text_corroboration_for_allow:
                    blockers.append(finding)
                else:
                    warnings.append(finding)
        step_number = _as_int(step_id, 0)
        if self.min_steps_before_allow and step_number < self.min_steps_before_allow:
            blockers.append(
                "before_min_steps:{}<{}".format(
                    step_number, self.min_steps_before_allow
                )
            )
        if self.require_full_coverage_for_allow and not sample_info.get(
            "sampled_all", False
        ):
            blockers.append("sample_limited")
        if (
            source == "completion_gate"
            and self.block_generic_final_terms_for_allow
            and self._all_final_terms_generic(final_terms)
        ):
            blockers.append(
                "generic_final_terms:{}".format(
                    ",".join(_norm(term) for term in final_terms)
                )
            )
        return {
            "blockers": blockers,
            "warnings": warnings,
        }

    def _estimation_supports_stop(self, estimation: str) -> bool:
        normalized = _norm(estimation)
        if not normalized or normalized == "none":
            return False
        negative_markers = (
            "not executed",
            "not completed",
            "not done",
            "has not",
            "have not",
            "not yet",
            "incomplete",
        )
        if any(marker in normalized for marker in negative_markers):
            return False
        return True

    def _verdict(
        self,
        stop_proposal: bool,
        parse_error: Any,
        candidate_verdicts: List[Dict[str, Any]],
        sampled_all: bool,
        selected_candidate_verdict: Optional[Dict[str, Any]] = None,
        allow_blockers: Optional[List[str]] = None,
    ) -> str:
        if not stop_proposal:
            return "not_applicable"
        if parse_error or not candidate_verdicts:
            return "uncertain"
        allow_blockers = allow_blockers or []
        if selected_candidate_verdict is not None:
            selected_verdict = selected_candidate_verdict.get("verdict")
            if selected_verdict == "allow":
                if allow_blockers:
                    return "uncertain"
                return "allow"
            if selected_verdict == "reject":
                return "reject"
            return "uncertain"
        if any(candidate["verdict"] == "allow" for candidate in candidate_verdicts):
            return "uncertain"
        if not sampled_all:
            return "uncertain"
        if any(candidate["verdict"] == "reject" for candidate in candidate_verdicts):
            return "reject"
        return "uncertain"

    def _reason(
        self,
        verdict: str,
        stop_proposal: bool,
        final_target_visible: bool,
        arrival_evidence: bool,
        missing_terms: List[str],
        confidence: float,
        sampled_all: bool,
        allow_blockers: Optional[List[str]] = None,
    ) -> str:
        if verdict == "not_applicable":
            return "No STOP proposal was present for this verifier call."
        if verdict == "allow":
            return (
                "Visual evidence supports final target visibility and arrival "
                "with confidence {:.2f}."
            ).format(confidence)
        if allow_blockers:
            return (
                "Visual evidence has an allow candidate, but conservative STOP "
                "allow rules blocked it: {}."
            ).format(", ".join(allow_blockers))
        if verdict == "reject":
            return (
                "Visual evidence does not support STOP: "
                "final_target_visible={}, arrival_evidence={}, missing={}"
            ).format(final_target_visible, arrival_evidence, missing_terms)
        if not sampled_all:
            return (
                "Only a sampled subset of candidates was checked; no single "
                "sampled candidate supports the STOP proposal."
            )
        if not stop_proposal:
            return "No STOP proposal."
        return "Visual evidence is insufficient to verify the STOP proposal."

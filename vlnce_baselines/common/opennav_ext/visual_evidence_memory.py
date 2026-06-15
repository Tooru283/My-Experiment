import re
from typing import Any, Dict, List, Optional

from vlnce_baselines.common.opennav_ext.visual_evidence_schema import (
    candidate_evidence_from_result,
)


def _position_to_list(position: Any) -> Optional[List[float]]:
    if position is None:
        return None
    try:
        return [float(position[0]), float(position[1]), float(position[2])]
    except Exception:
        return None


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _unique_sorted(values: List[str]) -> List[str]:
    return sorted({value for value in values if value})


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _term_present(term: str, values: List[str]) -> bool:
    term_norm = _norm(term)
    if not term_norm:
        return False
    for value in values:
        value_norm = _norm(value)
        if term_norm in value_norm or value_norm in term_norm:
            return True
    return False


def _landmark_terms(landmarks: str) -> List[str]:
    chunks = re.split(r"[,;\n]+", str(landmarks or ""))
    terms = []
    for chunk in chunks:
        text = re.sub(r"^\s*[-*\d.)]+\s*", "", chunk).strip()
        if text:
            terms.append(text)
    return _unique_sorted(terms)


class VisualEvidenceMemory:
    """Logging-only episode memory for Qwen-VL visual evidence."""

    def __init__(
        self,
        max_history: int = 64,
        max_notes_chars: int = 240,
    ) -> None:
        self.max_history = max_history
        self.max_notes_chars = max_notes_chars
        self.history: List[Dict[str, Any]] = []

    def reset_episode(self) -> None:
        self.history = []

    def update(
        self,
        step_id: int,
        position: Any,
        heading: Any,
        visual_evidence: Dict[str, Any],
        landmarks: str = "",
    ) -> Dict[str, Any]:
        parse_error = (
            visual_evidence.get("parse_error")
            if isinstance(visual_evidence, dict)
            else None
        )
        candidates = candidate_evidence_from_result(visual_evidence)

        records_added = []
        current_position = _position_to_list(position)
        heading_value = None
        try:
            heading_value = float(heading) if heading is not None else None
        except (TypeError, ValueError):
            heading_value = None

        for candidate in candidates:
            record = self._record_from_candidate(
                step_id,
                current_position,
                heading_value,
                candidate,
                _landmark_terms(landmarks),
            )
            self.history.append(record)
            records_added.append(record)

        if self.max_history > 0 and len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        return self._summary(
            step_id,
            parse_error,
            records_added,
        )

    def _record_from_candidate(
        self,
        step_id: int,
        position: Optional[List[float]],
        heading: Optional[float],
        candidate: Dict[str, Any],
        final_terms: List[str],
    ) -> Dict[str, Any]:
        visible_landmarks = _as_list(candidate.get("visible_landmarks"))
        matched_instruction_terms = _as_list(
            candidate.get("matched_instruction_terms")
        )
        evidence_terms = visible_landmarks + matched_instruction_terms
        matched_final_landmarks = [
            term for term in final_terms if _term_present(term, evidence_terms)
        ]
        missing_final_landmarks = [
            term
            for term in final_terms
            if not _term_present(term, matched_final_landmarks)
        ]
        raw_final_target_visible = _as_bool(candidate.get("final_target_visible"))
        raw_arrival_evidence = _as_bool(candidate.get("arrival_evidence"))
        verified_final_target_visible = (
            raw_final_target_visible and not missing_final_landmarks
        )
        verified_arrival_evidence = (
            verified_final_target_visible and raw_arrival_evidence
        )
        return {
            "step_id": step_id,
            "candidate_id": str(candidate.get("candidate_id", "")),
            "position": position,
            "heading": heading,
            "visible_landmarks": visible_landmarks,
            "matched_instruction_terms": matched_instruction_terms,
            "missing_instruction_terms": _as_list(
                candidate.get("missing_instruction_terms")
            ),
            "matched_final_landmarks": matched_final_landmarks,
            "missing_final_landmarks": missing_final_landmarks,
            "final_target_visible": raw_final_target_visible,
            "arrival_evidence": raw_arrival_evidence,
            "verified_final_target_visible": verified_final_target_visible,
            "verified_arrival_evidence": verified_arrival_evidence,
            "confidence": _as_float(candidate.get("confidence")),
            "spatial_notes": str(candidate.get("spatial_notes", ""))[
                : self.max_notes_chars
            ],
            "source": "qwen_vl_visual_evidence",
            "image_ref": "direction:{}".format(candidate.get("candidate_id", "")),
        }

    def _summary(
        self,
        step_id: int,
        parse_error: Any,
        records_added: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raw_target_records = [
            record for record in self.history if record["final_target_visible"]
        ]
        verified_target_records = [
            record
            for record in self.history
            if record["verified_final_target_visible"]
        ]
        raw_arrival_records = [
            record for record in self.history if record["arrival_evidence"]
        ]
        verified_arrival_records = [
            record for record in self.history if record["verified_arrival_evidence"]
        ]
        best_target_record = self._best_record(verified_target_records)
        best_visual_record = self._best_record(self.history)
        latest_target_step = (
            verified_target_records[-1]["step_id"]
            if verified_target_records
            else None
        )
        first_target_step = (
            verified_target_records[0]["step_id"]
            if verified_target_records
            else None
        )

        seen_landmarks: List[str] = []
        matched_terms: List[str] = []
        recent_missing_terms: List[str] = []
        for record in self.history:
            seen_landmarks.extend(record["visible_landmarks"])
            matched_terms.extend(record["matched_instruction_terms"])
        for record in self.history[-5:]:
            recent_missing_terms.extend(record["missing_instruction_terms"])

        return {
            "step_id": step_id,
            "records_added": len(records_added),
            "history_size": len(self.history),
            "parse_error": parse_error,
            "target_seen_count": len(verified_target_records),
            "arrival_evidence_count": len(verified_arrival_records),
            "raw_target_seen_count": len(raw_target_records),
            "raw_arrival_evidence_count": len(raw_arrival_records),
            "verified_target_seen_count": len(verified_target_records),
            "verified_arrival_evidence_count": len(verified_arrival_records),
            "first_target_visible_step": first_target_step,
            "latest_target_visible_step": latest_target_step,
            "best_target_evidence": best_target_record,
            "best_visual_evidence": best_visual_record,
            "seen_landmarks": _unique_sorted(seen_landmarks),
            "matched_instruction_terms": _unique_sorted(matched_terms),
            "recent_missing_instruction_terms": _unique_sorted(
                recent_missing_terms
            ),
            "candidates_this_step": [
                self._compact_record(record) for record in records_added
            ],
        }

    def _best_record(
        self, records: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not records:
            return None
        best = max(
            records,
            key=lambda record: (
                bool(record["final_target_visible"]),
                bool(record["arrival_evidence"]),
                float(record["confidence"]),
            ),
        )
        return self._compact_record(best)

    def _compact_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "step_id": record["step_id"],
            "candidate_id": record["candidate_id"],
            "final_target_visible": record["final_target_visible"],
            "arrival_evidence": record["arrival_evidence"],
            "verified_final_target_visible": record[
                "verified_final_target_visible"
            ],
            "verified_arrival_evidence": record["verified_arrival_evidence"],
            "confidence": record["confidence"],
            "visible_landmarks": record["visible_landmarks"],
            "matched_instruction_terms": record["matched_instruction_terms"],
            "missing_instruction_terms": record["missing_instruction_terms"],
            "matched_final_landmarks": record["matched_final_landmarks"],
            "missing_final_landmarks": record["missing_final_landmarks"],
            "image_ref": record["image_ref"],
        }

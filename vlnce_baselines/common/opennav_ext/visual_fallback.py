import re
from typing import Any, Dict, List, Optional, Tuple

from vlnce_baselines.common.opennav_ext.visual_evidence_schema import (
    candidate_evidence_from_result,
)


_GENERIC_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "continue",
    "down",
    "enter",
    "exit",
    "for",
    "from",
    "go",
    "head",
    "in",
    "into",
    "is",
    "left",
    "move",
    "of",
    "on",
    "out",
    "past",
    "right",
    "room",
    "stop",
    "the",
    "then",
    "through",
    "to",
    "toward",
    "towards",
    "turn",
    "up",
    "walk",
    "with",
}

_WEAK_FINAL_TARGET_TERMS = {
    "area",
    "archway",
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
}


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return bool(value)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def _extract_terms(*texts: Any) -> List[str]:
    seen = set()
    terms: List[str] = []
    for text in texts:
        normalized = _normalize_text(text)
        if not normalized:
            continue
        chunks = re.findall(r"[a-z0-9][a-z0-9\- ]{1,40}[a-z0-9]", normalized)
        for chunk in chunks:
            for term in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", chunk):
                if term in _GENERIC_TERMS or term in seen:
                    continue
                seen.add(term)
                terms.append(term)
    return terms[:32]


def _final_instruction_terms(landmarks: str) -> List[str]:
    chunks = [
        chunk.strip()
        for chunk in re.split(r"[,;\n]+", str(landmarks or ""))
        if chunk.strip()
    ]
    if not chunks:
        return []
    final_chunk = re.sub(
        r"^\s*(?:[-*]|\d+[\.\)]|landmark\s*\d+\s*:)\s*",
        "",
        chunks[-1],
        flags=re.I,
    ).strip()
    alternatives = [
        part.strip()
        for part in re.split(r"\s*(?:/|\||\bor\b)\s*", final_chunk)
        if part.strip()
    ]
    return alternatives or ([final_chunk] if final_chunk else [])


def _all_final_terms_weak(landmarks: str) -> bool:
    final_terms = _final_instruction_terms(landmarks)
    if not final_terms:
        return False
    return all(_normalize_text(term) in _WEAK_FINAL_TARGET_TERMS for term in final_terms)


def _local_term_hits(
    terms: List[str],
    observe_text: str,
    evidence_terms: List[str],
) -> List[str]:
    haystack = _normalize_text("{} {}".format(observe_text, " ".join(evidence_terms)))
    hits = []
    for term in terms:
        if term and re.search(r"(?<![a-z0-9]){}(?![a-z0-9])".format(re.escape(term)), haystack):
            hits.append(term)
    return hits[:8]


def _candidate_evidence_by_id(
    visual_evidence_results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(candidate.get("candidate_id", "")): candidate
        for candidate in candidate_evidence_from_result(visual_evidence_results)
    }


class VisualEvidenceFallbackRanker:
    """Rank movement candidates when selector returns no usable prediction."""

    def rank(
        self,
        observe_dict: Dict[str, str],
        visual_evidence_results: Dict[str, Any],
        instruction: str = "",
        actions: str = "",
        landmarks: str = "",
        source_stage: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(observe_dict, dict):
            observe_dict = {}
        evidence_by_id = _candidate_evidence_by_id(visual_evidence_results)
        required_terms = _extract_terms(instruction, actions, landmarks)
        stop_rejected_stage = str(source_stage or "") == "stop_rejected"
        weak_final_target = _all_final_terms_weak(landmarks)
        ranked: List[Dict[str, Any]] = []
        for order, candidate_id in enumerate(str(key) for key in observe_dict.keys()):
            evidence = evidence_by_id.get(candidate_id, {})
            final_target_visible = _as_bool(
                evidence.get("final_target_visible", False)
            )
            arrival_evidence = _as_bool(evidence.get("arrival_evidence", False))
            matched_terms = _as_list(evidence.get("matched_instruction_terms"))
            visible_terms = _as_list(evidence.get("visible_landmarks"))
            missing_terms = _as_list(evidence.get("missing_instruction_terms"))
            confidence = _as_float(evidence.get("confidence", 0.0))
            observe_text = observe_dict.get(candidate_id, "")
            local_hits = _local_term_hits(
                required_terms,
                observe_text,
                matched_terms + visible_terms,
            )
            missing_penalty = len(
                [
                    term
                    for term in missing_terms
                    if term.lower() not in {hit.lower() for hit in local_hits}
                ]
            )
            weak_target_penalty = 1 if weak_final_target and final_target_visible else 0
            if stop_rejected_stage:
                score: Tuple[Any, ...] = (
                    len(matched_terms),
                    len(local_hits),
                    -weak_target_penalty,
                    confidence,
                    len(visible_terms),
                    -missing_penalty,
                    1 if final_target_visible else 0,
                    1 if arrival_evidence else 0,
                    -order,
                )
                score_policy = "stop_rejected_continue_movement"
            else:
                score = (
                    1 if final_target_visible else 0,
                    1 if arrival_evidence else 0,
                    len(matched_terms),
                    len(local_hits),
                    confidence,
                    len(visible_terms),
                    -weak_target_penalty,
                    -missing_penalty,
                    -order,
                )
                score_policy = "visual_goal_tracking"
            ranked.append(
                {
                    "candidate_id": candidate_id,
                    "score": list(score),
                    "score_policy": score_policy,
                    "final_target_visible": final_target_visible,
                    "arrival_evidence": arrival_evidence,
                    "matched_instruction_terms": matched_terms,
                    "visible_landmarks": visible_terms,
                    "missing_instruction_terms": missing_terms,
                    "local_instruction_hits": local_hits,
                    "missing_penalty": missing_penalty,
                    "weak_target_penalty": weak_target_penalty,
                    "confidence": confidence,
                    "has_visual_evidence": candidate_id in evidence_by_id,
                    "original_order": order,
                }
            )
        ranked.sort(key=lambda item: tuple(item["score"]), reverse=True)
        selected: Optional[str] = ranked[0]["candidate_id"] if ranked else None
        return {
            "fallback_strategy": "visual_evidence_ranked",
            "selected_candidate": selected,
            "ranked_candidates": ranked,
            "candidate_count": len(observe_dict),
            "candidate_evidence_count": len(evidence_by_id),
            "required_terms": required_terms,
            "source_stage": source_stage,
            "reason": reason,
            "weak_final_target": weak_final_target,
            "parse_error": (
                visual_evidence_results.get("parse_error")
                if isinstance(visual_evidence_results, dict)
                else None
            ),
            "schema_error": (
                visual_evidence_results.get("schema_error")
                if isinstance(visual_evidence_results, dict)
                else None
            ),
            "schema_warnings": (
                visual_evidence_results.get("schema_warnings")
                if isinstance(visual_evidence_results, dict)
                else []
            ),
        }

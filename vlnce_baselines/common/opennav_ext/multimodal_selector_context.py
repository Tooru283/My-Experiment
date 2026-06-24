from typing import Any, Dict, List, Optional

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


def _candidate_visual_summary(
    candidate_id: str,
    evidence: Dict[str, Any],
    max_chars: int,
) -> str:
    matched = _as_list(evidence.get("matched_instruction_terms"))
    missing = _as_list(evidence.get("missing_instruction_terms"))
    target = _as_bool(evidence.get("final_target_visible", False))
    arrival = _as_bool(evidence.get("arrival_evidence", False))
    conf = _as_float(evidence.get("confidence", 0.0))
    notes = str(evidence.get("spatial_notes") or "").strip()
    parts = ["cid={}".format(candidate_id)]
    if matched:
        parts.append("match={}".format(",".join(str(m) for m in matched[:3])))
    if missing:
        parts.append("miss={}".format(",".join(str(m) for m in missing[:3])))
    parts.append("target={}".format("1" if target else "0"))
    parts.append("arrival={}".format("1" if arrival else "0"))
    parts.append("conf={:.2f}".format(conf))
    if notes:
        parts.append("note={}".format(notes[:48]))
    return ("[V: {}]".format("; ".join(parts)))[:max_chars]


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.6:
        return "medium"
    if confidence > 0:
        return "low"
    return "none"


def _candidate_selector_safe_summary(
    candidate_id: str,
    evidence: Dict[str, Any],
    max_chars: int,
    suppress_target_arrival: bool,
    min_confidence_for_target_hint: float,
) -> str:
    matched = _as_list(evidence.get("matched_instruction_terms"))
    missing = _as_list(evidence.get("missing_instruction_terms"))
    target = _as_bool(evidence.get("final_target_visible", False))
    conf = _as_float(evidence.get("confidence", 0.0))
    parts = ["cid={}".format(candidate_id)]
    if matched:
        parts.append("match={}".format(",".join(str(m) for m in matched[:3])))
    if missing:
        parts.append("miss={}".format(",".join(str(m) for m in missing[:3])))
    parts.append("conf={}".format(_confidence_band(conf)))
    if (
        not suppress_target_arrival
        and target
        and conf >= float(min_confidence_for_target_hint)
    ):
        parts.append("target_hint=visible")
    return ("[V4: {}]".format("; ".join(parts)))[:max_chars]


def _suppressed_selector_fields(
    evidence: Dict[str, Any],
    suppress_target_arrival: bool,
) -> List[str]:
    suppressed: List[str] = []
    if suppress_target_arrival:
        suppressed.extend(["final_target_visible", "arrival_evidence"])
    if str(evidence.get("spatial_notes") or "").strip():
        suppressed.append("spatial_notes")
    return suppressed


def _memory_suffix(mem: Dict[str, Any]) -> str:
    seen = _as_list(mem.get("seen_landmarks"))
    missing_recent = _as_list(mem.get("recent_missing_instruction_terms"))
    target_count = mem.get("target_seen_count", 0)
    parts = []
    if seen:
        parts.append("seen_ever={}".format(",".join(str(s) for s in seen[:5])))
    if missing_recent:
        parts.append(
            "missing_recent={}".format(
                ",".join(str(m) for m in missing_recent[:3])
            )
        )
    parts.append("target_seen={}".format(target_count))
    return " [VISUAL_MEMORY: {}]".format("; ".join(parts))


class MultimodalSelectorContext:
    """Augment observe_dict with compressed V1/V3 visual evidence summaries.

    When LOG_ONLY=true: computes augmented dict but does not replace observe_dict.
    When LOG_ONLY=false: caller replaces observe_dict with augmented_observe_dict.
    """

    def __init__(
        self,
        max_summary_chars: int = 200,
        include_memory_suffix: bool = False,
        decision_mode: str = "phase_gated_u1",
        suppress_target_arrival_for_selector: bool = True,
        min_confidence_for_target_hint: float = 0.9,
    ) -> None:
        self.max_summary_chars = max(0, int(max_summary_chars))
        self.include_memory_suffix = bool(include_memory_suffix)
        self.decision_mode = str(decision_mode or "phase_gated_u1")
        self.suppress_target_arrival_for_selector = bool(
            suppress_target_arrival_for_selector
        )
        self.min_confidence_for_target_hint = float(
            min_confidence_for_target_hint
        )

    def build(
        self,
        observe_dict: Dict[str, str],
        visual_evidence_results: Dict[str, Any],
        visual_evidence_memory_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(observe_dict, dict):
            observe_dict = {}
        if not isinstance(visual_evidence_results, dict):
            visual_evidence_results = {}
        per_candidate: Dict[str, Dict[str, Any]] = {
            str(c.get("candidate_id", "")): c
            for c in candidate_evidence_from_result(visual_evidence_results)
        }

        mem_sfx = (
            _memory_suffix(visual_evidence_memory_results)
            if self.include_memory_suffix
            and isinstance(visual_evidence_memory_results, dict)
            and visual_evidence_memory_results
            else ""
        )

        augmented: Dict[str, str] = {}
        raw_summaries: Dict[str, str] = {}
        selector_safe_summaries: Dict[str, str] = {}
        suppressed_fields_by_candidate: Dict[str, List[str]] = {}
        for cid, text_obs in observe_dict.items():
            cid = str(cid)
            if cid in per_candidate:
                raw_summary = _candidate_visual_summary(
                    cid,
                    per_candidate[cid],
                    self.max_summary_chars,
                )
                selector_safe_summary = _candidate_selector_safe_summary(
                    cid,
                    per_candidate[cid],
                    self.max_summary_chars,
                    self.suppress_target_arrival_for_selector,
                    self.min_confidence_for_target_hint,
                )
                suppressed_fields_by_candidate[cid] = _suppressed_selector_fields(
                    per_candidate[cid],
                    self.suppress_target_arrival_for_selector,
                )
            else:
                raw_summary = ""
                selector_safe_summary = ""
                suppressed_fields_by_candidate[cid] = []
            raw_summaries[cid] = raw_summary
            selector_safe_summaries[cid] = selector_safe_summary
            augmented[cid] = "{}{}{}".format(
                str(text_obs),
                " " + selector_safe_summary if selector_safe_summary else "",
                mem_sfx,
            )

        return {
            "augmented_observe_dict": augmented,
            "augmented_observation": list(augmented.values()),
            "summaries": selector_safe_summaries,
            "raw_summaries": raw_summaries,
            "selector_safe_summaries": selector_safe_summaries,
            "suppressed_fields_by_candidate": suppressed_fields_by_candidate,
            "selector_summary_policy": {
                "decision_mode": self.decision_mode,
                "suppress_target_arrival_for_selector": (
                    self.suppress_target_arrival_for_selector
                ),
                "min_confidence_for_target_hint": (
                    self.min_confidence_for_target_hint
                ),
            },
            "mem_suffix": mem_sfx,
            "include_memory_suffix": self.include_memory_suffix,
            "candidates_with_evidence": list(per_candidate.keys()),
            "candidate_count": len(observe_dict),
            "candidate_evidence_count": len(per_candidate),
            "parse_error": visual_evidence_results.get("parse_error"),
        }

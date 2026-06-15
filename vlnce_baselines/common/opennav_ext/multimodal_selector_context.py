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
    visible = _as_list(evidence.get("visible_landmarks"))
    matched = _as_list(evidence.get("matched_instruction_terms"))
    missing = _as_list(evidence.get("missing_instruction_terms"))
    target = _as_bool(evidence.get("final_target_visible", False))
    arrival = _as_bool(evidence.get("arrival_evidence", False))
    conf = _as_float(evidence.get("confidence", 0.0))
    notes = str(evidence.get("spatial_notes") or "").strip()
    parts = ["candidate_id={}".format(candidate_id)]
    if matched:
        parts.append("matched={}".format(",".join(str(m) for m in matched[:4])))
    if visible:
        parts.append("visible={}".format(",".join(str(v) for v in visible[:5])))
    if missing:
        parts.append("missing={}".format(",".join(str(m) for m in missing[:4])))
    parts.append("target={}".format("Y" if target else "N"))
    if arrival:
        parts.append("arrival=Y")
    parts.append("conf={:.2f}".format(conf))
    if notes:
        parts.append("notes={}".format(notes[:80]))
    return ("[VISUAL_EVIDENCE: {}]".format("; ".join(parts)))[:max_chars]


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

    def __init__(self, max_summary_chars: int = 200) -> None:
        self.max_summary_chars = max(0, int(max_summary_chars))

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
            if isinstance(visual_evidence_memory_results, dict)
            and visual_evidence_memory_results
            else ""
        )

        augmented: Dict[str, str] = {}
        summaries: Dict[str, str] = {}
        for cid, text_obs in observe_dict.items():
            cid = str(cid)
            summary = (
                _candidate_visual_summary(
                    cid,
                    per_candidate[cid],
                    self.max_summary_chars,
                )
                if cid in per_candidate
                else ""
            )
            summaries[cid] = summary
            augmented[cid] = "{}{}{}".format(
                str(text_obs),
                " " + summary if summary else "",
                mem_sfx,
            )

        return {
            "augmented_observe_dict": augmented,
            "augmented_observation": list(augmented.values()),
            "summaries": summaries,
            "mem_suffix": mem_sfx,
            "candidates_with_evidence": list(per_candidate.keys()),
            "candidate_count": len(observe_dict),
            "candidate_evidence_count": len(per_candidate),
            "parse_error": visual_evidence_results.get("parse_error"),
        }

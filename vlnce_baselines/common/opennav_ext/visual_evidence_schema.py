from typing import Any, Dict, List


def candidate_list_from_parsed(parsed: Any) -> List[Dict[str, Any]]:
    """Return candidate evidence for both supported V1 JSON shapes."""
    if isinstance(parsed, dict):
        candidates = parsed.get("candidates") or []
    elif isinstance(parsed, list):
        candidates = parsed
    else:
        candidates = []
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def normalize_visual_evidence_parsed(parsed: Any) -> Dict[str, Any]:
    """Normalize parsed V1 evidence to the canonical object schema."""
    if isinstance(parsed, dict):
        normalized = dict(parsed)
        normalized["candidates"] = candidate_list_from_parsed(parsed)
        return normalized
    if isinstance(parsed, list):
        return {"candidates": candidate_list_from_parsed(parsed)}
    return {}


def candidate_evidence_from_result(
    visual_evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(visual_evidence, dict):
        return []
    return candidate_list_from_parsed(visual_evidence.get("parsed"))

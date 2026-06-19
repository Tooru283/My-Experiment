from typing import Any, Dict, List, Tuple


def _candidate_container(parsed: Any) -> Tuple[Any, List[str], str]:
    warnings: List[str] = []
    root_type = type(parsed).__name__ if parsed is not None else None
    if isinstance(parsed, dict):
        if "candidates" in parsed:
            return parsed.get("candidates"), warnings, root_type
        for alias in ("candidate", "results"):
            if alias in parsed:
                warnings.append("candidate_key_alias:{}".format(alias))
                return parsed.get(alias), warnings, root_type
        warnings.append("missing_candidates_key")
        return [], warnings, root_type
    elif isinstance(parsed, list):
        warnings.append("root_list_normalized")
        return parsed, warnings, root_type
    else:
        warnings.append("unsupported_root_type:{}".format(root_type))
        return [], warnings, root_type


def candidate_list_from_parsed(parsed: Any) -> List[Dict[str, Any]]:
    """Return valid candidate evidence for supported V1 JSON shapes."""
    candidates, _, _ = _candidate_container(parsed)
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


def visual_evidence_schema_diagnostics(parsed: Any) -> Dict[str, Any]:
    candidates, warnings, root_type = _candidate_container(parsed)
    raw_candidate_count = 0
    valid_candidate_count = 0
    invalid_candidate_count = 0
    schema_error = None

    if isinstance(candidates, list):
        raw_candidate_count = len(candidates)
        valid_candidate_count = len(
            [candidate for candidate in candidates if isinstance(candidate, dict)]
        )
        invalid_candidate_count = raw_candidate_count - valid_candidate_count
        if invalid_candidate_count:
            warnings.append("invalid_candidate_items:{}".format(invalid_candidate_count))
    elif candidates is not None:
        schema_error = "candidates_not_list:{}".format(type(candidates).__name__)
        warnings.append(schema_error)

    if parsed is not None and not schema_error and valid_candidate_count == 0:
        warnings.append("no_valid_candidates")

    return {
        "schema_error": schema_error,
        "schema_warnings": warnings,
        "normalized_from_root_type": root_type,
        "raw_candidate_count": raw_candidate_count,
        "valid_candidate_count": valid_candidate_count,
        "invalid_candidate_count": invalid_candidate_count,
    }


def candidate_evidence_from_result(
    visual_evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(visual_evidence, dict):
        return []
    return candidate_list_from_parsed(visual_evidence.get("parsed"))

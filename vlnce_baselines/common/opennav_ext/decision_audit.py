from typing import Any, Dict


def build_decision_audit(
    unit_enabled: str,
    original_action: Any,
    proposed_action: Any,
    final_action: Any,
    override_reason: str = "",
    expected_failure_addressed: str = "none",
    source_stage: str = "",
    extra: Dict[str, Any] = None,
) -> Dict[str, Any]:
    original = None if original_action is None else str(original_action)
    proposed = None if proposed_action is None else str(proposed_action)
    final = None if final_action is None else str(final_action)
    payload = {
        "unit_enabled": str(unit_enabled or "none"),
        "original_action": original,
        "proposed_action": proposed,
        "final_action": final,
        "override": final != original if original is not None else False,
        "override_reason": str(override_reason or ""),
        "expected_failure_addressed": str(expected_failure_addressed or "none"),
        "source_stage": str(source_stage or ""),
    }
    if extra:
        payload["extra"] = extra
    return payload

from typing import Any, Optional


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    try:
        return getattr(node, key)
    except Exception:
        return default


def _harness_node(config: Any) -> Optional[Any]:
    return _cfg_get(config, "OPENNAV_HARNESS", None)


def harness_logging_enabled(config: Any) -> bool:
    harness = _harness_node(config)
    if harness is None:
        return False
    return bool(_cfg_get(harness, "ENABLED", False)) and bool(
        _cfg_get(harness, "ENABLE_HARNESS_LOGGING", False)
    )


def decision_effect_enabled(config: Any) -> bool:
    harness = _harness_node(config)
    if harness is None:
        return False
    return bool(_cfg_get(harness, "ENABLE_DECISION_EFFECT", False))


def get_trace_dir(config: Any) -> str:
    harness = _harness_node(config)
    return str(_cfg_get(harness, "TRACE_DIR", "logs/harness_traces"))


def fail_open_enabled(config: Any) -> bool:
    harness = _harness_node(config)
    return bool(_cfg_get(harness, "FAIL_OPEN", True))


def module_enabled(config: Any, module_name: str) -> bool:
    harness = _harness_node(config)
    if harness is None or not harness_logging_enabled(config):
        return False
    section = _cfg_get(harness, module_name, None)
    if section is None:
        return False
    return bool(_cfg_get(section, "ENABLED", False))


def module_log_only(config: Any, module_name: str) -> bool:
    harness = _harness_node(config)
    section = _cfg_get(harness, module_name, None)
    if section is None:
        return True
    return bool(_cfg_get(section, "LOG_ONLY", True))


def validate_a1_harness_config(config: Any) -> None:
    visual_target_verifier_decision_effect = module_enabled(
        config, "VISUAL_TARGET_VERIFIER"
    ) and not module_log_only(config, "VISUAL_TARGET_VERIFIER")
    multimodal_selector_decision_effect = module_enabled(
        config, "MULTIMODAL_SELECTOR_CONTEXT"
    ) and not module_log_only(config, "MULTIMODAL_SELECTOR_CONTEXT")
    decision_effect_modules = (
        visual_target_verifier_decision_effect
        or multimodal_selector_decision_effect
    )
    if decision_effect_enabled(config) and not decision_effect_modules:
        raise ValueError(
            "OPENNAV_HARNESS.ENABLE_DECISION_EFFECT=true currently requires "
            "at least one supported decision-effect module to be enabled with "
            "LOG_ONLY=false."
        )
    if decision_effect_modules and not decision_effect_enabled(config):
        raise ValueError(
            "Harness modules with LOG_ONLY=false require "
            "OPENNAV_HARNESS.ENABLE_DECISION_EFFECT=true."
        )
    if not harness_logging_enabled(config):
        return
    harness = _harness_node(config)
    trace_format = str(_cfg_get(harness, "TRACE_FORMAT", "jsonl")).lower()
    if trace_format != "jsonl":
        raise ValueError("A1 harness only supports TRACE_FORMAT=jsonl.")
    for module_name in (
        "GEOMETRY_QUERY",
        "GROUNDER_DIAGNOSTIC",
        "MEMORY_DIAGNOSTIC",
        "CONTEXT_BUILDER",
        "ORACLE_METRICS",
        "VISUAL_EVIDENCE",
        "VISUAL_TARGET_VERIFIER",
        "VISUAL_EVIDENCE_MEMORY",
        "MULTIMODAL_SELECTOR_CONTEXT",
    ):
        if module_enabled(config, module_name) and not module_log_only(
            config, module_name
        ):
            if (
                module_name
                in ("VISUAL_TARGET_VERIFIER", "MULTIMODAL_SELECTOR_CONTEXT")
                and decision_effect_enabled(config)
            ):
                continue
            raise ValueError(
                "A1 harness module {} must keep LOG_ONLY=true.".format(
                    module_name
                )
            )
    if (
        module_enabled(config, "VISUAL_TARGET_VERIFIER")
        or module_enabled(config, "VISUAL_EVIDENCE_MEMORY")
    ) and not module_enabled(config, "VISUAL_EVIDENCE"):
        raise ValueError(
            "VISUAL_TARGET_VERIFIER and VISUAL_EVIDENCE_MEMORY require "
            "OPENNAV_HARNESS.VISUAL_EVIDENCE.ENABLED=true."
        )
    if module_enabled(config, "MULTIMODAL_SELECTOR_CONTEXT") and not module_enabled(
        config, "VISUAL_EVIDENCE"
    ):
        raise ValueError(
            "MULTIMODAL_SELECTOR_CONTEXT requires "
            "OPENNAV_HARNESS.VISUAL_EVIDENCE.ENABLED=true."
        )

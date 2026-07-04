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


def u_series_node(config: Any) -> Optional[Any]:
    harness = _harness_node(config)
    return _cfg_get(harness, "U_SERIES", None)


def u_series_enabled(config: Any) -> bool:
    node = u_series_node(config)
    return bool(harness_logging_enabled(config)) and bool(
        _cfg_get(node, "ENABLED", False)
    )


def u_decision_effect_unit(config: Any) -> str:
    node = u_series_node(config)
    return str(_cfg_get(node, "DECISION_EFFECT_UNIT", "none") or "none")


def u_module_enabled(config: Any, module_name: str) -> bool:
    node = u_series_node(config)
    if node is None or not u_series_enabled(config):
        return False
    section = _cfg_get(node, module_name, None)
    if section is None:
        return False
    return bool(_cfg_get(section, "ENABLED", False))


def u_module_log_only(config: Any, module_name: str) -> bool:
    node = u_series_node(config)
    section = _cfg_get(node, module_name, None)
    if section is None:
        return True
    return bool(_cfg_get(section, "LOG_ONLY", True))


def arrival_gate_enabled(config: Any) -> bool:
    harness = _harness_node(config)
    if harness is None or not harness_logging_enabled(config):
        return False
    section = _cfg_get(harness, "ARRIVAL_GATE", None)
    if section is None:
        return False
    return bool(_cfg_get(section, "ENABLED", False))


def arrival_gate_config(config: Any) -> dict:
    harness = _harness_node(config)
    section = _cfg_get(harness, "ARRIVAL_GATE", None) if harness is not None else None
    if section is None:
        return {}
    return {
        "dist_threshold": float(_cfg_get(section, "DIST_THRESHOLD", 4.0)),
        "allowed_phases": list(_cfg_get(section, "ALLOWED_PHASES", ["approach", "verify", "unknown"]) or []),
        "min_trigger_step": int(_cfg_get(section, "MIN_TRIGGER_STEP", 1)),
    }


def proactive_stop_gate_enabled(config: Any) -> bool:
    harness = _harness_node(config)
    if harness is None or not harness_logging_enabled(config):
        return False
    section = _cfg_get(harness, "PROACTIVE_STOP_GATE", None)
    if section is None:
        return False
    return bool(_cfg_get(section, "ENABLED", False))


def proactive_stop_gate_config(config: Any) -> dict:
    harness = _harness_node(config)
    section = _cfg_get(harness, "PROACTIVE_STOP_GATE", None) if harness is not None else None
    if section is None:
        return {}
    return {
        "dist_threshold": float(_cfg_get(section, "DIST_THRESHOLD", 3.5)),
        "commit_dist_threshold": float(_cfg_get(section, "COMMIT_DIST_THRESHOLD", 0.0)),
    }


def validate_a1_harness_config(config: Any) -> None:
    u_node = u_series_node(config)
    u_config_enabled = bool(_cfg_get(u_node, "ENABLED", False))
    visual_target_verifier_decision_effect = module_enabled(
        config, "VISUAL_TARGET_VERIFIER"
    ) and not module_log_only(config, "VISUAL_TARGET_VERIFIER")
    multimodal_selector_decision_effect = module_enabled(
        config, "MULTIMODAL_SELECTOR_CONTEXT"
    ) and not module_log_only(config, "MULTIMODAL_SELECTOR_CONTEXT")
    u_unit = u_decision_effect_unit(config)
    u_phase_decision_effect = u_module_enabled(
        config, "PHASE_EVIDENCE"
    ) and not u_module_log_only(config, "PHASE_EVIDENCE")
    u_stop_decision_effect = u_module_enabled(
        config, "STOP_EVIDENCE_VERIFIER"
    ) and not u_module_log_only(config, "STOP_EVIDENCE_VERIFIER")
    u_recovery_decision_effect = u_module_enabled(
        config, "FAILURE_RECOVERY"
    ) and not u_module_log_only(config, "FAILURE_RECOVERY")
    u_decision_effect_modules = (
        u_phase_decision_effect or u_stop_decision_effect or u_recovery_decision_effect
    )
    decision_effect_modules = (
        visual_target_verifier_decision_effect
        or multimodal_selector_decision_effect
        or u_decision_effect_modules
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
    if u_config_enabled and not harness_logging_enabled(config):
        raise ValueError(
            "OPENNAV_HARNESS.U_SERIES.ENABLED=true requires "
            "OPENNAV_HARNESS.ENABLED=true and ENABLE_HARNESS_LOGGING=true."
        )
    if not harness_logging_enabled(config):
        return
    if u_node is not None:
        valid_units = {"none", "U1", "U2", "U3", "combined"}
        if u_unit not in valid_units:
            raise ValueError(
                "OPENNAV_HARNESS.U_SERIES.DECISION_EFFECT_UNIT must be one of "
                "{}.".format(sorted(valid_units))
            )
        if u_series_enabled(config) and u_unit == "none":
            for module_name in (
                "PHASE_EVIDENCE",
                "STOP_EVIDENCE_VERIFIER",
                "FAILURE_RECOVERY",
            ):
                if u_module_enabled(config, module_name) and not u_module_log_only(
                    config, module_name
                ):
                    raise ValueError(
                        "U0-log requires OPENNAV_HARNESS.U_SERIES.{}.LOG_ONLY=true.".format(
                            module_name
                        )
                    )
        unit_allowed_modules = {
            "U1": {"PHASE_EVIDENCE"},
            "U2": {"STOP_EVIDENCE_VERIFIER"},
            "U3": {"FAILURE_RECOVERY"},
            "combined": {
                "PHASE_EVIDENCE",
                "STOP_EVIDENCE_VERIFIER",
                "FAILURE_RECOVERY",
            },
        }
        unit_required_module = {
            "U1": "PHASE_EVIDENCE",
            "U2": "STOP_EVIDENCE_VERIFIER",
            "U3": "FAILURE_RECOVERY",
        }
        if u_series_enabled(config) and u_unit in unit_required_module:
            required_module = unit_required_module[u_unit]
            if (
                not u_module_enabled(config, required_module)
                or u_module_log_only(config, required_module)
            ):
                raise ValueError(
                    "DECISION_EFFECT_UNIT={} requires "
                    "OPENNAV_HARNESS.U_SERIES.{}.ENABLED=true and LOG_ONLY=false.".format(
                        u_unit,
                        required_module,
                    )
                )
        if u_series_enabled(config) and u_unit == "combined" and not (
            u_phase_decision_effect
            or u_stop_decision_effect
            or u_recovery_decision_effect
        ):
            raise ValueError(
                "DECISION_EFFECT_UNIT=combined requires at least one U-series "
                "module to be enabled with LOG_ONLY=false."
            )
        if u_unit in unit_allowed_modules:
            allowed = unit_allowed_modules[u_unit]
            for module_name in (
                "PHASE_EVIDENCE",
                "STOP_EVIDENCE_VERIFIER",
                "FAILURE_RECOVERY",
            ):
                if (
                    u_module_enabled(config, module_name)
                    and not u_module_log_only(config, module_name)
                    and module_name not in allowed
                ):
                    raise ValueError(
                        "DECISION_EFFECT_UNIT={} does not allow {}.LOG_ONLY=false.".format(
                            u_unit, module_name
                        )
                    )
            if u_unit in {"U3", "combined"} and u_recovery_decision_effect:
                recovery_node = _cfg_get(
                    u_series_node(config),
                    "FAILURE_RECOVERY",
                    None,
                )
                if not bool(_cfg_get(recovery_node, "ENABLE_RESELECT", False)):
                    raise ValueError(
                        "U3 decision-effect requires "
                        "OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.ENABLE_RESELECT=true."
                    )
            if u_unit in {"U1", "combined"} and u_phase_decision_effect:
                if multimodal_selector_decision_effect:
                    raise ValueError(
                        "U1 decision-effect cannot run with "
                        "MULTIMODAL_SELECTOR_CONTEXT.LOG_ONLY=false; keep V4 unified "
                        "context as log-only input for U1."
                    )
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

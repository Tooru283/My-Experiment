"""Single authoritative ACN-L1 progress provider.

The project no longer supports LLM completion estimation.  This adapter converts the
owned L1 reducer state into the M0 ProgressUpdate contract consumed by legacy prompt and
STOP APIs during migration.
"""

from dataclasses import replace
import math
from typing import Any, Dict, Optional, Sequence

from vlnce_baselines.common.opennav_ext.pipeline_contracts import ProgressUpdate


PROGRESS_PROVIDER_NAME = "acn_l1"


_GENERIC_TERMINAL_TERMS = frozenset((
    "area", "archway", "door", "doorway", "entry way", "entryway",
    "floor", "hall", "hallway", "room", "stair", "stairs", "staircase",
))


def is_weak_generic_terminal(verifier_result: Dict[str, Any]) -> bool:
    """True when a generic final target has no distinctive context landmark."""
    result = dict(verifier_result or {})
    required = [
        str(v).strip().lower()
        for v in (result.get("required_landmark_terms") or [])
        if str(v).strip()
    ]
    context = [
        str(v).strip().lower()
        for v in (result.get("all_landmark_terms") or [])
        if str(v).strip()
    ]
    if not required or not all(v in _GENERIC_TERMINAL_TERMS for v in required):
        return False
    distinctive = [v for v in context if v not in _GENERIC_TERMINAL_TERMS]
    return not distinctive


def direction_aligned_terminal_result(verifier_result: Dict[str, Any]) -> Dict[str, Any]:
    """Restrict terminal evidence to one selected panorama candidate and direction."""
    raw = dict(verifier_result or {})
    result = dict(raw)
    selected = dict(raw.get("selected_candidate_verdict") or {})
    required = {
        str(v).strip().lower()
        for v in (raw.get("required_landmark_terms") or [])
        if str(v).strip()
    }
    matched = {
        str(v).strip().lower()
        for v in (selected.get("matched_final_landmarks") or [])
        if str(v).strip()
    }
    missing = sorted(required - matched)
    direction_id = selected.get("target_direction_id")
    aligned = bool(
        selected
        and direction_id is not None
        and selected.get("final_target_visible") is True
        and selected.get("arrival_evidence") is True
        and not missing
    )
    result["aggregate_final_target_visible"] = raw.get("final_target_visible")
    result["aggregate_arrival_evidence"] = raw.get("arrival_evidence")
    result["final_target_visible"] = bool(
        selected.get("final_target_visible") is True and aligned
    )
    result["arrival_evidence"] = bool(
        selected.get("arrival_evidence") is True and aligned
    )
    result["direction_aligned_terminal_support"] = {
        "aligned": aligned,
        "candidate_id": selected.get("candidate_id"),
        "target_direction_id": direction_id,
        "matched_final_landmarks": sorted(matched),
        "missing_final_landmarks": missing,
    }
    result["aggregate_current_view_corroboration"] = dict(
        raw.get("current_view_corroboration") or {}
    )
    result["current_view_corroboration"] = {
        "required": True,
        "scope": "selected_candidate_direction",
        "target_direction_id": direction_id,
        "uncorroborated_terms": missing,
    }
    if not aligned:
        result["verdict"] = "reject"
        result["reason"] = (
            "Selected panorama candidate lacks direction-aligned terminal evidence."
        )
        blockers = list(result.get("allow_blockers") or [])
        if "direction_aligned_terminal_evidence_missing" not in blockers:
            blockers.append("direction_aligned_terminal_evidence_missing")
        result["allow_blockers"] = blockers
    return result


def weak_generic_close_is_confirmed(
    verifier_result: Dict[str, Any], *, route_progress_complete: bool,
    route_maturity_satisfied: bool, local_surface_depth_m: Optional[float],
    max_depth_m: float = 1.5,
) -> bool:
    """Allow one strong, very-close frame only for a mature weak-generic route."""
    result = dict(verifier_result or {})
    direction = result.get("direction_aligned_terminal_support") or {}
    raw_text = result.get("aggregate_current_view_corroboration") or {}
    try:
        depth = float(local_surface_depth_m)
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(
        is_weak_generic_terminal(result)
        and route_progress_complete
        and route_maturity_satisfied
        and direction.get("aligned") is True
        and not (raw_text.get("uncorroborated_terms") or [])
        and result.get("arrival_evidence") is True
        and (
            not (result.get("terminal_relation_evidence") or {}).get("required")
            or (result.get("terminal_relation_evidence") or {}).get("satisfied")
            is True
        )
        and confidence >= 0.9
        and 0.0 < depth <= float(max_depth_m)
    )


def terminal_target_is_confirmed(
    verifier_result: Dict[str, Any], *, persistent_visual_confirm: bool,
    spatial_distance_confirmed: bool,
    weak_generic_route_matured: bool = False,
) -> bool:
    """Accept a current-frame terminal observation under conservative rules."""
    result = dict(verifier_result or {})
    verdict = str(result.get("verdict") or "").lower()
    blockers = [str(v) for v in (result.get("allow_blockers") or [])]
    weak_generic = is_weak_generic_terminal(result)
    generic_only_uncertain = bool(
        verdict == "uncertain"
        and blockers
        and all(v.startswith("generic_final_terms:") for v in blockers)
        and not (
            (result.get("current_view_corroboration") or {}).get(
                "uncorroborated_terms"
            )
            or []
        )
        and (not weak_generic or weak_generic_route_matured)
    )
    return bool(
        persistent_visual_confirm
        and spatial_distance_confirmed
        and (verdict == "allow" or generic_only_uncertain)
        and result.get("final_target_visible") is True
        and result.get("arrival_evidence") is True
    )


class TerminalInstanceTracker:
    """Bind repeated terminal observations to an ego-local target instance."""

    def __init__(self, max_position_delta_m: float = 2.25):
        self.max_position_delta_m = max(0.0, float(max_position_delta_m))
        self.episode_id = None
        self.serial = 0
        self.track = None

    def reset(self, episode_id: Any) -> None:
        self.episode_id = str(episode_id)
        self.serial = 0
        self.track = None

    @staticmethod
    def _terms(values: Any) -> tuple:
        return tuple(sorted({
            str(v).strip().lower() for v in (values or []) if str(v).strip()
        }))


    @staticmethod
    def _is_weak_generic_terms(terms: Any) -> bool:
        values = tuple(terms or ())
        return bool(values) and all(v in _GENERIC_TERMINAL_TERMS for v in values)


    @staticmethod
    def _context_overlap(left: Any, right: Any) -> float:
        a = set(left or ())
        b = set(right or ())
        union = a | b
        return float(len(a & b)) / float(len(union)) if union else 0.0

    @staticmethod
    def _position(position: Any) -> Optional[tuple]:
        try:
            values = tuple(float(v) for v in position)
            return values if len(values) >= 3 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _target_point(
        position: Any, heading: Any, direction_id: Any, depth_m: Any
    ) -> Optional[tuple]:
        try:
            pos = TerminalInstanceTracker._position(position)
            if pos is None:
                return None
            direction = int(direction_id)
            if direction < 0 or direction > 11:
                return None
            distance = float(depth_m)
            if not math.isfinite(distance) or distance <= 0:
                return None
            bearing = float(heading) + direction * (2.0 * math.pi / 12.0)
            return (
                pos[0] + distance * math.sin(bearing),
                pos[2] + distance * math.cos(bearing),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _point_distance(left: Any, right: Any) -> Optional[float]:
        try:
            return (
                (float(left[0]) - float(right[0])) ** 2
                + (float(left[1]) - float(right[1])) ** 2
            ) ** 0.5
        except (TypeError, ValueError, IndexError):
            return None

    def update(
        self,
        *,
        step_id: int,
        position: Any,
        heading: Any,
        direction_id: Any,
        depth_m: Any,
        target_terms: Any,
        supporting_landmarks: Any,
        visually_supported: bool,
    ) -> Dict[str, Any]:
        terms = self._terms(target_terms)
        context = self._terms(supporting_landmarks)
        point = self._target_point(position, heading, direction_id, depth_m)
        previous_key = self.track.get("instance_key") if self.track else None
        delta = self._point_distance(
            point, self.track.get("target_point") if self.track else None
        )
        previous_terms = self.track.get("target_terms") if self.track else ()
        previous_context = (
            self.track.get("supporting_landmarks") if self.track else ()
        )
        weak_generic = self._is_weak_generic_terms(terms)
        context_overlap = self._context_overlap(context, previous_context)
        spatial_match = bool(
            delta is not None and delta <= self.max_position_delta_m
        )
        distinctive_match = bool(
            not weak_generic
            and delta is not None
            and delta <= 5.0
            and (delta <= 3.5 or context_overlap >= 0.5)
        )
        same_instance = bool(
            visually_supported
            and terms
            and point is not None
            and self.track is not None
            and terms == previous_terms
            and (spatial_match or distinctive_match)
        )
        confirmed_instance_switch = bool(
            previous_key
            and (terms != previous_terms or (delta is not None and delta > 5.0))
        )
        if not visually_supported or not terms or point is None:
            self.track = None
            return {
                "instance_key": None,
                "previous_instance_key": previous_key,
                "same_instance": False,
                "instance_changed": bool(previous_key),
                "consecutive_observations": 0,
                "target_point": point,
                "position_delta_m": delta,
                "target_terms": terms,
                "supporting_landmarks": context,
                "reason": "insufficient_instance_geometry",
                "weak_generic_target": self._is_weak_generic_terms(terms),
                "context_overlap": 0.0,
                "confirmed_instance_switch": False,
                "oracle_free_goal": True,
            }
        if same_instance:
            instance_key = self.track["instance_key"]
            consecutive = int(self.track["consecutive_observations"]) + 1
        else:
            self.serial += 1
            instance_key = "{}:terminal-instance:{}".format(
                self.episode_id or "unknown", self.serial
            )
            consecutive = 1
        self.track = {
            "instance_key": instance_key,
            "target_terms": terms,
            "supporting_landmarks": context,
            "target_point": point,
            "step_id": int(step_id),
            "consecutive_observations": consecutive,
        }
        return {
            "instance_key": instance_key,
            "previous_instance_key": previous_key,
            "same_instance": same_instance,
            "instance_changed": bool(previous_key and previous_key != instance_key),
            "consecutive_observations": consecutive,
            "target_point": point,
            "position_delta_m": delta,
            "target_terms": terms,
            "supporting_landmarks": context,
            "reason": "matched_terminal_track" if same_instance else "new_spatial_track",
            "weak_generic_target": weak_generic,
            "context_overlap": context_overlap,
            "distinctive_context_match": distinctive_match,
            "confirmed_instance_switch": confirmed_instance_switch,
            "max_position_delta_m": self.max_position_delta_m,
            "distinctive_position_delta_m": 3.5,
            "hard_switch_position_delta_m": 5.0,
            "oracle_free_goal": True,
        }


class TerminalEvidenceMemory:
    """Short-lived, ego-local bridge between terminal and route completion."""

    def __init__(self, max_age_steps: int = 2, max_displacement_m: float = 2.25):
        self.max_age_steps = max(0, int(max_age_steps))
        self.max_displacement_m = max(0.0, float(max_displacement_m))
        self.last = None

    def reset(self) -> None:
        self.last = None

    @staticmethod
    def _distance(left: Any, right: Any) -> Optional[float]:
        try:
            a = [float(v) for v in left]
            b = [float(v) for v in right]
            if len(a) != len(b) or not a:
                return None
            return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
        except (TypeError, ValueError):
            return None

    def update(
        self,
        *,
        step_id: int,
        position: Any,
        direct_confirmed: bool,
        current_visual_support: bool,
        instance_key: Optional[str] = None,
        confirmed_instance_switch: bool = False,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if direct_confirmed:
            self.last = {
                "step_id": int(step_id),
                "position": tuple(float(v) for v in position),
                "instance_key": str(instance_key) if instance_key else None,
                "evidence": dict(evidence or {}),
            }
        age = None
        displacement = None
        reused = False
        cache_cleared_reason = None
        if self.last is not None:
            age = int(step_id) - int(self.last["step_id"])
            displacement = self._distance(position, self.last["position"])
            cached_instance_key = self.last.get("instance_key")
            instance_matches = bool(
                instance_key
                and cached_instance_key
                and str(instance_key) == cached_instance_key
            )
            reused = bool(
                not direct_confirmed
                and current_visual_support
                and instance_matches
                and 0 <= age <= self.max_age_steps
                and displacement is not None
                and displacement <= self.max_displacement_m
            )
            if (
                current_visual_support
                and instance_key
                and cached_instance_key
                and not instance_matches
                and confirmed_instance_switch
            ):
                self.last = None
                reused = False
                cache_cleared_reason = "confirmed_instance_changed"
            elif (
                current_visual_support
                and instance_key
                and cached_instance_key
                and not instance_matches
                and 0 <= age <= self.max_age_steps
                and displacement is not None
                and displacement <= self.max_displacement_m
            ):
                reused = False
                cache_cleared_reason = "uncertain_instance_switch_cache_retained"
            elif age > self.max_age_steps or (
                displacement is not None
                and displacement > self.max_displacement_m
            ):
                self.last = None
                cache_cleared_reason = "expired_or_displaced"
        return {
            "confirmed": bool(direct_confirmed or reused),
            "direct_confirmed": bool(direct_confirmed),
            "reused": reused,
            "instance_key": instance_key,
            "cached_instance_key": self.last.get("instance_key") if self.last else None,
            "cache_cleared_reason": cache_cleared_reason,
            "age_steps": age,
            "displacement_m": displacement,
            "max_age_steps": self.max_age_steps,
            "max_displacement_m": self.max_displacement_m,
            "oracle_free": True,
        }


class ACNL1ProgressProvider:
    name = PROGRESS_PROVIDER_NAME

    def __init__(self, state_reducer: Any) -> None:
        self.state_reducer = state_reducer
        self.episode_id: Optional[str] = None
        self.previous_index: Optional[int] = None

    def reset_episode(self, episode_id: Any) -> None:
        self.episode_id = str(episode_id)
        self.previous_index = None

    def update(
        self,
        locator_state: Dict[str, Any],
        *,
        evidence_refs: Sequence[str] = (),
    ) -> ProgressUpdate:
        if self.episode_id is None:
            raise RuntimeError("ACNL1ProgressProvider must be reset before update")
        if not isinstance(locator_state, dict):
            raise TypeError("ACN L1 locator state must be a dict")

        degenerate = bool(locator_state.get("degenerate"))
        raw_index = locator_state.get("j")
        raw_total = locator_state.get("n_anchors")
        current_index = int(raw_index) if raw_index is not None else None
        total = int(raw_total) if raw_total is not None else None

        if degenerate or current_index is None:
            transition = "abstain"
            abstained = True
        elif self.previous_index is None:
            transition = "hold" if current_index == 0 else "advance"
            abstained = bool(locator_state.get("abstained_this_step"))
        elif current_index > self.previous_index:
            transition = "advance"
            abstained = False
        elif current_index == self.previous_index:
            transition = (
                "abstain"
                if locator_state.get("abstained_this_step")
                else "hold"
            )
            abstained = bool(locator_state.get("abstained_this_step"))
        else:
            raise ValueError(
                "ACN L1 regressed from {} to {}".format(
                    self.previous_index, current_index
                )
            )

        if current_index is not None:
            self.previous_index = current_index

        display_text = self.state_reducer.completion_text()
        route_complete = (
            bool(locator_state.get("complete")) if not degenerate else None
        )
        plan = getattr(self.state_reducer, "plan", {})
        terminal_policy = (
            plan.get("terminal_policy", {}) if isinstance(plan, dict) else {}
        )
        terminal_required = bool(terminal_policy.get("present"))
        initial_terminal_confirmed = False if terminal_required else None
        initial_goal_complete = (
            bool(route_complete) if not terminal_required else False
        ) if route_complete is not None else None
        current_evaluation = locator_state.get("last_evaluation") or {}
        if current_evaluation.get("anchor_index") != current_index:
            current_evaluation = {}
        update = ProgressUpdate(
            provider=self.name,
            current_index=current_index,
            total=total,
            completed_indices=tuple(
                int(item) for item in (locator_state.get("ever_satisfied") or [])
            ),
            current_target=(
                str(locator_state.get("current_raw"))
                if locator_state.get("current_raw") is not None
                else None
            ),
            # Compatibility alias: complete means route progress only. STOP must
            # consume goal_complete, never this field.
            complete=route_complete,
            route_progress_complete=route_complete,
            terminal_target_confirmed=initial_terminal_confirmed,
            goal_complete=initial_goal_complete,
            confidence=None,
            abstained=abstained,
            transition=transition,
            evidence_refs=tuple(str(item) for item in evidence_refs),
            display_text=str(display_text or ""),
            authoritative=True,
            shadow=None,
            current_action={
                "raw": locator_state.get("current_raw"),
                "requirements": locator_state.get("current_action") or [],
                "evaluation": dict(current_evaluation),
            },
            verified_actions=tuple(
                {
                    "anchor_index": int(item),
                    "raw": (
                        plan.get("anchors", [])[int(item)].get("raw")
                        if isinstance(plan, dict)
                        and int(item) < len(plan.get("anchors", []))
                        else None
                    ),
                }
                for item in (locator_state.get("ever_satisfied") or [])
            ),
            missing_evidence=tuple(
                str(item) for item in (
                    current_evaluation.get(
                        "missing_evidence"
                    ) or []
                )
            ),
            completion_events=tuple(
                dict(item) for item in (
                    locator_state.get("completion_events") or []
                )
            ),
        )
        update.to_dict()
        return update
    def apply_terminal_evidence(
        self,
        update: ProgressUpdate,
        *,
        terminal_target_confirmed: bool,
        evidence_refs: Sequence[str] = (),
    ) -> ProgressUpdate:
        """Combine route completion with terminal evidence into goal completion."""
        terminal_required = update.terminal_target_confirmed is not None
        confirmed = bool(terminal_target_confirmed) if terminal_required else None
        goal_complete = bool(
            update.route_progress_complete
            and (confirmed if terminal_required else True)
        )
        enriched = replace(
            update,
            terminal_target_confirmed=confirmed,
            goal_complete=goal_complete,
            evidence_refs=update.evidence_refs + tuple(str(v) for v in evidence_refs),
        )
        enriched.to_dict()
        return enriched

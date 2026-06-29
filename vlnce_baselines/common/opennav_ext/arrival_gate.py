from typing import Any, Dict, List, Optional


class ArrivalGate:
    """Log-only gate that detects when the agent enters the near-goal zone.

    Trigger condition (both must hold):
      1. latest observed goal distance <= dist_threshold
      2. current navigation phase is in allowed_phases

    The gate is purely diagnostic: it emits a structured event but never
    alters the selected action. Decision-effect wiring (M2+) is added later.
    """

    def __init__(
        self,
        dist_threshold: float = 4.0,
        allowed_phases: Optional[List[str]] = None,
        min_trigger_step: int = 1,
    ) -> None:
        self.dist_threshold = float(dist_threshold)
        self.allowed_phases: set = set(
            allowed_phases
            if allowed_phases is not None
            else ["approach", "verify", "unknown"]
        )
        self.min_trigger_step = max(1, int(min_trigger_step))

    def check(
        self,
        latest_goal_dist: Optional[float],
        current_phase: Optional[str],
        current_step: int,
        arrival_evidence_seen: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Return a gate result dict; never raises."""
        reasons: List[str] = []
        triggered = False

        dist_ok = latest_goal_dist is not None and latest_goal_dist <= self.dist_threshold
        phase_ok = current_phase is None or current_phase in self.allowed_phases
        step_ok = current_step >= self.min_trigger_step

        if dist_ok and phase_ok and step_ok:
            triggered = True
            reasons.append(
                f"dist={latest_goal_dist:.2f}m <= threshold={self.dist_threshold}m"
            )
        else:
            if not dist_ok:
                dist_str = f"{latest_goal_dist:.2f}m" if latest_goal_dist is not None else "unknown"
                reasons.append(f"dist={dist_str} > threshold={self.dist_threshold}m or unknown")
            if not phase_ok:
                reasons.append(f"phase={current_phase!r} not in allowed={sorted(self.allowed_phases)}")
            if not step_ok:
                reasons.append(f"step={current_step} < min_trigger_step={self.min_trigger_step}")

        return {
            "triggered": triggered,
            "reasons": reasons,
            "latest_goal_dist": latest_goal_dist,
            "dist_threshold": self.dist_threshold,
            "current_phase": current_phase,
            "allowed_phases": sorted(self.allowed_phases),
            "current_step": current_step,
            "arrival_evidence_seen": arrival_evidence_seen,
            "decision_effect": False,
        }

"""Bounded, oracle-free low-translation rechecks of a plausible terminal."""
import math


class TerminalRecheck:
    def __init__(self, max_holds=2, max_depth_m=3.0, recheck_step_m=0.25):
        self.max_holds = max(0, int(max_holds))
        self.max_depth_m = max(0.0, float(max_depth_m))
        self.recheck_step_m = max(0.01, float(recheck_step_m))
        self.reset()

    def reset(self):
        self.used = 0
        self.pending_step = None
        self.pending_position = None

    def decide(self, *, step_id, position, route_complete, stop_committed,
               verifier, spatial, evidence_unavailable=False):
        """A recheck never authorizes STOP or counts as route execution.

        Missing evidence may extend one immediately preceding micro-step, but it
        cannot start a recheck. Explicit contrary evidence ends the recheck.
        """
        selected = verifier.get("selected_candidate_verdict") or {}
        depth = spatial.get("local_surface_depth_m")
        near = (isinstance(depth, (int, float)) and not isinstance(depth, bool)
                and math.isfinite(depth) and 0 < depth <= self.max_depth_m)
        blockers = verifier.get("allow_blockers") or []
        compatible = (not verifier.get("contradictions")
                      and all(str(b).startswith("generic_final_terms:") for b in blockers))
        plausible = (compatible and selected.get("final_target_visible") is True
                     and selected.get("arrival_evidence") is True
                     and not selected.get("missing_final_landmarks")
                     and selected.get("target_direction_id") is not None and near)
        locally_stable = False
        if position is not None and self.pending_position is not None:
            try:
                locally_stable = (
                    len(position) == len(self.pending_position) == 3
                    and sum((float(a) - float(b)) ** 2
                            for a, b in zip(position, self.pending_position))
                    <= (self.recheck_step_m + 0.1) ** 2
                )
            except (TypeError, ValueError):
                pass
        continue_pending = (evidence_unavailable and locally_stable
                            and self.pending_step == step_id - 1)
        hold = bool(route_complete and not stop_committed
                    and self.used < self.max_holds and (plausible or continue_pending))
        if hold:
            self.used += 1
            self.pending_step = step_id
            self.pending_position = list(position) if position is not None else None
        else:
            self.pending_step = self.pending_position = None
        return {"hold": hold, "used": self.used, "max_holds": self.max_holds,
                "reason": ("terminal_evidence_recheck" if plausible else "retry_unavailable_evidence")
                if hold else "no_recheck_or_budget_exhausted",
                "plausible_terminal": bool(plausible),
                "recheck_step_m": self.recheck_step_m, "oracle_free": True}

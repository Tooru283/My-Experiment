"""L4 -- triple-conjunction termination (ACN §3, 20260805).

STOP is accepted only when all three hold:

    1. the final anchor is located          L1's j == N
    2. every earlier anchor was verified    the state machine's history, not this frame
    3. the final anchor has landmark support M2 entry with arrived=True

Condition 2 is the one the current stack has no analogue for: today "stop requests account
for 72/100 terminations" with no requirement that anything earlier was ever confirmed.

⚠ Depth is deliberately NOT a conjunct. At the moment of stopping, depth's AUROC for
"actually arrived" measured 0.531 -- no information. Opening-class goals (doorways,
stair heads) are legitimately >3 m away. `DEPTH_STOP_VETO` stays a separate switch; L4
does not fold it in.

⚠ This gate can only ever *block* a stop, never cause one. Combined with the standing
count of 45 false stops that is the intended direction -- but it also means a run with L4
decision-active must watch exhaustion count, which is the failure mode it converts into.
"""
from typing import Any, Dict, Optional


class TerminalGate:
    def __init__(
        self,
        require_chain_complete: bool = True,
        require_all_verified: bool = True,
        require_landmark_evidence: bool = True,
        allow_abstain_when_degenerate: bool = True,
    ) -> None:
        self.require_chain_complete = require_chain_complete
        self.require_all_verified = require_all_verified
        self.require_landmark_evidence = require_landmark_evidence
        self.allow_abstain_when_degenerate = allow_abstain_when_degenerate

    def evaluate(
        self,
        locator_state: Optional[Dict[str, Any]],
        landmark_pool: Any,
        final_landmark_term: Any = None,
    ) -> Dict[str, Any]:
        if not isinstance(locator_state, dict) or locator_state.get("degenerate"):
            # No anchor chain -> no opinion. Abstaining (not blocking) is the safe
            # default: 9/100 episodes build no chain, and silently blocking every stop
            # in them would manufacture exhaustion failures.
            return {
                "verdict": "abstain",
                "allow_stop": True,
                "reason": "no anchor chain available; L4 abstains",
                "conjuncts": {},
                "degenerate": True,
            }

        n = locator_state.get("n_anchors") or 0
        j = locator_state.get("j") or 0
        ever = set(locator_state.get("ever_satisfied") or [])

        c1 = bool(n and j >= n)
        c2 = all(i in ever for i in range(max(0, n - 1)))
        entry = None
        if final_landmark_term is not None and landmark_pool is not None:
            entry = landmark_pool.query(final_landmark_term)
        c3 = bool(entry and entry.get("arrived"))

        checks = {
            "chain_complete": c1 if self.require_chain_complete else None,
            "all_previous_verified": c2 if self.require_all_verified else None,
            "final_landmark_arrived": c3 if self.require_landmark_evidence else None,
        }
        required = [v for v in checks.values() if v is not None]
        allow = all(required) if required else True

        missing = [k for k, v in checks.items() if v is False]
        return {
            "verdict": "allow" if allow else "block",
            "allow_stop": allow,
            "reason": (
                "all conjuncts satisfied"
                if allow
                else "blocked by: " + ", ".join(missing)
            ),
            "conjuncts": checks,
            "j": j,
            "n_anchors": n,
            "final_landmark_term": final_landmark_term,
            "final_landmark_entry": (
                {k: v for k, v in entry.items() if k != "fused_pos"} if entry else None
            ),
            "degenerate": False,
        }

def gate_stop_request(
    stop_requested: bool,
    reason: str,
    gate_result: Optional[Dict[str, Any]],
    decision_effect_enabled: bool,
):
    """Return ``(stop_requested, reason, blocked)`` for L4 action wiring.

    The function is intentionally independent of the trainer so a unit test can
    prove that a logged L4 block becomes a non-STOP action when enabled.
    """
    if not stop_requested or not decision_effect_enabled:
        return bool(stop_requested), reason, False
    gate = gate_result if isinstance(gate_result, dict) else {}
    if gate.get("allow_stop", True):
        return True, reason, False
    return False, "", True

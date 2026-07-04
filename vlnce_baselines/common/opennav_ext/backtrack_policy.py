"""Backtracking policy: relative reverse-move recovery for dead-end / progress stall.

Motivation
----------
`RecoveryPolicy` (U3) re-selects an *alternate candidate at the current viewpoint* --
it cannot rescue an agent that has already walked into a dead-end corridor, because the
current viewpoint's candidates all lie down the wrong branch. SmartWay's backtracking
(real-robot +12pp) instead physically returns the agent toward a previous branch point.

In this codebase a viewpoint is not an absolute graph node: a selected candidate is
executed as a *relative polar move* (see base_il_trainer_llm.py:3130)::

    {'action': 4, 'action_args': {'angle': <turn deg>, 'distance': <meters>}}

Therefore backtracking needs no teleport/graph API: the reverse of a step that moved
`d` meters is simply ``{'angle': 180.0, 'distance': d}`` (turn around, retrace). After
returning, the abandoned heading is recorded so the regenerated candidates at the
returned position are de-prioritised (blocked) rather than re-selected.

The trigger must be *observable* and *zero-calibration*. It is NOT ``recent_distance_gains``:
that quantity is ``distances[-2]-distances[-1]`` over ``info["position"]["distance"]``, i.e.
the simulator's geodesic distance-to-goal (oracle_metrics.selected_distance_gain). Reading it
at inference is oracle leakage -- fatal for a zero-shot claim. Instead we dead-reckon net
displacement purely from the agent's own executed ``(angle, distance)`` moves (odometry the
agent already emits): lots of path length but little net displacement == physically circling
/ thrashing at a dead-end. A dead-end can also be signalled observably by the waypoint
predictor running out of forward candidates (passed in as ``dead_end``). Both are behavioural
ground truth, need no goal knowledge and no calibration -- the whole point vs the P1
dispersion/confusion triggers.

This module is pure and dependency-free so it can be unit-tested off-GPU and dropped
into the harness the same way RecoveryPolicy is (propose/apply returning a log dict).
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple


MOVE_BACK_CANDIDATE = "MOVE_BACK"


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _clean_moves(recent_moves: Optional[List[Dict[str, Any]]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for mv in recent_moves or []:
        if not isinstance(mv, dict):
            continue
        ang = mv.get("angle")
        dist = mv.get("distance")
        if _is_number(dist):
            out.append({"angle": float(ang) if _is_number(ang) else 0.0,
                        "distance": float(dist)})
    return out


def deadreckon(recent_moves: Optional[List[Dict[str, Any]]]) -> Tuple[float, float]:
    """Ego dead-reckoning from executed ``(angle, distance)`` moves. Observable, no oracle.

    ``angle`` is the relative turn (deg) applied before moving ``distance`` (m) forward,
    matching the env action ``{'action':4,'action_args':{'angle','distance'}}``. Returns
    ``(net_displacement, path_length)`` in the agent's own frame; heading is integrated
    relative so the absolute start orientation is irrelevant.
    """
    moves = _clean_moves(recent_moves)
    heading = 0.0
    x = y = path = 0.0
    for mv in moves:
        heading += math.radians(mv["angle"])
        d = mv["distance"]
        x += d * math.cos(heading)
        y += d * math.sin(heading)
        path += abs(d)
    return math.hypot(x, y), path


def _xy(p: Any) -> Optional[Tuple[float, float]]:
    """Accept (x,y), (x,y,z) or habitat's (x, height, z); use the ground plane (x, last)."""
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        try:
            return float(p[0]), float(p[-1])
        except (TypeError, ValueError):
            return None
    return None


def net_path_from_positions(recent_positions: Optional[List[Any]]) -> Tuple[float, float]:
    """(net_displacement, path_length) from a history of the agent's own ego positions.

    ``net`` = straight-line distance between the first and last pose in the window; ``path``
    = summed step-to-step distance. Uses only position *differences* (odometry-equivalent),
    never absolute coordinates or goal distance -- observable, no oracle.
    """
    pts = [xy for xy in (_xy(p) for p in (recent_positions or [])) if xy is not None]
    if len(pts) < 2:
        return 0.0, 0.0
    path = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    net = math.dist(pts[0], pts[-1])
    return net, path


def displacement_stalled(
    recent_positions: Optional[List[Any]] = None,
    window: int = 3,
    disp_eps: float = 1.0,
    ratio: float = 0.35,
    recent_moves: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Observable stall: over the last ``window`` steps the agent covered path but went nowhere.

    Primary input is ``recent_positions`` (the agent's own ego poses from get_agent_info);
    if absent, falls back to dead-reckoning ``recent_moves``. Stall when net displacement is
    tiny in absolute terms (``<= disp_eps`` m) OR highly inefficient (``net/path <= ratio``)
    -- both mean physical circling / thrashing typical of a dead-end, from ego signals only
    (no goal, no calibration). Needs a full window so we don't fire on one noisy step.
    Thresholds are heuristics to be tuned on the 72-lost-episode smoke.
    """
    if recent_positions is not None:
        pts = [p for p in recent_positions if _xy(p) is not None]
        if len(pts) < max(2, window):
            return False
        net, path = net_path_from_positions(pts[-(window + 1):])
    else:
        moves = _clean_moves(recent_moves)
        if len(moves) < max(1, window):
            return False
        net, path = deadreckon(moves[-window:])
    if path <= 0.0:
        return False
    return net <= disp_eps or (net / path) <= ratio


def reverse_action_args(last_move: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Relative move that retraces ``last_move``: turn 180 deg, same distance.

    ``last_move`` is the previously executed ``action_args`` ({'angle','distance'}).
    Returns None when there is no reversible prior move (episode start, or a zero-distance
    step such as a pure rotation).
    """
    if not isinstance(last_move, dict):
        return None
    if not _is_number(last_move.get("distance")):
        return None
    distance = float(last_move["distance"])
    if distance <= 0.0:
        return None
    return {"angle": 180.0, "distance": distance}


class BacktrackPolicy:
    """Decides whether MOVE_BACK is offered, and executes it when the navigator picks it.

    Budgeted (per-episode cap) and anti-oscillation guarded (never two backtracks in a
    row, never immediately after arriving via a backtrack) so it cannot loop forever.
    Mirrors RecoveryPolicy's return-a-log-dict contract for harness logging / ablation.
    """

    def __init__(
        self,
        max_backtracks_per_episode: int = 2,
        stall_window: int = 3,
        disp_eps: float = 1.0,
        disp_ratio: float = 0.35,
    ) -> None:
        self.max_backtracks_per_episode = max(0, int(max_backtracks_per_episode))
        self.stall_window = max(1, int(stall_window))
        self.disp_eps = float(disp_eps)
        self.disp_ratio = float(disp_ratio)

    def available(
        self,
        last_move: Optional[Dict[str, Any]],
        recent_positions: Optional[List[Any]],
        budget_remaining: int,
        just_backtracked: bool = False,
        dead_end: bool = False,
        recent_moves: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Whether to inject MOVE_BACK into the action space this step (offer, not force).

        The navigator still freely chooses among {candidates, STOP, MOVE_BACK}; we only
        gate whether MOVE_BACK is *on the menu*, using observable preconditions: an ego
        displacement stall (from ``recent_positions`` ego poses, or dead-reckoned
        ``recent_moves``) OR a waypoint-predictor ``dead_end`` (no forward candidates).
        Neither reads goal distance.
        """
        reverse = reverse_action_args(last_move)
        stalled = displacement_stalled(
            recent_positions, self.stall_window, self.disp_eps, self.disp_ratio,
            recent_moves=recent_moves,
        )
        triggered = bool(stalled or dead_end)
        reasons: List[str] = []
        if int(budget_remaining) <= 0:
            reasons.append("budget_exhausted")
        if reverse is None:
            reasons.append("no_reversible_last_move")
        if just_backtracked:
            reasons.append("anti_oscillation_just_backtracked")
        if not triggered:
            reasons.append("no_observable_stall_or_dead_end")
        offered = not reasons
        if recent_positions is not None:
            net, path = net_path_from_positions(
                [p for p in recent_positions if _xy(p) is not None][-(self.stall_window + 1):]
            )
        else:
            net, path = deadreckon((recent_moves or [])[-self.stall_window:])
        return {
            "offered": offered,
            "reverse_action_args": reverse,
            "stalled": stalled,
            "dead_end": bool(dead_end),
            "net_displacement": round(net, 3),
            "path_length": round(path, 3),
            "budget_remaining": int(budget_remaining),
            "block_reasons": reasons,
        }

    def apply(
        self,
        chosen_candidate: Any,
        last_move: Optional[Dict[str, Any]],
        budget_remaining: int,
        last_move_heading: Optional[float] = None,
        blocked_headings: Optional[Set[float]] = None,
    ) -> Dict[str, Any]:
        """Execute a navigator-selected MOVE_BACK: emit reverse action + book-keeping.

        Returns applied=False (a no-op passthrough) when the navigator did not select
        MOVE_BACK, so the caller can call this unconditionally. On apply it decrements
        the budget and records the abandoned heading so the next step at the returned
        position can de-prioritise re-entering the failed branch.
        """
        blocked = set(blocked_headings or set())
        if str(chosen_candidate) != MOVE_BACK_CANDIDATE:
            return {
                "applied": False,
                "reason": "not_move_back",
                "action_args": None,
                "budget_before": int(budget_remaining),
                "budget_after": int(budget_remaining),
                "blocked_headings": sorted(blocked),
            }
        reverse = reverse_action_args(last_move)
        if reverse is None or int(budget_remaining) <= 0:
            return {
                "applied": False,
                "reason": (
                    "no_reversible_last_move"
                    if reverse is None
                    else "budget_exhausted"
                ),
                "action_args": None,
                "budget_before": int(budget_remaining),
                "budget_after": int(budget_remaining),
                "blocked_headings": sorted(blocked),
            }
        if last_move_heading is not None and _is_number(last_move_heading):
            blocked.add(round(float(last_move_heading), 1))
        return {
            "applied": True,
            "reason": "backtrack_reverse_move",
            "action_args": reverse,
            "budget_before": int(budget_remaining),
            "budget_after": max(0, int(budget_remaining) - 1),
            "blocked_headings": sorted(blocked),
        }


if __name__ == "__main__":
    # Off-GPU sanity checks (no framework deps).
    policy = BacktrackPolicy(max_backtracks_per_episode=2)

    # deadreckon: straight line -> net == path; U-turn oscillation -> net ~ 0.
    net, path = deadreckon([{"angle": 0, "distance": 1.0}, {"angle": 0, "distance": 1.0}])
    assert abs(net - 2.0) < 1e-6 and abs(path - 2.0) < 1e-6            # straight
    net2, _ = deadreckon([{"angle": 0, "distance": 1.0}, {"angle": 180, "distance": 1.0}])
    assert net2 < 1e-6                                                # there and back -> nowhere

    # displacement_stalled via positions (primary): needs window+1 poses.
    pos_circle = [(0.0, 0.0), (1.0, 0.0), (0.2, 0.0), (1.0, 0.0)]   # thrashing near x~0-1
    assert displacement_stalled(pos_circle) is True                 # net 1.0 <= eps
    pos_straight = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)] # net 6 == path
    assert displacement_stalled(pos_straight) is False
    assert displacement_stalled([(0.0, 0.0), (1.0, 0.0)]) is False  # window not full
    # habitat 3-tuple (x, height, z) -> ground plane (x, z)
    assert displacement_stalled([(0.0, 1.5, 0.0), (0.3, 1.5, 0.1),
                                 (0.1, 1.5, 0.2), (0.2, 1.5, 0.0)]) is True

    # displacement_stalled via moves fallback (no positions).
    moves_circle = [{"angle": 0, "distance": 1.0}, {"angle": 180, "distance": 1.0},
                    {"angle": 0, "distance": 1.0}]
    assert displacement_stalled(recent_moves=moves_circle) is True

    # reverse_action_args
    assert reverse_action_args({"angle": 30, "distance": 2.5}) == {"angle": 180.0, "distance": 2.5}
    assert reverse_action_args({"angle": 30, "distance": 0.0}) is None
    assert reverse_action_args(None) is None

    # available: reversible last move + observable stall -> offered
    a = policy.available({"angle": 30, "distance": 2.0}, pos_circle, 2)
    assert a["offered"] is True, a
    # dead_end alone triggers even without displacement stall
    a_de = policy.available({"angle": 30, "distance": 2.0}, pos_straight, 2, dead_end=True)
    assert a_de["offered"] is True, a_de
    # blocked when just backtracked
    a2 = policy.available({"angle": 30, "distance": 2.0}, pos_circle, 2, just_backtracked=True)
    assert a2["offered"] is False and "anti_oscillation_just_backtracked" in a2["block_reasons"]
    # not offered when moving efficiently and no dead-end
    a3 = policy.available({"angle": 30, "distance": 2.0}, pos_straight, 2)
    assert a3["offered"] is False and "no_observable_stall_or_dead_end" in a3["block_reasons"]

    # apply: navigator picked MOVE_BACK
    r = policy.apply(MOVE_BACK_CANDIDATE, {"angle": 30, "distance": 2.0}, 2, last_move_heading=95.0)
    assert r["applied"] and r["action_args"] == {"angle": 180.0, "distance": 2.0}
    assert r["budget_after"] == 1 and 95.0 in r["blocked_headings"]
    # apply passthrough when not chosen
    r2 = policy.apply("3", {"angle": 30, "distance": 2.0}, 2)
    assert r2["applied"] is False and r2["budget_after"] == 2

    print("backtrack_policy self-tests passed")

import math
from typing import Any, Dict, Iterable, List, Optional

from vlnce_baselines.common.opennav_ext.agent_state import CandidateState


def _position_to_list(position: Any) -> Optional[List[float]]:
    if position is None:
        return None
    try:
        return [float(position[0]), float(position[1]), float(position[2])]
    except Exception:
        return None


def _distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _candidate_next_position(
    position: List[float], heading: Any, candidate: CandidateState
) -> Optional[List[float]]:
    if candidate.angle_rad is None or candidate.distance is None:
        return None
    try:
        heading_value = float(heading) if heading is not None else 0.0
        # Match M2 waypoint geometry: Habitat heading and candidate angles use the inverse sign.
        angle = -heading_value - float(candidate.angle_rad)
        distance = float(candidate.distance)
        return [
            position[0] + distance * math.sin(angle),
            position[1],
            position[2] - distance * math.cos(angle),
        ]
    except Exception:
        return None


LOOP_ALERT_TEMPLATE = (
    " Spatial memory alert: you have already stood at or within {radius:.1f} m of this "
    "exact position {count} times before (earlier at step {first}, most recently at step "
    "{last}). Repeatedly returning to the same place means the branch you have been "
    "taking from here does not lead onward. Prefer a direction you have NOT already "
    "taken from this position. This alert is about which direction to pick; it is NOT "
    "evidence that you have arrived, and it is NOT a reason to predict STOP."
)

HEIGHT_ALERT_TEMPLATE = (
    " Height status: you are approximately {delta:.1f} m {relation} your starting "
    "position ({label})."
)


class VisualGraphMemoryDiagnostic:
    """Open-Nav revisit diagnostic + GTA-style topological node graph.

    The original diagnostic (``visit_count`` / ``revisit_score`` / ``candidate_novelty`` /
    ``loop_flag``) is preserved byte-for-byte so historical traces stay comparable. The
    GTA additions are new keys layered on top:

    ``G_topo``  (GTA arXiv:2602.15400 §IV-A-b)
        Each node = <metric position p_i, visit count c_i>. On a new pose, merge into the
        nearest node when d_min < delta_merge (paper: 0.8 m), else open a new node.
        A loop alert fires when the CURRENT node's count exceeds tau_loop.

    Two deliberate deviations from the paper, both documented:

    1. **3D merge distance, not the paper's 2D p_i in R^2.** In Habitat, y is up. A 2D
       (x, z) test would merge a landing with the floor directly above it, so a staircase
       would look like a loop. Full 3D distance costs nothing here and cannot merge
       across floors; within a floor the two are numerically the same.
    2. **Alert counts node visits, never raw steps within a radius.** The legacy
       ``visit_count`` counts every historical step inside ``revisit_radius``, so standing
       still for three steps already reads as "3 visits". Measured on the 100-episode
       clean_baseline_v1 trace, the legacy ``loop_flag`` fires on 41.2% of all steps --
       far too dense to inject as an alert. Node counting is what the paper specifies and
       what the threshold table in docs/ was calibrated against.

    ``alert_text`` is assembled here but has NO effect unless the caller is running with
    ``MEMORY_DIAGNOSTIC.LOG_ONLY=False``. With LOG_ONLY=True the field is logged only, so
    a disabled run stays byte-identical to clean_baseline_v1.
    """

    def __init__(
        self,
        revisit_radius: float = 1.0,
        merge_radius: float = 0.8,
        loop_alert_threshold: int = 3,
        vertical_alert_m: float = 0.3,
        enable_vertical_alert: bool = False,
        floor_height_m: float = 1.5,
    ) -> None:
        self.revisit_radius = revisit_radius
        self.merge_radius = merge_radius
        self.loop_alert_threshold = loop_alert_threshold
        self.vertical_alert_m = vertical_alert_m
        self.enable_vertical_alert = enable_vertical_alert
        self.floor_height_m = floor_height_m
        self.history = []
        self.nodes: List[Dict[str, Any]] = []
        self.start_height: Optional[float] = None
        self.current_node: Optional[Dict[str, Any]] = None

    def reset_episode(self) -> None:
        self.history = []
        self.nodes = []
        self.start_height = None
        self.current_node = None

    def _update_topological_graph(
        self, step_id: int, current: List[float]
    ) -> Dict[str, Any]:
        """GTA Eq. (4): cluster the new pose into G_topo and return the current node."""
        nearest = None
        nearest_distance = None
        for node in self.nodes:
            dist = _distance(current, node["position"])
            if nearest_distance is None or dist < nearest_distance:
                nearest_distance = dist
                nearest = node
        if nearest is not None and nearest_distance < self.merge_radius:
            n = nearest["count"]
            # running mean keeps the node centred on everywhere it was actually observed
            nearest["position"] = [
                (nearest["position"][i] * n + current[i]) / (n + 1) for i in range(3)
            ]
            nearest["count"] = n + 1
            nearest["last_step"] = step_id
            nearest["steps"].append(step_id)
            return nearest
        node = {
            "node_id": len(self.nodes),
            "position": list(current),
            "count": 1,
            "first_step": step_id,
            "last_step": step_id,
            "steps": [step_id],
            # ACN M1: floor index from the pose's y component, quantised. No model.
            "floor": int(round(current[1] / self.floor_height_m)),
            # ACN M1: directions that were OFFERED here. `taken` fills in once the agent
            # moves, so `offered - taken` = unexplored branches at this node. This is the
            # "return to the fork" substrate; it is NOT a backtracking policy (backtracking
            # measured 1.9% adoption / zero contribution -- do not reopen that).
            "offered_directions": set(),
            "taken_directions": set(),
        }
        self.nodes.append(node)
        return node

    def record_edges(
        self, node: Dict[str, Any], candidates: Iterable[Any], taken: Any = None
    ) -> None:
        """ACN M1 reachability: log which directions this node offered, and which was used."""
        for candidate in candidates or []:
            cid = getattr(candidate, "candidate_id", None)
            if cid is None and isinstance(candidate, dict):
                cid = candidate.get("candidate_id")
            if cid is not None:
                node["offered_directions"].add(str(cid))
        if taken is not None:
            node["taken_directions"].add(str(taken))

    def note_taken(self, direction_id: Any) -> None:
        """Called after the selector commits, so the CURRENT node records what was used.

        Uses ``self.current_node`` (the node this step merged into), never ``nodes[-1]``:
        after a merge the newest node in the list is not the one we are standing on.
        """
        if self.current_node is not None and direction_id is not None:
            self.current_node["taken_directions"].add(str(direction_id))

    def _build_alert_text(
        self, node: Dict[str, Any], height_delta: Optional[float]
    ) -> str:
        parts = []
        if node["count"] >= self.loop_alert_threshold:
            parts.append(
                LOOP_ALERT_TEMPLATE.format(
                    radius=self.merge_radius,
                    count=node["count"] - 1,
                    first=node["first_step"],
                    last=node["steps"][-2] if len(node["steps"]) >= 2 else node["first_step"],
                )
            )
        if (
            self.enable_vertical_alert
            and height_delta is not None
            and abs(height_delta) > self.vertical_alert_m
        ):
            parts.append(
                HEIGHT_ALERT_TEMPLATE.format(
                    delta=abs(height_delta),
                    relation="above" if height_delta > 0 else "below",
                    label="upstairs" if height_delta > 0 else "downstairs",
                )
            )
        return "".join(parts)

    def update(
        self,
        step_id: int,
        position: Any,
        heading: Any,
        candidates: Iterable[CandidateState],
    ) -> Dict[str, object]:
        current = _position_to_list(position)
        if current is None:
            return {
                "visit_count": 0,
                "nearest_previous_step": None,
                "revisit_score": None,
                "candidate_novelty": {},
                "loop_flag": False,
                "node_id": None,
                "node_visit_count": 0,
                "node_count": len(self.nodes),
                "loop_alert": False,
                "height_delta": None,
                "alert_text": "",
                "floor": None,
                "floor_count": len({n["floor"] for n in self.nodes}),
                "offered_directions": [],
                "taken_directions": [],
                "unexplored_branch_count": 0,
            }
        if self.start_height is None:
            self.start_height = current[1]

        nearest_step = None
        nearest_distance = None
        for record in self.history:
            dist = _distance(current, record["position"])
            if nearest_distance is None or dist < nearest_distance:
                nearest_distance = dist
                nearest_step = record["step_id"]

        visit_count = 1
        if nearest_distance is not None:
            visit_count += sum(
                1
                for record in self.history
                if _distance(current, record["position"]) <= self.revisit_radius
            )
        revisit_score = (
            math.exp(-nearest_distance)
            if nearest_distance is not None
            else 0.0
        )

        candidate_novelty = {}
        for candidate in candidates:
            next_position = _candidate_next_position(current, heading, candidate)
            if next_position is None or not self.history:
                candidate_novelty[candidate.candidate_id] = None
                continue
            min_dist = min(
                _distance(next_position, record["position"])
                for record in self.history
            )
            candidate_novelty[candidate.candidate_id] = 1.0 - math.exp(
                -min_dist
            )

        self.history.append({"step_id": step_id, "position": current})

        # --- GTA G_topo (added 20260805). Runs after the legacy block so every legacy
        # field keeps its exact previous value; these keys are purely additive. ---
        node = self._update_topological_graph(step_id, current)
        self.current_node = node
        self.record_edges(node, candidates)
        height_delta = (
            current[1] - self.start_height if self.start_height is not None else None
        )
        alert_text = self._build_alert_text(node, height_delta)
        unexplored = node["offered_directions"] - node["taken_directions"]

        return {
            "visit_count": visit_count,
            "nearest_previous_step": nearest_step,
            "revisit_score": revisit_score,
            "candidate_novelty": candidate_novelty,
            "loop_flag": bool(
                nearest_distance is not None
                and nearest_distance <= self.revisit_radius
            ),
            "node_id": node["node_id"],
            "node_visit_count": node["count"],
            "node_count": len(self.nodes),
            "loop_alert": bool(node["count"] >= self.loop_alert_threshold),
            "height_delta": height_delta,
            "alert_text": alert_text,
            # ACN M1 additions
            "floor": node["floor"],
            "floor_count": len({n["floor"] for n in self.nodes}),
            "offered_directions": sorted(node["offered_directions"]),
            "taken_directions": sorted(node["taken_directions"]),
            "unexplored_branch_count": len(unexplored),
        }

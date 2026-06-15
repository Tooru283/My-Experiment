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
        angle = heading_value + float(candidate.angle_rad)
        distance = float(candidate.distance)
        return [
            position[0] + distance * math.sin(angle),
            position[1],
            position[2] - distance * math.cos(angle),
        ]
    except Exception:
        return None


class VisualGraphMemoryDiagnostic:
    def __init__(self, revisit_radius: float = 1.0) -> None:
        self.revisit_radius = revisit_radius
        self.history = []

    def reset_episode(self) -> None:
        self.history = []

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
            }

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

        return {
            "visit_count": visit_count,
            "nearest_previous_step": nearest_step,
            "revisit_score": revisit_score,
            "candidate_novelty": candidate_novelty,
            "loop_flag": bool(
                nearest_distance is not None
                and nearest_distance <= self.revisit_radius
            ),
        }

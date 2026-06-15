import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if hasattr(value, "numel") and value.numel() == 1:
            value = value.item()
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class CandidateState:
    candidate_id: str
    direction_id: int
    angle_rad: Optional[float]
    angle_deg: Optional[float]
    distance: Optional[float]
    raw_rank: int
    image_ref: Optional[str] = None
    geometry_score: Optional[float] = None
    grounding_score: Optional[float] = None
    novelty_score: Optional[float] = None
    final_prompt_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    episode: Dict[str, Any]
    environment: Dict[str, Any]
    candidates: List[CandidateState]
    perception_state: Dict[str, Any]
    memory_state: Dict[str, Any]
    selector_output: Dict[str, Any]
    action_state: Dict[str, Any]
    verifier_output: Dict[str, Any]
    fallback_state: Dict[str, Any]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return data


def build_candidate_records(
    radius_dict: Dict[str, Any],
    distance_dict: Dict[str, Any],
    images_dict: Optional[Dict[str, Any]] = None,
) -> List[CandidateState]:
    candidates = []
    for raw_rank, candidate_id in enumerate(radius_dict.keys()):
        angle_rad = _to_float(radius_dict.get(candidate_id))
        angle_deg = math.degrees(angle_rad) if angle_rad is not None else None
        distance = _to_float(distance_dict.get(candidate_id))
        direction_id = int(candidate_id) if str(candidate_id).isdigit() else -1
        image_ref = None
        if images_dict is not None and candidate_id in images_dict:
            image_ref = "direction:{}".format(candidate_id)
        candidates.append(
            CandidateState(
                candidate_id=str(candidate_id),
                direction_id=direction_id,
                angle_rad=angle_rad,
                angle_deg=angle_deg,
                distance=distance,
                raw_rank=raw_rank,
                image_ref=image_ref,
            )
        )
    return candidates


def candidates_to_dicts(candidates: List[CandidateState]) -> List[Dict[str, Any]]:
    return [candidate.to_dict() for candidate in candidates]

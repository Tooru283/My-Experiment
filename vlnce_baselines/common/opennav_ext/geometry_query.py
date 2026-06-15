import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from vlnce_baselines.common.opennav_ext.agent_state import CandidateState


def _has_valid_depth(image_entry: Any) -> Optional[bool]:
    if not isinstance(image_entry, dict) or "depth" not in image_entry:
        return None
    depth = image_entry.get("depth")
    if depth is None:
        return False
    try:
        return bool(getattr(depth, "size", None))
    except Exception:
        return True


def _world_point(position: Any, heading: Any, candidate: CandidateState):
    if (
        position is None
        or candidate.angle_rad is None
        or candidate.distance is None
    ):
        return None
    try:
        heading_value = float(heading) if heading is not None else 0.0
        angle = heading_value + float(candidate.angle_rad)
        distance = float(candidate.distance)
        x = float(position[0]) + distance * math.sin(angle)
        y = float(position[1]) if len(position) > 1 else 0.0
        z = float(position[2]) - distance * math.cos(angle)
        return [x, y, z]
    except Exception:
        return None


@dataclass
class GeometryQueryResult:
    query_type: str
    candidate_id: str
    depth_valid: Optional[bool]
    projected_pixel: Optional[List[int]]
    world_point: Optional[List[float]]
    confidence: Optional[float]
    failure_reason: Optional[str]
    approx_world_point: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GeometryQueryLogger:
    def run(
        self,
        candidates: List[CandidateState],
        position: Any,
        heading: Any,
        images_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        images_dict = images_dict or {}
        for candidate in candidates:
            image_entry = images_dict.get(candidate.candidate_id)
            depth_valid = _has_valid_depth(image_entry)
            point = _world_point(position, heading, candidate)
            confidence_parts = [
                depth_valid is True,
                point is not None,
                candidate.angle_rad is not None,
                candidate.distance is not None,
            ]
            confidence = float(sum(confidence_parts)) / float(
                len(confidence_parts)
            )
            failure_reason = None
            if depth_valid is False:
                failure_reason = "missing_depth"
            elif point is None:
                failure_reason = "missing_pose_or_candidate_geometry"
            result = GeometryQueryResult(
                query_type="candidate_direction_depth",
                candidate_id=candidate.candidate_id,
                depth_valid=depth_valid,
                projected_pixel=None,
                world_point=point,
                confidence=confidence,
                failure_reason=failure_reason,
            )
            results.append(result.to_dict())
        return results

from typing import Any, Dict, List, Optional


def _distance_history(info: Dict[str, Any]) -> List[float]:
    try:
        distances = info.get("position", {}).get("distance", [])
    except Exception:
        return []
    try:
        return [float(value) for value in distances]
    except Exception:
        return []


def latest_distance_to_goal(info: Dict[str, Any]) -> Optional[float]:
    distances = _distance_history(info)
    if not distances:
        return None
    return distances[-1]


def selected_distance_gain(info: Dict[str, Any]) -> Optional[float]:
    distances = _distance_history(info)
    if len(distances) < 2:
        return None
    return distances[-2] - distances[-1]


def _collision_summary(info: Dict[str, Any]) -> Optional[float]:
    collisions = info.get("collisions")
    if collisions is None:
        return None
    try:
        if isinstance(collisions, dict) and "is_collision" in collisions:
            return float(bool(collisions["is_collision"]))
        if isinstance(collisions, (list, tuple)) and collisions:
            return float(collisions[-1])
        return float(collisions)
    except Exception:
        return None


def summarize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "steps_taken": info.get("steps_taken"),
        "distance_to_goal": latest_distance_to_goal(info),
        "distance_gain_selected": selected_distance_gain(info),
        "collision": _collision_summary(info),
    }


def summarize_step_outputs(outputs: Any) -> List[Dict[str, Any]]:
    summaries = []
    for output in outputs:
        if len(output) < 4:
            summaries.append({"raw_output_len": len(output)})
            continue
        _, _, done, info = output
        summary = summarize_info(info if isinstance(info, dict) else {})
        summary["done"] = bool(done)
        summaries.append(summary)
    return summaries

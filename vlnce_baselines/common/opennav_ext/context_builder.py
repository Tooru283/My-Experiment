from typing import Any, Dict, List

from vlnce_baselines.common.opennav_ext.agent_state import CandidateState


def _count(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except Exception:
        return 1


class ContextBuilder:
    def build_diagnostic(
        self,
        candidates: List[CandidateState],
        geometry: Any = None,
        grounding: Any = None,
        memory: Any = None,
        selected_candidate: Any = None,
    ) -> Dict[str, Any]:
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        selected = str(selected_candidate) if selected_candidate is not None else None
        return {
            "candidate_count": len(candidates),
            "candidate_ids": candidate_ids,
            "has_geometry": geometry is not None,
            "has_grounder": grounding is not None,
            "has_memory": memory is not None,
            "geometry_result_count": _count(geometry),
            "grounding_result_count": _count(grounding),
            "memory_keys": sorted(memory.keys()) if isinstance(memory, dict) else [],
            "selected_candidate": selected_candidate,
            "selected_in_candidates": selected in candidate_ids
            if selected is not None
            else None,
        }

import re
from typing import Dict, Iterable, List

from vlnce_baselines.common.opennav_ext.agent_state import CandidateState


def _split_constraints(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[\n,;]+", text)
    constraints = []
    for part in parts:
        cleaned = re.sub(r"^\s*[-*\d.)]+\s*", "", part).strip()
        if len(cleaned) >= 2:
            constraints.append(cleaned)
    return constraints


def _matches(constraint: str, text: str) -> bool:
    lowered = text.lower()
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9_]+", constraint.lower())
        if len(token) > 2
    ]
    if not tokens:
        return False
    return any(token in lowered for token in tokens)


class GrounderDiagnostic:
    def run(
        self,
        instruction: str,
        actions: str,
        landmarks: str,
        observe_dict: Dict[str, str],
        candidates: Iterable[CandidateState],
    ) -> List[Dict[str, object]]:
        del instruction, actions
        constraints = _split_constraints(landmarks)
        results = []
        for candidate in candidates:
            observation = observe_dict.get(candidate.candidate_id, "")
            matched = [
                constraint
                for constraint in constraints
                if _matches(constraint, observation)
            ]
            failed = [
                constraint
                for constraint in constraints
                if constraint not in matched
            ]
            total = max(len(constraints), 1)
            score = float(len(matched)) / float(total)
            results.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "grounding_score": score,
                    "matched_constraints": matched,
                    "failed_constraints": failed,
                    "score_breakdown": {
                        "object": score,
                        "room": 0.0,
                        "direction": 0.0,
                        "relation": 0.0,
                        "distance": 0.0,
                    },
                }
            )
        return results

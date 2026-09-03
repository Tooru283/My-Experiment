"""M0-M3 contracts for the RouteState-driven navigation pipeline.

These objects separate observable inputs, progress, decision data, emitted actions, and
observable action receipts.  They are intentionally pure Python and do not import
Habitat, Torch, PIL, or model clients.  M0-M3 define and validate the boundary only; it
does not change the trainer's action path.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PIPELINE_CONTRACT_SCHEMA_VERSION = "opennav.pipeline_contracts.v1"
FORBIDDEN_ONLINE_KEYS = frozenset(
    {
        "latest_goal_dist",
        "distance_to_goal",
        "distance_gain_selected",
        "recent_distance_gains",
        "success",
        "oracle_success",
        "spl",
        "ndtw",
        "sdtw",
        "oracle_metrics",
    }
)


class OracleFieldError(ValueError):
    """Raised when simulator-only data crosses an online contract boundary."""


def _primitive(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _primitive(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_primitive(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def forbidden_online_paths(value: Any, prefix: str = "") -> List[str]:
    """Return recursive paths of simulator-only keys in an online payload."""
    paths: List[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            current = "{}.{}".format(prefix, key) if prefix else key
            if key in FORBIDDEN_ONLINE_KEYS:
                paths.append(current)
            paths.extend(forbidden_online_paths(item, current))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(
                forbidden_online_paths(item, "{}[{}]".format(prefix, index))
            )
    return paths


def require_oracle_free(value: Any, contract_name: str = "online contract") -> None:
    paths = forbidden_online_paths(value)
    if paths:
        raise OracleFieldError(
            "{} contains simulator-only fields: {}".format(
                contract_name, ", ".join(paths)
            )
        )


class ContractMixin:
    """Serializable contract with a strict oracle-free boundary."""

    def to_dict(self) -> Dict[str, Any]:
        data = _primitive(asdict(self))
        require_oracle_free(data, type(self).__name__)
        return data


@dataclass(frozen=True)
class ObservationFrame(ContractMixin):
    episode_id: str
    step_id: int
    instruction: str
    plan_ref: Optional[str]
    position: Optional[Tuple[float, float, float]]
    heading: Optional[float]
    candidate_ids: Tuple[str, ...]
    candidate_observations: Dict[str, str]
    candidate_geometry: Dict[str, Dict[str, Optional[float]]]
    previous_action_receipt_id: Optional[str] = None
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class ProgressUpdate(ContractMixin):
    provider: str
    current_index: Optional[int]
    total: Optional[int]
    completed_indices: Tuple[int, ...]
    current_target: Optional[str]
    complete: Optional[bool]
    route_progress_complete: Optional[bool]
    terminal_target_confirmed: Optional[bool]
    goal_complete: Optional[bool]
    confidence: Optional[float]
    abstained: bool
    transition: str
    evidence_refs: Tuple[str, ...]
    display_text: str
    authoritative: bool = True
    shadow: Optional[Dict[str, Any]] = None
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class DecisionContext(ContractMixin):
    route_state_id: str
    current_target: Optional[str]
    verified_history: Tuple[str, ...]
    progress: ProgressUpdate
    candidate_observations: Dict[str, str]
    candidate_ids: Tuple[str, ...]
    landmark_summary: Dict[str, Any] = field(default_factory=dict)
    loop_summary: Dict[str, Any] = field(default_factory=dict)
    budget_summary: Dict[str, Any] = field(default_factory=dict)
    failure_summary: Dict[str, Any] = field(default_factory=dict)
    allowed_action_types: Tuple[str, ...] = ("move", "stop")
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class StopProposal(ContractMixin):
    source: str
    reason: str
    route_state_id: str
    evidence_refs: Tuple[str, ...] = ()
    requested: bool = True
    proposal_id: str = ""
    kind: str = "goal_stop"
    step_id: Optional[int] = None
    priority: int = 0
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class StopEvidenceItem(ContractMixin):
    evidence_id: str
    proposal_id: str
    evaluator: str
    verdict: str
    confidence: Optional[float]
    reason: str
    observable_inputs: Tuple[str, ...] = ()
    thresholds_version: str = ""
    evidence_refs: Tuple[str, ...] = ()
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class StopDecision(ContractMixin):
    decision_id: str
    route_state_id: str
    step_id: int
    outcome: str
    selected_proposal_id: Optional[str]
    reason: str
    consumed_evidence_ids: Tuple[str, ...] = ()
    rejected_proposal_ids: Tuple[str, ...] = ()
    fallback_required: bool = False
    policy_version: str = "coordinated_v1"
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class OverrideEntry(ContractMixin):
    override_id: str
    stage: str
    policy: str
    reason: str
    input_candidate: Optional[str]
    output_candidate: Optional[str]
    input_action_type: str
    output_action_type: str
    evidence_refs: Tuple[str, ...] = ()
    action_affecting: bool = True
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class ResolvedAction(ContractMixin):
    action_type: str
    route_state_id: str
    decision_record_id: str
    candidate_id: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None
    stop_decision_id: Optional[str] = None
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class ActionCommand(ContractMixin):
    command_id: str
    route_state_id: str
    decision_record_id: str
    action_code: int
    candidate_id: Optional[str]
    action_args: Optional[Dict[str, Any]]
    stop_requested: bool
    stop_decision_id: Optional[str] = None
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class DecisionRecord(ContractMixin):
    decision_record_id: str
    route_state_id: str
    step_id: int
    progress_provider: str
    planner_output: Dict[str, Any]
    stop_proposals: Tuple[StopProposal, ...]
    override_ledger: Tuple[OverrideEntry, ...]
    final_command: ActionCommand
    stop_evidence_items: Tuple[StopEvidenceItem, ...] = ()
    stop_decision: Optional[StopDecision] = None
    planner_skipped_reason: Optional[str] = None
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class ActionReceipt(ContractMixin):
    receipt_id: str
    command_id: str
    episode_id: str
    step_id: int
    executed_action: Dict[str, Any]
    candidate_id: Optional[str]
    pose_before: Optional[Tuple[float, float, float]] = None
    pose_after: Optional[Tuple[float, float, float]] = None
    collision: Optional[float] = None
    displacement: Optional[float] = None
    done: Optional[bool] = None
    steps_taken: Optional[int] = None
    observable_failure_events: Tuple[str, ...] = ()
    schema_version: str = PIPELINE_CONTRACT_SCHEMA_VERSION


def _position(value: Any) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return None


def build_observation_frame(
    *,
    episode_id: Any,
    step_id: int,
    instruction: Any,
    candidates: Iterable[Any],
    observe_dict: Optional[Mapping[Any, Any]],
    position: Any = None,
    heading: Any = None,
    plan_ref: Optional[str] = None,
    previous_action_receipt_id: Optional[str] = None,
) -> ObservationFrame:
    candidate_ids: List[str] = []
    geometry: Dict[str, Dict[str, Optional[float]]] = {}
    for candidate in candidates or []:
        if isinstance(candidate, Mapping):
            candidate_id = candidate.get("candidate_id")
            angle = candidate.get("angle_rad")
            distance = candidate.get("distance")
        else:
            candidate_id = getattr(candidate, "candidate_id", None)
            angle = getattr(candidate, "angle_rad", None)
            distance = getattr(candidate, "distance", None)
        if candidate_id is None:
            continue
        candidate_key = str(candidate_id)
        candidate_ids.append(candidate_key)
        geometry[candidate_key] = {
            "angle_rad": float(angle) if angle is not None else None,
            "distance": float(distance) if distance is not None else None,
        }
    observations = {
        str(key): str(value) for key, value in (observe_dict or {}).items()
    }
    frame = ObservationFrame(
        episode_id=str(episode_id),
        step_id=int(step_id),
        instruction=str(instruction or ""),
        plan_ref=str(plan_ref) if plan_ref is not None else None,
        position=_position(position),
        heading=float(heading) if heading is not None else None,
        candidate_ids=tuple(candidate_ids),
        candidate_observations=observations,
        candidate_geometry=geometry,
        previous_action_receipt_id=previous_action_receipt_id,
    )
    frame.to_dict()
    return frame



def build_action_receipt(
    *,
    receipt_id: str,
    command_id: str,
    episode_id: Any,
    step_id: int,
    executed_action: Mapping[str, Any],
    candidate_id: Any = None,
    step_result: Optional[Mapping[str, Any]] = None,
    pose_before: Any = None,
    pose_after: Any = None,
    displacement: Optional[float] = None,
    observable_failure_events: Sequence[str] = (),
) -> ActionReceipt:
    """Build a receipt from observable fields only; reject mixed oracle summaries."""
    require_oracle_free(executed_action, "executed_action")
    result = dict(step_result or {})
    require_oracle_free(result, "step_result")
    receipt = ActionReceipt(
        receipt_id=str(receipt_id),
        command_id=str(command_id),
        episode_id=str(episode_id),
        step_id=int(step_id),
        executed_action=_primitive(executed_action),
        candidate_id=str(candidate_id) if candidate_id is not None else None,
        pose_before=_position(pose_before),
        pose_after=_position(pose_after),
        collision=result.get("collision"),
        displacement=float(displacement) if displacement is not None else None,
        done=bool(result["done"]) if "done" in result else None,
        steps_taken=int(result["steps_taken"])
        if result.get("steps_taken") is not None
        else None,
        observable_failure_events=tuple(
            str(item) for item in observable_failure_events
        ),
    )
    receipt.to_dict()
    return receipt

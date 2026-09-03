"""Pure M3 STOP arbitration; it does not encode or execute environment actions."""

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from vlnce_baselines.common.opennav_ext.pipeline_contracts import (
    StopDecision,
    StopEvidenceItem,
    StopProposal,
    require_oracle_free,
)


STOP_COORDINATOR_POLICY_VERSION = "coordinated_v1"
GOAL_STOP = "goal_stop"
FORCED_TERMINATION = "forced_termination"
VALID_KINDS = frozenset((GOAL_STOP, FORCED_TERMINATION))
VALID_VERDICTS = frozenset(("support", "oppose", "abstain"))
SOURCE_PRIORITY = {
    "progress_completion": 300,
    "proactive_visual": 200,
    "selector": 100,
}


class StopCoordinationError(ValueError):
    """Raised when inputs violate the single-step STOP contract."""


def _policy_values(policy: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(policy or {})
    required = tuple(str(v) for v in values.get("required_evaluators", ("v2",)))
    opposing = set(str(v) for v in values.get("opposing_evaluators", required))
    if bool(values.get("depth_oppose_enabled", False)):
        opposing.add("depth")
    if bool(values.get("l4_action_effect", False)):
        if "l4" not in required:
            required += ("l4",)
        opposing.add("l4")
    return {
        "required_evaluators": required,
        "opposing_evaluators": frozenset(opposing),
        "abstain_outcome": str(values.get("abstain_outcome", "reject")),
        "policy_version": str(
            values.get("policy_version", STOP_COORDINATOR_POLICY_VERSION)
        ),
    }


def _validate_inputs(
    route_state_id: str,
    step_id: int,
    proposals: Sequence[StopProposal],
    evidence_items: Sequence[StopEvidenceItem],
) -> None:
    require_oracle_free(
        {
            "route_state_id": route_state_id,
            "proposals": [item.to_dict() for item in proposals],
            "evidence_items": [item.to_dict() for item in evidence_items],
        },
        "StopCoordinator inputs",
    )
    proposal_ids = []
    for proposal in proposals:
        if not proposal.proposal_id:
            raise StopCoordinationError("active StopProposal requires proposal_id")
        if proposal.route_state_id != route_state_id:
            raise StopCoordinationError("StopProposal route_state_id mismatch")
        if proposal.step_id != step_id:
            raise StopCoordinationError("StopProposal step_id mismatch")
        if proposal.kind not in VALID_KINDS:
            raise StopCoordinationError("unsupported STOP kind: {}".format(proposal.kind))
        proposal_ids.append(proposal.proposal_id)
    if len(set(proposal_ids)) != len(proposal_ids):
        raise StopCoordinationError("duplicate StopProposal proposal_id")
    known_ids = set(proposal_ids)
    evidence_ids = []
    for evidence in evidence_items:
        if evidence.proposal_id not in known_ids:
            raise StopCoordinationError("StopEvidenceItem references unknown proposal")
        if evidence.verdict not in VALID_VERDICTS:
            raise StopCoordinationError(
                "unsupported evidence verdict: {}".format(evidence.verdict)
            )
        if not evidence.evidence_id:
            raise StopCoordinationError("StopEvidenceItem requires evidence_id")
        evidence_ids.append(evidence.evidence_id)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise StopCoordinationError("duplicate StopEvidenceItem evidence_id")


def _goal_allowed(
    proposal: StopProposal,
    evidence: Sequence[StopEvidenceItem],
    policy: Mapping[str, Any],
) -> bool:
    by_evaluator = {item.evaluator: item for item in evidence}
    for evaluator in policy["required_evaluators"]:
        item = by_evaluator.get(evaluator)
        verdict = item.verdict if item is not None else "abstain"
        if verdict == "oppose":
            return False
        if verdict == "abstain" and policy["abstain_outcome"] == "reject":
            return False
        if verdict != "support" and policy["abstain_outcome"] != "allow":
            return False
    return not any(
        item.evaluator in policy["opposing_evaluators"]
        and item.verdict == "oppose"
        for item in evidence
    )


def _proposal_rank(proposal: StopProposal) -> Tuple[int, int, str]:
    return (
        SOURCE_PRIORITY.get(proposal.source, 0),
        int(proposal.priority),
        proposal.proposal_id,
    )


def build_m3_2_evidence_items(
    proposal: StopProposal,
    *,
    allowed: bool,
    verifier_result: Mapping[str, Any] = None,
    terminal_gate_result: Mapping[str, Any] = None,
    progress_update: Mapping[str, Any] = None,
    proactive_policy: Mapping[str, Any] = None,
) -> Tuple[StopEvidenceItem, ...]:
    """Adapt current gates into explicit evidence without changing M3.2 policy."""
    verifier = dict(verifier_result or {})
    terminal = dict(terminal_gate_result or {})
    progress = dict(progress_update or {})
    proactive = dict(proactive_policy or {})
    base = proposal.proposal_id
    legacy = StopEvidenceItem(
        evidence_id=base + ":legacy_combined",
        proposal_id=base,
        evaluator="legacy_combined",
        verdict="support" if allowed else "oppose",
        confidence=None,
        reason=(
            "Existing V2/depth/L4 gate chain allowed STOP."
            if allowed
            else "Existing V2/depth/L4 gate chain rejected STOP."
        ),
        evidence_refs=(proposal.source,),
        thresholds_version="m3_2_legacy_gate_adapter",
    )
    raw_v2 = str(verifier.get("verdict") or "").lower()
    v2_verdict = (
        "support" if raw_v2 == "allow"
        else "oppose" if raw_v2 == "reject"
        else "abstain"
    )
    v2 = StopEvidenceItem(
        evidence_id=base + ":v2",
        proposal_id=base,
        evaluator="v2",
        verdict=v2_verdict,
        confidence=verifier.get("confidence"),
        reason=str(verifier.get("reason") or "V2 did not return a verdict."),
        observable_inputs=(
            "final_target_visible={}".format(
                bool(verifier.get("final_target_visible"))
            ),
            "arrival_evidence={}".format(
                bool(verifier.get("arrival_evidence"))
            ),
        ),
        evidence_refs=(proposal.source + ":visual_target_verifier",),
        thresholds_version="v2_runtime",
    )
    raw_l4 = str(terminal.get("verdict") or "").lower()
    l4_verdict = (
        "support" if raw_l4 == "allow"
        else "oppose" if raw_l4 == "block"
        else "abstain"
    )
    l4 = StopEvidenceItem(
        evidence_id=base + ":l4",
        proposal_id=base,
        evaluator="l4",
        verdict=l4_verdict,
        confidence=None,
        reason=str(terminal.get("reason") or "L4 did not return a verdict."),
        observable_inputs=(
            "decision_effect={}".format(
                bool(terminal.get("decision_effect_enabled"))
            ),
        ),
        evidence_refs=(proposal.source + ":terminal_gate",),
        thresholds_version="l4_runtime",
    )
    progress_complete = progress.get("goal_complete") is True
    progress_item = StopEvidenceItem(
        evidence_id=base + ":progress",
        proposal_id=base,
        evaluator="progress",
        verdict="support" if progress_complete else "abstain",
        confidence=progress.get("confidence"),
        reason=(
            "Route progress and terminal target are both confirmed."
            if progress_complete
            else "Goal completion is not confirmed."
        ),
        observable_inputs=(
            "provider={}".format(progress.get("provider") or "unknown"),
            "current_index={}".format(progress.get("current_index")),
            "total={}".format(progress.get("total")),
        ),
        evidence_refs=(proposal.source + ":progress_update",),
        thresholds_version="acn_l1_runtime",
    )
    items = [legacy, v2, l4, progress_item]
    if proposal.source == "proactive_visual":
        depth_ok = proactive.get("depth_ok")
        depth = StopEvidenceItem(
            evidence_id=base + ":depth",
            proposal_id=base,
            evaluator="depth",
            verdict=(
                "support" if depth_ok is True
                else "oppose" if depth_ok is False
                else "abstain"
            ),
            confidence=None,
            reason=(
                "Depth veto passed." if depth_ok is True
                else "Depth veto blocked proactive STOP." if depth_ok is False
                else "Depth result was unavailable."
            ),
            observable_inputs=(
                "depth_ok={}".format(depth_ok),
            ),
            evidence_refs=(proposal.source + ":depth_stop_veto",),
            thresholds_version="depth_runtime",
        )
        persistent = bool(proactive.get("persistent_visual_confirm"))
        e3_allow = bool(proactive.get("e3_b_allow"))
        v2_allow = bool(proactive.get("v2_allow"))
        policy_pass = bool(
            depth_ok is True
            and ((persistent and v2_allow) or e3_allow)
        )
        proactive_item = StopEvidenceItem(
            evidence_id=base + ":proactive_policy",
            proposal_id=base,
            evaluator="proactive_policy",
            verdict="support" if policy_pass else "oppose",
            confidence=None,
            reason=(
                "Proactive persistence/E3 commit policy passed."
                if policy_pass
                else "Proactive persistence/E3 commit policy did not pass."
            ),
            observable_inputs=(
                "persistent_visual_confirm={}".format(persistent),
                "v2_allow={}".format(v2_allow),
                "e3_b_allow={}".format(e3_allow),
                "depth_ok={}".format(depth_ok),
            ),
            evidence_refs=(proposal.source + ":commit_policy",),
            thresholds_version="proactive_commit_runtime",
        )
        items.extend((depth, proactive_item))
    items = tuple(items)
    for item in items:
        item.to_dict()
    return items


class StopCoordinator:
    """Resolve all STOP intents for one route-state revision exactly once."""

    def __init__(self, policy: Mapping[str, Any] = None):
        self.policy = _policy_values(policy or {})

    def resolve(
        self,
        *,
        route_state_id: str,
        step_id: int,
        proposals: Iterable[StopProposal],
        evidence_items: Iterable[StopEvidenceItem],
        movement_candidate_ids: Iterable[str],
        decision_id: str = None,
    ) -> StopDecision:
        active = tuple(item for item in proposals if item.requested)
        evidence = tuple(evidence_items)
        movements = tuple(str(item) for item in movement_candidate_ids if item is not None)
        _validate_inputs(route_state_id, int(step_id), active, evidence)
        decision_id = decision_id or "{}:stop".format(route_state_id)
        by_proposal = {
            proposal.proposal_id: tuple(
                item for item in evidence if item.proposal_id == proposal.proposal_id
            )
            for proposal in active
        }
        consumed = tuple(item.evidence_id for item in evidence)
        goals = tuple(item for item in active if item.kind == GOAL_STOP)
        forced = tuple(item for item in active if item.kind == FORCED_TERMINATION)
        allowed = tuple(
            item
            for item in goals
            if _goal_allowed(item, by_proposal[item.proposal_id], self.policy)
        )
        rejected = tuple(
            item.proposal_id for item in goals if item not in allowed
        )

        if allowed:
            selected = max(allowed, key=_proposal_rank)
            return StopDecision(
                decision_id=decision_id,
                route_state_id=route_state_id,
                step_id=int(step_id),
                outcome="commit_goal_stop",
                selected_proposal_id=selected.proposal_id,
                reason=selected.reason,
                consumed_evidence_ids=consumed,
                rejected_proposal_ids=rejected,
                policy_version=self.policy["policy_version"],
            )
        if not movements and (forced or goals):
            selected = max(forced, key=_proposal_rank) if forced else None
            return StopDecision(
                decision_id=decision_id,
                route_state_id=route_state_id,
                step_id=int(step_id),
                outcome="commit_forced_termination",
                selected_proposal_id=(selected.proposal_id if selected else None),
                reason=(
                    selected.reason
                    if selected
                    else "No legal movement remained after goal STOP rejection."
                ),
                consumed_evidence_ids=consumed,
                rejected_proposal_ids=rejected,
                policy_version=self.policy["policy_version"],
            )
        if forced:
            raise StopCoordinationError(
                "forced_termination is illegal while movement candidates exist"
            )
        if goals:
            return StopDecision(
                decision_id=decision_id,
                route_state_id=route_state_id,
                step_id=int(step_id),
                outcome="reject",
                selected_proposal_id=None,
                reason="Required goal-stop evidence did not pass.",
                consumed_evidence_ids=consumed,
                rejected_proposal_ids=rejected,
                fallback_required=True,
                policy_version=self.policy["policy_version"],
            )
        return StopDecision(
            decision_id=decision_id,
            route_state_id=route_state_id,
            step_id=int(step_id),
            outcome="no_request",
            selected_proposal_id=None,
            reason="No STOP proposal was requested.",
            policy_version=self.policy["policy_version"],
        )

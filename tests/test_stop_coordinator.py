import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contracts = load_module(
    "vlnce_baselines.common.opennav_ext.pipeline_contracts",
    "vlnce_baselines/common/opennav_ext/pipeline_contracts.py",
)
coordinator_module = load_module(
    "stop_coordinator_under_test",
    "vlnce_baselines/common/opennav_ext/stop_coordinator.py",
)


class StopCoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.coordinator = coordinator_module.StopCoordinator()

    def proposal(self, proposal_id="p1", source="progress_completion", kind="goal_stop", **kwargs):
        return contracts.StopProposal(
            proposal_id=proposal_id,
            source=source,
            kind=kind,
            reason=kwargs.pop("reason", "stop requested"),
            route_state_id=kwargs.pop("route_state_id", "state-1"),
            step_id=kwargs.pop("step_id", 4),
            **kwargs
        )

    def evidence(self, verdict="support", proposal_id="p1", evaluator="v2", **kwargs):
        return contracts.StopEvidenceItem(
            evidence_id=kwargs.pop("evidence_id", "e1"),
            proposal_id=proposal_id,
            evaluator=evaluator,
            verdict=verdict,
            confidence=kwargs.pop("confidence", 0.9),
            reason=kwargs.pop("reason", verdict),
            **kwargs
        )

    def resolve(self, proposals=(), evidence=(), movements=("7",)):
        return self.coordinator.resolve(
            route_state_id="state-1",
            step_id=4,
            proposals=proposals,
            evidence_items=evidence,
            movement_candidate_ids=movements,
        )

    def test_no_request_with_movement(self):
        result = self.resolve()
        self.assertEqual(result.outcome, "no_request")
        self.assertFalse(result.fallback_required)

    def test_goal_with_required_support_commits(self):
        result = self.resolve((self.proposal(),), (self.evidence(),))
        self.assertEqual(result.outcome, "commit_goal_stop")
        self.assertEqual(result.selected_proposal_id, "p1")

    def test_goal_with_oppose_rejects_and_requires_fallback(self):
        result = self.resolve(
            (self.proposal(),), (self.evidence("oppose"),)
        )
        self.assertEqual(result.outcome, "reject")
        self.assertTrue(result.fallback_required)
        self.assertEqual(result.rejected_proposal_ids, ("p1",))

    def test_missing_required_evidence_is_abstain_and_rejects(self):
        result = self.resolve((self.proposal(),), ())
        self.assertEqual(result.outcome, "reject")

    def test_rejected_goal_without_movement_is_forced_termination(self):
        result = self.resolve(
            (self.proposal(),), (self.evidence("oppose"),), movements=()
        )
        self.assertEqual(result.outcome, "commit_forced_termination")
        self.assertIsNone(result.selected_proposal_id)
        self.assertIn("p1", result.rejected_proposal_ids)

    def test_explicit_forced_termination_requires_empty_movement(self):
        forced = self.proposal(
            proposal_id="forced-1",
            source="empty_action_space",
            kind="forced_termination",
        )
        result = self.resolve((forced,), (), movements=())
        self.assertEqual(result.outcome, "commit_forced_termination")
        self.assertEqual(result.selected_proposal_id, "forced-1")
        with self.assertRaises(coordinator_module.StopCoordinationError):
            self.resolve((forced,), (), movements=("7",))

    def test_fixed_source_priority_selects_progress(self):
        proactive = self.proposal("p-pro", "proactive_visual")
        progress = self.proposal("p-progress", "progress_completion")
        result = self.resolve(
            (proactive, progress),
            (
                self.evidence(proposal_id="p-pro", evidence_id="e-pro"),
                self.evidence(proposal_id="p-progress", evidence_id="e-progress"),
            ),
        )
        self.assertEqual(result.selected_proposal_id, "p-progress")

    def test_l4_is_required_only_when_action_effect_enabled(self):
        coordinator = coordinator_module.StopCoordinator({"l4_action_effect": True})
        proposal = self.proposal()
        result = coordinator.resolve(
            route_state_id="state-1",
            step_id=4,
            proposals=(proposal,),
            evidence_items=(self.evidence(),),
            movement_candidate_ids=("7",),
        )
        self.assertEqual(result.outcome, "reject")
        result = coordinator.resolve(
            route_state_id="state-1",
            step_id=4,
            proposals=(proposal,),
            evidence_items=(
                self.evidence(),
                self.evidence(
                    proposal_id="p1", evaluator="l4", evidence_id="e-l4"
                ),
            ),
            movement_candidate_ids=("7",),
        )
        self.assertEqual(result.outcome, "commit_goal_stop")

    def test_state_step_and_reference_mismatches_fail(self):
        with self.assertRaises(coordinator_module.StopCoordinationError):
            self.resolve((self.proposal(route_state_id="other"),), ())
        with self.assertRaises(coordinator_module.StopCoordinationError):
            self.resolve((self.proposal(step_id=5),), ())
        with self.assertRaises(coordinator_module.StopCoordinationError):
            self.resolve(
                (self.proposal(),),
                (self.evidence(proposal_id="unknown"),),
            )

    def test_progress_evidence_requires_goal_complete(self):
        proposal = self.proposal()
        common = {
            "provider": "acn_l1", "current_index": 3, "total": 3,
            "complete": True, "route_progress_complete": True,
            "terminal_target_confirmed": False, "goal_complete": False,
        }
        items = coordinator_module.build_m3_2_evidence_items(
            proposal, allowed=True, verifier_result={"verdict": "allow"},
            progress_update=common,
        )
        progress = {item.evaluator: item for item in items}["progress"]
        self.assertEqual(progress.verdict, "abstain")

        common.update(terminal_target_confirmed=True, goal_complete=True)
        items = coordinator_module.build_m3_2_evidence_items(
            proposal, allowed=True, verifier_result={"verdict": "allow"},
            progress_update=common,
        )
        progress = {item.evaluator: item for item in items}["progress"]
        self.assertEqual(progress.verdict, "support")

    def test_trainer_wires_m3_2_as_authoritative_goal_stop_path(self):
        source = (
            ROOT / "vlnce_baselines/common/base_il_trainer_llm.py"
        ).read_text(encoding="utf-8")
        self.assertIn("resolve_m3_goal_stop", source)
        self.assertIn("pipeline_stop_proposal", source)
        self.assertIn("pipeline_stop_evidence", source)
        self.assertIn("pipeline_stop_decision", source)
        self.assertIn("stop_decision=m3_stop_decision", source)
        self.assertNotIn("resolved_legacy_pipeline", source)
        self.assertNotIn("test_decision_after_stop_rejection", source)
        self.assertIn("stop_rejected_no_second_selector", source)
        self.assertIn("and not m3_preplanner_stop_decided", source)

    def test_runtime_policy_v2_oppose_blocks_legacy_support(self):
        coordinator = coordinator_module.StopCoordinator({
            "required_evaluators": ("legacy_combined", "v2"),
            "opposing_evaluators": ("legacy_combined", "v2"),
        })
        proposal = self.proposal()
        decision = coordinator.resolve(
            route_state_id="state-1", step_id=4, proposals=(proposal,),
            evidence_items=(
                self.evidence(evaluator="legacy_combined", evidence_id="legacy"),
                self.evidence("oppose", evaluator="v2", evidence_id="v2"),
            ), movement_candidate_ids=("7",),
        )
        self.assertEqual(decision.outcome, "reject")

    def test_trainer_uses_single_v2_and_proactive_is_diagnostic_only(self):
        source = (ROOT / "vlnce_baselines/common/base_il_trainer_llm.py").read_text(encoding="utf-8")
        self.assertIn("_route_endgame = bool(", source)
        self.assertEqual(source.count("record_visual_target_verifier("), 2)
        self.assertIn("proactive_terminal_hint", source)
        self.assertIn("proactive_stop_requested = False", source)

    def test_m3_2_adapter_exposes_diagnostics_without_changing_policy(self):
        proposal = self.proposal(source="proactive_visual")
        items = coordinator_module.build_m3_2_evidence_items(
            proposal,
            allowed=True,
            verifier_result={
                "verdict": "allow",
                "confidence": 0.9,
                "final_target_visible": True,
                "arrival_evidence": True,
            },
            terminal_gate_result={
                "verdict": "block",
                "reason": "chain incomplete",
                "decision_effect_enabled": False,
            },
            progress_update={
                "provider": "acn_l1",
                "current_index": 1,
                "total": 4,
                "complete": False,
            },
            proactive_policy={
                "persistent_visual_confirm": False,
                "v2_allow": True,
                "e3_b_allow": False,
                "depth_ok": True,
            },
        )
        by_name = {item.evaluator: item for item in items}
        self.assertEqual(
            set(by_name),
            {"legacy_combined", "v2", "l4", "progress", "depth", "proactive_policy"},
        )
        self.assertEqual(by_name["legacy_combined"].verdict, "support")
        self.assertEqual(by_name["v2"].verdict, "support")
        self.assertEqual(by_name["l4"].verdict, "oppose")
        self.assertEqual(by_name["progress"].verdict, "abstain")
        self.assertEqual(by_name["depth"].verdict, "support")
        self.assertEqual(by_name["proactive_policy"].verdict, "oppose")
        self.assertIn(
            "persistent_visual_confirm=False",
            by_name["proactive_policy"].observable_inputs,
        )
        coordinator = coordinator_module.StopCoordinator({
            "required_evaluators": ("legacy_combined",),
            "opposing_evaluators": ("legacy_combined",),
        })
        decision = coordinator.resolve(
            route_state_id="state-1", step_id=4, proposals=(proposal,),
            evidence_items=items, movement_candidate_ids=("7",),
        )
        self.assertEqual(decision.outcome, "commit_goal_stop")

    def test_trainer_has_single_action_compiler_and_no_direct_emitters(self):
        source = (
            ROOT / "vlnce_baselines/common/base_il_trainer_llm.py"
        ).read_text(encoding="utf-8")
        self.assertIn("action_compiler.compile(", source)
        self.assertIn("action_compiler.to_env_action", source)
        self.assertIn("resolve_m3_forced_termination", source)
        self.assertNotIn("forced termination conflicts with existing StopDecision", source)
        self.assertIn("proposals=tuple(m3_stop_proposals)", source)
        self.assertIn("cumulative_actual_displacement_m", source)
        self.assertIn('"required_evaluators": ("v2", "progress")', source)
        self.assertIn("pipeline_action_command", source)
        self.assertNotIn("env_actions.append", source)
        self.assertNotIn("{\"action\": {\"action\": 0", source)
        self.assertNotIn("{\"action\": {\"action\": 4", source)

    def test_oracle_evidence_is_rejected(self):
        evidence = self.evidence(
            observable_inputs=({"distance_to_goal": 1.0},)
        )
        with self.assertRaises(contracts.OracleFieldError):
            self.resolve((self.proposal(),), (evidence,))


if __name__ == "__main__":
    unittest.main()

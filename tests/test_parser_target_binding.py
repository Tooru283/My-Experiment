"""EP10 target-binding regressions, including completion and formal STOP gates."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "vlnce_baselines/common/opennav_ext" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_anchor_chain = load("target_binding_parser_test", "anchor_chain.py").build_anchor_chain
ConstraintQueueLocator = load("target_binding_locator_test", "progress_locator.py").ConstraintQueueLocator
contracts = load("vlnce_baselines.common.opennav_ext.pipeline_contracts", "pipeline_contracts.py")
load("vlnce_baselines.common.opennav_ext.landmark_matching", "landmark_matching.py")
VisualTargetVerifier = load("target_binding_verifier_test", "visual_target_verifier.py").VisualTargetVerifier
StopCoordinator = load("target_binding_coordinator_test", "stop_coordinator.py").StopCoordinator


class ParserTargetBindingTest(unittest.TestCase):
    def requirements(self, action):
        return build_anchor_chain(action, "closet\nbed room\nbath room\nbedroom")[
            "anchors"
        ][0]["action_spec"]["requirements"]

    def test_ep371_reference_uses_terminal_clause_not_longer_route_landmark(self):
        terminal = "Wait by the door at the top of the stairs"
        policy = build_anchor_chain(terminal,
            "dining room\nkitchen air\nshort flight of stairs\ndoor at the top of the stairs"
        )["terminal_policy"]
        self.assertEqual(policy["target"], "door at the top of the stairs")
        self.assertEqual(policy["directives"], [terminal])
        self.assertEqual(policy["relations"], [{
            "type": "at_top_of", "reference": "stairs", "subject": "terminal_target",
        }])

    def test_ep1092_agent_relation_can_reference_primary_target(self):
        policy = build_anchor_chain(
            "Stop in front of door on right", "black door\nstairs\ndoor on right"
        )["terminal_policy"]
        self.assertEqual(policy["target"], "door on right")
        self.assertEqual(policy["relations"], [{
            "type": "in_front_of", "reference": "door on right", "subject": "agent",
        }])

    def test_relation_does_not_borrow_reference_from_following_clause(self):
        policy = build_anchor_chain(
            "Stop in the doorway to the left of it, next to the red chair",
            "doorway\nred chair",
        )["terminal_policy"]
        self.assertIsNone(policy["relations"][0]["reference"])
        self.assertEqual(policy["relations"][1]["reference"], "red chair")
        result = self.verify(policy, ["doorway", "red chair"])
        self.assertTrue(result["terminal_relation_evidence"]["unresolved_reference"])
        self.assertNotEqual(result["verdict"], "allow")

    def test_agent_position_and_nested_entity_relations_keep_distinct_subjects(self):
        policy = build_anchor_chain(
            "Stop in front of the door at the top of the stairs, next to the red chair",
            "stairs\nred chair\ndoor at the top of the stairs",
        )["terminal_policy"]
        self.assertEqual(policy["relations"], [
            {"type": "in_front_of", "reference": "door", "subject": "agent"},
            {"type": "at_top_of", "reference": "stairs", "subject": "terminal_target"},
            {"type": "next_to", "reference": "red chair", "subject": "agent"},
        ])

    def test_relation_preserves_reference_modifiers_and_whole_words(self):
        policy = build_anchor_chain(
            "Stop in the doorway to the left of the red armchair",
            "arm\nchair\ndoorway",
        )["terminal_policy"]
        self.assertEqual(policy["relations"][0]["reference"], "red armchair")

    def test_repeated_relation_keeps_each_reference(self):
        policy = build_anchor_chain(
            "Stop by the table next to the red chair, next to the blue sofa",
            "table\nchair\nsofa",
        )["terminal_policy"]
        self.assertEqual([r["reference"] for r in policy["relations"]],
                         ["red chair", "blue sofa"])

    def test_shared_reference_does_not_become_unresolved_after_deduplication(self):
        policy = build_anchor_chain(
            "Stop in front of the sofa, next to the sofa", "sofa"
        )["terminal_policy"]
        result = self.verify(policy, ["sofa"])
        report = result["terminal_relation_evidence"]
        self.assertEqual(report["required_references"], ["sofa"])
        self.assertFalse(report["unresolved_reference"])
        self.assertTrue(report["satisfied"])

    def test_ep116_keeps_turn_exit_and_elided_enter_targets(self):
        for destination in ("bedroom", "bed room", "bath room"):
            with self.subTest(destination=destination):
                self.assertEqual(self.requirements(
                    "Turn right to walk out of closet into a " + destination
                ), [
                    {"type": "turn", "direction": "right"},
                    {"type": "exit", "target": "closet"},
                    {"type": "enter", "target": destination.replace(" ", "")},
                ])

    def test_exit_into_synonyms_keep_both_targets(self):
        for verb in ("Walk out of", "Go out of", "Leave", "Exit"):
            with self.subTest(verb=verb):
                self.assertEqual(self.requirements(verb + " the closet into the bedroom"), [
                    {"type": "exit", "target": "closet"},
                    {"type": "enter", "target": "bedroom"},
                ])

    def test_missing_exit_target_cannot_borrow_enter_destination(self):
        self.assertEqual(self.requirements("Walk out of it into the bedroom"), [
            {"type": "exit", "target": None}, {"type": "enter", "target": "bedroom"},
        ])

    def test_later_room_modifier_does_not_replace_exit_object(self):
        self.assertEqual(self.requirements("Leave the closet beside the bedroom"), [
            {"type": "exit", "target": "closet"},
        ])

    def test_looking_into_another_room_is_not_an_elided_enter(self):
        self.assertEqual(self.requirements("Exit the closet and look into the bedroom"), [
            {"type": "exit", "target": "closet"},
        ])

    def test_compound_completion_requires_exit_and_enter_with_real_movement(self):
        action = "Turn right to walk out of closet into a bedroom"
        for after_room, displacement, collision, expected in (
            ("bedroom", 1.0, False, True),
            ("hallway", 1.0, False, False),
            ("closet", 1.0, False, False),
            ("bedroom", 0.0, False, False),
            ("bedroom", 1.0, True, False),
        ):
            with self.subTest(after_room=after_room, displacement=displacement, collision=collision):
                locator = ConstraintQueueLocator(enable_location=True)
                locator.reset_episode(build_anchor_chain(action, "closet\nbedroom")["anchors"])
                locator.update(1, [0, 0, 0], 0, {str(i): ["closet"] for i in range(12)}, {})
                result = locator.update(2, [displacement, 0, 0], -math.pi / 2,
                    {str(i): [after_room] for i in range(12)}, {},
                    {"displacement": displacement, "collision": collision})
                self.assertEqual(result["complete"], expected)

    def verify(self, policy, visible):
        evidence = {
            "parsed": {"candidates": [{
                "candidate_id": "__current_view__", "visible_landmarks": visible,
                "matched_instruction_terms": visible, "missing_instruction_terms": [],
                "final_target_visible": True, "arrival_evidence": True,
                "target_direction_id": 0, "confidence": 0.95,
            }]},
            "requested_candidate_ids": ["__current_view__"],
            "total_candidate_ids": ["__current_view__"], "sampled_all": True,
            "parse_error": None, "schema_error": None,
        }
        return VisualTargetVerifier().verify(
            "selector_stop_gate", policy["directives"][0], policy["directives"][0],
            policy["target"], "completed", "history", [], evidence, True,
            "selector proposed STOP", "__current_view__", 8, "current_pano", policy,
        )

    def test_fixed_references_reach_the_visual_gate(self):
        for text, landmarks, reference in (
            ("Wait by the door at the top of the stairs", "short flight of stairs\ndoor at the top of the stairs", "stairs"),
            ("Stop in front of door on right", "black door\nstairs\ndoor on right", "door on right"),
        ):
            with self.subTest(text=text):
                policy = build_anchor_chain(text, landmarks)["terminal_policy"]
                result = self.verify(policy, [policy["target"], reference])
                self.assertTrue(result["terminal_relation_evidence"]["satisfied"])
                self.assertEqual(result["verdict"], "allow")
                self.assertFalse(result["terminal_relation_evidence"]["metric_geometry_verified"])

    def test_missing_reference_still_rejects_formal_stop(self):
        policy = build_anchor_chain(
            "Stop in the area between the two white sofas, next to the entrance to the dining room",
            "two white sofas\nentrance to the dining room\narea between the two white sofas",
        )["terminal_policy"]
        for include_reference in (False, True):
            with self.subTest(include_reference=include_reference):
                visible = [policy["target"], "two white sofas"]
                if include_reference:
                    visible.append("entrance to the dining room")
                result = self.verify(policy, visible)
                evidence = [contracts.StopEvidenceItem(
                    evidence_id=name, proposal_id="p", evaluator=name,
                    verdict=({"allow": "support", "reject": "oppose", "uncertain": "abstain"}[result["verdict"]]
                             if name == "v2" else "support"), reason="test evidence", confidence=None,
                ) for name in ("legacy_combined", "v2", "progress")]
                decision = StopCoordinator({
                    "required_evaluators": ("legacy_combined", "v2", "progress"),
                }).resolve(route_state_id="state", step_id=8, proposals=[contracts.StopProposal(
                    proposal_id="p", source="selector", kind="goal_stop", reason="test",
                    route_state_id="state", step_id=8,
                )], evidence_items=evidence, movement_candidate_ids=["1"])
                self.assertEqual(decision.outcome,
                    "commit_goal_stop" if include_reference else "reject")


if __name__ == "__main__":
    unittest.main()

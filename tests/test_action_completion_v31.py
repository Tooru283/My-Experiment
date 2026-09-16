"""Behavioral regressions from the 2026-09-07 EP10, without Habitat/LLM imports."""
import math
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_module(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_anchor_chain = load_module(
    "action_v31_parser_test", "vlnce_baselines/common/opennav_ext/anchor_chain.py"
).build_anchor_chain
ConstraintQueueLocator = load_module(
    "action_v31_locator_test", "vlnce_baselines/common/opennav_ext/progress_locator.py"
).ConstraintQueueLocator

guidance = load_module(
    "vlnce_baselines.common.opennav_ext.navigation_guidance",
    "vlnce_baselines/common/opennav_ext/navigation_guidance.py",
)
prompts = load_module("action_prompts_test", "vlnce_baselines/common/navigator/prompts.py")


class ActionCompletionV31Test(unittest.TestCase):
    def locator(self, text, landmarks=""):
        locator = ConstraintQueueLocator(enable_location=True)
        locator.reset_episode(build_anchor_chain(text, landmarks)["anchors"])
        return locator

    def test_exit_synonyms_and_target_scope(self):
        for text in ("Walk out of the closet", "Go out of closet", "Leave the closet", "Exit the closet"):
            with self.subTest(text=text):
                a = build_anchor_chain(text, "closet")["anchors"][0]
                self.assertEqual(a["action_spec"]["requirements"], [{"type": "exit", "target": "closet"}])
        a = build_anchor_chain("Leave the bedroom and walk into the kitchen", "kitchen\nbedroom")["anchors"][0]
        self.assertEqual(a["action_spec"]["requirements"], [
            {"type": "exit", "target": "bedroom"}, {"type": "enter", "target": "kitchen"},
        ])

    def test_region_traversal_is_distinct_from_opening_crossing(self):
        for text, landmarks, expected in (
            ("Walk through the kitchen to the hallway", "kitchen\nhallway", "traverse"),
            ("Go through the kitchen doorway", "kitchen doorway\nkitchen", "cross"),
            ("Go through the door", "door", "cross"),
        ):
            with self.subTest(text=text):
                a = build_anchor_chain(text, landmarks)["anchors"][0]
                self.assertEqual(a["action_spec"]["requirements"][0]["type"], expected)

    def test_missing_pass_target_is_not_borrowed_from_later_destination(self):
        a = build_anchor_chain("Walk past it to the stairs", "stairs")["anchors"][0]
        self.assertIsNone(a["action_spec"]["requirements"][0]["target"])

    def test_living_area_alias_does_not_lose_enter_target(self):
        a = build_anchor_chain("Go into the living area with the fireplace", "living area with the fireplace")["anchors"][0]
        self.assertEqual(a["action_spec"]["requirements"], [{"type": "enter", "target": "living room"}])

    def test_turn_into_room_keeps_both_requirements(self):
        a = build_anchor_chain("Take a left into the bedroom", "bedroom")["anchors"][0]
        self.assertEqual(a["action_spec"]["requirements"], [
            {"type": "turn", "direction": "left"}, {"type": "enter", "target": "bedroom"},
        ])

    def test_crossing_kitchen_door_does_not_target_kitchen(self):
        a = build_anchor_chain("Go through the kitchen door", "kitchen\ndoor")["anchors"][0]
        self.assertEqual(a["action_spec"]["requirements"], [{"type": "cross", "target": "door"}])

    def test_room_head_noun_cannot_make_dining_room_a_living_room(self):
        loc = self.locator("Enter the living room", "living room")
        self.assertFalse(loc._tag_matches_room(["dining room", "room"], "living room", "room"))
        self.assertTrue(loc._tag_matches_room(["living room;"], "living room", "room"))

    def test_spatial_destination_is_not_rewritten_as_turn(self):
        a = build_anchor_chain("Go to the left of the stairs", "stairs")["anchors"][0]
        self.assertEqual(a["action_spec"]["spatial_relations"], ["left of"])
        self.assertNotIn("turn", [r["type"] for r in a["action_spec"]["requirements"]])

    def test_chain_completed_at_step_two_survives_next_eight_updates(self):
        loc = self.locator("Find the sofa", "sofa")
        tags, geometry = {"0": ["sofa"]}, {"0": {"distance": 2}}
        loc.update(1, [0, 0, 0], 0, tags, geometry)
        loc.update(2, [0, 0, 0], 0, tags, geometry)
        for step in range(3, 11):
            state = loc.update(step, [0, 0, 0], 0, {}, {})
            self.assertTrue(state["complete"])
            self.assertFalse(state["abstained_this_step"])
            self.assertEqual(len(state["completion_events"]), 1)

    def test_turn_sign_matches_actual_left_positive_controller(self):
        for direction, delta in (("left", math.radians(48)), ("right", math.radians(-54))):
            with self.subTest(direction=direction):
                loc = self.locator("Turn " + direction)
                loc.update(1, [0, 0, 0], 0, {}, {})
                self.assertTrue(loc.update(2, [0, 0, 0], delta, {}, {})["complete"])
                loc = self.locator("Turn " + direction)
                loc.update(1, [0, 0, 0], 0, {}, {})
                self.assertFalse(loc.update(2, [0, 0, 0], -delta, {}, {})["complete"])

    def test_left_turn_across_pi_boundary(self):
        loc = self.locator("Turn left")
        loc.update(1, [0, 0, 0], math.radians(170), {}, {})
        self.assertTrue(loc.update(2, [0, 0, 0], math.radians(-140), {}, {})["complete"])

    def test_first_movement_after_switch_is_counted(self):
        loc = self.locator("Find the sofa\nTurn left", "sofa")
        tags, geometry = {"0": ["sofa"]}, {"0": {"distance": 2}}
        loc.update(1, [0, 0, 0], 0, tags, geometry)
        loc.update(2, [0, 0, 0], 0, tags, geometry)
        state = loc.update(3, [0, 0, 0], math.radians(48), {}, {})
        self.assertTrue(state["complete"])
        self.assertEqual(state["completion_events"][-1]["event_interval"], [2, 3])

    def test_forward_does_not_borrow_old_movement_or_accept_backwards(self):
        loc = self.locator("Find the sofa\nWalk forward", "sofa")
        tags, geometry = {"0": ["sofa"]}, {"0": {"distance": 2}}
        loc.update(1, [0, 0, 0], 0, tags, geometry)
        loc.update(2, [0, 0, -2], 0, tags, geometry)
        self.assertEqual(loc.update(3, [0, 0, -2], 0, {}, {})["j"], 1)
        self.assertEqual(loc.update(4, [0, 0, -0.5], 0, {}, {})["j"], 1)
        self.assertTrue(loc.update(5, [0, 0, -3.5], 0, {}, {})["complete"])


    def test_surface_motion_does_not_depend_on_landmark_extraction(self):
        anchor = build_anchor_chain("Walk across the floor", "archway")["anchors"][0]
        self.assertEqual(anchor["kind"], "motion")
        self.assertEqual(anchor["verification"]["predicate"], "odometry_displacement")
        loc = self.locator("Walk across the floor", "archway")
        self.assertFalse(loc.update(1, [0, 0, 0], 0, {}, {})["complete"])
        self.assertFalse(loc.update(2, [0.4, 0, 0], 0, {}, {})["complete"])
        self.assertTrue(loc.update(3, [0.6, 0, 0], 0, {}, {})["complete"])

    def test_same_id_after_rotation_is_not_comparable(self):
        loc = self.locator("Exit the bedroom", "bedroom")
        loc.update(1, [0, 0, 0], 0, {"0": ["bedroom"]}, {})
        state = loc.update(2, [0, 0, -1], math.pi / 2, {"0": ["hallway"]}, {})
        self.assertFalse(state["complete"])
        self.assertIn("room_observations_not_comparable", state["last_evaluation"]["missing_evidence"])

    def test_changed_id_with_same_world_bearing_can_be_compared(self):
        loc = self.locator("Exit the bedroom", "bedroom")
        loc.update(1, [0, 0, 0], 0, {"0": ["bedroom"]}, {})
        state = loc.update(2, [0, 0, -1], math.pi / 2, {"9": ["hallway"]}, {})
        self.assertTrue(state["complete"])
        self.assertEqual(state["last_evaluation"]["requirements"][0]["matched_direction_pairs"], [["0", "9"]])

    def test_missing_room_label_alone_does_not_prove_exit(self):
        loc = self.locator("Exit the bedroom", "bedroom")
        loc.update(1, [0, 0, 0], 0, {"0": ["bedroom"]}, {})
        state = loc.update(2, [0, 0, -1], 0, {"0": ["floor", "wall"]}, {})
        self.assertFalse(state["complete"])
        self.assertEqual(state["last_evaluation"]["status"], "unknown")
        self.assertIn("room_state_unobserved", state["last_evaluation"]["missing_evidence"])

    def test_room_transition_cannot_borrow_movement_from_an_earlier_step(self):
        loc = self.locator("Exit the bedroom", "bedroom")
        loc.update(1, [0, 0, 0], 0, {"0": ["bedroom"]}, {})
        loc.update(2, [0, 0, -1], 0, {"0": ["bedroom"]}, {})
        state = loc.update(3, [0, 0, -1], 0, {"0": ["hallway"]}, {})
        self.assertFalse(state["complete"])
        self.assertIn("effective_movement_missing", state["last_evaluation"]["missing_evidence"])

    def test_noun_sighting_does_not_complete_cross_or_pass(self):
        for text in ("Go through the door", "Walk past the door"):
            loc = self.locator(text, "door")
            for step in range(1, 5):
                state = loc.update(step, [0, 0, -step], 0, {"0": ["door"]}, {"0": {"distance": 1}})
                self.assertFalse(state["complete"])
                self.assertTrue(any(m.startswith("verifier_not_implemented:") for m in state["last_evaluation"]["missing_evidence"]))

    def test_feedback_exposes_uncertainty_and_excludes_stale_evaluation(self):
        progress = {"current_index": 1, "route_progress_complete": False, "current_action": {
            "raw": "Walk past the piano", "requirements": [{"type": "pass", "target": "piano"}],
            "evaluation": {"anchor_index": 1, "status": "unknown", "missing_evidence": ["verifier_not_implemented:pass"]},
        }}
        text = guidance.format_navigation_feedback(progress)
        self.assertNotIn("verifier_not_implemented:pass", text)
        self.assertIn("unsupported_requirements", text)
        self.assertIn("Physically continue the named route action", text)
        progress["current_action"]["evaluation"]["anchor_index"] = 0
        text = guidance.format_navigation_feedback(progress)
        self.assertNotIn("verifier_not_implemented:pass", text)
        self.assertIn("not_evaluated", text)

    def test_prompt_format_remains_compatible_with_existing_selector(self):
        text = prompts.NAVIGATOR["user"].format("0, 3, STOP", 2, "instruction", "actions", "landmarks", "history", "progress", "observations")
        self.assertIn("Prediction:", text)
        self.assertIn("Verified Route Progress: progress", text)

    def test_decomposition_cache_changes_with_prompt_and_model(self):
        original = guidance.action_cache_filename("model-a", "local", {"system": "old"}, {})
        self.assertEqual(original, guidance.action_cache_filename("model-a", "local", {"system": "old"}, {}))
        self.assertNotEqual(original, guidance.action_cache_filename("model-b", "local", {"system": "old"}, {}))
        self.assertNotEqual(original, guidance.action_cache_filename("model-a", "local", {"system": "new"}, {}))

    def test_actual_selector_method_sends_feedback_in_its_existing_call(self):
        root = Path(__file__).resolve().parents[1]
        source = ast.parse((root / "vlnce_baselines/common/navigator/spatialNavigator.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Open_Nav")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "move_to_next_vp")
        namespace = {
            "NAVIGATOR": prompts.NAVIGATOR,
            "STOP_CANDIDATE": "STOP",
            "format_navigation_feedback": guidance.format_navigation_feedback,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "selector_method", "exec"), namespace)
        calls = []
        def infer(system, user, **kwargs):
            calls.append((system, user))
            return "Thought: The listed view leads left. Prediction: 2"
        nav = SimpleNamespace(llm=SimpleNamespace(gpt_infer=infer), _parse_prediction=lambda text, ids: text.strip())
        progress = {"current_index": 0, "current_action": {"raw": "Turn left", "requirements": [{"type": "turn", "direction": "left"}], "evaluation": {"anchor_index": 0, "status": "incomplete", "missing_evidence": ["required_turn_not_reached"]}}}
        result = namespace["move_to_next_vp"](
            nav, SimpleNamespace(info=lambda *a: None), 2, "instruction", "actions", "landmarks", "history",
            "progress", "observation", {"2": "left view"}, return_prompt=True, progress_feedback=progress,
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("required_turn_not_reached", calls[0][1])
        self.assertEqual(result[0], ["2"])
        self.assertEqual(result[3], calls[0][1])


if __name__ == "__main__":
    unittest.main()

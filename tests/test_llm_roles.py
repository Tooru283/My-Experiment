"""Actual navigator methods exercised without Habitat, GPUs, or model requests."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from typing import Dict, Iterable
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guidance = load_file("vlnce_baselines.common.opennav_ext.navigation_guidance",
                     "vlnce_baselines/common/opennav_ext/navigation_guidance.py")
parsing = load_file("vlnce_baselines.common.opennav_ext.instruction_parsing",
                    "vlnce_baselines/common/opennav_ext/instruction_parsing.py")
prompts = load_file("llm_roles_prompts", "vlnce_baselines/common/navigator/prompts.py")
NAV_SOURCE = ast.parse((ROOT / "vlnce_baselines/common/navigator/spatialNavigator.py").read_text())
NAV_CLASS = next(n for n in NAV_SOURCE.body if isinstance(n, ast.ClassDef) and n.name == "Open_Nav")
namespace = dict(vars(guidance), **{k: v for k, v in vars(prompts).items() if not k.startswith("__")})
namespace.update(STOP_CANDIDATE="STOP", MOVE_BACK_CANDIDATE="MOVE_BACK", re=re, math=math,
                 json=json, time=time, parse_actions=parsing.parse_actions,
                 parse_landmarks=parsing.parse_landmarks)
exec(compile(ast.Module(body=[NAV_CLASS], type_ignores=[]), "actual_navigator", "exec"), namespace)
Navigator = namespace["Open_Nav"]
LOGGER = SimpleNamespace(info=lambda *args, **kwargs: None)


def no_llm(*args, **kwargs):
    raise AssertionError("Unexpected auxiliary model request")


class LLMRolesTest(unittest.TestCase):
    def nav(self, infer=no_llm):
        nav = Navigator.__new__(Navigator)
        nav.llm = SimpleNamespace(gpt_infer=infer)
        return nav

    def test_only_plan_and_selector_contain_text_calls(self):
        callers = [method.name for method in NAV_CLASS.body if isinstance(method, ast.FunctionDef)
                   for node in ast.walk(method) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "gpt_infer"]
        self.assertEqual(callers, ["_extract_plan_stage", "move_to_next_vp"])

    def test_two_stage_plan_preserves_terminal_reference(self):
        calls = []
        plan = {
            "actions": ["Go through the door", "Turn left", "Go to the left of the stairs",
                        "Stop in the doorway to the left of the white double doors"],
            "landmarks": ["door", "stairs", "white double doors", "doorway"],
        }
        def infer(system, user, **kwargs):
            calls.append((system, user, kwargs))
            return json.dumps({"actions": plan["actions"]} if len(calls) == 1 else
                              {"landmarks": plan["landmarks"], "terminal_target": "doorway"})
        instruction = ". ".join(plan["actions"])
        nav = self.nav(infer)
        actions, landmarks = nav.get_plan(instruction)
        self.assertEqual(len(calls), 2)
        self.assertIn(instruction, calls[0][1])
        self.assertIn(instruction, calls[1][1])
        self.assertIn(json.dumps(plan["actions"]), calls[1][1])
        self.assertEqual(nav.last_plan_metadata["llm_calls"], 2)
        self.assertEqual(actions.splitlines(), plan["actions"])
        self.assertEqual(landmarks.splitlines(), plan["landmarks"])
        self.assertEqual(landmarks.splitlines()[-1], "doorway")
        self.assertEqual(calls[0][2]["temperature"], 0)

    def test_invalid_plan_is_rejected_not_guessed(self):
        for raw in ("Go forward", '[]', '{"actions":[]}',
                    '{"actions":[],"landmarks":[]}',
                    '{"actions":[true],"landmarks":[]}',
                    '{"actions":["Turn left"],"landmarks":"left"}',
                    '{"actions":["Turn left"],"landmarks":[],"complete":true}',
                    '{"actions":["Turn left\\nStop"],"landmarks":[]}',
                    'Thought: plan\n{"actions":["Turn left"],"landmarks":[]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.nav(lambda *a, **kw: raw).get_plan("Turn left")

    def test_single_fenced_json_and_no_landmark_instruction(self):
        actions, landmarks = guidance.parse_instruction_plan(
            '```json\n{"actions":["Turn left"],"landmarks":[]}\n```')
        self.assertEqual((actions, landmarks), ("Turn left", ""))

    def test_two_stage_cache_fingerprints_both_prompts_and_schema(self):
        old = guidance.action_cache_filename("qwen", "local", {"system": "old"}, {})
        new = guidance.action_cache_filename("qwen", "local", prompts.ACTION_DETECTION,
                                             prompts.LANDMARK_DETECTION)
        self.assertNotEqual(old, new)
        changed = guidance.action_cache_filename("qwen", "local", prompts.ACTION_DETECTION,
                                                 {"system": "changed"})
        self.assertNotEqual(new, changed)

    def test_ep513_split_stop_retried_before_landmark_stage(self):
        instruction = "Go into the living area with the fireplace. Stop in the area between the two white sofas, next to the entrance to the dining room."
        route = "Go into the living area with the fireplace"
        terminal = "Stop in the area between the two white sofas, next to the entrance to the dining room"
        good_landmarks = ["living area with the fireplace", "two white sofas", "entrance to the dining room", "area between the two white sofas"]
        replies = iter([
            {"actions": [route, "Stop in the area between the two white sofas", "Stop next to the entrance to the dining room"]},
            {"actions": [route, terminal]},
            {"landmarks": good_landmarks, "terminal_target": good_landmarks[-1]},
        ])
        calls = []
        def infer(system, user, **kw):
            calls.append((system, user))
            return json.dumps(next(replies))
        nav = self.nav(infer)
        actions, landmarks = nav.get_plan(instruction)
        self.assertEqual(actions.splitlines(), [route, terminal])
        self.assertEqual(nav.last_plan_metadata["llm_calls"], 3)
        self.assertIn("STOP/wait count changed", calls[1][1])
        chain = load_file("llm_roles_anchor_chain", "vlnce_baselines/common/opennav_ext/anchor_chain.py").build_anchor_chain(actions, landmarks)
        self.assertEqual(chain["terminal_policy"]["directives"], [terminal])
        self.assertEqual(chain["terminal_policy"]["target"], "area between the two white sofas")
        matching = load_file("llm_roles_landmark_matching", "vlnce_baselines/common/opennav_ext/landmark_matching.py")
        self.assertEqual(matching.final_landmark_terms(landmarks), [chain["terminal_policy"]["target"]])

    def test_redundant_wait_at_same_terminal_may_collapse(self):
        instruction = ("Go straight past the pool. Stop when you get to the corner "
                       "of the bar. That's where you will wait.")
        parsed = parsing.parse_actions(json.dumps({"actions": [
            "Go straight past the pool",
            "Stop when you get to the corner of the bar",
        ]}), instruction)
        self.assertEqual(parsed[-1], "Stop when you get to the corner of the bar")
        separate = parsing.parse_actions(json.dumps({"actions": [
            "Go straight past the pool",
            "Stop when you get to the corner of the bar",
            "That's where you will wait",
        ]}), instruction)
        chain = load_file("llm_roles_anaphoric_wait_chain",
            "vlnce_baselines/common/opennav_ext/anchor_chain.py").build_anchor_chain(
                separate, ["pool", "corner of the bar"])
        self.assertEqual(chain["n_anchors"], 1)
        self.assertEqual(len(chain["terminal_policy"]["directives"]), 2)
        self.assertEqual(chain["terminal_policy"]["target"], "corner of the bar")
        self.assertEqual(parsing.primary_terminal_place(instruction), "corner of the bar")

    def test_invalid_action_stage_never_calls_landmark_stage(self):
        calls = []
        def infer(system, user, **kw):
            calls.append(system)
            return '{"actions":["Stop by the door"]}'
        nav = self.nav(infer)
        with self.assertRaisesRegex(ValueError, "action_extraction failed after 2 attempts"):
            nav.get_plan("Turn left")
        self.assertEqual(calls, [prompts.ACTION_DETECTION["system"]] * 2)

    def test_landmark_reference_is_deterministically_rebound_to_primary_target(self):
        instruction = "Stop in the doorway to the room."
        replies = iter([
            {"actions": ["Stop in the doorway to the room"]},
            {"landmarks": ["room"], "terminal_target": "room"},
        ])
        nav = self.nav(lambda *a, **kw: json.dumps(next(replies)))
        actions, landmarks = nav.get_plan(instruction)
        self.assertEqual(landmarks.splitlines(), ["room", "doorway to the room"])
        self.assertEqual(nav.last_plan_metadata["llm_calls"], 2)
        self.assertTrue(nav.last_plan_metadata["stages"][1]["valid"])

    def test_terminal_qualifiers_and_joint_target_preserved(self):
        for instruction, invalid_action in [
            ("Wait by the door at the top of the stairs.", "Wait by the door"),
            ("Stop in the doorway to the left of the white double doors.", "Stop in the doorway to the right of the white double doors"),
            ("Wait near the tub and sink.", "Wait near the tub"),
        ]:
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                parsing.parse_actions(json.dumps({"actions": [invalid_action]}), instruction)
        target = "tub and sink"
        self.assertEqual(parsing.parse_landmarks(json.dumps({"landmarks": ["tub", "sink", target],
                         "terminal_target": target}), "Wait near the tub and sink."), ["tub", "sink", target])

    def test_terminal_reference_words_must_survive(self):
        with self.assertRaisesRegex(ValueError, "reference/modifier"):
            parsing.parse_landmarks('{"landmarks":["doorway"],"terminal_target":"doorway"}',
                "Stop in the doorway to the left of the white double doors.")

    def test_temporal_stop_binds_immediately_preceding_route_destination(self):
        instruction = (
            "Go through living room, through the door on the to the right, "
            "through the den, through the dining to, to the outdoor foyer. "
            "Stop before going outside."
        )
        landmarks = [
            "living room", "door on the to the right", "den", "dining to",
            "outdoor foyer",
        ]
        parsed = parsing.parse_landmarks(json.dumps({
            "landmarks": landmarks, "terminal_target": "outdoor foyer",
        }), instruction)
        replies = iter([
            {"actions": [
                "Go through living room", "Go through the door on the to the right",
                "Go through the den", "Go through the dining to",
                "Go to the outdoor foyer", "Stop before going outside",
            ]},
            {"landmarks": landmarks, "terminal_target": "outdoor foyer"},
        ])
        nav = self.nav(lambda *args, **kwargs: json.dumps(next(replies)))
        actual_actions, actual_landmarks = nav.get_plan(instruction)
        self.assertEqual(nav.last_plan_metadata["llm_calls"], 2)
        self.assertEqual(actual_actions.splitlines()[-1], "Stop before going outside")
        self.assertEqual(actual_landmarks.splitlines(), landmarks)
        self.assertEqual(parsing.primary_terminal_place(instruction), "outdoor foyer")
        self.assertEqual(parsed, landmarks)
        chain = load_file(
            "llm_roles_temporal_stop_chain",
            "vlnce_baselines/common/opennav_ext/anchor_chain.py",
        ).build_anchor_chain(
            ["Go to the outdoor foyer", "Stop before going outside"], parsed
        )
        self.assertEqual(chain["terminal_policy"]["target"], "outdoor foyer")

    def test_temporal_stop_does_not_borrow_an_earlier_landmark(self):
        instruction = "Pass the table, then turn right. Stop before going outside."
        response = '{"landmarks":["table"],"terminal_target":null}'
        self.assertIsNone(parsing.primary_terminal_place(instruction))
        self.assertEqual(parsing.parse_landmarks(response, instruction), ["table"])
        with self.assertRaisesRegex(ValueError, "must be null"):
            parsing.parse_landmarks(
                '{"landmarks":["table"],"terminal_target":"table"}', instruction
            )

    def test_spatial_before_keeps_literal_terminal_place(self):
        instruction = "Walk down the hallway. Stop before the stairs."
        response = '{"landmarks":["stairs","before the stairs"],"terminal_target":"before the stairs"}'
        self.assertEqual(parsing.primary_terminal_place(instruction), "before the stairs")
        self.assertEqual(
            parsing.parse_landmarks(response, instruction),
            ["stairs", "before the stairs"],
        )

    def test_no_stop_and_fenced_stage_outputs(self):
        nav = self.nav(lambda system, *a, **kw: '```json\n{"actions":["Turn left"]}\n```'
                       if system == prompts.ACTION_DETECTION["system"] else
                       '{"landmarks":[],"terminal_target":null}')
        self.assertEqual(nav.get_plan("Turn left"), ("Turn left", ""))

    def test_embedded_terminal_and_independent_verbs_are_separated_losslessly(self):
        instruction = "Leave the bedroom and take a left in the hallway. Enter the room and stop in the doorway."
        parsed = parsing.parse_actions(json.dumps({"actions": [
            "Leave the bedroom and take a left in the hallway", "Enter the room and stop in the doorway",
        ]}), instruction)
        self.assertEqual(parsed, ["Leave the bedroom", "take a left in the hallway",
                                  "Enter the room", "stop in the doorway"])
        compound = "Turn right to walk past stairs"
        self.assertEqual(parsing.parse_actions(json.dumps({"actions": [compound]}), compound), [compound])
        conditional = "Walk forward until you reach the sofa and turn left"
        self.assertEqual(parsing.parse_actions(json.dumps({"actions": [conditional]}), conditional), [conditional])
        leave = "Leave the bathroom and closet"
        self.assertEqual(
            parsing.parse_actions(json.dumps({"actions": [leave]}), leave),
            ["Leave the bathroom", "Leave the closet"],
        )
        approach = "Go toward the house and through the sliding doors"
        self.assertEqual(
            parsing.parse_actions(json.dumps({"actions": [approach]}), approach),
            ["Go toward the house", "Go through the sliding doors"],
        )



    def test_ep100_terminal_grammar_is_normalized_from_source(self):
        cases = (
            ("Take the doorway directly left of the staircase and wait there.",
             "doorway directly left of the staircase"),
            ("Walk into the TV room. Stop once you enter the room.", "TV room"),
            ("Walk through the bathroom to the other door. You will then be in a gym. Stop when you get into this room and wait.", "gym"),
            ("Wait right at the bathroom door.", "bathroom door"),
            ("Stand on the platform at the end of the pool near the lounge chairs.", "platform at the end of the pool"),
            ("Stop on the rug before you pass the first bed.", "rug"),
        )
        for instruction, expected in cases:
            with self.subTest(instruction=instruction):
                self.assertEqual(parsing.primary_terminal_place(instruction), expected)

    def test_intermediate_stop_does_not_replace_later_route_destination(self):
        instruction = ("Turn left and go forward and stop next to the bed. "
                       "Then turn left and go into the bathroom.")
        actions = parsing.parse_actions(json.dumps({"actions": [
            "Turn left and go forward and stop next to the bed",
            "Then turn left and go into the bathroom",
        ]}), instruction)
        self.assertEqual(actions, ["Turn left", "go forward", "stop next to the bed",
                                   "Then turn left", "go into the bathroom"])
        self.assertIsNone(parsing.primary_terminal_place(instruction))
        chain = load_file("llm_roles_intermediate_stop_chain",
            "vlnce_baselines/common/opennav_ext/anchor_chain.py").build_anchor_chain(
                actions, ["bed", "bathroom"])
        self.assertFalse(chain["terminal_policy"]["present"])
        self.assertEqual(chain["terminal_policy"]["intermediate_directives"],
                         ["stop next to the bed"])
        self.assertEqual(chain["terminal_policy"]["target"], "bathroom")

    def test_full_target_description_normalized_without_losing_reference(self):
        instruction = "Stop in the doorway to the left of the white double doors."
        verbose = "doorway to the left of the white double doors"
        parsed = parsing.parse_landmarks(json.dumps({"landmarks": [verbose],
                                         "terminal_target": verbose}), instruction)
        self.assertEqual(parsed, ["white double doors", "doorway"])
        chain = load_file("llm_roles_verbose_chain", "vlnce_baselines/common/opennav_ext/anchor_chain.py").build_anchor_chain(
            [instruction], parsed)
        self.assertEqual(chain["terminal_policy"]["target"], parsed[-1])

    def receipt(self, candidate="3", step=7):
        return {"step_id": step, "candidate_id": candidate, "displacement": 0.0,
                "collision": True, "distance_to_goal": 1.2,
                "executed_action": {"action": {"action": 4,
                    "action_args": {"angle": 1.57, "distance": 2.0, "goal_distance": 1.2}}}}

    def test_history_does_not_convert_intent_or_command_to_execution(self):
        nav = self.nav()
        history = nav.save_history(LOGGER, 7, "3", "I will pass the door.",
                                   "Scene Description: door ahead Scene Objects: door; wall", [],
                                   action_receipt=self.receipt())
        item = history[0]
        self.assertEqual(item["observation_phase"], "before_action")
        self.assertEqual(item["thought_role"], "unverified_selection_rationale")
        self.assertEqual(item["execution"]["command_distance_m"], 2.0)
        self.assertEqual(item["execution"]["measured_displacement_m"], 0.0)
        text = nav.review_history(LOGGER, history)
        self.assertIn("Step 7", text)
        self.assertNotIn("Step 1", text)
        self.assertIn("not post-action evidence", text)
        self.assertIn("not semantic completion", text)
        self.assertNotIn("distance_to_goal", text)
        self.assertNotIn("goal_distance", text)

    def test_missing_or_mismatched_receipt_is_unknown_not_zero(self):
        for receipt in (None, {}, self.receipt("4"), self.receipt(step=6)):
            item = guidance.build_navigation_history_item(7, "3", "", "", receipt)
            self.assertFalse(item["execution"]["receipt_available"])
            self.assertIsNone(item["execution"]["measured_displacement_m"])

    def test_backtrack_history_needs_no_camera_parser_or_llm(self):
        history = self.nav().save_history(LOGGER, 7, "MOVE_BACK", "Try another branch", "", [],
                                         self.receipt("MOVE_BACK"))
        self.assertTrue(history[0]["synthetic_action"])
        self.assertTrue(history[0]["execution"]["receipt_available"])

    def test_long_description_does_not_erase_ram_tags(self):
        item = guidance.build_navigation_history_item(7, "3", "x" * 2000,
                    "Scene Description: " + "x" * 4000 + " Scene Objects: door; stairs")
        self.assertIn("door; stairs", item["observation"])
        self.assertLess(len(item["observation"]), 700)
        self.assertLessEqual(len(item["thought"]), 240)

    def test_disagreement_uses_stable_vote_without_auxiliary_llm(self):
        nav = self.nav()
        fused = nav.thought_fusion(LOGGER, ["3", "1", "1"], ["left", "forward", "forward again"])
        decision = nav.test_decisions(LOGGER, fused, "obs", "instruction", 0, {"1": "front", "3": "left"})
        self.assertEqual(decision[:2], ("1", "forward"))
        self.assertEqual(nav.last_test_decision_metadata["decision_source"], "deterministic_vote")
        self.assertEqual(nav.last_test_decision_metadata["llm_calls"], 0)
        tied = nav.thought_fusion(LOGGER, ["3", "1"], ["left", "forward"])
        self.assertEqual(list(tied), ["3", "1"])

    def test_candidate_validation_does_not_mutate_input_or_offer_backtrack(self):
        nav = self.nav()
        fused = {"MOVE_BACK": "recover", "99": "invalid", "3": "left"}
        result = nav.test_decisions(LOGGER, fused, "", "", 0, {"3": "left"})
        self.assertEqual(result[0], "3")
        self.assertEqual(len(fused), 3)
        result = nav.test_decisions(LOGGER, fused, "", "", 0, {"3": "left"}, offer_move_back=True)
        self.assertEqual(result[0], "MOVE_BACK")

    def test_empty_predictions_fall_back_without_llm(self):
        nav = self.nav()
        result = nav.test_decisions(LOGGER, {}, "", "", 0, {"3": "left"})
        self.assertEqual(result, ("3", "left", 1))
        self.assertTrue(nav.last_test_decision_metadata["fallback_used"])

    def test_stop_is_only_a_selected_proposal(self):
        nav = self.nav()
        result = nav.test_decisions(LOGGER, {"STOP": "request"}, "", "", 0, {"3": "left"})
        self.assertEqual(result[0], "STOP")
        self.assertNotIn("goal_complete", nav.last_test_decision_metadata)
        self.assertIn("STOP is a proposal", prompts.NAVIGATOR["system"])

    def test_visual_prompts_keep_same_time_and_metric_boundaries(self):
        source = ast.parse((ROOT / "vlnce_baselines/common/opennav_ext/visual_evidence.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "VisualEvidenceLogger")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in {"_build_prompt", "_build_current_view_prompt"}]
        constants = [n for n in source.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in {"VISUAL_EVIDENCE_SCOPE", "STOP_CURRENT_VIEW_CANDIDATE_ID"}
                             for t in n.targets)]
        scope = {"Dict": Dict, "Iterable": Iterable, "CandidateState": object,
                 "_candidate_id": lambda c: str(c.candidate_id)}
        exec(compile(ast.Module(body=constants + methods, type_ignores=[]), "actual_visual_prompts", "exec"), scope)
        owner = SimpleNamespace(compact_json=True, metadata_observation_chars=300)
        candidate = SimpleNamespace(candidate_id="3", angle_deg=90, distance=2)
        texts = [scope["_build_prompt"](owner, "instruction", "actions", "landmarks", [candidate], {})]
        owner.compact_json = False
        texts.append(scope["_build_prompt"](owner, "instruction", "actions", "landmarks", [candidate], {}))
        texts.append(scope["_build_current_view_prompt"](owner, "instruction", "actions", "landmarks", "obs", ["3"]))
        for text in texts:
            self.assertTrue(text.startswith(scope["VISUAL_EVIDENCE_SCOPE"]))
            self.assertIn("NOT target distance", text)
            self.assertIn("not images from", text)
            self.assertIn('"target_direction_id":', text)
            self.assertNotIn('"target_direction_id":"string|null"', text)


if __name__ == "__main__":
    unittest.main()

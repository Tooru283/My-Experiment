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


anchor_module = load_module(
    "anchor_chain_v2_under_test",
    "vlnce_baselines/common/opennav_ext/anchor_chain.py",
)
locator_module = load_module(
    "progress_locator_v2_under_test",
    "vlnce_baselines/common/opennav_ext/progress_locator.py",
)
build_anchor_chain = anchor_module.build_anchor_chain
ConstraintQueueLocator = locator_module.ConstraintQueueLocator


class AnchorChainV2Test(unittest.TestCase):
    def test_terminal_directive_is_not_a_progress_anchor(self):
        chain = build_anchor_chain(
            "1. Go to the living room\n2. Find the sofa\n3. Stop",
            "1. living room\n2. sofa",
        )
        self.assertEqual(chain["schema_version"], "opennav.anchor_chain.v3")
        self.assertEqual(chain["n_anchors"], 2)
        self.assertEqual([a["kind"] for a in chain["anchors"]], ["location", "object"])
        self.assertEqual(chain["terminal_policy"]["directives"], ["Stop"])
        self.assertFalse(chain["terminal_policy"]["in_progress_queue"])
        self.assertTrue(chain["anchors"][-1]["terminal_target"])

    def test_target_bearing_stop_is_terminal_policy_not_progress(self):
        chain = build_anchor_chain(
            "Go through the door\nTurn left\nGo to the left of the stairs\n"
            "Stop in the doorway to the left of the white double doors",
            "door\nstairs\nwhite double doors\ndoorway",
        )
        self.assertEqual(chain["n_anchors"], 3)
        self.assertEqual(chain["terminal_policy"]["target"], "doorway")
        self.assertEqual(
            chain["terminal_policy"]["directives"],
            ["Stop in the doorway to the left of the white double doors"],
        )
        self.assertFalse(any(a["action"] == "stop" for a in chain["anchors"]))
        self.assertTrue(chain["anchors"][-1]["terminal_predecessor"])
        self.assertFalse(chain["anchors"][-1]["terminal_target"])

    def test_terminal_comma_clause_stays_out_of_progress_queue(self):
        chain = build_anchor_chain(
            "Go into the living area with the fireplace, "
            "Stop in the area between the two white sofas, "
            "next to the entrance to the dining room",
            "living area with the fireplace, two white sofas, "
            "entrance to the dining room, area between the two white sofas",
        )
        self.assertEqual(chain["n_anchors"], 1)
        self.assertEqual(
            chain["terminal_policy"]["directives"],
            [
                "Stop in the area between the two white sofas, "
                "next to the entrance to the dining room"
            ],
        )
        self.assertEqual(
            chain["terminal_policy"]["target"],
            "area between the two white sofas",
        )
        policy = chain["terminal_policy"]
        self.assertEqual(policy["target_kind"], "relational_region")
        self.assertEqual(policy["relations"], [
            {"type": "between", "reference": "two white sofas", "subject": "terminal_target"},
            {"type": "next_to", "reference": "entrance to the dining room", "subject": "terminal_target"},
        ])

    def test_background_surface_uses_displacement_not_object_proximity(self):
        anchor = build_anchor_chain(
            "Walk across the floor\nWait by the archway",
            "floor\narchway",
        )["anchors"][0]
        self.assertEqual(anchor["kind"], "motion")
        self.assertEqual(anchor["verification"]["predicate"], "odometry_displacement")
        locator = ConstraintQueueLocator()
        locator.reset_episode([anchor])
        first = locator.update(1, [0, 0, 0], 0.0, {}, {})
        second = locator.update(2, [0.6, 0, 0], 0.0, {}, {})
        self.assertEqual(first["j"], 0)
        self.assertEqual(second["j"], 1)

    def test_background_surface_is_object_when_explicitly_searched(self):
        found = build_anchor_chain("Find the rug", "rug")["anchors"][0]
        self.assertEqual(found["kind"], "object")

    def test_reference_bound_terminal_records_relation(self):
        chain = build_anchor_chain(
            "Stop in the doorway to the left of the white double doors",
            "white double doors\ndoorway",
        )
        self.assertEqual(chain["terminal_policy"]["target_kind"], "reference_bound_entity")
        self.assertEqual(chain["terminal_policy"]["relations"][0],
                         {"type": "left_of", "reference": "white double doors", "subject": "terminal_target"})

    def test_object_anchor_requires_consecutive_evidence(self):
        anchor = build_anchor_chain("Find the sofa", "sofa")["anchors"][0]
        locator = ConstraintQueueLocator()
        locator.reset_episode([anchor])
        tags = {"0": ["sofa"]}
        geometry = {"0": {"distance": 2.0}}
        first = locator.update(0, [0, 0, 0], 0.0, tags, geometry)
        second = locator.update(1, [0, 0, 0], 0.0, tags, geometry)
        self.assertEqual(first["j"], 0)
        self.assertEqual(first["hit_streaks"], {0: 1})
        self.assertEqual(second["j"], 1)
        self.assertTrue(second["complete"])

    def test_parse_failure_abstains_instead_of_auto_advancing(self):
        anchor = build_anchor_chain("Admire the view", "")["anchors"][0]
        self.assertEqual(anchor["parse_status"], "parse_failure")
        self.assertEqual(anchor["verification"]["on_unverifiable"], "abstain")
        locator = ConstraintQueueLocator(vacuous_unknown=True)
        locator.reset_episode([anchor])
        state = locator.update(0, [0, 0, 0], 0.0, {}, {})
        self.assertEqual(state["j"], 0)
        self.assertFalse(state["complete"])

    def test_unsupported_action_event_abstains_instead_of_noun_fallback(self):
        anchor = build_anchor_chain("Cross carefully", "")["anchors"][0]
        self.assertEqual(anchor["kind"], "action_event")
        self.assertEqual(anchor["action_spec"]["requirements"][0]["type"], "cross")
        self.assertEqual(anchor["verification"]["on_unverifiable"], "abstain")
        locator = ConstraintQueueLocator()
        locator.reset_episode([anchor])
        state = locator.update(0, [0, 0, 0], 0.0, {}, {})
        self.assertEqual(state["j"], 0)
        self.assertFalse(state["complete"])
        self.assertEqual(state["last_evaluation"]["status"], "unknown")

    def test_enter_and_exit_compile_to_opposite_state_transitions(self):
        enter = build_anchor_chain("Enter the bedroom", "bedroom")["anchors"][0]
        exit_ = build_anchor_chain("Exit the bedroom", "bedroom")["anchors"][0]
        self.assertEqual(
            enter["action_spec"]["requirements"],
            [{"type": "enter", "target": "bedroom"}],
        )
        self.assertEqual(
            exit_["action_spec"]["requirements"],
            [{"type": "exit", "target": "bedroom"}],
        )

    def test_enter_requires_outside_to_inside_with_motion(self):
        anchor = build_anchor_chain("Enter the bedroom", "bedroom")["anchors"][0]
        locator = ConstraintQueueLocator(enable_location=True)
        locator.reset_episode([anchor])
        outside = {"0": ["hallway"], "1": ["corridor"]}
        inside = {"0": ["bedroom"], "1": ["bedroom"]}
        first = locator.update(0, [0, 0, 0], 0.0, outside, {})
        second = locator.update(1, [0.2, 0, 0], 0.0, inside, {})
        self.assertEqual(first["j"], 0)
        self.assertEqual(second["j"], 1)
        self.assertEqual(second["last_evaluation"]["status"], "completed")
        self.assertEqual(second["completion_events"][0]["event_interval"], [0, 1])

    def test_collision_receipt_blocks_room_transition_completion(self):
        anchor = build_anchor_chain("Enter the bedroom", "bedroom")["anchors"][0]
        locator = ConstraintQueueLocator(enable_location=True)
        locator.reset_episode([anchor])
        locator.update(
            0, [0, 0, 0], 0.0,
            {"0": ["hallway"], "1": ["corridor"]}, {},
        )
        state = locator.update(
            1, [0.2, 0, 0], 0.0,
            {"0": ["bedroom"], "1": ["bedroom"]}, {},
            {
                "receipt_id": "ep:0:receipt",
                "displacement": 0.2,
                "collision": True,
            },
        )
        self.assertEqual(state["j"], 0)
        self.assertEqual(state["last_evaluation"]["status"], "incomplete")
        self.assertIn(
            "effective_movement_missing",
            state["last_evaluation"]["missing_evidence"],
        )

    def test_exit_requires_inside_to_outside_with_motion(self):
        anchor = build_anchor_chain("Exit the bedroom", "bedroom")["anchors"][0]
        locator = ConstraintQueueLocator(enable_location=True)
        locator.reset_episode([anchor])
        inside = {"0": ["bedroom"], "1": ["bedroom"]}
        outside = {"0": ["hallway"], "1": ["corridor"]}
        locator.update(0, [0, 0, 0], 0.0, inside, {})
        state = locator.update(1, [0.2, 0, 0], 0.0, outside, {})
        self.assertTrue(state["complete"])
        self.assertEqual(
            state["completion_events"][0]["requirements"][0]["type"], "exit"
        )

    def test_room_transition_without_common_views_is_unknown(self):
        anchor = build_anchor_chain("Exit the bedroom", "bedroom")["anchors"][0]
        locator = ConstraintQueueLocator(enable_location=True)
        locator.reset_episode([anchor])
        locator.update(0, [0, 0, 0], 0.0, {"0": ["bedroom"]}, {})
        state = locator.update(1, [0.2, 0, 0], 0.0, {"4": ["hallway"]}, {})
        self.assertEqual(state["j"], 0)
        self.assertEqual(state["last_evaluation"]["status"], "unknown")
        self.assertIn(
            "room_observations_not_comparable",
            state["last_evaluation"]["missing_evidence"],
        )

    def test_turn_does_not_borrow_heading_change_from_previous_anchor(self):
        anchors = build_anchor_chain(
            "Find the sofa\nTurn left", "sofa"
        )["anchors"]
        locator = ConstraintQueueLocator()
        locator.reset_episode(anchors)
        tags = {"0": ["sofa"]}
        geometry = {"0": {"distance": 2.0}}
        locator.update(0, [0, 0, 0], 0.0, tags, geometry)
        after_sofa = locator.update(1, [0, 0, 0], 1.0, tags, geometry)
        self.assertEqual(after_sofa["j"], 1)
        no_new_turn = locator.update(2, [0, 0, 0], 1.0, {}, {})
        self.assertEqual(no_new_turn["j"], 1)
        completed = locator.update(3, [0, 0, 0], 2.0, {}, {})
        self.assertEqual(completed["j"], 2)

    def test_future_anchor_does_not_bank_early_object_hits(self):
        anchors = build_anchor_chain(
            "Find the sofa\nFind the table", "sofa\ntable"
        )["anchors"]
        locator = ConstraintQueueLocator()
        locator.reset_episode(anchors)
        tags = {"0": ["sofa", "table"]}
        geometry = {"0": {"distance": 2.0}}
        locator.update(0, [0, 0, 0], 0.0, tags, geometry)
        first_done = locator.update(1, [0, 0, 0], 0.0, tags, geometry)
        self.assertEqual(first_done["j"], 1)
        table_first_hit = locator.update(2, [0, 0, 0], 0.0, tags, geometry)
        self.assertEqual(table_first_hit["j"], 1)
        table_done = locator.update(3, [0, 0, 0], 0.0, tags, geometry)
        self.assertEqual(table_done["j"], 2)

    def test_pass_target_binds_after_relation_not_first_landmark(self):
        anchor = build_anchor_chain(
            "Walk down the hallway past the piano to the end",
            "hallway\npiano",
        )["anchors"][0]
        self.assertEqual(
            anchor["action_spec"]["requirements"],
            [{"type": "pass", "target": "piano"}],
        )

    def test_compound_turn_and_pass_requires_both_events(self):
        anchor = build_anchor_chain(
            "Turn right to walk past stairs", "stairs"
        )["anchors"][0]
        self.assertEqual(
            [item["type"] for item in anchor["action_spec"]["requirements"]],
            ["turn", "pass"],
        )
        locator = ConstraintQueueLocator()
        locator.reset_episode([anchor])
        locator.update(0, [0, 0, 0], 0.0, {}, {})
        state = locator.update(1, [0, 0, 0], -1.0, {}, {})
        self.assertEqual(state["j"], 0)
        self.assertEqual(state["last_evaluation"]["status"], "unknown")
        self.assertEqual(
            state["last_evaluation"]["unsupported_requirements"], ["pass"]
        )


if __name__ == "__main__":
    unittest.main()

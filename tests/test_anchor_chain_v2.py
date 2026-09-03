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
        self.assertEqual(chain["schema_version"], "opennav.anchor_chain.v2")
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
            "living area with the fireplace, area between the two white sofas, "
            "entrance to the dining room",
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

    def test_supported_but_unverifiable_action_is_explicit_skip(self):
        anchor = build_anchor_chain("Cross carefully", "")["anchors"][0]
        self.assertEqual(anchor["parse_status"], "unsupported")
        self.assertEqual(anchor["verification"]["on_unverifiable"], "skip")
        locator = ConstraintQueueLocator()
        locator.reset_episode([anchor])
        state = locator.update(0, [0, 0, 0], 0.0, {}, {})
        self.assertEqual(state["j"], 1)
        self.assertTrue(state["complete"])


if __name__ == "__main__":
    unittest.main()

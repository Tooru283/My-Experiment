import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "terminal_recheck_under_test",
    ROOT / "vlnce_baselines/common/opennav_ext/terminal_recheck.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
TerminalRecheck = module.TerminalRecheck


def verifier(visible=True, arrived=True):
    return {
        "selected_candidate_verdict": {
            "final_target_visible": visible,
            "arrival_evidence": arrived,
            "missing_final_landmarks": [],
            "target_direction_id": "6",
        },
        "allow_blockers": ["generic_final_terms:archway"],
        "contradictions": [],
    }


class TerminalRecheckTest(unittest.TestCase):
    def test_plausible_terminal_gets_two_bounded_micro_steps(self):
        policy = TerminalRecheck(max_holds=2, recheck_step_m=0.25)
        first = policy.decide(
            step_id=8, position=[0, 0, 0], route_complete=True,
            stop_committed=False, verifier=verifier(),
            spatial={"local_surface_depth_m": 2.8},
        )
        self.assertTrue(first["hold"])
        second = policy.decide(
            step_id=9, position=[0.25, 0, 0], route_complete=True,
            stop_committed=False, verifier={}, spatial={},
            evidence_unavailable=True,
        )
        self.assertTrue(second["hold"])
        third = policy.decide(
            step_id=10, position=[0.5, 0, 0], route_complete=True,
            stop_committed=False, verifier=verifier(),
            spatial={"local_surface_depth_m": 2.0},
        )
        self.assertFalse(third["hold"])

    def test_no_hold_without_route_completion_or_directional_depth(self):
        policy = TerminalRecheck()
        for route_complete, depth in ((False, 2.0), (True, 3.5), (True, None)):
            policy.reset()
            result = policy.decide(
                step_id=3, position=[0, 0, 0],
                route_complete=route_complete, stop_committed=False,
                verifier=verifier(), spatial={"local_surface_depth_m": depth},
            )
            self.assertFalse(result["hold"])

    def test_negative_visual_evidence_does_not_start_recheck(self):
        result = TerminalRecheck().decide(
            step_id=3, position=[0, 0, 0], route_complete=True,
            stop_committed=False, verifier=verifier(False, False),
            spatial={"local_surface_depth_m": 1.0},
        )
        self.assertFalse(result["hold"])


if __name__ == "__main__":
    unittest.main()

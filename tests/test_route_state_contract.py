import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

phase_module = load_module("phase_evidence_under_test", "vlnce_baselines/common/opennav_ext/phase_evidence.py")
route_state_module = load_module("route_state_under_test", "vlnce_baselines/common/opennav_ext/route_state.py")
terminal_gate_module = load_module("terminal_gate_under_test", "vlnce_baselines/common/opennav_ext/terminal_gate.py")

def contains_forbidden_key(value, forbidden):
    if isinstance(value, dict):
        return any(key in forbidden or contains_forbidden_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden_key(item, forbidden) for item in value)
    return False

class RouteStateContractTest(unittest.TestCase):
    def test_distance_gains_are_audit_only_for_phase(self):
        result = phase_module.PhaseEvidenceTracker().build(
            instruction="go forward", actions="go forward", landmarks="door",
            estimation="", history="", current_step=8,
            recent_distance_gains=[-2.0, -1.0, 0.0],
        )
        self.assertNotEqual(result["phase"], "recover")
        self.assertFalse(result["recent_progress"]["decision_effect"])
        self.assertEqual(result["recent_progress"]["source"], "simulator_goal_distance_audit_only")

    def test_route_state_excludes_oracle_fields_recursively(self):
        state = route_state_module.build_route_state(
            episode_id="ep-1", step_id=3, stage="post_action",
            instruction="walk to the door", actions="walk forward", landmarks="door",
            anchors=[{"idx": 0, "key": "door", "latest_goal_dist": 4.0}],
            progress_locator_result={"j": 1, "n_anchors": 1, "complete": True, "ever_satisfied": [0]},
            phase_evidence={"phase": "verify", "confidence": 0.7},
            spatial_memory_result={"node_id": 1, "debug": {"distance_to_goal": 7.0}},
            landmark_pool_result={"pool_size": 1, "entries": [{"category": "door", "success": True}]},
            failure_status={"evidence": {"recent_distance_gains": [-1.0], "distance_to_goal": 9.0}},
            terminal_gate_result={"allow_stop": True, "latest_goal_dist": 2.0},
            candidate_ids=["1", "2"], selected_candidate="1",
            action_result={"distance_gain_selected": 1.0, "distance_to_goal": 8.0, "collision": 0.0},
        )
        self.assertEqual(state["schema_version"], "route_state.v1")
        self.assertFalse(contains_forbidden_key(state, set(route_state_module.ORACLE_FIELDS_EXCLUDED)))
        json.dumps(state)

    def test_route_state_names_coordinator_as_authoritative_stop_source(self):
        state = route_state_module.build_route_state(
            episode_id="ep-1", step_id=5, stage="post_action",
            instruction="stop at the door", actions="stop", landmarks="door",
            terminal_gate_result={
                "verdict": "block", "allow_stop": False,
                "decision_effect_enabled": False,
            },
            stop_decision_result={
                "decision_id": "stop-1",
                "outcome": "commit_goal_stop",
                "selected_proposal_id": "p1",
            },
            stop_evidence_items=[{
                "evidence_id": "e1", "evaluator": "legacy_combined",
                "verdict": "support",
            }],
            stop_requested=True,
            action_result={"stop_committed": True},
        )
        stop = state["stop_evidence"]
        self.assertEqual(stop["authoritative_source"], "stop_coordinator")
        self.assertEqual(stop["coordinator_decision"]["outcome"], "commit_goal_stop")
        self.assertFalse(stop["terminal_gate_action_effect"])
        self.assertEqual(state["provenance"]["stop_evidence"]["source"], "stop_coordinator")
        self.assertEqual(state["progress"]["source"], "acn_l1_abstain")

    def test_terminal_gate_block_changes_stop_to_movement(self):
        result = terminal_gate_module.TerminalGate().evaluate(
            {"j": 0, "n_anchors": 2, "ever_satisfied": [], "degenerate": False},
            landmark_pool=None, final_landmark_term="door",
        )
        self.assertFalse(result["allow_stop"])
        self.assertEqual(terminal_gate_module.gate_stop_request(True, "candidate STOP", result, True), (False, "", True))
        self.assertEqual(terminal_gate_module.gate_stop_request(True, "candidate STOP", result, False), (True, "candidate STOP", False))

    def test_candidate_projection_uses_calibrated_sign_convention(self):
        graph_source = (ROOT / "vlnce_baselines/common/opennav_ext/visual_graph_memory.py").read_text(encoding="utf-8")
        self.assertIn("angle = -heading_value - float(candidate.angle_rad)", graph_source)

    def test_acn_resets_are_owned_by_episode_reducer(self):
        trainer_source = (ROOT / "vlnce_baselines/common/base_il_trainer_llm.py").read_text(encoding="utf-8")
        boundary = trainer_source.index("if active_navigation_episode_id != current_episode_id:")
        reducer_reset = trainer_source.index("route_state_reducer.reset_episode")
        self.assertLess(boundary, reducer_reset)
        self.assertEqual(trainer_source.count("route_state_reducer.reset_episode"), 1)
        self.assertNotIn("progress_locator.reset_episode", trainer_source)
        self.assertNotIn("landmark_pool.reset_episode", trainer_source)

if __name__ == "__main__":
    unittest.main()

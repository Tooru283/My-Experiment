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


contracts = load_module(
    "pipeline_contracts_under_test",
    "vlnce_baselines/common/opennav_ext/pipeline_contracts.py",
)


class Candidate:
    candidate_id = "7"
    angle_rad = 0.25
    distance = 1.5


class PipelineContractsTest(unittest.TestCase):
    def test_observation_frame_normalizes_legacy_inputs(self):
        frame = contracts.build_observation_frame(
            episode_id=42,
            step_id=3,
            instruction="go to the door",
            candidates=[Candidate()],
            observe_dict={7: "Scene Objects: door"},
            position=[1, 2, 3],
            heading=0.5,
            plan_ref="plan-42",
        )
        payload = frame.to_dict()
        self.assertEqual(payload["episode_id"], "42")
        self.assertEqual(payload["candidate_ids"], ["7"])
        self.assertEqual(payload["candidate_observations"]["7"], "Scene Objects: door")
        self.assertEqual(payload["candidate_geometry"]["7"]["distance"], 1.5)
        json.dumps(payload)

    def test_online_contract_rejects_oracle_fields_recursively(self):
        with self.assertRaises(contracts.OracleFieldError):
            contracts.require_oracle_free(
                {"nested": [{"distance_to_goal": 2.0}]},
                "test payload",
            )

    def test_action_receipt_accepts_observable_result(self):
        receipt = contracts.build_action_receipt(
            receipt_id="receipt-1",
            command_id="command-1",
            episode_id="ep-1",
            step_id=4,
            executed_action={
                "action": {"action": 4, "action_args": {"angle": 0.2, "distance": 1.0}}
            },
            candidate_id="7",
            step_result={"done": False, "collision": 0.0, "steps_taken": 5},
        )
        payload = receipt.to_dict()
        self.assertEqual(payload["candidate_id"], "7")
        self.assertFalse(payload["done"])
        self.assertEqual(payload["steps_taken"], 5)
        json.dumps(payload)

    def test_action_receipt_rejects_mixed_audit_summary(self):
        with self.assertRaises(contracts.OracleFieldError):
            contracts.build_action_receipt(
                receipt_id="receipt-1",
                command_id="command-1",
                episode_id="ep-1",
                step_id=4,
                executed_action={"action": {"action": 0, "action_args": None}},
                step_result={"done": True, "distance_gain_selected": 1.0},
            )

    def test_decision_record_is_serializable_and_referential(self):
        command = contracts.ActionCommand(
            command_id="command-2",
            route_state_id="state-2",
            decision_record_id="decision-2",
            action_code=0,
            candidate_id=None,
            action_args=None,
            stop_requested=True,
        )
        proposal = contracts.StopProposal(
            source="completion",
            reason="legacy stop",
            route_state_id="state-2",
        )
        record = contracts.DecisionRecord(
            decision_record_id="decision-2",
            route_state_id="state-2",
            step_id=2,
            progress_provider="acn_l1",
            planner_output={"selected": "STOP"},
            stop_proposals=(proposal,),
            override_ledger=(),
            final_command=command,
        )
        payload = record.to_dict()
        self.assertEqual(payload["final_command"]["decision_record_id"], "decision-2")
        self.assertEqual(payload["stop_proposals"][0]["source"], "completion")
        json.dumps(payload)


    def test_trainer_wires_m0_contracts_without_replacing_action_path(self):
        source = (
            ROOT / "vlnce_baselines/common/base_il_trainer_llm.py"
        ).read_text(encoding="utf-8")
        for event_name in (
            "pipeline_observation_frame",
            "pipeline_progress_update",
            "pipeline_decision_record",
            "pipeline_action_receipt",
        ):
            self.assertIn(event_name, source)
        self.assertLess(
            source.index("outputs = envs.step(env_actions)"),
            source.index("_contract_receipt = run_harness_tool"),
        )
        self.assertIn("progress_update_contract = None", source)
        self.assertIn(
            "def run_harness_tool(tool_name, trace_step_id", source
        )


if __name__ == "__main__":
    unittest.main()

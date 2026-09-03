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
compiler_module = load_module(
    "action_compiler_under_test",
    "vlnce_baselines/common/opennav_ext/action_compiler.py",
)


class ActionCompilerTest(unittest.TestCase):
    def setUp(self):
        self.compiler = compiler_module.ActionCompiler()

    def decision(self, outcome="commit_goal_stop", decision_id="state-1:stop"):
        return contracts.StopDecision(
            decision_id=decision_id,
            route_state_id="state-1",
            step_id=3,
            outcome=outcome,
            selected_proposal_id="p1",
            reason="resolved",
        )

    def resolved(self, action_type, **kwargs):
        return contracts.ResolvedAction(
            action_type=action_type,
            route_state_id=kwargs.pop("route_state_id", "state-1"),
            decision_record_id="decision-1",
            **kwargs
        )

    def test_goal_stop_compiles_to_action_zero(self):
        decision = self.decision()
        command = self.compiler.compile(
            command_id="command-1",
            resolved_action=self.resolved(
                "stop", candidate_id="STOP", stop_decision_id=decision.decision_id
            ),
            stop_decision=decision,
        )
        self.assertEqual(command.action_code, 0)
        self.assertTrue(command.stop_requested)
        self.assertEqual(
            self.compiler.to_env_action(command),
            {"action": {"action": 0, "action_args": None}},
        )

    def test_forced_termination_also_compiles_to_action_zero(self):
        decision = self.decision("commit_forced_termination")
        command = self.compiler.compile(
            command_id="command-1",
            resolved_action=self.resolved(
                "stop", stop_decision_id=decision.decision_id
            ),
            stop_decision=decision,
        )
        self.assertEqual(command.action_code, 0)

    def test_move_and_backtrack_compile_to_action_four(self):
        for action_type in ("move", "backtrack"):
            command = self.compiler.compile(
                command_id="command-1",
                resolved_action=self.resolved(
                    action_type,
                    candidate_id="7" if action_type == "move" else "MOVE_BACK",
                    action_args={"angle": 1, "distance": 1.5},
                ),
            )
            self.assertEqual(command.action_code, 4)
            self.assertFalse(command.stop_requested)
            self.assertEqual(command.action_args, {"angle": 1.0, "distance": 1.5})

    def test_stop_requires_committed_matching_decision(self):
        resolved = self.resolved("stop", stop_decision_id="state-1:stop")
        with self.assertRaises(compiler_module.ActionCompilationError):
            self.compiler.compile(command_id="c", resolved_action=resolved)
        with self.assertRaises(compiler_module.ActionCompilationError):
            self.compiler.compile(
                command_id="c", resolved_action=resolved,
                stop_decision=self.decision("reject"),
            )
        with self.assertRaises(compiler_module.ActionCompilationError):
            self.compiler.compile(
                command_id="c", resolved_action=resolved,
                stop_decision=self.decision(decision_id="other"),
            )

    def test_movement_rejects_invalid_geometry_or_committed_stop(self):
        with self.assertRaises(compiler_module.ActionCompilationError):
            self.compiler.compile(
                command_id="c",
                resolved_action=self.resolved(
                    "move", candidate_id="7", action_args={"angle": 1}
                ),
            )
        with self.assertRaises(compiler_module.ActionCompilationError):
            self.compiler.compile(
                command_id="c",
                resolved_action=self.resolved(
                    "move", candidate_id="7",
                    action_args={"angle": 1, "distance": 1},
                ),
                stop_decision=self.decision(),
            )

    def test_trainer_handles_synthetic_move_back_history(self):
        source = (ROOT / "vlnce_baselines/common/base_il_trainer_llm.py").read_text(encoding="utf-8")
        self.assertIn("if next_vp == MOVE_BACK_CANDIDATE:", source)
        self.assertIn("synthetic_action\": True", source)
        self.assertIn("nav_history.append({", source)


if __name__ == "__main__":
    unittest.main()

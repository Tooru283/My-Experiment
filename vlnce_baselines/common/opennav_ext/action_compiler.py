"""M3.3 single compiler from resolved navigation actions to Habitat actions."""

from typing import Any, Dict, Optional

from vlnce_baselines.common.opennav_ext.pipeline_contracts import (
    ActionCommand,
    ResolvedAction,
    StopDecision,
    require_oracle_free,
)


ACTION_COMPILER_VERSION = "action_compiler.v1"
STOP_OUTCOMES = frozenset(("commit_goal_stop", "commit_forced_termination"))


class ActionCompilationError(ValueError):
    """Raised when a resolved action cannot be safely compiled."""


class ActionCompiler:
    """The only component allowed to encode Habitat action 0 or action 4."""

    def compile(
        self,
        *,
        command_id: str,
        resolved_action: ResolvedAction,
        stop_decision: Optional[StopDecision] = None,
    ) -> ActionCommand:
        resolved_action.to_dict()
        if not command_id:
            raise ActionCompilationError("command_id is required")
        action_type = str(resolved_action.action_type or "").lower()
        if action_type == "stop":
            command = self._compile_stop(command_id, resolved_action, stop_decision)
        elif action_type in {"move", "backtrack"}:
            command = self._compile_move(command_id, resolved_action, stop_decision)
        else:
            raise ActionCompilationError(
                "unsupported resolved action type: {}".format(action_type)
            )
        command.to_dict()
        return command

    def _compile_stop(
        self,
        command_id: str,
        resolved: ResolvedAction,
        decision: Optional[StopDecision],
    ) -> ActionCommand:
        if decision is None or decision.outcome not in STOP_OUTCOMES:
            raise ActionCompilationError(
                "STOP requires a committed StopDecision"
            )
        if decision.route_state_id != resolved.route_state_id:
            raise ActionCompilationError("STOP route_state_id mismatch")
        if not resolved.stop_decision_id:
            raise ActionCompilationError("STOP requires stop_decision_id")
        if resolved.stop_decision_id != decision.decision_id:
            raise ActionCompilationError("STOP decision id mismatch")
        if resolved.candidate_id not in (None, "STOP"):
            raise ActionCompilationError("STOP cannot reference a movement candidate")
        if resolved.action_args is not None:
            raise ActionCompilationError("STOP cannot carry action_args")
        return ActionCommand(
            command_id=command_id,
            route_state_id=resolved.route_state_id,
            decision_record_id=resolved.decision_record_id,
            action_code=0,
            candidate_id="STOP",
            action_args=None,
            stop_requested=True,
            stop_decision_id=decision.decision_id,
        )

    def _compile_move(
        self,
        command_id: str,
        resolved: ResolvedAction,
        decision: Optional[StopDecision],
    ) -> ActionCommand:
        if resolved.stop_decision_id is not None:
            raise ActionCompilationError("movement cannot reference stop_decision_id")
        if decision is not None and decision.outcome in STOP_OUTCOMES:
            raise ActionCompilationError(
                "movement conflicts with a committed StopDecision"
            )
        if resolved.candidate_id is None:
            raise ActionCompilationError("movement requires candidate_id")
        args = dict(resolved.action_args or {})
        try:
            angle = float(args["angle"])
            distance = float(args["distance"])
        except (KeyError, TypeError, ValueError):
            raise ActionCompilationError(
                "movement requires numeric angle and distance"
            )
        if distance <= 0:
            raise ActionCompilationError("movement distance must be positive")
        normalized_args = {"angle": angle, "distance": distance}
        require_oracle_free(normalized_args, "ActionCompiler movement args")
        return ActionCommand(
            command_id=command_id,
            route_state_id=resolved.route_state_id,
            decision_record_id=resolved.decision_record_id,
            action_code=4,
            candidate_id=str(resolved.candidate_id),
            action_args=normalized_args,
            stop_requested=False,
            stop_decision_id=None,
        )

    @staticmethod
    def to_env_action(command: ActionCommand) -> Dict[str, Dict[str, Any]]:
        command.to_dict()
        if command.action_code not in (0, 4):
            raise ActionCompilationError("unsupported compiled action code")
        return {
            "action": {
                "action": int(command.action_code),
                "action_args": command.action_args,
            }
        }

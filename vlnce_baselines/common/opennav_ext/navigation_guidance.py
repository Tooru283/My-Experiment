"""Navigation guidance, deterministic history and legacy plan replay helpers."""
import json
import hashlib
import re
import math


PLAN_SCHEMA_VERSION = "opennav.instruction_plan.v2.2.two_stage"


def action_cache_filename(model, base_url, action_prompt, landmark_prompt):
    """Old plans must not hide a change in the decomposition prompt or model."""
    payload = json.dumps({
        "model": model, "base_url": base_url, "schema_version": PLAN_SCHEMA_VERSION,
        "action_prompt": action_prompt, "landmark_prompt": landmark_prompt,
    }, ensure_ascii=False, sort_keys=True)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return "actions_cache_{}.json".format(fingerprint)


# Historical v1 prompt for old trace readers. Runtime uses instruction_parsing.py.
LEGACY_INSTRUCTION_PLAN_SYSTEM = """Extract the ordered actions AND landmarks in one request.
Return exactly one JSON object with two keys: actions (array of nonempty strings)
and landmarks (array of nonempty strings). No markdown, numbering or commentary.
These are intended actions, NOT a record of completed actions. Do not output
completion flags, confidence, verification predicates or invented observations.
Keep the original wording, targets, spatial relations and stop conditions.
Keep then/until/while/after/before attached to the actions they constrain; do not
split a conditional or overlapping action into unrelated commands.
Descriptions of the starting position are context, not extra movement commands.
For independent sequential commands, keep their original order. Retain compound
commands when splitting would lose a destination, condition or spatial relation.

Meaning distinctions to preserve:
- Leave the closet / Walk out of the closet: leaving a region.
- Enter the bedroom / Walk into the bedroom: entering a region.
- Go through the doorway: crossing an opening.
- Walk through the kitchen to the hallway: travelling through a region.
- Turn right to walk past the stairs: BOTH the turn and passing the stairs.
- Go to the left of the stairs: a spatial destination, not simply Turn left.

Do not invent targets, distances, actions, or claims that an action is completed.
Do not replace a state-changing phrase with just its landmark noun.
Landmarks name physical objects/regions, not directions or verbs. Preserve
identifying modifiers (white double doors, kitchen doorway). Keep route landmarks
in route order, then any terminal reference objects, and put the FINAL destination
last. In 'Stop in the doorway to the left of the white double doors', doorway is
the destination; white double doors is a reference, not the stop target. Do not
invent a STOP instruction when none was given. Use [] if no landmark is named.

Examples:
Instruction: Go through the door. Turn left. Go to the left of the stairs.
Stop in the doorway to the left of the white double doors.
Output: {"actions":["Go through the door","Turn left","Go to the left of the stairs","Stop in the doorway to the left of the white double doors"],"landmarks":["door","stairs","white double doors","doorway"]}
Instruction: You are in a closet. Walk out of the closet, then take a left into
the bedroom. Stop beside the bed.
Output: {"actions":["Walk out of the closet","Take a left into the bedroom","Stop beside the bed"],"landmarks":["closet","bedroom","bed"]}
Instruction: Turn right to walk past the stairs, then walk forward until you
reach the sofa.
Output: {"actions":["Turn right to walk past the stairs","Walk forward until you reach the sofa"],"landmarks":["stairs","sofa"]}"""


def parse_instruction_plan(response):
    """Read a legacy v1 joint plan for replay; runtime uses two stage parsers.

    Invalid output fails explicitly, never becoming a guessed/completed route.
    A single surrounding JSON fence is tolerated; prose and partial JSON are not.
    """
    raw = str(response or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.S | re.I)
    if fenced:
        raw = fenced.group(1)
    try:
        plan = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid instruction plan: expected actions/landmarks JSON") from exc
    if not isinstance(plan, dict) or set(plan) != {"actions", "landmarks"}:
        raise ValueError("Invalid instruction plan: only actions and landmarks are allowed")
    for key in ("actions", "landmarks"):
        items = plan[key]
        if not isinstance(items, list) or len(items) > 64 or (key == "actions" and not items):
            raise ValueError("Invalid instruction plan: {} must be a list of phrases".format(key))
        if any(not isinstance(item, str) or not item.strip() or "\n" in item or "\r" in item for item in items):
            raise ValueError("Invalid instruction plan: {} contains an invalid phrase".format(key))
    return tuple("\n".join(item.strip() for item in plan[key]) for key in ("actions", "landmarks"))


NAVIGATION_SYSTEM = """Follow the navigation instruction by selecting one available waypoint.
Use the supplied ACN verified progress as the completion record. Your job is to
choose the next move; do not revise completed actions or declare an unverified
action completed. STOP is a proposal subject to the system's terminal checks.

Use the current action, its verification feedback, recent history, and candidate
observations together. Prefer a waypoint that executes the current action or helps
obtain missing evidence. Seeing a later landmark does not prove earlier actions
were completed. A direction with more instruction landmarks is not automatically
better than a direction that fulfils the current action's spatial relation.

unknown means evidence or a verifier is missing, not proof the robot has failed
to move. Do not repeatedly turn or return to a doorway just to recreate a missing
label. Use the history to choose a useful, traversable move while leaving verified
progress unchanged. Do not invent observation tools or unlisted viewpoints.
If feedback says verifier_not_implemented, another view cannot implement that
verifier. Use recent measured movement and observations to avoid repeating a
manoeuvre; you may choose a useful onward move without marking the action complete.
The original instruction defines the task; the parsed plan is an interpretation.
If they conflict, preserve the instruction's target/relation, not an invented plan
detail, and do not edit ACN state. History records pre-move observations and
selection rationales separately from measured execution. A rationale describes
intent, not proof of arrival, crossing or passing. Candidate views are simultaneous
views from the current position, not images taken at future waypoint positions.

Candidate IDs refer to camera directions relative to your CURRENT heading:
0 front; 1,2,3,4,5 left by approximately 30,60,90,120,150 degrees;
6 behind; 7,8,9,10,11 right by approximately 150,120,90,60,30 degrees.
The actual waypoint angle can differ from the camera center. A left turn normally
uses a listed ID in 1..5; a right turn normally uses a listed ID in 7..11. Check
the intended destination and available route as well as turn direction.
An object's left/right position inside an image is relative to that camera view,
not automatically left/right of the agent's forward heading.
Waypoint distance is a movement length, not distance to the named object.
Distances in scene descriptions are estimates, not measured target distances.

Keep the final stopping instruction in view. Propose STOP only when the route is
verified complete and current observations support the required terminal position.
Otherwise choose a listed waypoint (or MOVE_BACK only when explicitly offered).
Output exactly these two fields:
Thought: One or two sentences citing the relevant observation and expected effect.
Prediction: One available candidate ID, STOP, or an explicitly offered MOVE_BACK.
Do not output confidence scores or a replacement completion assessment."""


def build_navigation_history_item(step, candidate_id, thought, observation, receipt=None):
    """No model summaries: separate pre-action perception, intent and measured motion.

    Whitelist receipt fields so evaluation-only goal distances never enter prompts.
    A missing/mismatched receipt is unknown execution, not evidence of zero motion.
    """
    receipt = receipt if isinstance(receipt, dict) else {}
    matched = (str(receipt.get("step_id")) == str(step)
               and str(receipt.get("candidate_id")) == str(candidate_id))
    if not matched:
        receipt = {}

    def number(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return round(value, 4)
        return None

    executed = receipt.get("executed_action") or {}
    command = executed.get("action") if isinstance(executed, dict) else None
    command = command if isinstance(command, dict) else {}
    args = command.get("action_args") or {}
    args = args if isinstance(args, dict) else {}
    execution = {
        "receipt_available": matched,
        "action_code": number(command.get("action")),
        "command_angle_rad": number(args.get("angle")),
        "command_distance_m": number(args.get("distance")),
        "measured_displacement_m": number(receipt.get("displacement")),
        "collision": receipt.get("collision") if isinstance(receipt.get("collision"), bool)
        else number(receipt.get("collision")),
    }
    # Preserve the compact RAM tags even if the preceding scene description is long.
    description, marker, tags = str(observation or "").partition("Scene Objects:")
    compact = description[:450] + (" Scene Objects:" + tags[:200] if marker else "")
    return {
        "step": step, "viewpoint": str(candidate_id),
        "observation": compact, "observation_phase": "before_action",
        "thought": str(thought or "")[:240], "thought_role": "unverified_selection_rationale",
        "execution": execution, "synthetic_action": str(candidate_id) == "MOVE_BACK",
    }


def format_navigation_history(history):
    return " -> ".join(
        "Step {} Candidate {} Pre-action observation (not post-action evidence): {} "
        "Execution (not semantic completion): {} Selection rationale (unverified intent): {}".format(
            item.get("step", "unknown"), item.get("viewpoint", "unknown"),
            item.get("observation", ""),
            json.dumps(item.get("execution", {"receipt_available": False}), separators=(",", ":")),
            item.get("thought", ""),
        ) for item in history
    )


def format_navigation_feedback(progress):
    """Read-only view of ProgressUpdate; excludes stale previous-anchor evaluation."""
    if not isinstance(progress, dict):
        return ""
    action = progress.get("current_action") or {}
    evaluation = action.get("evaluation") or {}
    if evaluation.get("anchor_index") != progress.get("current_index"):
        evaluation = {}
    raw_missing = evaluation.get("missing_evidence") or []
    unsupported = list(evaluation.get("unsupported_requirements") or [])
    unsupported.extend(
        str(item).split(":", 1)[1]
        for item in raw_missing
        if str(item).startswith("verifier_not_implemented:")
        and ":" in str(item)
    )
    unsupported = list(dict.fromkeys(unsupported))
    missing = [
        item for item in raw_missing
        if not str(item).startswith("verifier_not_implemented:")
    ]
    feedback = {
        "current_action": action.get("raw"),
        "completion_conditions": action.get("requirements") or [],
        "verification_status": evaluation.get("status", "not_evaluated"),
        "observed_requirements": evaluation.get("requirements") or [],
        "missing_evidence": missing,
        "unsupported_requirements": unsupported,
        "route_progress_complete": progress.get("route_progress_complete"),
    }
    if unsupported:
        feedback["navigation_instruction"] = (
            "Physically continue the named route action using the original instruction "
            "and recent movement. Do not rotate or revisit solely to make the unavailable "
            "verifier succeed; ACN will keep progress unconfirmed."
        )
    if progress.get("route_progress_complete"):
        feedback["verification_status"] = "route_verified_terminal_check_required"
    return "\nACN action feedback (read-only): " + json.dumps(
        feedback, ensure_ascii=False, separators=(",", ":")
    )

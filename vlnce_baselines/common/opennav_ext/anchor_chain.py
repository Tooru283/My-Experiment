"""Build ordered action completion conditions from the episode-level action plan.

Parsing adds no LLM calls. An understood action is not necessarily verifiable:
unsupported cross/pass/traverse requirements remain explicit. Terminal directives
are separate from the route queue. Version: state_change.v3.4.
"""
import re
from typing import Any, Dict, List, Optional

ROOM_TYPES = (
    "living room", "dining room", "laundry room", "storage area", "bedroom",
    "bathroom", "kitchen", "hallway", "closet", "garage", "office", "corridor",
    "foyer", "den", "stairwell", "balcony", "patio", "lobby", "hall", "room",
)

_DIRECTION_VERB = re.compile(
    r"\b(turn (?:left|right|around)|take a (?:left|right)|go (?:straight|forward)|"
    r"walk (?:straight|forward)|head (?:left|right|straight)|continue straight)\b",
    re.I,
)
_TURN_LEFT = re.compile(r"\b(turn left|take a left|head left)\b", re.I)
_TURN_RIGHT = re.compile(r"\b(turn right|take a right|head right)\b", re.I)
_TURN_AROUND = re.compile(r"\bturn around\b", re.I)
_ACTION_VERB = re.compile(
    r"\b(walk|go|move|head|turn|take|enter|exit|climb|descend|continue|stop|wait|"
    r"proceed|leave|cross|pass)\b",
    re.I,
)
_SPLIT = re.compile(r"[,\n;]| then ", re.I)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*]\s*|\d+\s*[.):、-]\s*)")
_TERMINAL_DIRECTIVE = re.compile(
    r"^\s*(?:"
    r"(?:(?:when|once|after|before|until|while)\b[^,;.!?]*,\s*)?"
    r"(?:stop|wait|stay|stand|pause)\b|"
    r"(?:(?:that['’]?s|that is|this is)\s+)?where\b[^,;.!?]*"
    r"\b(?:stop|wait|stay|stand|pause)\b)", re.I
)
_TEMPORAL_TERMINAL_CONDITION = re.compile(
    r"^\s*(?:stop|wait|stay|stand|pause)\s+"
    r"(?:before|after|until|while)\s+"
    r"(?:(?:you|we|the\s+agent|the\s+robot)\s+)?"
    r"(?:go(?:ing)?|walk(?:ing)?|move|moving|head(?:ing)?|proceed(?:ing)?|"
    r"enter(?:ing)?|exit(?:ing)?|leave|leaving|step(?:ping)?|cross(?:ing)?|"
    r"pass(?:ing)?)\b",
    re.I,
)

ANCHOR_CHAIN_SCHEMA_VERSION = "opennav.anchor_chain.v3"
ACTION_SEMANTICS_VERSION = "state_change.v3.4"
DEFAULT_CONSECUTIVE_HITS = 2
_EXIT_PHRASE = r"\b(?:exit|leave|(?:walk|go|move|head|step|proceed|continue)\s+out\s+of)\b"
_ENTER_PHRASE = r"\b(?:enter|(?:walk|go|move|head|step|proceed|continue)\s+into|(?:turn|take)\s+(?:a\s+)?(?:left|right)\s+into)\b"
_ROOM_TARGET_BOUNDARY = re.compile(
    r"[,;.!?]|\b(?:then|until|while|after|before|and|past|through|into|out\s+of|to)\b"
)
_TERMINAL_RELATION = re.compile("|".join(
    "(?P<{}>{})".format(name, marker)
    for name, marker in (
        ("between", r"\bbetween\b"),
        ("next_to", r"\bnext\s+to\b"),
        ("left_of", r"\bto\s+the\s+left\s+of\b"),
        ("right_of", r"\bto\s+the\s+right\s+of\b"),
        ("in_front_of", r"\bin\s+front\s+of\b"),
        ("at_top_of", r"\bat\s+the\s+top\s+of\b"),
    )
))

BACKGROUND_SURFACES = frozenset((
    "floor", "ground", "carpet", "rug", "wall", "ceiling",
))


def _split_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        parts = _SPLIT.split(str(value))
    cleaned = []
    for part in parts:
        if not part or not str(part).strip():
            continue
        item = _LIST_PREFIX.sub("", str(part)).strip()
        if item:
            cleaned.append(item)
    return cleaned


def _verification(kind: str, parse_status: str) -> Dict[str, Any]:
    if kind == "location":
        return {
            "predicate": "location_dominant",
            "parameters": {"min_view_ratio": 0.5, "min_consecutive_hits": 2},
            "on_unverifiable": "abstain",
        }
    if kind == "object":
        return {
            "predicate": "object_nearby",
            "parameters": {
                "max_waypoint_distance_m": 3.0,
                "min_consecutive_hits": 2,
            },
            "on_unverifiable": "abstain",
        }
    if kind == "direction":
        return {
            "predicate": "odometry_motion",
            "parameters": {"min_consecutive_hits": 1},
            "on_unverifiable": "abstain",
        }
    if kind == "motion":
        return {
            "predicate": "odometry_displacement",
            "parameters": {"min_displacement_m": 0.5, "min_consecutive_hits": 1},
            "on_unverifiable": "abstain",
        }
    return {
        "predicate": "unverifiable",
        "parameters": {"min_consecutive_hits": 1},
        "on_unverifiable": "skip" if parse_status == "unsupported" else "abstain",
    }


def _room_in(text: str, *, first_mention: bool = False) -> Optional[str]:
    low = text.lower()
    for alias, canonical in (
        ("living area", "living room"), ("dining area", "dining room"),
        ("bed room", "bedroom"), ("bath room", "bathroom"),
    ):
        low = re.sub(r"\b" + alias + r"\b", canonical, low)
    # State-change objects bind the first room in their scoped phrase. Keep the
    # existing longest-category preference for other, whole-action callers.
    matches = [
        (match.start() if first_mention else 0, -len(room), room)
        for room in ROOM_TYPES
        for match in re.finditer(r"\b" + re.escape(room) + r"\b", low)
    ]
    return min(matches)[2] if matches else None


def _direction_key(text: str) -> str:
    if _TURN_AROUND.search(text):
        return "around"
    if _TURN_LEFT.search(text):
        return "left"
    if _TURN_RIGHT.search(text):
        return "right"
    return "forward"


def _action_verb(text: str) -> Optional[str]:
    m = _ACTION_VERB.search(text)
    return m.group(1).lower() if m else None


def _relation_target(
    text: str, marker: str, landmarks: List[str]
) -> Optional[str]:
    match = re.search(marker, text, re.I)
    if not match:
        return None
    tail = text[match.end():].lower()
    # A missing object must not bind to a later action's reference object.
    tail = re.split(r"\b(?:then|until|while|past|through|into|out of|to)\b", tail)[0]
    candidates = [
        (tail.find(item), -len(item), item)
        for item in landmarks
        if item and re.search(r"\b" + re.escape(item) + r"\b", tail)
    ]
    return min(candidates)[2] if candidates else None


def _room_change_requirements(text: str) -> List[Dict[str, Any]]:
    """Bind each state change to its own phrase, including 'out of A into B'."""
    changes = [
        (match.start(), match.end(), kind)
        for kind, marker in (("exit", _EXIT_PHRASE), ("enter", _ENTER_PHRASE))
        for match in re.finditer(marker, text)
    ]
    for _, end, kind in list(changes):
        if kind != "exit":
            continue
        # A bare 'into' inherits the exit's movement verb only in this clause.
        # 'Exit A and look into B' must not become an enter requirement.
        tail = text[end:]
        into = re.search(r"\binto\b", tail)
        if into is None:
            continue
        source = tail[:into.start()]
        if (_ROOM_TARGET_BOUNDARY.search(source) or _ACTION_VERB.search(source)
                or re.search(r"\b(?:look|see|face|watch)\b", source)):
            continue
        changes.append((end + into.start(), end + into.end(), "enter"))

    changes.sort()
    requirements = []
    for index, (_, end, kind) in enumerate(changes):
        limit = changes[index + 1][0] if index + 1 < len(changes) else len(text)
        target_text = _ROOM_TARGET_BOUNDARY.split(text[end:limit], maxsplit=1)[0]
        requirements.append({
            "type": kind, "target": _room_in(target_text, first_mention=True),
        })
    return requirements


def _action_spec(
    text: str, room: Optional[str], landmark: Optional[str],
    landmarks: List[str],
) -> Dict[str, Any]:
    """Compile action semantics instead of reducing an action to its head noun."""
    low = text.lower()
    verb = _action_verb(low)
    requirements: List[Dict[str, Any]] = []
    if _TURN_LEFT.search(low):
        requirements.append({"type": "turn", "direction": "left"})
    elif _TURN_RIGHT.search(low):
        requirements.append({"type": "turn", "direction": "right"})
    elif _TURN_AROUND.search(low):
        requirements.append({"type": "turn", "direction": "around"})
    requirements.extend(_room_change_requirements(low))
    if re.search(r"\b(?:past|pass(?:ing)?)\b", low):
        requirements.append({
            "type": "pass",
            "target": _relation_target(
                low, r"\b(?:past|pass(?:ing)?)\b", landmarks
            ),
        })
    if verb == "cross" or re.search(r"\bthrough\b", low):
        marker = r"\b(?:through|cross)\b"
        target = _relation_target(low, marker, landmarks)
        match = re.search(marker, low)
        target_text = re.split(
            r"\b(?:then|until|while|past|to)\b", low[match.end():]
        )[0] if match else ""
        target_room = _room_in(target_text)
        opening = re.search(r"\b(?:doors?|doorways?|openings?|archways?|entrances?)\b", target_text)
        if opening and (not target or not re.search(
            r"\b(?:doors?|doorways?|openings?|archways?|entrances?)\b", target
        )):
            # 'through the kitchen door' refers to an opening, not to the kitchen.
            # The raw phrase retains modifiers when the landmark list omitted them.
            target = opening.group(0)
        requirements.append({
            "type": "traverse" if target_room and not opening else "cross",
            "target": target or target_room,
        })
    return {
        "verb": verb,
        "requirements": requirements,
        "completion_semantics": "all_requirements",
        "temporal_markers": re.findall(r"\b(?:then|until|while|after|before)\b", low),
        "spatial_relations": re.findall(r"\b(?:left of|right of|between|next to|in front of)\b", low),
    }


def _coalesce_terminal_clauses(parts: List[str]) -> List[str]:
    """Keep descriptive comma clauses attached to a preceding terminal action."""
    merged: List[str] = []
    for part in parts:
        low = part.lower()
        continuation = (
            bool(merged)
            and bool(_TERMINAL_DIRECTIVE.match(merged[-1].lower()))
            and not _TERMINAL_DIRECTIVE.match(low)
            and _action_verb(low) is None
            and not _DIRECTION_VERB.search(low)
        )
        if continuation:
            merged[-1] = merged[-1] + ", " + part
        else:
            merged.append(part)
    return merged


def _terminal_target(text: str, landmarks: List[str]) -> Optional[str]:
    """Choose the earliest mentioned target; prefer the longest phrase on a tie."""
    matches = []
    for landmark in landmarks:
        if not landmark:
            continue
        pos = text.find(landmark)
        if pos >= 0:
            matches.append((pos, -len(landmark), landmark))
    return min(matches)[2] if matches else None


def _terminal_relation_policy(text: str, target: Optional[str]) -> Dict[str, Any]:
    """Represent reference-bound destinations without pretending they are objects."""
    low = str(text or "").lower()
    relations = []
    target_match = re.search(r"\b" + re.escape(target) + r"\b", low) if target else None
    matches = list(_TERMINAL_RELATION.finditer(low))
    agent_relative_place = bool(
        target_match and matches and matches[0].start() < target_match.start()
    )
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(low)
        # Use only the source span owned by this relation. This preserves 'stairs'
        # even if route landmarks say 'short flight of stairs', and never borrows
        # an unrelated reference from a subsequent relation or conditional clause.
        tail = re.split(
            r"[,;.!?]|\b(?:then|until|while|after|before)\b", low[match.end():end],
            maxsplit=1,
        )[0].strip()
        tail = re.sub(r"\s+and\s*$", "", tail).strip()
        reference = re.sub(r"^(?:the|a|an)\s+", "", tail).strip()
        if not reference or reference in {"it", "them", "there", "this", "that", "these", "those"}:
            reference = None
        # 'Stop in front of X' places the agent relative to X, even when X is
        # the primary target. 'Door at the top of stairs' qualifies the target.
        qualifies_target = bool(
            target_match and target_match.start() <= match.start() < target_match.end()
        )
        subject = "agent" if agent_relative_place and not qualifies_target else "terminal_target"
        if target_match is None:
            subject = "agent"
        relations.append({
            "type": match.lastgroup, "reference": reference, "subject": subject,
        })
    relational_region = bool(
        target and re.search(r"\b(?:area|space|position|spot)\s+between\b", target)
    )
    return {
        "target_kind": "relational_region" if relational_region else (
            "reference_bound_entity" if relations else "entity"
        ),
        "relations": relations,
        "verification_source": "current_view_relation_evidence"
        if relational_region or relations else "current_rgbd_entity_evidence",
    }


def build_anchor_chain(actions: Any, landmarks: Any) -> Dict[str, Any]:
    """actions/landmarks -> ordered anchor chain. Pure function, no side effects."""
    acts = _coalesce_terminal_clauses(_split_list(actions))
    lms = [l.lower() for l in _split_list(landmarks)]
    anchors: List[Dict[str, Any]] = []
    terminal_directives: List[str] = []
    intermediate_directives: List[str] = []
    terminal_targets: List[Optional[str]] = []
    used_landmarks = set()
    terminal_flags = [bool(_TERMINAL_DIRECTIVE.match(raw.lower())) for raw in acts]
    last_route_position = max(
        (i for i, is_terminal in enumerate(terminal_flags) if not is_terminal),
        default=-1,
    )

    for idx, raw in enumerate(acts):
        low = raw.lower()
        if terminal_flags[idx]:
            if idx <= last_route_position:
                intermediate_directives.append(raw)
                continue
            terminal_directives.append(raw)
            terminal_target = _terminal_target(low, lms)
            # In "Stop before going outside", the STOP phrase is a temporal
            # condition. The two-stage parser stores the preceding destination
            # as the final landmark, so retain that full phrase here.
            if terminal_target is None and _TEMPORAL_TERMINAL_CONDITION.match(low) and lms:
                terminal_target = lms[-1]
            terminal_targets.append(terminal_target)
            if terminal_target is not None:
                used_landmarks.add(terminal_target)
            continue
        room = _room_in(low)
        # a landmark counts as aligned to this action only if it literally occurs in it;
        # this is conservative on purpose -- a wrong alignment is worse than an abstain
        landmark = None
        for cand in lms:
            if cand and cand in low:
                landmark = cand
                break
        if landmark is not None:
            used_landmarks.add(landmark)

        action_spec = _action_spec(low, room, landmark, lms)
        requirements = action_spec["requirements"]
        motion_expression = bool(
            re.search(r"\b(?:walk|go|move|head|proceed|continue)\b", low)
            and re.search(r"\b(?:across|along|over)\b", low)
            and not re.search(r"\b(?:until|reach|toward|towards)\b", low)
        )
        if requirements:
            kind = "action_event"
            first = requirements[0]
            key = first.get("target") or first.get("direction")
        elif motion_expression and (
            landmark is None or landmark in BACKGROUND_SURFACES
        ):
            # Motion semantics come from the action text. Landmark extraction may
            # legitimately omit background surfaces such as ``floor``.
            kind, key = "motion", "displacement"
        elif room is not None and (landmark is None or room in landmark):
            kind, key = "location", room
        elif landmark is not None:
            kind, key = "object", landmark
        elif _DIRECTION_VERB.search(low):
            kind, key = "direction", _direction_key(low)
        else:
            kind, key = "unknown", None

        parse_status = "parsed" if kind != "unknown" else (
            "unsupported" if _action_verb(low) is not None else "parse_failure"
        )
        anchors.append(
            {
                "idx": len(anchors),
                "landmark": landmark,
                "room": room,
                "action": _action_verb(low),
                "action_spec": action_spec,
                "raw": raw,
                "terminal": False,
                "kind": kind,
                "key": key,
                "parse_status": parse_status,
                "verification": (
                    {
                        "predicate": "action_event",
                        "parameters": {
                            "min_turn_deg": 35.0,
                            "min_displacement_m": 0.05,
                            "min_common_views": 1,
                        },
                        "on_unverifiable": "abstain",
                    }
                    if kind == "action_event"
                    else _verification(kind, parse_status)
                ),
                "terminal_target": False,
            }
        )

    if anchors:
        anchors[-1]["terminal"] = True
        anchors[-1]["terminal_target"] = not any(terminal_targets)
        anchors[-1]["terminal_predecessor"] = bool(terminal_directives)

    aligned = sum(1 for a in anchors if a["landmark"] is not None or a["room"] is not None)
    return {
        "schema_version": ANCHOR_CHAIN_SCHEMA_VERSION,
        "action_semantics_version": ACTION_SEMANTICS_VERSION,
        "anchors": anchors,
        "n_anchors": len(anchors),
        "n_landmarks": len(lms),
        "n_landmarks_aligned": len(used_landmarks),
        # coverage = share of anchors carrying a groundable target. Anchors below this
        # bar are exactly the ones L1 must abstain on.
        "alignment_coverage": (aligned / len(anchors)) if anchors else 0.0,
        "kind_counts": {
            k: sum(1 for a in anchors if a["kind"] == k)
            for k in (
                "action_event", "location", "object", "direction", "motion", "unknown"
            )
        },
        "degenerate": not anchors,
        "terminal_policy": {
            "present": bool(terminal_directives),
            "directives": terminal_directives,
            "intermediate_directives": intermediate_directives,
            "target": (
                next((target for target in reversed(terminal_targets) if target), None)
                or (anchors[-1].get("key") if anchors else None)
            ),
            "predicate": "coordinated_stop",
            **_terminal_relation_policy(
                " ".join(terminal_directives),
                next((target for target in reversed(terminal_targets) if target), None),
            ),
            "requires": [
                "chain_complete", "all_previous_verified",
                "final_target_evidence", "stop_coordinator_allow",
            ],
            "in_progress_queue": False,
        },
    }

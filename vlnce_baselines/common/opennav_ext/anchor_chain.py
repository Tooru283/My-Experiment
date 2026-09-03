"""L0 -- anchor chain construction (ACN §3, 20260805).

`ACTION_DETECTION` and `LANDMARK_DETECTION` already run once per episode and produce two
*separate* lists: an ordered sub-action list and an ordered landmark list. Nothing aligns
them into "go where -> look for what". L0 is exactly that alignment, and nothing more.

Deliberately **zero new LLM calls**: the alignment is deterministic string work over two
lists the harness already has. One episode-level pass, cacheable for the whole split.

Output schema (strict, so offline spot-checks are cheap):

    {"anchors": [{"idx": 0, "landmark": "the kitchen counter", "room": "kitchen",
                  "action": "walk past", "terminal": false,
                  "kind": "object|location|direction|unknown", "key": "..."}, ...]}

`kind`/`key` are what L1's constraint queue consumes; the other fields are for auditing.

Alignment failures are explicit in v2: supported-but-unverifiable actions use a logged
``skip`` policy, while malformed parse failures abstain and cannot silently advance. Pure
terminal directives are removed from the progress queue and recorded in terminal_policy.
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
    r"^\s*(?:stop|wait|stay|stand|pause)\b", re.I
)

ANCHOR_CHAIN_SCHEMA_VERSION = "opennav.anchor_chain.v2"
DEFAULT_CONSECUTIVE_HITS = 2


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
    return {
        "predicate": "unverifiable",
        "parameters": {"min_consecutive_hits": 1},
        "on_unverifiable": "skip" if parse_status == "unsupported" else "abstain",
    }


def _room_in(text: str) -> Optional[str]:
    low = text.lower()
    # longest first so "living room" wins over "room"
    for room in sorted(ROOM_TYPES, key=len, reverse=True):
        if room in low:
            return room
    return None


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


def build_anchor_chain(actions: Any, landmarks: Any) -> Dict[str, Any]:
    """actions/landmarks -> ordered anchor chain. Pure function, no side effects."""
    acts = _coalesce_terminal_clauses(_split_list(actions))
    lms = [l.lower() for l in _split_list(landmarks)]
    anchors: List[Dict[str, Any]] = []
    terminal_directives: List[str] = []
    terminal_targets: List[Optional[str]] = []
    used_landmarks = set()

    for idx, raw in enumerate(acts):
        low = raw.lower()
        if _TERMINAL_DIRECTIVE.match(low):
            terminal_directives.append(raw)
            terminal_target = _terminal_target(low, lms)
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

        if room is not None and (landmark is None or room in landmark):
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
                "raw": raw,
                "terminal": False,
                "kind": kind,
                "key": key,
                "parse_status": parse_status,
                "verification": _verification(kind, parse_status),
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
        "anchors": anchors,
        "n_anchors": len(anchors),
        "n_landmarks": len(lms),
        "n_landmarks_aligned": len(used_landmarks),
        # coverage = share of anchors carrying a groundable target. Anchors below this
        # bar are exactly the ones L1 must abstain on.
        "alignment_coverage": (aligned / len(anchors)) if anchors else 0.0,
        "kind_counts": {
            k: sum(1 for a in anchors if a["kind"] == k)
            for k in ("location", "object", "direction", "unknown")
        },
        "degenerate": not anchors,
        "terminal_policy": {
            "present": bool(terminal_directives),
            "directives": terminal_directives,
            "target": (
                next((target for target in reversed(terminal_targets) if target), None)
                or (anchors[-1].get("key") if anchors else None)
            ),
            "predicate": "coordinated_stop",
            "requires": [
                "chain_complete", "all_previous_verified",
                "final_target_evidence", "stop_coordinator_allow",
            ],
            "in_progress_queue": False,
        },
    }

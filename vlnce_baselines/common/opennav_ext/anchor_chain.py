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

⚠ Alignment rate is an input precondition for L1, not an output of it. Measured on the
100-episode trace: landmarks align to the action sequence in 91/100 episodes. The other 9
produce anchors with ``landmark=None`` and MUST take L1's abstain branch rather than being
silently advanced past.
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


def _split_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        parts = _SPLIT.split(str(value))
    return [p.strip() for p in parts if p and p.strip()]


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


def build_anchor_chain(actions: Any, landmarks: Any) -> Dict[str, Any]:
    """actions/landmarks -> ordered anchor chain. Pure function, no side effects."""
    acts = _split_list(actions)
    lms = [l.lower() for l in _split_list(landmarks)]
    anchors: List[Dict[str, Any]] = []
    used_landmarks = set()

    for idx, raw in enumerate(acts):
        low = raw.lower()
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

        anchors.append(
            {
                "idx": idx,
                "landmark": landmark,
                "room": room,
                "action": _action_verb(low),
                "raw": raw,
                "terminal": False,
                "kind": kind,
                "key": key,
            }
        )

    if anchors:
        anchors[-1]["terminal"] = True

    aligned = sum(1 for a in anchors if a["landmark"] is not None or a["room"] is not None)
    return {
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
    }

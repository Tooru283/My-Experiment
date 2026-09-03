"""L1 -- constraint-queue progress location (ACN §5 / L1 spec 20260802, coded 20260805).

Replaces `completion_estimation`'s per-step LLM call with a monotone state machine.
Blueprint = CA-Nav's CSM (arXiv 2412.10137); only CSM is ported, not CVM (CVM needs a
value map we do not have).

WHY A QUEUE BREAKS THE CIRCULARITY
    Generic anchors (doorway/door/hallway/stairs) are 40% of all anchor words, appear in a
    median of 3 directions in the same step, and are unique only 14.6% of the time -- so
    "which doorway" cannot be settled by appearance. CA-Nav's answer is to never ask.
    Because the queue advances monotonically, satisfying constraint i only asks "is there
    *a* doorway within r right now"; *which* one is fixed implicitly by "constraints
    0..i-1 are already satisfied". Instance identification degrades to category detection
    plus order.

THE THREE STRUCTURAL GUARANTEES (each maps to a measured defect of the current module)
    75 progress regressions      -> j is monotone non-decreasing: structurally zero
    69/94 episodes report 100% at step 1 -> cannot skip a slot, must satisfy in order
    29/100 episodes never change -> j is driven by external evidence, not by wording

DETECTORS
    object    : RAM tag hit AND that direction's waypoint distance <= r   (no new model)
    direction : odometry pose window, signed yaw delta                    (no model at all)
    location  : NOT IMPLEMENTED -> abstain. The abstain rate IS the measured value of the
                location channel; do not paper over it with a guess.

⚠ Two things this module must never do, both from prior burns:
  1. Never use SpatialBot's reported object distance (r=0.171, 86% integers, measured a
     negative asset). Waypoint distance only.
  2. Never judge direction from a single frame's heading. The 20260720e single-frame
     direction prior measured net -0.91/-0.74. Window over poses.
"""
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# r must be calibrated, NOT copied from CA-Nav's 5 m: our candidate step length has a
# median of 1.50 m, so a 5 m gate would let nearly every candidate satisfy every object
# constraint. 2x the step median is the documented starting point.
DEFAULT_OBJECT_RADIUS_M = 3.0
DEFAULT_DIRECTION_WINDOW = 2
DEFAULT_TURN_DEG = 35.0
DEFAULT_AROUND_DEG = 120.0
DEFAULT_FORWARD_M = 1.0

# --- location detector (added 20260805 after the 51.9% stall measurement) ---
# RAM ALREADY EMITS ROOM CATEGORIES. Measured on the 100-episode trace: for 93.8% of
# location constraints the required room word appears in RAM's tags at some step. No new
# model is needed for recall. The problem is entirely SELECTIVITY:
#
#   detector rule                     satisfied-step share (median) | first satisfied at
#   any direction has the word                  88%                 |   step 0
#   >= 50% of directions                        60%                 |   step 0
#   >= 50% of directions + generic stoplist     33%                 |   9% into episode
#   >= 75% of directions + generic stoplist     25%                 |  18% into episode
#
# The naive "any direction" rule fires at step 0 for the median constraint -- it would
# reproduce the exact defect L1 exists to fix (the old module reported 100% at step 1 in
# 69/94 episodes). DOMINANCE is the fix and it has a physical reading: if you are *in* the
# kitchen most directions look like kitchen; if you are looking *into* it through a door,
# one direction does.
DEFAULT_LOCATION_DOMINANCE = 0.5
# RAM emits these indiscriminately -- "room" appears in 73.2% of ALL direction views,
# "hall" is similar. A constraint keyed on them carries no information, so it is treated
# as VACUOUSLY satisfied rather than abstained on: abstaining would stall the queue
# forever on a slot that never had any content. 43/162 location constraints are generic.
GENERIC_ROOMS = frozenset({"room", "hall"})

# --- term matching (rewritten 20260805 after the self-audit) ---
# The first version matched with `head in tag or tag in key`, i.e. raw substrings in both
# directions. Audited on the 100-episode trace: **44.7% of all object matches (767/1714)
# were substring artifacts** that a token-level rule rejects. The worst were not marginal:
#     bed  <- bedroom     x119   (being in a bedroom is not being at the bed)
#     red carpet <- red    x22
#     white rug  <- white  x19
#     bar  <- barber shop  x10
#     bed  <- bedcover     x12
# Because a satisfied constraint ADVANCES the queue, every one of these is a premature
# advance -- which is L1's designated primary failure mode (pre-registered gate: <=15%).
#
# But a pure token rule also loses matches that are genuinely right (door <- doorway x194,
# stairs <- stair x41). So: token boundaries + a light stemmer + an explicit, auditable
# variant table. A hand-listed pair is reviewable; a substring rule is not.
_VARIANTS = (
    ("door", "doorway"), ("door", "entryway"), ("door", "entry way"),
    ("stairs", "staircase"), ("stairs", "stairway"), ("stairs", "stairwell"),
    ("stairs", "steps"), ("rug", "carpet"), ("couch", "sofa"),
    ("tub", "bathtub"), ("counter", "countertop"), ("hall", "hallway"),
    ("corridor", "hallway"), ("picture", "painting"), ("fridge", "refrigerator"),
    ("tv", "television"), ("closet", "wardrobe"), ("sink", "washbasin"),
)
VARIANT_PAIRS = frozenset(
    frozenset(pair) for pair in _VARIANTS
)


def _stem(word: str) -> str:
    """Deliberately minimal: plural -s only. Anything cleverer needs its own audit."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def term_matches(head: str, key: str, tag: str) -> bool:
    """Does RAM tag `tag` satisfy a constraint whose phrase is `key` (head noun `head`)?"""
    if not head or not tag:
        return False
    if tag == key or tag == head:
        return True
    tag_words = tag.split()
    if head in tag_words:
        return True
    sh = _stem(head)
    if sh == _stem(tag) or any(sh == _stem(w) for w in tag_words):
        return True
    for w in tag_words + [tag]:
        if frozenset((head, w)) in VARIANT_PAIRS:
            return True
    return False


def _norm_angle(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def _planar_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[2] - b[2])


class ConstraintQueueLocator:
    """Monotone constraint queue over an L0 anchor chain.

    `j` = index of the first *unsatisfied* constraint = how far along the chain we are.
    `j` never decreases. When evidence is missing the queue abstains (holds `j`); it
    never guesses.
    """

    def __init__(
        self,
        object_radius_m: float = DEFAULT_OBJECT_RADIUS_M,
        direction_window: int = DEFAULT_DIRECTION_WINDOW,
        turn_deg: float = DEFAULT_TURN_DEG,
        around_deg: float = DEFAULT_AROUND_DEG,
        forward_m: float = DEFAULT_FORWARD_M,
        heading_sign: float = -1.0,
        enable_location: bool = False,
        location_dominance: float = DEFAULT_LOCATION_DOMINANCE,
        generic_rooms: Optional[Iterable[str]] = None,
        vacuous_unknown: bool = True,
    ) -> None:
        self.location_dominance = location_dominance
        self.vacuous_unknown = vacuous_unknown
        self.generic_rooms = frozenset(
            generic_rooms if generic_rooms is not None else GENERIC_ROOMS
        )
        self.object_radius_m = object_radius_m
        self.direction_window = direction_window
        self.turn_deg = turn_deg
        self.around_deg = around_deg
        self.forward_m = forward_m
        # sign convention calibrated as -1 (log §一quater by-product); exposed so a future
        # re-calibration is a config change, not a code edit
        self.heading_sign = heading_sign
        self.enable_location = enable_location
        self.reset_episode()

    def reset_episode(self, anchors: Optional[List[Dict[str, Any]]] = None) -> None:
        self.anchors: List[Dict[str, Any]] = list(anchors or [])
        self.j = 0
        self.ever_satisfied: Set[int] = set()
        self.hit_streaks: Dict[int, int] = {}
        self.pose_history: List[Dict[str, Any]] = []
        self.j_history: List[int] = []
        self.abstain_steps = 0
        self.stall_kind_counts: Dict[str, int] = {}

    # ---------------- detectors ----------------

    @staticmethod
    def _distance_of(entry: Any) -> Optional[float]:
        """Accepts either {"distance": d, ...} (the shared view_geometry shape) or a bare
        number, so the trainer extracts geometry once and both L1 and M2 consume it."""
        if entry is None:
            return None
        if isinstance(entry, dict):
            entry = entry.get("distance")
        try:
            return float(entry)
        except (TypeError, ValueError):
            return None

    def _satisfy_object(
        self, key: str, view_tags: Dict[str, Iterable[str]], view_geometry: Dict[str, Any],
        max_distance_m: Optional[float] = None,
    ) -> bool:
        if not key:
            return False
        head = key.split()[-1]
        radius = self.object_radius_m if max_distance_m is None else float(max_distance_m)
        for direction_id, tags in view_tags.items():
            dist = self._distance_of(view_geometry.get(direction_id))
            if dist is None or dist > radius:
                continue
            for tag in tags:
                t = str(tag).strip().lower()
                if not t:
                    continue
                if term_matches(head, key, t):
                    return True
        return False

    def _satisfy_direction(self, key: str) -> bool:
        w = self.direction_window
        if len(self.pose_history) < w + 1:
            return False
        h0 = self.pose_history[-(w + 1)].get("heading")
        h1 = self.pose_history[-1].get("heading")
        p0 = self.pose_history[-(w + 1)].get("position")
        p1 = self.pose_history[-1].get("position")
        if key == "forward":
            if p0 is None or p1 is None:
                return False
            return _planar_distance(p0, p1) > self.forward_m
        if h0 is None or h1 is None:
            return False
        deg = math.degrees(_norm_angle(float(h1) - float(h0))) * self.heading_sign
        if key == "around":
            return abs(deg) > self.around_deg
        if key == "left":
            return deg > self.turn_deg
        if key == "right":
            return deg < -self.turn_deg
        return False

    @staticmethod
    def _tag_matches_room(tags: Iterable[str], room: str, head: str) -> bool:
        # Rooms keep a looser rule than objects on purpose: RAM writes "living room;" with
        # punctuation and "dining room area", and a room constraint is region-level -- over-firing
        # on a room is far less harmful than over-firing on an object, because the
        # dominance threshold already requires it in >=50% of directions.
        for tag in tags:
            t = str(tag).strip().lower()
            if not t:
                continue
            if room in t or term_matches(head, room, t):
                return True
        return False

    def _satisfy_location(
        self, key: str, view_tags: Dict[str, Iterable[str]],
        min_view_ratio: Optional[float] = None,
    ) -> Optional[bool]:
        """Dominance rule. Returns True/False, or None to abstain (no evidence at all)."""
        if not key:
            return None
        if key in self.generic_rooms:
            # vacuous: carries no information, must not block the queue
            return True
        if not view_tags:
            return None
        head = key.split()[-1]
        hits = sum(
            1 for tags in view_tags.values() if self._tag_matches_room(tags, key, head)
        )
        threshold = self.location_dominance if min_view_ratio is None else float(min_view_ratio)
        return (hits / len(view_tags)) >= threshold

    def _satisfy(
        self,
        anchor: Dict[str, Any],
        view_tags: Dict[str, Iterable[str]],
        view_geometry: Dict[str, Any],
    ) -> bool:
        kind = anchor.get("kind")
        key = anchor.get("key")
        verification = anchor.get("verification")
        parameters = verification.get("parameters", {}) if isinstance(verification, dict) else {}
        if kind == "object":
            return self._satisfy_object(
                str(key or ""), view_tags, view_geometry,
                parameters.get("max_waypoint_distance_m"),
            )
        if kind == "direction":
            return self._satisfy_direction(str(key or ""))
        if kind == "location" and self.enable_location:
            return bool(self._satisfy_location(
                str(key or ""), view_tags, parameters.get("min_view_ratio")
            ))
        # Unknown anchors are handled by their explicit verification policy in update().
        return False

    @staticmethod
    def _verification_policy(anchor: Dict[str, Any]) -> str:
        verification = anchor.get("verification")
        if isinstance(verification, dict):
            return str(verification.get("on_unverifiable") or "abstain")
        return "skip" if anchor.get("kind") == "unknown" else "abstain"

    @staticmethod
    def _required_hits(anchor: Dict[str, Any]) -> int:
        verification = anchor.get("verification")
        parameters = verification.get("parameters", {}) if isinstance(verification, dict) else {}
        try:
            return max(1, int(parameters.get("min_consecutive_hits", 1)))
        except (TypeError, ValueError):
            return 1

    # ---------------- step ----------------

    def update(
        self,
        step_id: int,
        position: Any,
        heading: Any,
        view_tags: Optional[Dict[str, Iterable[str]]] = None,
        view_geometry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        view_tags = view_tags or {}
        view_geometry = view_geometry or {}
        pos = None
        if position is not None:
            try:
                pos = [float(position[0]), float(position[1]), float(position[2])]
            except Exception:
                pos = None
        self.pose_history.append(
            {"step_id": step_id, "position": pos, "heading": heading}
        )

        if not self.anchors:
            self.j_history.append(0)
            return self._emit(step_id, degenerate=True)

        # Satisfaction is PERSISTENT and evaluated for every slot, not just the head.
        # Evaluating only the head loses evidence: if slot i is stalled, the agent can
        # walk straight past slot i+1's landmark and the window is gone forever.
        for idx, anchor in enumerate(self.anchors):
            if idx in self.ever_satisfied:
                continue
            if anchor.get("kind") == "unknown":
                if self._verification_policy(anchor) == "skip":
                    self.ever_satisfied.add(idx)
                continue
            if self._satisfy(anchor, view_tags, view_geometry):
                self.hit_streaks[idx] = self.hit_streaks.get(idx, 0) + 1
            else:
                self.hit_streaks[idx] = 0
            if self.hit_streaks.get(idx, 0) >= self._required_hits(anchor):
                self.ever_satisfied.add(idx)

        advanced = 0
        while self.j < len(self.anchors) and self.j in self.ever_satisfied:
            self.j += 1
            advanced += 1
        if advanced == 0:
            self.abstain_steps += 1
        if self.j < len(self.anchors):
            kind = str(self.anchors[self.j].get("kind"))
            self.stall_kind_counts[kind] = self.stall_kind_counts.get(kind, 0) + 1

        self.j_history.append(self.j)
        return self._emit(step_id, degenerate=False)

    def _emit(self, step_id: int, degenerate: bool) -> Dict[str, Any]:
        n = len(self.anchors)
        current = self.anchors[self.j] if 0 <= self.j < n else None
        return {
            "j": self.j,
            "n_anchors": n,
            "progress": (self.j / n) if n else None,
            "complete": bool(n and self.j >= n),
            "current_kind": current.get("kind") if current else None,
            "current_key": current.get("key") if current else None,
            "current_raw": current.get("raw") if current else None,
            "abstained_this_step": bool(self.j_history[-2:] and len(self.j_history) >= 2
                                        and self.j_history[-1] == self.j_history[-2]),
            "abstain_steps": self.abstain_steps,
            "ever_satisfied": sorted(self.ever_satisfied),
            "hit_streaks": dict(self.hit_streaks),
            "distinct_j_values": len(set(self.j_history)),
            # structural guarantee, asserted rather than hoped for
            "regressions": sum(
                1
                for a, b in zip(self.j_history, self.j_history[1:])
                if b < a
            ),
            "stall_kind_counts": dict(self.stall_kind_counts),
            "degenerate": degenerate,
            "step_id": step_id,
        }

    def completion_text(self) -> str:
        """Drop-in replacement for the LLM's `estimation` string.

        Deliberately states only what the state machine verified. It does NOT say
        "you have completed X%" -- the old module's percentage was the thing that
        regressed 75 times and read 100% at step 1 in 69/94 episodes.
        """
        n = len(self.anchors)
        if not n:
            return "No decomposed sub-actions are available for this instruction."
        done = [a["raw"] for a in self.anchors[: self.j]]
        if self.j >= n:
            return (
                "All {} sub-actions have been verified as executed: {}.".format(
                    n, "; ".join(done)
                )
            )
        current = self.anchors[self.j]
        head = (
            "Verified as executed so far ({} of {}): {}. ".format(
                self.j, n, "; ".join(done)
            )
            if done
            else "No sub-action has been verified as executed yet. "
        )
        return head + (
            "The sub-action still in progress is: {}.".format(current.get("raw"))
        )

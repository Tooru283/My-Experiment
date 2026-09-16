"""Verify route actions using an ordered queue and observable execution evidence.

Only the current action may complete. Its start pose is captured when activated,
not on the following observation. The controller and heading use left-positive yaw.
Room transitions compare heading-corrected RGB view centers and require movement;
RAM room classifications remain hypotheses, not instance or boundary ground truth.
Cross/pass/traverse are explicitly unsupported until spatial evidence is available.
Legacy object proximity uses waypoint distance, which is NOT object distance.
"""
import math
import re
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
        heading_sign: float = 1.0,
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
        # get_agent_info() and MoveHighToLowActionEval use left-positive yaw.
        # Verified against 88 executed EP10 movements on 2026-09-08.
        self.heading_sign = heading_sign
        self.enable_location = enable_location
        self.reset_episode()

    def reset_episode(self, anchors: Optional[List[Dict[str, Any]]] = None) -> None:
        self.anchors: List[Dict[str, Any]] = list(anchors or [])
        self.j = 0
        self.ever_satisfied: Set[int] = set()
        self.hit_streaks: Dict[int, int] = {}
        self.pose_history: List[Dict[str, Any]] = []
        self.observation_history: List[Dict[str, Any]] = []
        self.action_states: Dict[int, Dict[str, Any]] = {}
        self.completion_events: List[Dict[str, Any]] = []
        self.last_evaluation: Dict[str, Any] = {}
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
        start = self.action_states.get(self.j, {})
        history = [p for p in self.pose_history if p["step_id"] >= start.get("started_at", 0)]
        if len(history) < 2:
            return False
        h0, h1 = history[0].get("heading"), history[-1].get("heading")
        p0, p1 = history[0].get("position"), history[-1].get("position")
        if key == "forward":
            if p0 is None or p1 is None or h0 is None:
                return False
            # A lateral/backward displacement is not evidence of walking forward.
            forward = -(p1[0] - p0[0]) * math.sin(h0) - (p1[2] - p0[2]) * math.cos(h0)
            return forward > self.forward_m
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

    def _satisfy_motion(self, min_displacement_m: float) -> bool:
        state = self.action_states.get(self.j, {})
        start = state.get("start_position")
        current = self.pose_history[-1].get("position") if self.pose_history else None
        if start is None or current is None:
            return False
        return _planar_distance(start, current) > float(min_displacement_m)


    @staticmethod
    def _tag_matches_room(tags: Iterable[str], room: str, head: str) -> bool:
        # Keep full room types: the generic head 'room' cannot distinguish a living
        # room from a dining room. Punctuation and explicit aliases remain accepted.
        aliases = {
            "living room": ("living room", "living area"),
            "dining room": ("dining room", "dining area"),
            "hallway": ("hallway", "corridor"),
            "corridor": ("hallway", "corridor"),
        }.get(room, (room,))
        for tag in tags:
            t = str(tag).strip().lower()
            if not t:
                continue
            if any(re.search(r"\b" + re.escape(alias) + r"\b", t) for alias in aliases):
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

    def _room_ratio(
        self, room: str, view_tags: Dict[str, Iterable[str]],
        direction_ids: Optional[Set[str]] = None,
    ) -> Optional[float]:
        if not room or room in self.generic_rooms or not view_tags:
            return None
        ids = set(view_tags) if direction_ids is None else set(direction_ids)
        ids &= set(view_tags)
        if not ids:
            return None
        head = room.split()[-1]
        hits = sum(
            1 for direction_id in ids
            if self._tag_matches_room(view_tags[direction_id], room, head)
        )
        return hits / len(ids)

    def _activate_action(self, idx, step_id, position, heading, view_tags):
        """Capture the legal start before the first command for the new queue head."""
        if idx not in self.action_states:
            self.action_states[idx] = {
                "started_at": step_id,
                "start_position": position,
                "last_heading": heading,
                "signed_turn_deg": 0.0,
                "last_tags": dict(view_tags),
                "last_view_heading": heading,
                "previous_position": position,
                "previous_step": step_id,
            }

    @staticmethod
    def _matched_views(before, after, before_heading, after_heading, tolerance_deg=15.0):
        """Match RGB view centers in world yaw, not waypoint bearings or raw IDs.

        This pipeline's RGB IDs are the 12 camera orientations spaced by 30 degrees.
        A waypoint within a camera view need not lie at that image's center.
        """
        def bearings(tags, heading):
            if heading is None or not math.isfinite(float(heading)):
                return {}
            result = {}
            for key in tags:
                try:
                    direction = int(key)
                    if 0 <= direction < 12 and str(direction) == str(key):
                        result[key] = float(heading) + direction * math.pi / 6.0
                except (TypeError, ValueError):
                    continue
            return result
        old, new = bearings(before, before_heading), bearings(after, after_heading)
        pairs = sorted(
            (abs(math.degrees(_norm_angle(a - b))), i, j)
            for i, a in old.items() for j, b in new.items()
        )
        used_old, used_new, matches = set(), set(), []
        for error, i, j in pairs:
            if error <= tolerance_deg and i not in used_old and j not in used_new:
                matches.append((i, j))
                used_old.add(i)
                used_new.add(j)
        return matches

    def _room_presence(self, room, tags, ids):
        ratio = self._room_ratio(room, tags, ids)
        if ratio is None:
            return None, ratio
        if ratio >= self.location_dominance:
            return True, ratio
        # Absence of the requested RAM label is not positive outside evidence.
        other_rooms = (
            "bedroom", "bathroom", "kitchen", "hallway", "corridor", "closet",
            "living room", "dining room", "office", "garage", "lobby", "stairwell",
        )
        outside_hits = sum(any(
            self._tag_matches_room(tags[i], other, other.split()[-1])
            for other in other_rooms if other != room
        ) for i in ids)
        return (False if outside_hits / len(ids) >= self.location_dominance else None), ratio

    def _evaluate_action_event(
        self,
        idx: int,
        anchor: Dict[str, Any],
        step_id: int,
        position: Any,
        heading: Any,
        view_tags: Dict[str, Iterable[str]],
        action_receipt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        spec = anchor.get("action_spec") or {}
        requirements = list(spec.get("requirements") or [])
        verification = anchor.get("verification") or {}
        parameters = verification.get("parameters") or {}
        self._activate_action(idx, step_id, position, heading, view_tags)
        state = self.action_states[idx]
        if heading is not None and state.get("last_heading") is not None:
            delta = math.degrees(
                _norm_angle(float(heading) - float(state["last_heading"]))
            ) * self.heading_sign
            state["signed_turn_deg"] += delta
        state["last_heading"] = heading

        min_move = float(parameters.get("min_displacement_m", 0.05))
        receipt = dict(action_receipt or {})
        receipt_displacement = receipt.get("displacement")
        try:
            receipt_displacement = (
                float(receipt_displacement)
                if receipt_displacement is not None else None
            )
        except (TypeError, ValueError):
            receipt_displacement = None
        displacement = (
            receipt_displacement
            if receipt_displacement is not None
            else (
                _planar_distance(state["previous_position"], position)
                if state.get("previous_position") is not None and position is not None
                else None
            )
        )
        collision = receipt.get("collision")
        movement_observed = bool(
            displacement is not None
            and displacement > min_move
            and not bool(collision)
        )
        results: List[Dict[str, Any]] = []
        missing: List[str] = []
        unsupported: List[str] = []

        for requirement in requirements:
            req_type = str(requirement.get("type") or "")
            if req_type == "turn":
                direction = requirement.get("direction")
                signed = float(state["signed_turn_deg"])
                threshold = float(parameters.get("min_turn_deg", self.turn_deg))
                if heading is None or not math.isfinite(float(heading)):
                    missing.append("heading_missing")
                    complete = False
                elif direction == "left":
                    complete = signed >= threshold
                elif direction == "right":
                    complete = signed <= -threshold
                else:
                    complete = abs(signed) >= self.around_deg
                results.append({
                    "type": "turn", "direction": direction,
                    "completed": complete, "signed_turn_deg": signed,
                    "threshold_deg": self.around_deg
                    if direction == "around" else threshold,
                })
                if not complete:
                    missing.append("required_turn_not_reached")
                continue

            if req_type in ("enter", "exit"):
                room = str(requirement.get("target") or "")
                previous_tags = state.get("last_tags") or {}
                pairs = self._matched_views(
                    previous_tags, view_tags, state.get("last_view_heading"), heading
                )
                before_ids = {p[0] for p in pairs}
                after_ids = {p[1] for p in pairs}
                min_common = int(parameters.get("min_common_views", 1))
                before_inside, before_ratio = self._room_presence(room, previous_tags, before_ids)
                after_inside, after_ratio = self._room_presence(room, view_tags, after_ids)
                comparable = (
                    len(pairs) >= min_common
                    and before_ratio is not None
                    and after_ratio is not None
                )
                transition = (
                    before_inside is False and after_inside is True
                    if req_type == "enter"
                    else before_inside is True and after_inside is False
                )
                moved = movement_observed
                complete = bool(comparable and transition and moved)
                results.append({
                    "type": req_type, "target": room, "completed": complete,
                    "comparable": comparable,
                    "common_direction_ids": sorted(before_ids & after_ids),
                    "matched_direction_pairs": [list(pair) for pair in pairs],
                    "comparison": "world_yaw_rgb_centers",
                    "before_room_ratio": before_ratio,
                    "after_room_ratio": after_ratio,
                    "before_inside": before_inside,
                    "after_inside": after_inside,
                    "displacement_m": displacement,
                    "action_receipt_id": receipt.get("receipt_id"),
                    "collision": collision,
                })
                if not room:
                    missing.append("target_missing")
                elif not comparable:
                    missing.append("room_observations_not_comparable")
                elif before_inside is None or after_inside is None:
                    missing.append("room_state_unobserved")
                elif not moved:
                    missing.append("effective_movement_missing")
                elif not transition:
                    missing.append("{}_state_transition_missing".format(req_type))
                continue

            unsupported.append(req_type or "unknown")
            missing.append("verifier_not_implemented:" + (req_type or "unknown"))
            if not requirement.get("target"):
                missing.append("target_missing")

        state["last_tags"] = dict(view_tags)
        state["last_view_heading"] = heading
        state["previous_position"] = position
        completed = bool(results and all(item["completed"] for item in results)
                         and not unsupported)
        status = "completed" if completed else (
            "unknown" if unsupported or any(m in missing for m in (
                "target_missing", "room_state_unobserved", "heading_missing"
            )) or any(
                item.get("comparable") is False for item in results
            ) else "incomplete"
        )
        return {
            "anchor_index": idx,
            "status": status,
            "requirements": results,
            "unsupported_requirements": unsupported,
            "missing_evidence": sorted(set(missing)),
            "event_interval": [state["started_at"], step_id],
            "confirmed_at": step_id if completed else None,
        }

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
        if kind == "motion":
            return self._satisfy_motion(
                parameters.get("min_displacement_m", self.forward_m))
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
        action_receipt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        view_tags = view_tags or {}
        view_geometry = view_geometry or {}
        pos = None
        if position is not None:
            try:
                pos = [float(position[0]), float(position[1]), float(position[2])]
            except Exception:
                pos = None
        receipt = dict(action_receipt or {})
        self.pose_history.append(
            {
                "step_id": step_id,
                "position": pos,
                "heading": heading,
                "previous_action_receipt": receipt,
            }
        )

        if not self.anchors:
            self.j_history.append(0)
            return self._emit(step_id, degenerate=True)

        self.observation_history.append({
            "step_id": step_id,
            "position": pos,
            "heading": heading,
            "view_tags": {str(k): list(v) for k, v in view_tags.items()},
            "previous_action_receipt": receipt,
        })
        self.observation_history = self.observation_history[-8:]

        if self.j >= len(self.anchors):
            self.j_history.append(self.j)
            return self._emit(step_id, degenerate=False)

        # Only the current queue head may complete. Future anchors must not bank
        # noun sightings or turns that happened before their legal start time.
        advanced = 0
        anchor = self.anchors[self.j]
        self._activate_action(self.j, step_id, pos, heading, view_tags)
        if anchor.get("kind") == "unknown":
            self.last_evaluation = {
                "anchor_index": self.j,
                "status": "unknown",
                "missing_evidence": ["unsupported_or_unparsed_action"],
            }
            if self._verification_policy(anchor) == "skip":
                self.ever_satisfied.add(self.j)
        elif anchor.get("kind") == "action_event":
            self.last_evaluation = self._evaluate_action_event(
                self.j, anchor, step_id, pos, heading, view_tags,
                action_receipt=receipt,
            )
            if self.last_evaluation["status"] == "completed":
                self.ever_satisfied.add(self.j)
                self.completion_events.append(dict(self.last_evaluation))
        else:
            satisfied = self._satisfy(anchor, view_tags, view_geometry)
            self.hit_streaks[self.j] = (
                self.hit_streaks.get(self.j, 0) + 1 if satisfied else 0
            )
            completed = (
                self.hit_streaks.get(self.j, 0) >= self._required_hits(anchor)
            )
            self.last_evaluation = {
                "anchor_index": self.j,
                "status": "completed" if completed else "incomplete",
                "missing_evidence": [] if completed else ["predicate_not_satisfied"],
                "event_interval": [step_id, step_id],
                "confirmed_at": step_id if completed else None,
            }
            if completed:
                self.ever_satisfied.add(self.j)
                self.completion_events.append(dict(self.last_evaluation))

        if self.j in self.ever_satisfied:
            self.j += 1
            advanced = 1
        if advanced == 0:
            self.abstain_steps += 1
        if self.j < len(self.anchors):
            if advanced:
                self._activate_action(self.j, step_id, pos, heading, view_tags)
            kind = str(self.anchors[self.j].get("kind"))
            self.stall_kind_counts[kind] = self.stall_kind_counts.get(kind, 0) + 1

        self.j_history.append(self.j)
        return self._emit(step_id, degenerate=False)

    def _emit(self, step_id: int, degenerate: bool) -> Dict[str, Any]:
        n = len(self.anchors)
        current = self.anchors[self.j] if 0 <= self.j < n else None
        return {
            "j": self.j,
            "verification_version": "state_change.v3.4",
            "n_anchors": n,
            "progress": (self.j / n) if n else None,
            "complete": bool(n and self.j >= n),
            "current_kind": current.get("kind") if current else None,
            "current_key": current.get("key") if current else None,
            "current_raw": current.get("raw") if current else None,
            "abstained_this_step": bool(self.j < n and self.j_history[-2:] and len(self.j_history) >= 2
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
            "current_action": (
                (current.get("action_spec") or {}).get("requirements")
                if current else None
            ),
            "last_evaluation": dict(self.last_evaluation),
            "completion_events": list(self.completion_events),
            "evidence_buffer_size": len(self.observation_history),
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

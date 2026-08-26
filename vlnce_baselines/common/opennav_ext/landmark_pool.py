"""M2 -- cross-step landmark pool (ACN §3, 20260805).

`landmark_matching.py` is already referenced by three modules but is stateless: every step
re-asks "is this term in this view" and throws the answer away. M2 keeps it.

An entry is opened the first time a category is seen and then accumulates:
    category, fused_pos (running mean), n_obs, first/last step, min_distance,
    visible (seen this step), arrived (ever seen within ARRIVAL_RADIUS_M)

Position source is the **candidate waypoint world coordinate** of the direction the tag was
seen in, using the calibrated transform

    theta = -heading - angle_rad;  x += d*sin(theta);  z -= d*cos(theta)

(median reconstruction error 0.000 m). ⚠ This is a *direction-level* position, not an
object-level one: every tag seen in the same direction inherits the same coordinate. That
is a known, measured limitation -- it is what collapsed the 20260730 registration test --
so `fused_pos` is fit for "roughly where", never for "which instance".

⚠ Never fuse SpatialBot's reported distances here (measured negative asset). Waypoint
geometry only.

Pool is cleared on stage switch, per spec: it is evidence for the *current* sub-goal.
"""
import math
from typing import Any, Dict, Iterable, List, Optional

from vlnce_baselines.common.opennav_ext.progress_locator import term_matches

DEFAULT_ARRIVAL_RADIUS_M = 3.0
# tags too generic to carry evidential weight; identical to the registration test's list
STOPWORDS = frozenset(
    {"wall", "ceiling", "floor", "room", "lead to", "light", "shadow", "corner"}
)


def waypoint_world_position(
    position: List[float], heading: float, angle_rad: float, distance: float
) -> List[float]:
    theta = -float(heading) - float(angle_rad)
    return [
        position[0] + distance * math.sin(theta),
        position[1],
        position[2] - distance * math.cos(theta),
    ]


class LandmarkPool:
    def __init__(
        self,
        arrival_radius_m: float = DEFAULT_ARRIVAL_RADIUS_M,
        merge_radius_m: float = 2.0,
        drop_stopwords: bool = True,
        restrict_to_vocabulary: bool = True,
    ) -> None:
        self.arrival_radius_m = arrival_radius_m
        self.merge_radius_m = merge_radius_m
        self.drop_stopwords = drop_stopwords
        # ⚠ First offline replay (20260805): admitting EVERY RAM tag gave a median pool of
        # 92 entries per episode and made "the final landmark has arrival evidence" true in
        # 100/100 episodes -- i.e. L4's third conjunct carried no information at all. RAM
        # emits a dozen tags per direction, so an unrestricted pool is just a tag dump.
        # Restricting admission to the categories the anchor chain actually asks about is
        # what makes the pool evidence rather than noise.
        self.restrict_to_vocabulary = restrict_to_vocabulary
        self.vocabulary: Optional[frozenset] = None
        self.reset_stage()

    def set_vocabulary(self, terms: Optional[Iterable[str]]) -> None:
        """Categories worth remembering = the anchor chain's landmarks and rooms.

        Head nouns are stored too, because RAM emits "screen door" where the instruction
        says "the door". Call once per episode, after L0 builds the chain.
        """
        if not terms:
            self.vocabulary = None
            return
        vocab = set()
        for t in terms:
            s = str(t or "").strip().lower()
            if not s:
                continue
            vocab.add(s)
            parts = s.split()
            if parts:
                vocab.add(parts[-1])
        self.vocabulary = frozenset(vocab) or None

    def _in_vocabulary(self, category: str) -> bool:
        if not self.restrict_to_vocabulary or not self.vocabulary:
            return True
        for term in self.vocabulary:
            th = term.split()[-1] if term.split() else term
            if term_matches(th, term, category):
                return True
        return False

    def reset_stage(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.stage_id = getattr(self, "stage_id", -1) + 1

    def reset_episode(self, vocabulary: Optional[Iterable[str]] = None) -> None:
        self.entries = []
        self.stage_id = 0
        self.set_vocabulary(vocabulary)

    def _find(self, category: str, pos: Optional[List[float]]) -> Optional[Dict[str, Any]]:
        best, best_d = None, None
        for e in self.entries:
            if e["category"] != category:
                continue
            if pos is None or e["fused_pos"] is None:
                return e
            d = math.hypot(pos[0] - e["fused_pos"][0], pos[2] - e["fused_pos"][2])
            if d < self.merge_radius_m and (best_d is None or d < best_d):
                best, best_d = e, d
        return best

    def update(
        self,
        step_id: int,
        position: Any,
        heading: Any,
        view_tags: Optional[Dict[str, Iterable[str]]] = None,
        view_geometry: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """view_tags: {direction_id: [tag,...]}, view_geometry: {direction_id: {angle_rad, distance}}"""
        view_tags = view_tags or {}
        view_geometry = view_geometry or {}
        try:
            pos = [float(position[0]), float(position[1]), float(position[2])]
            head = float(heading)
        except Exception:
            pos, head = None, None

        for e in self.entries:
            e["visible"] = False

        seen_this_step = 0
        for direction_id, tags in view_tags.items():
            geo = view_geometry.get(direction_id) or {}
            dist = geo.get("distance")
            wp = None
            if pos is not None and head is not None and geo.get("angle_rad") is not None \
                    and dist is not None:
                wp = waypoint_world_position(pos, head, geo["angle_rad"], float(dist))
            for tag in tags:
                cat = str(tag).strip().lower()
                if not cat or (self.drop_stopwords and cat in STOPWORDS):
                    continue
                if not self._in_vocabulary(cat):
                    continue
                seen_this_step += 1
                entry = self._find(cat, wp)
                if entry is None:
                    self.entries.append(
                        {
                            "category": cat,
                            "fused_pos": list(wp) if wp else None,
                            "n_obs": 1,
                            "first_step": step_id,
                            "last_step": step_id,
                            "min_distance": float(dist) if dist is not None else None,
                            "visible": True,
                            "arrived": bool(
                                dist is not None
                                and float(dist) <= self.arrival_radius_m
                            ),
                        }
                    )
                    continue
                n = entry["n_obs"]
                if wp is not None:
                    if entry["fused_pos"] is None:
                        entry["fused_pos"] = list(wp)
                    else:
                        entry["fused_pos"] = [
                            (entry["fused_pos"][i] * n + wp[i]) / (n + 1)
                            for i in range(3)
                        ]
                entry["n_obs"] = n + 1
                entry["last_step"] = step_id
                entry["visible"] = True
                if dist is not None:
                    d = float(dist)
                    if entry["min_distance"] is None or d < entry["min_distance"]:
                        entry["min_distance"] = d
                    if d <= self.arrival_radius_m:
                        entry["arrived"] = True

        return {
            "stage_id": self.stage_id,
            "pool_size": len(self.entries),
            "visible_count": sum(1 for e in self.entries if e["visible"]),
            "arrived_count": sum(1 for e in self.entries if e["arrived"]),
            "tags_seen_this_step": seen_this_step,
            "entries": [
                {k: v for k, v in e.items() if k != "fused_pos"} for e in self.entries
            ],
        }

    # ---------------- queries used by L4 ----------------

    def query(self, term: Any) -> Optional[Dict[str, Any]]:
        """Match `term` against the pool. Returns the best-evidenced entry or None.

        ⚠ Uses `progress_locator.term_matches`, the SAME matcher L1 uses. The first
        version matched raw substrings here too; the 20260805 self-audit found that rule
        produced 44.7% false matches in L1 (bed <- bedroom, red carpet <- red, ...), and
        this query feeds L4's evidence conjunct, so the same inflation applied. Keeping
        one matcher means one audit covers both.
        """
        if not term:
            return None
        t = str(term).strip().lower()
        head = t.split()[-1] if t.split() else t
        hits = [e for e in self.entries if term_matches(head, t, e["category"])]
        if not hits:
            return None
        return max(hits, key=lambda e: (e["arrived"], e["n_obs"]))

    def has_arrived(self, term: Any) -> bool:
        entry = self.query(term)
        return bool(entry and entry["arrived"])

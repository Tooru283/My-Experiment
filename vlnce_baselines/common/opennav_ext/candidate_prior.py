"""Observable-feature prior over waypoint candidates (Arm C).

Scores each candidate with a fixed linear model over features the agent can see at
decision time. The weights were fit OFFLINE against an oracle label (which candidate
is euclidean-closest to the goal) on ep100_series_m420260719_001700; that is legal
offline supervision under the project's oracle rule -- nothing here reads goal
distance, goal position, or any simulator geodesic at inference time. The five
features are all present in the trace before the selector runs.

Measured on the fitting run, GroupKFold-5 by episode, out-of-fold:

    random                    27.1%
    grounding_score alone     32.9%
    longest-step heuristic    35.7%
    geometry-only (4 feats)   36.7%
    LLM selector (9B)         36.7%
    this model (5 feats)      44.1% +/- 1.6   (5 CV seeds)

Two caveats that the number does not carry:

1. "Hit rate" means picking the beeline-closest candidate. That target was validated
   (regret predicts success, AUROC 0.773 whole-episode / 0.662 first-3-steps, and
   survives stratification by initial difficulty) but it is not SR. Whether a better
   pick rate converts to SR can only be settled by a run.
2. The weights were fit on trajectories the LLM selector produced. Deploying the
   prior changes the state distribution it will see (covariate shift), so the
   out-of-fold number is an estimate under the old policy, not a guarantee.

The LLM's own choice was tested as a sixth feature and made the model WORSE
(40.4% vs 41.5% on the 14-feature variant): it carries no information beyond these
features.

Arbitration was then attempted directly and FAILED (20260720, n=654 decision points,
exact geometry). The prize is real and large: of 386 disagreements the LLM alone is
right 99 times and the prior alone 155, so a perfect arbiter would reach 60.4% against
45.3% for always deferring to the prior. But nothing observable separates the two
cases. Best single feature over the 254 steps where exactly one side is right:

    distance difference (LLM pick - prior pick)   AUROC 0.588
    prior's softmax confidence                          0.566 (inverted)
    hedging words in the LLM's own reasoning text        0.548
    landmark-noun density in that text                   0.538
    nmatch difference, final_target_visible, ncand      ~0.50

A 4-feature logistic arbiter, GroupKFold-5 by episode, gains +0.6 points at its best
threshold and goes negative one step either side -- noise. The LLM's correctness is
not predictable from anything currently logged, including its own stated reasoning.
Capturing the complementarity therefore needs a head reading the visual/text
representations directly, not more hand-built scalars. That is the training line.
"""
import json
import math
import os

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "candidate_prior_weights.json")

_MODEL = None


def _load_model():
    """Load and cache the fitted weights. Returns None if unusable (fail-open)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL or None
    try:
        with open(_WEIGHTS_PATH) as fh:
            m = json.load(fh)
        n = len(m["features"])
        if not (len(m["mean"]) == len(m["std"]) == len(m["weights"]) == n):
            _MODEL = {}
            return None
        _MODEL = m
        return m
    except (OSError, ValueError, KeyError, TypeError):
        _MODEL = {}
        return None


def _norm_deg(angle_rad):
    """|turn| in degrees, folded to [0, 180]."""
    d = (math.degrees(angle_rad) + 180.0) % 360.0 - 180.0
    return abs(d)


def _as_dict(candidate):
    """Accept either a CandidateState dataclass or an already-serialised dict.

    The runtime passes CandidateState (agent_state.py); the offline fit and the
    replay harness read the same records after they have been written to a trace,
    i.e. as plain dicts. Both must work -- the 20260719_130559 run scored zero
    candidates for 671/671 steps because this module assumed the dict form.
    """
    if isinstance(candidate, dict):
        return candidate
    to_dict = getattr(candidate, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return getattr(candidate, "__dict__", {}) or {}


def candidate_features(candidate, n_candidates, visual_evidence_entry):
    """Assemble the five observable features for one candidate.

    visual_evidence_entry is that candidate's dict from the visual_evidence parse
    (may be None -- an unsampled or unparsed candidate scores nmatch=0, which is
    what the fit saw for those rows too).
    """
    candidate = _as_dict(candidate)
    ve = visual_evidence_entry or {}
    matched = ve.get("matched_instruction_terms") or []
    return {
        "dist": float(candidate.get("distance") or 0.0),
        "ang": _norm_deg(float(candidate.get("angle_rad") or 0.0)),
        "rank": float(candidate.get("raw_rank") or 0),
        "ncand": float(n_candidates),
        "nmatch": float(len(matched)),
    }


def score_candidates(candidates, visual_evidence_by_id):
    """Return {candidate_id: prior_score} plus a rank map, or ({}, {}) if unusable.

    The score is the linear logit; only its ORDER within a decision point is
    meaningful, so it is never compared across steps.
    """
    m = _load_model()
    if not m or not candidates:
        return {}, {}
    feats = m["features"]
    scores = {}
    try:
        for c in candidates:
            raw_cid = _as_dict(c).get("candidate_id")
            if raw_cid is None:
                # No usable id: a score here could only ever be published as an
                # illegal next_vp downstream. Drop the candidate instead.
                continue
            cid = str(raw_cid)
            f = candidate_features(c, len(candidates), (visual_evidence_by_id or {}).get(cid))
            z = float(m["bias"])
            for i, name in enumerate(feats):
                z += m["weights"][i] * (f[name] - m["mean"][i]) / (m["std"][i] or 1e-9)
            scores[cid] = z
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}, {}
    order = sorted(scores, key=lambda k: -scores[k])
    return scores, {cid: i for i, cid in enumerate(order)}

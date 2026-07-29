#!/usr/bin/env python3
"""M1 step 1/3 — pick the decision points and freeze every non-image field.

Why M1 exists: the text-channel ceiling (20260724b) showed a *supervised* bag-of-words
probe over the SpatialBot descriptions (34.2%) ties the 9B zero-shot selector (33.2%),
i.e. the 9B has already squeezed the text dry and the text channel is worth ~9 points
over random. That leaves one untested hypothesis: the direction information is in the
*pixels* and is lost at the perception->text interface. M1 measures that directly by
feeding the same model the candidate view images instead of their descriptions.

Sampling deviates from the 20260724b pre-registration on one point, deliberately:

  pre-registered: "sample 60 decision points with regret > 1m"
  actual:         uniform (scene-stratified) sample over ALL decision points

Reason: regret is defined against the *logged 9B choice*, so on a regret>1m subset the
text arm scores 0% hit by construction and cannot be compared to anything -- and the
gate's reference numbers (BoW 34.2%, structured 44.2%, random 25.8%) were all measured
on the full set. A hardness filter would have been the alternative justification for
the subset, but it does not bite here: candidate spread (max-min distance-to-goal) is
median 2.44m and 93.0% of steps already exceed 1m of spread, and the LLM hit rate on
the spread>1m subset (37.4%) is within noise of the full set (36.7%). Every step is
already consequential, so the uniform sample measures the same quantity without the
selection artifact. The regret>1m points inside the sample are still reported as a
nested secondary readout (see m1_probe.py).

Output: m1_points.json -- everything the renderer and the probe need, with the goal
position kept in a separate `_scoring` block that is never shown to any model.

Usage:  python m1_select_points.py [--n 100] [--seed 0]
"""
import argparse, glob, gzip, json, math, os, re, collections

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DS = os.path.join(REPO, 'data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/'
                        'OpenNav_R2R-CE_100_bertidx.json.gz')
RUN = os.path.join(REPO, 'logs/harness_traces/ep100/20260719/clean_baseline_v1/'
                         'ep100_series_m420260719_130559_train_val_unseen_seed0_r0_w1_20260719_130623')
SCENES_DIR = os.path.join(REPO, 'data/scene_datasets')


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[2] - b[2])


def cand_pos(pos, head, c):
    """Candidate world position. theta = -heading - angle_rad is the calibrated
    convention (20260719 §N: 4 conventions tried, this one has 0.000m median error)."""
    th = -head - c['angle_rad']
    return (pos[0] + c['distance'] * math.sin(th), pos[1], pos[2] - c['distance'] * math.cos(th))


def parse_obs(s):
    """Split the SpatialBot observation string into per-direction description blocks.
    Same parser as text_channel_ceiling.py so the matched-text arm reads the exact
    text the online selector read."""
    out = {}
    for chunk in re.split(r'(?=Direction \d+ Direction Viewpoint ID)', s):
        m = re.match(r'Direction (\d+) ', chunk)
        if m:
            out[m.group(1)] = chunk.strip()
    return out


def load_points():
    ds = json.load(gzip.open(DS))
    meta = {str(e['episode_id']): (e['goals'][0]['position'], e['scene_id'])
            for e in ds['episodes']}
    pts = []
    for f in sorted(glob.glob(os.path.join(RUN, 'val_unseen/rank_0/*.jsonl'))):
        ep = os.path.basename(f)[:-6]
        if ep not in meta:
            continue
        goal, scene = meta[ep]
        steps, instr = {}, None
        for line in open(f):
            e = json.loads(line)
            st, et, p = e['step_id'], e['event_type'], e['payload']
            d = steps.setdefault(st, {})
            if et == 'episode_start':
                instr = p.get('instruction') or ''
            elif et == 'step_start':
                d['pos'] = p['positions'][0]
                d['head'] = p['headings'][0]
            elif et == 'waypoint_candidates':
                d['geo'] = {str(c['candidate_id']): c for c in p['candidates']}
            elif et == 'observation':
                o = p.get('observation')
                d['obs'] = parse_obs(' '.join(o) if isinstance(o, list) else str(o))
            elif et == 'post_action_progress':
                d['sel'] = p.get('selected_candidate')
        for st in sorted(steps):
            d = steps[st]
            if not all(d.get(k) is not None for k in ('pos', 'head', 'geo', 'sel')):
                continue
            ids = list(d['geo'])
            if len(ids) < 2 or str(d['sel']) not in ids:
                continue
            obs = d.get('obs') or {}
            dist = {c: euclid(cand_pos(d['pos'], d['head'], d['geo'][c]), goal) for c in ids}
            best = min(dist, key=dist.get)
            cands = [dict(cid=c,
                          angle_rad=d['geo'][c]['angle_rad'],
                          angle_deg=d['geo'][c]['angle_deg'],
                          distance=d['geo'][c]['distance'],
                          raw_rank=d['geo'][c].get('raw_rank'),
                          desc=obs.get(c))
                     for c in ids]
            pts.append(dict(
                ep=ep, step=st, scene=scene,
                scene_glb=os.path.join(SCENES_DIR, scene),
                instruction=instr.strip(),
                pos=d['pos'], head=d['head'],
                candidates=cands,
                n_desc=sum(1 for c in cands if c['desc']),
                _scoring=dict(goal=goal,
                              dist2goal={c: dist[c] for c in ids},
                              best_cid=best,
                              logged_sel=str(d['sel']),
                              logged_regret=dist[str(d['sel'])] - dist[best]),
            ))
    return pts


def stratified_sample(pts, n, seed):
    """Proportional allocation over scenes (unbiased for the pooled rate, lower
    variance than plain SRS), deterministic given the seed."""
    import random
    rng = random.Random(seed)
    by_scene = collections.defaultdict(list)
    for p in pts:
        by_scene[p['scene']].append(p)
    scenes = sorted(by_scene)
    total = len(pts)
    quota = {s: n * len(by_scene[s]) / total for s in scenes}
    take = {s: int(math.floor(quota[s])) for s in scenes}
    # hand out the remainder to the largest fractional parts (deterministic)
    rem = n - sum(take.values())
    for s in sorted(scenes, key=lambda s: (-(quota[s] - take[s]), s))[:rem]:
        take[s] += 1
    out = []
    for s in scenes:
        pool = sorted(by_scene[s], key=lambda p: (p['ep'], p['step']))
        out += rng.sample(pool, min(take[s], len(pool)))
    out.sort(key=lambda p: (p['scene'], p['ep'], p['step']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'm1_points.json'))
    args = ap.parse_args()

    pts = load_points()
    print(f'总决策点 n={len(pts)}  场景={len({p["scene"] for p in pts})}  集={len({p["ep"] for p in pts})}')
    miss_desc = [p for p in pts if p['n_desc'] < len(p['candidates'])]
    print(f'描述不全的点 {len(miss_desc)}（matched-text 臂会跳过缺描述的候选，故只采描述齐全的点）')
    pool = [p for p in pts if p['n_desc'] == len(p['candidates'])]
    print(f'可采样池 n={len(pool)}')

    sample = stratified_sample(pool, args.n, args.seed)
    reg = [p['_scoring']['logged_regret'] for p in sample]
    hit = sum(1 for p in sample if p['_scoring']['logged_sel'] == p['_scoring']['best_cid'])
    K = [len(p['candidates']) for p in sample]
    print(f'\n抽样 n={len(sample)}  场景={len({p["scene"] for p in sample})}  集={len({p["ep"] for p in sample})}')
    print(f'  平均候选数 K={sum(K)/len(K):.2f}   随机基线={sum(1/k for k in K)/len(K):.1%}')
    print(f'  9B(在线,完整harness prompt) 命中={hit/len(sample):.1%}  '
          f'米制regret={sum(reg)/len(reg):.3f} m/步')
    print(f'  其中 logged_regret>1m 的点 = {sum(1 for r in reg if r > 1)}（预注册子集的嵌套读数）')

    with open(args.out, 'w') as f:
        json.dump(dict(run=RUN, seed=args.seed, n=len(sample),
                       n_pool=len(pool), n_all=len(pts), points=sample), f, indent=1)
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()

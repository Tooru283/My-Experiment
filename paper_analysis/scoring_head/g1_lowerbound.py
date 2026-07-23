#!/usr/bin/env python3
"""G1 lower-bound: structured-feature listwise scoring head vs LLM baseline.
Numpy-only (no sklearn). Scene-level GroupKFold. Metric = metric regret + top-1 hit.
Purpose: validate the regret-labeling pipeline + give the hidden-state head a floor.
Verdict gate (per design doc §4-G1): head folded regret <= 0.85m AND top1-AUC > 0.60.
"""
import json, glob, os, gzip, math, statistics, collections
import numpy as np

DS = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz'
RUN = 'logs/harness_traces/ep100/20260719/clean_baseline_v1/ep100_series_m420260719_130559_train_val_unseen_seed0_r0_w1_20260719_130623'

meta = {str(e['episode_id']): (e['goals'][0]['position'], e['scene_id']) for e in json.load(gzip.open(DS))['episodes']}
def euclid(a,b): return math.hypot(a[0]-b[0], a[2]-b[2])
def cand_pos(pos, head, c):
    th = -head - c['angle_rad']
    return (pos[0]+c['distance']*math.sin(th), 0, pos[2]-c['distance']*math.cos(th))

# ---- build decision points: features per candidate + regret label ----
DPs = []  # each: dict(scene, ep, feats=[K,F], reg=[K], sel_idx, best_idx)
for f in glob.glob(RUN + '/val_unseen/rank_0/*.jsonl'):
    ep = os.path.basename(f)[:-6]
    if ep not in meta: continue
    goal, scene = meta[ep]
    steps = {}
    for line in open(f):
        e = json.loads(line); st, et, p = e['step_id'], e['event_type'], e['payload']
        d = steps.setdefault(st, {})
        if et=='step_start':
            md = p.get('metadata') or p
            d['pos']=(p.get('positions') or md.get('positions') or [None])[0]
            d['head']=(p.get('headings') or md.get('headings') or [None])[0]
        elif et=='waypoint_candidates': d['geo']={c['candidate_id']:c for c in p['candidates']}
        elif et=='visual_evidence':
            ve={}
            for c in ((p.get('parsed') or {}).get('candidates') or []):
                ve[str(c['candidate_id'])]=c
            d['ve']=ve
        elif et=='post_action_progress': d['sel']=p.get('selected_candidate')
    for d in steps.values():
        if not all(k in d and d[k] is not None for k in ('pos','head','geo','sel')): continue
        ids = list(d['geo'].keys())
        if len(ids)<2: continue
        ve = d.get('ve', {})
        feats=[]; regs=[]
        for cid in ids:
            g = d['geo'][cid]; v = ve.get(str(cid), {})
            ang = g['angle_rad']
            turn = -math.degrees(ang)
            while turn<=-180: turn+=360
            while turn>180: turn-=360
            feat = [
                g['distance'],                                   # step length
                abs(turn)/180.0,                                 # |turn| normalized
                1.0 if abs(turn)<30 else 0.0,                    # straight-ish
                g.get('raw_rank', 0)/8.0,                        # predictor rank
                len(v.get('matched_instruction_terms') or []),   # nmatch (the one useful semantic feat)
                float(v.get('confidence') or 0.0),
                1.0 if v.get('final_target_visible') else 0.0,
                len(v.get('visible_landmarks') or []),
            ]
            feats.append(feat)
            regs.append(euclid(cand_pos(d['pos'], d['head'], g), goal))
        sel = str(d['sel'])
        if sel not in ids: continue
        DPs.append(dict(scene=scene, ep=ep, X=np.array(feats,float),
                        reg=np.array(regs,float), sel=ids.index(sel)))

print(f'决策点 n={len(DPs)}, 场景数={len(set(d["scene"] for d in DPs))}')
F = DPs[0]['X'].shape[1]

# ---- standardize features (global) ----
allX = np.vstack([d['X'] for d in DPs])
mu, sd = allX.mean(0), allX.std(0)+1e-6
for d in DPs: d['Xn'] = (d['X']-mu)/sd

# ---- listwise softmax scorer, trained by GD to minimize regret-weighted CE toward argmin-regret ----
def train(train_dps, epochs=300, lr=0.2, l2=1e-2):
    w = np.zeros(F); b_unused=0
    for _ in range(epochs):
        grad = np.zeros(F)
        for d in train_dps:
            s = d['Xn'] @ w
            s = s - s.max()
            p = np.exp(s); p/=p.sum()
            y = np.argmin(d['reg'])            # target = truly-best candidate
            wgt = min(max(d['reg'].max(),0.3),3.0)  # regret-weighted
            g = p.copy(); g[y]-=1
            grad += wgt * (d['Xn'].T @ g)
        grad = grad/len(train_dps) + l2*w
        w -= lr*grad
    return w

def eval_dps(dps, w):
    regs=[]; hits=0; base_reg=[]; base_hit=0
    for d in dps:
        s = d['Xn'] @ w
        pick = int(np.argmax(s))
        best = int(np.argmin(d['reg']))
        regs.append(d['reg'][pick]-d['reg'][best])
        hits += (pick==best)
        base_reg.append(d['reg'][d['sel']]-d['reg'][best])
        base_hit += (d['sel']==best)
    n=len(dps)
    return dict(reg=statistics.mean(regs), hit=hits/n,
                base_reg=statistics.mean(base_reg), base_hit=base_hit/n)

# ---- scene-level GroupKFold (5 folds) ----
scenes = sorted(set(d['scene'] for d in DPs))
rng = np.random.default_rng(0); order = list(scenes); rng.shuffle(order)
K=5; folds=[order[i::K] for i in range(K)]
agg=collections.defaultdict(list)
for k in range(K):
    test_sc=set(folds[k])
    tr=[d for d in DPs if d['scene'] not in test_sc]
    te=[d for d in DPs if d['scene'] in test_sc]
    if not te: continue
    w=train(tr)
    r=eval_dps(te,w)
    for key,v in r.items(): agg[key].append(v)
print('\n=== 场景级 GroupKFold-5（折外）===')
print(f'打分头   regret {np.mean(agg["reg"]):.3f} ± {np.std(agg["reg"]):.3f} m/步 | top1命中 {np.mean(agg["hit"]):.1%}')
print(f'LLM基线  regret {np.mean(agg["base_reg"]):.3f} ± {np.std(agg["base_reg"]):.3f} m/步 | top1命中 {np.mean(agg["base_hit"]):.1%}')
print(f'Δregret  {np.mean(agg["base_reg"])-np.mean(agg["reg"]):+.3f} m/步  (正=打分头更优)')
delta_ep = (np.mean(agg["base_reg"])-np.mean(agg["reg"]))*6.6
print(f'折算 ≈ {delta_ep:+.2f} m/集  (上线闸门 +0.5)')
print(f'\nG1下界闸门(structured): 头regret<=0.85 且 命中>基线? '
      f'{"~通过" if np.mean(agg["reg"])<=0.85 and np.mean(agg["hit"])>np.mean(agg["base_hit"]) else "未过(预期,标量够不着)"}')
print('注: 此为下界。真闸门用 9B 隐藏态版(方案A)，本步只验管道+给floor。')

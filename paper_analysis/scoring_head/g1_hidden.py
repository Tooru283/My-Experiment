#!/usr/bin/env python3
"""G1 (方案A): hidden-state linear probe — the training-line life-or-death gate.

Reuses g1_lowerbound.py's regret-label pipeline verbatim (same traces, same
euclidean regret, same scene-level GroupKFold-5, same listwise softmax head) and
swaps the 8 structured scalars for the frozen 9B's candidate hidden states dumped
by collect_hidden_states.py. Adds train-fold PCA (numpy SVD, no leakage) so a linear
probe on 4096-dim survives ~525 train points. Sweeps layer x pooling.

Pre-registered gate (打分头设计 §4-G1):
  out-of-fold regret <= 0.85 m/step (beat LLM 0.99, beat scalar lower-bound 0.94)
  AND AUROC(top-1 correct) > 0.60.
Floor to clear (scalar version): +0.12 m/集; need ~5x to ~0.5 to be worth training.

Usage:
  python g1_hidden.py --hs hidden_states.npy
"""
import argparse, glob, gzip, json, math, os, statistics, collections
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DS = os.path.join(REPO, 'data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/'
                        'OpenNav_R2R-CE_100_bertidx.json.gz')
DEFAULT_RUN = os.path.join(
    REPO, 'logs/harness_traces/ep100/20260719/clean_baseline_v1/'
    'ep100_series_m420260719_130559_train_val_unseen_seed0_r0_w1_20260719_130623')


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[2] - b[2])


def cand_pos(pos, head, c):
    th = -head - c['angle_rad']
    return (pos[0] + c['distance'] * math.sin(th), 0, pos[2] - c['distance'] * math.cos(th))


def build_regret_dps(run):
    """Same label pipeline as g1_lowerbound, but keyed by (ep, step, cid) so hidden
    vectors can be joined. Returns list of dict(scene, ep, step, cids, reg, sel_idx)."""
    meta = {str(e['episode_id']): (e['goals'][0]['position'], e['scene_id'])
            for e in json.load(gzip.open(DS))['episodes']}
    DPs = []
    for f in glob.glob(os.path.join(run, 'val_unseen/rank_0/*.jsonl')):
        ep = os.path.basename(f)[:-6]
        if ep not in meta:
            continue
        goal, scene = meta[ep]
        steps = {}
        for line in open(f):
            e = json.loads(line); st, et, p = e['step_id'], e['event_type'], e['payload']
            d = steps.setdefault(st, {})
            if et == 'step_start':
                md = p.get('metadata') or p
                d['pos'] = (p.get('positions') or md.get('positions') or [None])[0]
                d['head'] = (p.get('headings') or md.get('headings') or [None])[0]
            elif et == 'waypoint_candidates':
                d['geo'] = {str(c['candidate_id']): c for c in p['candidates']}
            elif et == 'post_action_progress':
                d['sel'] = p.get('selected_candidate')
        for st, d in steps.items():
            if not all(k in d and d[k] is not None for k in ('pos', 'head', 'geo', 'sel')):
                continue
            cids = list(d['geo'].keys())
            if len(cids) < 2:
                continue
            regs = [euclid(cand_pos(d['pos'], d['head'], d['geo'][c]), goal) for c in cids]
            sel = str(d['sel'])
            if sel not in cids:
                continue
            DPs.append(dict(scene=scene, ep=ep, step=st, cids=cids,
                            reg=np.array(regs, float), sel_idx=cids.index(sel)))
    return DPs


def load_hidden(path):
    """(ep, step) -> {cid: {'mean':[L,H], 'last':[L,H]}}, the layer index list, and the
    set of (ep, step) keys whose offline greedy decision matched the online selector
    (fid_match). matched is used as the pre-registered primary subset; if records carry
    no fid_match (older dump), matched == all keys."""
    recs = np.load(path, allow_pickle=True)
    idx, layers, matched, has_fid = {}, None, set(), False
    for r in recs:
        layers = r['layers']
        cmap = {}
        for ci, cid in enumerate(r['candidate_ids']):
            cmap[str(cid)] = {'mean': r['hs_mean'][ci], 'last': r['hs_last'][ci]}
        key = (str(r['ep']), r['step'])
        idx[key] = cmap
        if 'fid_match' in r:            # each record is a plain dict
            has_fid = True
            if r['fid_match']:
                matched.add(key)
    if not has_fid:
        matched = set(idx)  # no fidelity info -> treat every point as matched
    return idx, layers, matched


def train(train_dps, F, epochs=300, lr=0.2, l2=1e-2):
    w = np.zeros(F)
    for _ in range(epochs):
        grad = np.zeros(F)
        for d in train_dps:
            s = d['Xn'] @ w; s -= s.max()
            p = np.exp(s); p /= p.sum()
            y = np.argmin(d['reg'])
            wgt = min(max(d['reg'].max(), 0.3), 3.0)
            g = p.copy(); g[y] -= 1
            grad += wgt * (d['Xn'].T @ g)
        grad = grad / len(train_dps) + l2 * w
        w -= lr * grad
    return w


def eval_dps(dps, w):
    regs, hits, conf, correct = [], 0, [], []
    for d in dps:
        s = d['Xn'] @ w; s -= s.max()
        p = np.exp(s); p /= p.sum()
        pick = int(np.argmax(s)); best = int(np.argmin(d['reg']))
        regs.append(d['reg'][pick] - d['reg'][best]); hits += (pick == best)
        srt = np.sort(p)[::-1]
        conf.append(srt[0] - (srt[1] if len(srt) > 1 else 0.0))
        correct.append(pick == best)
    return regs, hits / len(dps), conf, correct


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float('nan')
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg)


def pca_fit(X, k):
    mu = X.mean(0); Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:k]; sd = (Xc @ comp.T).std(0) + 1e-6
    return mu, comp, sd


def run_probe(DPs, hidden, cids_by_dp, layer_i, pool, ncomp):
    """Assemble X for one (layer, pooling), run scene-level GroupKFold-5."""
    dps = []
    for d in DPs:
        cmap = cids_by_dp.get((d['ep'], d['step']))
        if not cmap or any(c not in cmap for c in d['cids']):
            continue
        X = np.array([cmap[c][pool][layer_i] for c in d['cids']], dtype=np.float32)
        dps.append(dict(scene=d['scene'], reg=d['reg'], sel_idx=d['sel_idx'], X=X))
    if len(dps) < 20:
        return None
    scenes = sorted(set(d['scene'] for d in dps))
    rng = np.random.default_rng(0); order = list(scenes); rng.shuffle(order)
    folds = [order[i::5] for i in range(5)]
    ag = collections.defaultdict(list); all_conf, all_corr = [], []
    base_reg = [d['reg'][d['sel_idx']] - d['reg'][np.argmin(d['reg'])] for d in dps]
    base_hit = np.mean([d['sel_idx'] == np.argmin(d['reg']) for d in dps])
    for k in range(5):
        te_sc = set(folds[k])
        tr = [d for d in dps if d['scene'] not in te_sc]
        te = [d for d in dps if d['scene'] in te_sc]
        if not te:
            continue
        Xtr = np.vstack([d['X'] for d in tr])
        mu, comp, sd = pca_fit(Xtr, ncomp)
        for d in tr + te:
            d['Xn'] = ((d['X'] - mu) @ comp.T) / sd
        w = train(tr, ncomp)
        regs, hit, conf, corr = eval_dps(te, w)
        ag['reg'].append(statistics.mean(regs)); ag['hit'].append(hit)
        all_conf += conf; all_corr += corr
    return dict(n=len(dps), reg=np.mean(ag['reg']), reg_sd=np.std(ag['reg']),
                hit=np.mean(ag['hit']), auroc=auroc(all_conf, all_corr),
                base_reg=statistics.mean(base_reg), base_hit=base_hit)


def sweep(name, DPs, hidden, layers, ncomp):
    """Run the full layer×pooling sweep over one DP set; print table + gate verdict.
    Returns (best, gate_pass)."""
    print(f'\n########## 口径: {name}  (DPs={len(DPs)}) ##########')
    print(f'{"layer":>6} {"pool":>5} {"n":>5} {"reg":>7} {"±":>6} {"hit":>6} '
          f'{"AUROC":>6}   gate')
    best = None
    for li, layer in enumerate(layers):
        for pool in ('mean', 'last'):
            r = run_probe(DPs, hidden, hidden, li, pool, ncomp)
            if r is None:
                continue
            gate = 'PASS' if (r['reg'] <= 0.85 and r['auroc'] > 0.60) else '--'
            print(f'{layer:>6} {pool:>5} {r["n"]:>5} {r["reg"]:>7.3f} {r["reg_sd"]:>6.3f} '
                  f'{r["hit"]:>6.1%} {r["auroc"]:>6.3f}   {gate}')
            if best is None or r['reg'] < best['reg']:
                best = dict(layer=layer, pool=pool, **r)
    if not best:
        print('  (no probe had enough data)')
        return None, False
    b = best
    print(f'--- best: layer {b["layer"]} / {b["pool"]}-pool ---')
    print(f'打分头   regret {b["reg"]:.3f} m/步 | top1命中 {b["hit"]:.1%} | '
          f'AUROC(选对) {b["auroc"]:.3f}')
    print(f'LLM基线  regret {b["base_reg"]:.3f} m/步 | top1命中 {b["base_hit"]:.1%}')
    d_ep = (b['base_reg'] - b['reg']) * 6.6
    print(f'Δregret {b["base_reg"]-b["reg"]:+.3f} m/步  ≈ {d_ep:+.2f} m/集  '
          f'(scalar下界 +0.12; 上线闸门 +0.5)')
    gate_pass = (b['reg'] <= 0.85 and b['auroc'] > 0.60)
    print(f'G1 闸门({name}): regret<=0.85 且 AUROC>0.60  ->  '
          f'{"PASS" if gate_pass else "FAIL"}')
    return best, gate_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hs', default=os.path.join(os.path.dirname(__file__), 'hidden_states.npy'))
    ap.add_argument('--run', default=DEFAULT_RUN)
    ap.add_argument('--ncomp', type=int, default=64, help='PCA components')
    ap.add_argument('--subset', choices=['matched', 'full', 'both'], default='both',
                    help='matched=pre-registered primary (offline==online); full=sensitivity')
    args = ap.parse_args()

    DPs = build_regret_dps(args.run)
    hidden, layers, matched = load_hidden(args.hs)
    DPs_matched = [d for d in DPs if (d['ep'], d['step']) in matched]
    print(f'regret DPs={len(DPs)}  hidden-state DPs={len(hidden)}  '
          f'matched(offline==online)={len(matched)}  '
          f'matched∩regret={len(DPs_matched)}  layers={layers}  PCA={args.ncomp}')

    results = {}
    if args.subset in ('matched', 'both'):
        results['matched(主口径)'] = sweep('matched(主口径/预注册)', DPs_matched,
                                           hidden, layers, args.ncomp)
    if args.subset in ('full', 'both'):
        results['full(敏感性)'] = sweep('full(敏感性)', DPs, hidden, layers, args.ncomp)

    print('\n' + '=' * 60)
    verdicts = {k: ('PASS' if v[1] else 'FAIL') for k, v in results.items() if v[0]}
    for k, v in verdicts.items():
        print(f'  {k:22s} -> {v}')
    agree = len(set(verdicts.values())) == 1 if verdicts else False
    if agree and 'PASS' in verdicts.values():
        print('G1 生死闸门: 两口径一致 PASS  ->  训练线活')
    elif agree:
        print('G1 生死闸门: 两口径一致 FAIL  ->  方案A死 → backbone快筛/最小可发表版')
    else:
        print('G1 生死闸门: 两口径分歧 -> 以主口径(matched/预注册)为准,full 存疑需讨论')


if __name__ == '__main__':
    main()

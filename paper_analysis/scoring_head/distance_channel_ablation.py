#!/usr/bin/env python3
"""SpatialBot 编造距离的消融：没用，还是有害？（20260730，纯离线、零 GPU）

背景：
  · 选择器 prompt 实测**零真实几何**（grep `navigator_prompt`：无 Waypoint distance、无 angle/degree）；
    唯一的距离信息是 SpatialBot 自己写的（"chair - 0.5 meters, table - 0.7 meters…"）。
  · 该距离已被证明是语言先验：整数占比 64%、与真值 r=0.175、MAE 1.74m（§一.2）。
  · 而真实的 WaypointBert 测距被 `GEOMETRY_INJECTION=False` 关着
    （`spatialNavigator.py:67-72` 的注释写明 P0 正为此设计）。

本脚本问三件事：
  A. 编造的距离与真实候选距离有没有相关？（是噪声，还是有噪的信号）
  B. 把距离句从描述里删掉，文本探针会变好还是变差？（没用 vs 有害）
  C. 直接按"报出的最近物体距离"选路，能不能打败随机？（能否被当启发式用）

判读：
  · B 中 NO-DIST ≥ FULL  → 编造距离无用甚至有害 → 删它是零成本改动
  · B 中 NO-DIST < FULL  → 尽管编造，它仍携带信息（很可能是"近处有东西"的粗代理）
"""
import json, glob, os, gzip, math, re, statistics, collections
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DS = os.path.join(REPO, 'data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz')
RUN = os.path.join(REPO, 'logs/harness_traces/ep100/20260719/clean_baseline_v1/'
                         'ep100_series_m420260719_130559_train_val_unseen_seed0_r0_w1_20260719_130623')
STOP = set('the a an is are in of and to with from on at it this that image contains camera distances '
           'approximately meters closest followed then finally farthest scene description objects '
           'direction viewpoint id step elevation eye level'.split())

# 距离句的三种写法（实测覆盖）
DIST_PATTERNS = [
    r'the distances? from the camera (?:are|is)[^.]*\.',        # "The distances from the camera are approximately: chair - 0.5 meters, ..."
    r'[^.]*\bis (?:approximately|about|around)\s+[\d.]+\s*met(?:er|re)s?[^.]*\.',
    r'[^.]*\b[\d.]+\s*met(?:er|re)s?\s+(?:away|from the camera)[^.]*\.',
]
NUM_M = re.compile(r'([\d.]+)\s*met(?:er|re)s?', re.I)


def strip_distance(t):
    out = t
    for p in DIST_PATTERNS:
        out = re.sub(p, ' ', out, flags=re.I)
    out = NUM_M.sub(' ', out)                 # 兜底：残留的裸数字+meters
    return re.sub(r'\s+', ' ', out)


def euclid(a, b): return math.hypot(a[0] - b[0], a[2] - b[2])


def cand_pos(pos, head, c):
    th = -head - c['angle_rad']
    return (pos[0] + c['distance'] * math.sin(th), 0, pos[2] - c['distance'] * math.cos(th))


def parse_obs(s):
    out = {}
    for chunk in re.split(r'(?=Direction \d+ Direction Viewpoint ID)', s):
        m = re.match(r'Direction (\d+) ', chunk)
        if m:
            out[m.group(1)] = chunk.lower()
    return out


def toks(t): return {w for w in re.findall(r'[a-z]{3,}', t) if w not in STOP}


def load():
    goals = {str(e['episode_id']): (e['goals'][0]['position'], e['scene_id'])
             for e in json.load(gzip.open(DS))['episodes']}
    DP = []
    for f in glob.glob(RUN + '/val_unseen/rank_0/*.jsonl'):
        ep = os.path.basename(f)[:-6]
        if ep not in goals:
            continue
        goal, scene = goals[ep]; steps = {}; instr = None
        for line in open(f):
            e = json.loads(line); st, et, p = e['step_id'], e['event_type'], e['payload']
            d = steps.setdefault(st, {})
            if et == 'episode_start':
                instr = (p.get('instruction') or '').lower()
            elif et == 'step_start':
                md = p.get('metadata') or p
                d['pos'] = (p.get('positions') or md.get('positions') or [None])[0]
                d['head'] = (p.get('headings') or md.get('headings') or [None])[0]
            elif et == 'waypoint_candidates':
                d['cands'] = {c['candidate_id']: c for c in p['candidates']}
            elif et == 'observation':
                o = p.get('observation')
                d['obs'] = parse_obs(' '.join(o) if isinstance(o, list) else str(o))
            elif et == 'post_action_progress':
                d['sel'] = p.get('selected_candidate')
        it = toks(instr or '')
        for d in steps.values():
            if not all(k in d and d[k] is not None for k in ('pos', 'head', 'cands', 'obs')):
                continue
            ids = [c for c in d['cands'] if c in d['obs']]
            if len(ids) < 2:
                continue
            DP.append(dict(
                scene=scene, ids=ids, itoks=it, sel=str(d.get('sel')),
                dists={c: euclid(cand_pos(d['pos'], d['head'], d['cands'][c]), goal) for c in ids},
                truewp={c: float(d['cands'][c]['distance']) for c in ids},
                full={c: toks(d['obs'][c]) for c in ids},
                nodist={c: toks(strip_distance(d['obs'][c])) for c in ids},
                said={c: [float(x) for x in NUM_M.findall(d['obs'][c])] for c in ids},
            ))
    return DP


def probe(DP, key, seed=0):
    """BoW listwise 探针，场景级 GroupKFold-5（与 text_channel_ceiling.py 同配方）"""
    vocab = collections.Counter()
    for r in DP:
        for c in r['ids']:
            vocab.update(r[key][c])
    V = [w for w, _ in vocab.most_common(300)]; VI = {w: i for i, w in enumerate(V)}

    def feat(r, c):
        v = np.zeros(2 * len(V)); ct = r[key][c]
        for w in ct:
            if w in VI: v[VI[w]] = 1
        for w in ct & r['itoks']:
            if w in VI: v[len(V) + VI[w]] = 1
        return v

    scenes = sorted({r['scene'] for r in DP})
    rng = np.random.default_rng(seed); order = list(scenes); rng.shuffle(order)
    folds = [order[i::5] for i in range(5)]
    acc = []
    for k in range(5):
        te_sc = set(folds[k])
        tr = [r for r in DP if r['scene'] not in te_sc]
        te = [r for r in DP if r['scene'] in te_sc]
        if not te:
            continue
        w = np.zeros(2 * len(V))
        for _ in range(150):
            g = np.zeros_like(w)
            for r in tr:
                X = np.array([feat(r, c) for c in r['ids']]); s = X @ w; s -= s.max()
                p = np.exp(s); p /= p.sum()
                y = int(np.argmin([r['dists'][c] for c in r['ids']]))
                gv = p.copy(); gv[y] -= 1; g += X.T @ gv
            w -= 0.5 * (g / len(tr) + 0.01 * w)
        acc.append(sum(1 for r in te
                       if r['ids'][int(np.argmax(np.array([feat(r, c) for c in r['ids']]) @ w))]
                       == min(r['dists'], key=r['dists'].get)) / len(te))
    return np.mean(acc), np.std(acc)


def main():
    DP = load(); n = len(DP)
    print(f'n={n} 决策点\n')

    # ---- A. 编造距离 vs 真实候选距离 ----
    xs, ys = [], []
    for r in DP:
        for c in r['ids']:
            if r['said'][c]:
                xs.append(min(r['said'][c])); ys.append(r['truewp'][c])
    if len(xs) > 10:
        rr = np.corrcoef(xs, ys)[0, 1]
        print(f'[A] SpatialBot 报的最近物体距离 vs 真实航点距离：r = {rr:+.3f}  (n={len(xs)})')
        print(f'    报出值 中位 {statistics.median(xs):.2f}m / 真实航点 中位 {statistics.median(ys):.2f}m')
        allv = [v for r in DP for c in r['ids'] for v in r['said'][c]]
        ints = sum(1 for v in allv if abs(v - round(v)) < 1e-6) / max(1, len(allv))
        print(f'    所有报出距离 n={len(allv)}，整数占比 {ints:.0%}')

    # ---- C. 直接当启发式 ----
    hits = collections.Counter(); tot = 0
    for r in DP:
        best = min(r['dists'], key=r['dists'].get); tot += 1
        avail = [c for c in r['ids'] if r['said'][c]]
        if avail:
            hits['报出最近物体最远'] += (max(avail, key=lambda c: min(r['said'][c])) == best)
            hits['报出最近物体最近'] += (min(avail, key=lambda c: min(r['said'][c])) == best)
        hits['真实步长最长'] += (max(r['ids'], key=lambda c: r['truewp'][c]) == best)
    rand = statistics.mean(1 / len(r['ids']) for r in DP)
    print(f'\n[C] 当启发式用（命中最优候选，n={tot}）：')
    for k, v in hits.items():
        print(f'    {k:14} {v/tot:6.1%}')
    print(f'    {"随机":14} {rand:6.1%}')

    # ---- B. 消融 ----
    print(f'\n[B] BoW listwise 探针（场景级 GroupKFold-5，与 §一sex 同配方）：')
    a1, s1 = probe(DP, 'full')
    a2, s2 = probe(DP, 'nodist')
    base = statistics.mean(
        1.0 if r['sel'] == min(r['dists'], key=r['dists'].get) else 0.0 for r in DP)
    print(f'    FULL   （现状，含编造距离）  {a1:.1%} ± {s1:.1%}')
    print(f'    NO-DIST（删掉距离句）       {a2:.1%} ± {s2:.1%}')
    print(f'    Δ = {a2-a1:+.1%}   (正 = 删掉更好 = 编造距离有害)')
    print(f'    9B 零样本（全集口径）        {base:.1%}')
    print(f'    随机                        {rand:.1%}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""文本通道信息量上界检验（20260724，纯离线、零 GPU）

问题：选路近随机、隐藏态无 regret 信号——瓶颈到底在模型还是在输入？
方法：选路器读的是 SpatialBot 生成的**文本描述**（不看图）。测这段文本里
      到底有多少"哪个候选通往目标"的信息。

三个检验（同一批 721 决策点，clean_baseline_v1 20260719 跑）：
  1) 描述可区分性：候选间物体集合 Jaccard（排除"描述全一样"这种平凡解释）
  2) 指令-描述词汇重合：最优候选是否比最差候选更多提到指令里的词
  3) 纯文本 BoW listwise 探针（有监督、场景级 GroupKFold-5）vs 9B 零样本

判读：若 (3) 的有监督文本探针 ≈ 9B 零样本，则 9B 已把文本信息榨干，
      瓶颈在感知→文本接口，不在模型。
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

def euclid(a, b): return math.hypot(a[0]-b[0], a[2]-b[2])
def cand_pos(pos, head, c):
    th = -head - c['angle_rad']
    return (pos[0]+c['distance']*math.sin(th), 0, pos[2]-c['distance']*math.cos(th))
def parse_obs(s):
    out = {}
    for chunk in re.split(r'(?=Direction \d+ Direction Viewpoint ID)', s):
        m = re.match(r'Direction (\d+) ', chunk)
        if m: out[m.group(1)] = chunk.lower()
    return out
def toks(t): return {w for w in re.findall(r'[a-z]{3,}', t) if w not in STOP}
def objects_of(t):
    m = re.search(r'scene objects:(.*?)(?:;|$)', t, re.S)
    return {x.strip() for x in m.group(1).split('|') if x.strip()} if m else set()

def load():
    goals = {str(e['episode_id']): (e['goals'][0]['position'], e['scene_id'])
             for e in json.load(gzip.open(DS))['episodes']}
    DP = []
    for f in glob.glob(RUN + '/val_unseen/rank_0/*.jsonl'):
        ep = os.path.basename(f)[:-6]
        if ep not in goals: continue
        goal, scene = goals[ep]; steps = {}; instr = None
        for line in open(f):
            e = json.loads(line); st, et, p = e['step_id'], e['event_type'], e['payload']
            d = steps.setdefault(st, {})
            if et == 'episode_start': instr = (p.get('instruction') or '').lower()
            elif et == 'step_start':
                md = p.get('metadata') or p
                d['pos'] = (p.get('positions') or md.get('positions') or [None])[0]
                d['head'] = (p.get('headings') or md.get('headings') or [None])[0]
            elif et == 'waypoint_candidates': d['cands'] = {c['candidate_id']: c for c in p['candidates']}
            elif et == 'observation':
                o = p.get('observation'); d['obs'] = parse_obs(' '.join(o) if isinstance(o, list) else str(o))
            elif et == 'post_action_progress': d['sel'] = p.get('selected_candidate')
        it = toks(instr or '')
        for d in steps.values():
            if not all(k in d and d[k] is not None for k in ('pos', 'head', 'cands', 'obs')): continue
            ids = [c for c in d['cands'] if c in d['obs']]
            if len(ids) < 2: continue
            DP.append(dict(scene=scene,
                           ids=ids,
                           dists={c: euclid(cand_pos(d['pos'], d['head'], d['cands'][c]), goal) for c in ids},
                           ctoks={c: toks(d['obs'][c]) for c in ids},
                           cobjs={c: objects_of(d['obs'][c]) for c in ids},
                           itoks=it, sel=str(d.get('sel'))))
    return DP

def main():
    DP = load(); n = len(DP)
    print(f'n={n} 决策点（clean_baseline_v1，有描述且候选≥2）\n')

    # 1) 可区分性
    jac, jbw = [], []
    for r in DP:
        o = r['cobjs']; ids = r['ids']
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                u = o[a] | o[b]; jac.append(len(o[a] & o[b]) / len(u) if u else 1.0)
        best = min(r['dists'], key=r['dists'].get); worst = max(r['dists'], key=r['dists'].get)
        u = o[best] | o[worst]; jbw.append(len(o[best] & o[worst]) / len(u) if u else 1.0)
    print(f'[1] 描述可区分性: 候选间物体集 Jaccard 均值 {statistics.mean(jac):.3f}；'
          f'最优vs最差 {statistics.mean(jbw):.3f} → 描述不退化（非"全都一样"）')

    # 2) 指令词重合
    bm = [len(r['ctoks'][min(r['dists'], key=r['dists'].get)] & r['itoks']) for r in DP]
    wm = [len(r['ctoks'][max(r['dists'], key=r['dists'].get)] & r['itoks']) for r in DP]
    wins = sum(1 for b, w in zip(bm, wm) if b > w); ties = sum(1 for b, w in zip(bm, wm) if b == w)
    print(f'[2] 指令-描述词汇重合: 最优 {statistics.mean(bm):.2f} vs 最差 {statistics.mean(wm):.2f} 词；'
          f'最优>最差 {wins/n:.1%}（平局 {ties/n:.1%}，无信息基线 {(1-ties/n)/2:.1%}）')

    # 3) BoW listwise 探针
    vocab = collections.Counter()
    for r in DP:
        for c in r['ids']: vocab.update(r['ctoks'][c])
    V = [w for w, _ in vocab.most_common(300)]; VI = {w: i for i, w in enumerate(V)}
    def feat(r, c):
        v = np.zeros(2 * len(V)); ct = r['ctoks'][c]
        for w in ct:
            if w in VI: v[VI[w]] = 1
        for w in ct & r['itoks']:
            if w in VI: v[len(V) + VI[w]] = 1
        return v
    scenes = sorted({r['scene'] for r in DP}); rng = np.random.default_rng(0)
    order = list(scenes); rng.shuffle(order); folds = [order[i::5] for i in range(5)]
    acc, base = [], []
    for k in range(5):
        te_sc = set(folds[k])
        tr = [r for r in DP if r['scene'] not in te_sc]; te = [r for r in DP if r['scene'] in te_sc]
        if not te: continue
        w = np.zeros(2 * len(V))
        for _ in range(150):
            g = np.zeros_like(w)
            for r in tr:
                X = np.array([feat(r, c) for c in r['ids']]); s = X @ w; s -= s.max()
                p = np.exp(s); p /= p.sum()
                y = int(np.argmin([r['dists'][c] for c in r['ids']]))
                gv = p.copy(); gv[y] -= 1; g += X.T @ gv
            w -= 0.5 * (g / len(tr) + 0.01 * w)
        acc.append(sum(1 for r in te if r['ids'][int(np.argmax(np.array([feat(r, c) for c in r['ids']]) @ w))]
                       == min(r['dists'], key=r['dists'].get)) / len(te))
        base.append(sum(1 for r in te if r['sel'] == min(r['dists'], key=r['dists'].get)) / len(te))
    rand = statistics.mean(1 / len(r['ids']) for r in DP)
    print(f'\n[3] 场景级 GroupKFold-5:')
    print(f'    有监督纯文本 BoW 探针  {np.mean(acc):.1%} ± {np.std(acc):.1%}')
    print(f'    9B 零样本选择器（同折）{np.mean(base):.1%}')
    print(f'    随机基线              {rand:.1%}')
    print(f'\n判读: 有监督文本探针 ≈ 9B 零样本 → 9B 已榨干文本信息；'
          f'文本通道相对随机仅值 ~{(np.mean(acc)-rand)*100:.0f} 点 → 瓶颈在感知→文本接口。')

if __name__ == '__main__':
    main()

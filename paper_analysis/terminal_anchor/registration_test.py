#!/usr/bin/env python3
"""终点锚点注册检验（纯离线，零渲染，零 GPU）—— 20260730

问题（见 docs/新框架设计-ACN锚点链导航 §S1 前置）：
  把「这一帧像不像 landmark_i」换成「地标池里有没有一个已注册实体、其累积 3D 位置对应 landmark_i」，
  外观问题变成位置问题。**这条路值不值得建？**

做法（不需要检测器/深度/渲染）：
  · 实体位置代理 = 该方向的**候选航点世界坐标**（θ=−heading−angle_rad，误差已标定 0.000m）
  · 每步把「方向 × RAM Scene Objects 标签」注册进池；同标签且相距 <r 的合并（滑动均值）
  · 终点锚点短语来自 terminal_anchor_rule.py（v6，覆盖 93%，触发子集精度 96.8%）

判据（预注册）：
  **主对照 = agent 实际停止位置** —— 这是系统现在就在用的东西。
  注册出的终点锚点若不比它更接近目标，L4 拿它当证据即无价值。
  另设：随机实体（无信息下界）、末步候选航点（平凡几何）、oracle 最优匹配实体（上界）。
抗假阴性：合并半径扫 1/2/3/4m —— 曲线平且低 = 机制不行；随半径上升 = 位置代理太粗（该上检测器）。

⚠ 已知限制（必须随结果一起报）：
  1. observation 只覆盖**候选方向**（中位 4 个），不是 12 路全景 → 池子偏窄
  2. 实体位置 = 航点位置，误差 1–3m，与关联半径同量级
  3. 终点锚点短语与金标同源（我写规则我标注），精度 96.8% 未经独立复核
"""
import json, glob, os, re, math, gzip, random, statistics, collections, csv

TRACE = 'logs/harness_traces/ep100/20260719/clean_baseline_v1/*/val_unseen/rank_0/*.jsonl'
DS = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz'
ANNO = 'paper_analysis/terminal_anchor/terminal_anchor_annotation.csv'
SUCCESS_R = 3.0

STOPWORDS = {'wall', 'ceiling', 'floor', 'room', 'lead to', 'light', 'shadow', 'corner'}


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[2] - b[2])


def cand_pos(pos, head, angle_rad, dist):
    th = -head - angle_rad
    return (pos[0] + dist * math.sin(th), 0.0, pos[2] - dist * math.cos(th))


def head_noun(phrase):
    p = re.sub(r'[^a-z ]', ' ', (phrase or '').lower()).strip()
    return p.split()[-1] if p.split() else ''


def tags_of(view_text):
    m = re.search(r'scene objects:(.*)', view_text, re.I | re.S)
    if not m:
        return []
    return [t.strip().lower() for t in m.group(1).split('|') if t.strip()]


def load_goals():
    return {str(e['episode_id']): e['goals'][0]['position']
            for e in json.load(gzip.open(DS))['episodes']}


def load_anchors():
    out = {}
    for r in csv.DictReader(open(ANNO, encoding='utf-8-sig')):
        if r['rule_output']:
            out[r['episode_id']] = r['rule_output']
    return out


def build_pool(steps, merge_r):
    """entities: list of dict(tag, pos, n, first_step, last_step)"""
    pool = []
    for st in sorted(steps):
        d = steps[st]
        if not all(k in d for k in ('pos', 'head', 'geo', 'views')):
            continue
        for did, txt in d['views'].items():
            g = d['geo'].get(did)
            if not g:
                continue
            p = cand_pos(d['pos'], d['head'], g['angle_rad'], g['distance'])
            for t in tags_of(txt):
                if t in STOPWORDS:
                    continue
                hit = None
                for e in pool:
                    if e['tag'] == t and euclid(e['pos'], p) < merge_r:
                        hit = e
                        break
                if hit:
                    n = hit['n']
                    hit['pos'] = ((hit['pos'][0] * n + p[0]) / (n + 1), 0.0,
                                  (hit['pos'][2] * n + p[2]) / (n + 1))
                    hit['n'] = n + 1
                    hit['last_step'] = st
                else:
                    pool.append(dict(tag=t, pos=p, n=1, first_step=st, last_step=st))
    return pool


def parse_episode(f):
    steps = collections.defaultdict(dict)
    final_pos = None
    for line in open(f):
        e = json.loads(line)
        st, et, p = e.get('step_id'), e['event_type'], e['payload']
        if et == 'step_start':
            md = p.get('metadata') or p
            q = (p.get('positions') or md.get('positions') or [None])[0]
            h = (p.get('headings') or md.get('headings') or [None])[0]
            if q:
                steps[st]['pos'] = q
                steps[st]['head'] = h
                final_pos = q
        elif et == 'waypoint_candidates':
            steps[st]['geo'] = {str(c.get('direction_id', c['candidate_id'])): c
                                for c in p['candidates']}
        elif et == 'observation':
            views = {}
            for v in (p.get('observation') or []):
                m = re.match(r'Direction (\d+) ', str(v))
                if m:
                    views[m.group(1)] = str(v)
            steps[st]['views'] = views
    return steps, final_pos


def main():
    goals = load_goals()
    anchors = load_anchors()
    rng = random.Random(0)
    files = {}
    for f in glob.glob(TRACE):
        files.setdefault(os.path.basename(f)[:-6], f)

    print(f'轨迹集 {len(files)} | 规则触发的终点锚点 {len(anchors)}')
    print(f'成功半径 {SUCCESS_R}m\n')

    header = f'{"合并r":>5} {"n":>4} | {"停止位置":>9} {"最多证据":>9} {"最近":>8} {"末步航点":>9} {"随机实体":>9} {"oracle":>8}'
    print(header)
    print('-' * len(header))

    detail_rows = []
    for merge_r in (1.0, 2.0, 3.0, 4.0):
        res = collections.defaultdict(list)
        nmatch = 0
        for ep, f in files.items():
            if ep not in anchors or ep not in goals:
                continue
            steps, final_pos = parse_episode(f)
            if final_pos is None:
                continue
            goal = goals[ep]
            pool = build_pool(steps, merge_r)
            hn = head_noun(anchors[ep])
            cand = [e for e in pool if hn and (hn == head_noun(e['tag']) or hn in e['tag'])]
            res['stop'].append(euclid(final_pos, goal))
            last_geo = steps[max(steps)].get('geo') or {}
            if last_geo:
                c0 = list(last_geo.values())[0]
                res['lastwp'].append(euclid(
                    cand_pos(steps[max(steps)]['pos'], steps[max(steps)]['head'],
                             c0['angle_rad'], c0['distance']), goal))
            if not cand:
                continue
            nmatch += 1
            res['most'].append(euclid(max(cand, key=lambda e: e['n'])['pos'], goal))
            res['near'].append(euclid(min(cand, key=lambda e: euclid(e['pos'], final_pos))['pos'], goal))
            res['oracle'].append(min(euclid(e['pos'], goal) for e in cand))
            res['rand'].append(euclid(rng.choice(pool)['pos'], goal))
            detail_rows.append(dict(merge_r=merge_r, ep=ep, anchor=anchors[ep],
                                    n_cand=len(cand), pool=len(pool),
                                    d_stop=round(res['stop'][-1], 2),
                                    d_most=round(res['most'][-1], 2),
                                    d_oracle=round(res['oracle'][-1], 2)))

        def med(k):
            return statistics.median(res[k]) if res[k] else float('nan')
        print(f'{merge_r:5.1f} {nmatch:4d} | {med("stop"):9.2f} {med("most"):9.2f} '
              f'{med("near"):8.2f} {med("lastwp"):9.2f} {med("rand"):9.2f} {med("oracle"):8.2f}')
        if merge_r == 2.0:
            print()
            print('  ↑ 中位到目标距离 (m)，越小越好。以下为 r=2.0 的 <3m 命中率：')
            for k, name in (('stop', '停止位置'), ('most', '最多证据'), ('near', '最近'),
                            ('lastwp', '末步航点'), ('rand', '随机实体'), ('oracle', 'oracle上界')):
                if res[k]:
                    hit = sum(1 for x in res[k] if x <= SUCCESS_R) / len(res[k])
                    print(f'     {name:8}: {hit:5.1%}   (n={len(res[k])})')
            print()

    out = 'paper_analysis/terminal_anchor/registration_test_detail.csv'
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail_rows[0].keys()))
        w.writeheader(); w.writerows(detail_rows)
    print(f'逐集明细: {out}')
    print('\n判据提醒：「最多证据/最近」必须显著优于「停止位置」，否则 L4 拿注册实体当证据无价值。')
    print('        曲线随 r 平坦且高 → 机制不行；随 r 明显下降 → 位置代理太粗，该上检测器。')


if __name__ == '__main__':
    main()


def subgroup():
    """按「agent 是否到过目标 3m 内」分组 —— 区分『注册不行』与『根本没走到』。
    这是本检验最关键的一段：全集 FAIL 的死因必须归对。"""
    import glob as _g
    EV = json.load(open(_g.glob('logs/eval_results/ep100/20260719/*/stats_ep_ckpt_val_unseen_r0_w1.json')[0]))
    goals, anchors = load_goals(), load_anchors()
    files = {os.path.basename(f)[:-6]: f for f in glob.glob(TRACE)}
    G = collections.defaultdict(lambda: collections.defaultdict(list))
    nomatch = collections.Counter()
    for ep, f in files.items():
        if ep not in anchors or ep not in goals or ep not in EV:
            continue
        grp = 'OSR成功(到过3m内)' if EV[ep]['oracle_success'] == 1.0 else 'OSR失败(从没到过)'
        steps, final = parse_episode(f)
        if final is None:
            continue
        goal = goals[ep]; pool = build_pool(steps, 2.0); hn = head_noun(anchors[ep])
        cand = [e for e in pool if hn and (hn == head_noun(e['tag']) or hn in e['tag'])]
        G[grp]['stop'].append(euclid(final, goal))
        if not cand:
            nomatch[grp] += 1; continue
        G[grp]['most'].append(euclid(max(cand, key=lambda e: e['n'])['pos'], goal))
        G[grp]['oracle'].append(min(euclid(e['pos'], goal) for e in cand))
    print(f'\n{"分组":22} {"集数":>4} {"锚点无匹配":>8} | {"停止位置":>9} {"最多证据":>9} {"oracle上界":>10}')
    print('-' * 76)
    for g in ('OSR成功(到过3m内)', 'OSR失败(从没到过)'):
        d = G[g]
        med = lambda k: statistics.median(d[k]) if d[k] else float('nan')
        n = len(d['oracle']) or 1
        print(f'{g:22} {len(d["stop"]):4d} {nomatch[g]:8d} | {med("stop"):9.2f} {med("most"):9.2f} {med("oracle"):10.2f}')
        print(f'{"":22} {"":4} {"":8} |   <3m: {sum(1 for x in d["stop"] if x<=3)/len(d["stop"]):5.1%}   '
              f'{sum(1 for x in d["most"] if x<=3)/n:5.1%}      {sum(1 for x in d["oracle"] if x<=3)/n:5.1%}')

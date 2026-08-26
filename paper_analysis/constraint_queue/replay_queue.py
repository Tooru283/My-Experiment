#!/usr/bin/env python3
"""L1 约束队列离线重放（S1-a，纯离线、零 GPU）—— 20260802

规格见 docs/L1规格-约束队列进度定位-20260802.md。
本脚本做规格 §6 实施顺序的第 1 步：**在现有 trace 上建队列并重放**，
测三个结构性指标 + 一个相关性指标，与 `completion_estimation` 现状对照。

判定器（本轮只实现两类，location 走弃权，用以量化它的价值）：
  · object    : RAM 标签命中 ∧ 该方向航点距离 ≤ r        —— 零新模型
  · direction : odometry 位姿窗口，叉积定左右/点积定前后 —— 零模型
  · location  : ❌ 本轮未实现 → 弃权（停在该格）。**弃权率即 location 通道的价值下界**

对照（completion_estimation 现状，日志 §2.2）：
  倒退 75 次 · 每集不同输出中位 2 · None 率 4% · 与真实进度 r=0.120
"""
import json, glob, os, re, math, gzip, statistics, collections

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RUN = os.path.join(REPO, 'logs/harness_traces/ep100/20260719/clean_baseline_v1/'
                         'ep100_series_m420260719_130559_train_val_unseen_seed0_r0_w1_20260719_130623')
DS = os.path.join(REPO, 'data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/'
                        'OpenNav_R2R-CE_100_bertidx.json.gz')
R_OBJ = 3.0          # object 约束距离阈值（规格 §3.1：2×步长中位≈3m 起步，待标定）
W_DIR = 2            # direction 约束的位姿窗口（步）

ROOM = ['living room', 'dining room', 'laundry room', 'storage area', 'bedroom', 'bathroom',
        'kitchen', 'hallway', 'closet', 'garage', 'office', 'corridor', 'foyer', 'den',
        'hall', 'room']
DIRV = re.compile(r'\b(turn (?:left|right|around)|take a (?:left|right)|'
                  r'go (?:straight|forward)|walk (?:straight|forward)|head (?:left|right|straight))\b', re.I)
TURN_L = re.compile(r'\b(turn left|take a left|head left)\b', re.I)
TURN_R = re.compile(r'\b(turn right|take a right|head right)\b', re.I)
TURN_A = re.compile(r'\bturn around\b', re.I)


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[2] - b[2])


def cand_pos(pos, head, c):
    th = -head - c['angle_rad']
    return (pos[0] + c['distance'] * math.sin(th), 0.0, pos[2] - c['distance'] * math.cos(th))


def tags_of(view_text):
    m = re.search(r'scene objects:(.*)', view_text, re.I | re.S)
    return {t.strip().lower() for t in m.group(1).split('|') if t.strip()} if m else set()


def build_queue(actions, landmarks):
    """→ [ {kind, key, raw} ]，按指令顺序"""
    acts = [a.strip() for a in re.split(r'[,\n]', actions or '') if a.strip()]
    lms = [x.strip().lower() for x in re.split(r'[,\n]', landmarks or '') if x.strip()]
    q = []
    for a in acts:
        al = a.lower()
        room = next((r for r in ROOM if r in al), None)
        lm = next((l for l in lms if l and l in al), None)
        if room and (lm is None or room in lm):
            q.append(dict(kind='location', key=room, raw=a))
        elif lm:
            q.append(dict(kind='object', key=lm, raw=a))
        elif DIRV.search(al):
            d = 'around' if TURN_A.search(al) else ('left' if TURN_L.search(al)
                 else ('right' if TURN_R.search(al) else 'forward'))
            q.append(dict(kind='direction', key=d, raw=a))
        else:
            q.append(dict(kind='unknown', key=None, raw=a))
    return q


def sat_object(key, step):
    """RAM 标签命中 ∧ 航点距离 ≤ R_OBJ"""
    head = key.split()[-1]
    for did, txt in step['views'].items():
        g = step['geo'].get(did)
        if not g:
            continue
        if float(g['distance']) > R_OBJ:
            continue
        tags = tags_of(txt)
        if any(head in t or t in key for t in tags):
            return True
    return False


def sat_direction(key, hist):
    """位姿窗口内的朝向变化（零模型）"""
    if len(hist) < W_DIR + 1:
        return False
    h0, h1 = hist[-(W_DIR + 1)]['head'], hist[-1]['head']
    d = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi     # 有符号转角
    deg = math.degrees(d)
    if key == 'around':
        return abs(deg) > 120
    if key == 'left':
        return deg > 35          # 符号约定 sign=-1 已标定（日志 §一quater 副产品）
    if key == 'right':
        return deg < -35
    p0, p1 = hist[-(W_DIR + 1)]['pos'], hist[-1]['pos']
    return euclid(p0, p1) > 1.0  # forward：净位移


def main():
    goals = {str(e['episode_id']): e['goals'][0]['position']
             for e in json.load(gzip.open(DS))['episodes']}
    rows = []
    for f in glob.glob(RUN + '/val_unseen/rank_0/*.jsonl'):
        ep = os.path.basename(f)[:-6]
        if ep not in goals:
            continue
        meta = None; steps = collections.defaultdict(dict)
        for line in open(f):
            e = json.loads(line); st, et, p = e.get('step_id'), e['event_type'], e['payload']
            if et == 'episode_metadata':
                meta = p
            elif et == 'step_start':
                md = p.get('metadata') or p
                q = (p.get('positions') or md.get('positions') or [None])[0]
                h = (p.get('headings') or md.get('headings') or [None])[0]
                if q is not None:
                    steps[st]['pos'] = q; steps[st]['head'] = h
            elif et == 'waypoint_candidates':
                steps[st]['geo'] = {str(c.get('direction_id', c['candidate_id'])): c
                                    for c in p['candidates']}
            elif et == 'observation':
                v = {}
                for s in (p.get('observation') or []):
                    m = re.match(r'Direction (\d+) ', str(s))
                    if m:
                        v[m.group(1)] = str(s)
                steps[st]['views'] = v
        if meta is None:
            continue
        Q = build_queue(meta.get('actions', ''), meta.get('landmarks', ''))
        if not Q:
            continue
        j = 0; hist = []; js = []; stalled_by = collections.Counter()
        ever = set()   # ★ v2 修复：约束满足是**持久**的。逐步累积"曾满足过"的格号，
                       # 否则前格阻塞会让后格的证据窗口被错过（agent 已走过 piano 才轮到检查它）。
        ks = sorted(k for k in steps if all(x in steps[k] for x in ('pos', 'head', 'geo', 'views')))
        for st in ks:
            s = steps[st]; hist.append(s)
            # ★ v2：每步先更新**所有**格子的"曾满足"记录（证据不因队列阻塞而丢失）
            for idx, c in enumerate(Q):
                if idx in ever:
                    continue
                if c['kind'] == 'object' and sat_object(c['key'], s):
                    ever.add(idx)
                elif c['kind'] == 'direction' and sat_direction(c['key'], hist):
                    ever.add(idx)
                # location / unknown 无判定器 → 永不进 ever
            # 再按单向队列推进：只看第一个未满足的格
            while j < len(Q) and j in ever:
                j += 1
            if j < len(Q):
                stalled_by[Q[j]['kind']] += 1
            js.append(j)
        if not js:
            continue
        prog = js[-1] / len(Q)
        rows.append(dict(ep=ep, N=len(Q), nstep=len(js), jfinal=js[-1],
                         nuniq=len(set(js)), prog=prog,
                         stall=stalled_by.most_common(1)[0][0] if stalled_by else None,
                         kinds=collections.Counter(c['kind'] for c in Q)))
    n = len(rows)
    print(f'集数 {n}   队列长度 中位 {statistics.median([r["N"] for r in rows]):.0f}')
    print()
    print('=== S1-a 结构性指标（对照 = completion_estimation 现状）===')
    print(f'{"指标":28}{"本方案":>10}{"现状":>10}{"闸门":>10}')
    print(f'{"进度倒退次数":28}{0:>10}{75:>10}{"≤5":>10}   ← 单向队列，结构上归零')
    med_u = statistics.median([r['nuniq'] for r in rows])
    print(f'{"每集不同进度取值数 中位":28}{med_u:>10.0f}{2:>10}{"≥4":>10}')
    print(f'{"每集步数 中位":28}{statistics.median([r["nstep"] for r in rows]):>10.0f}{"—":>10}{"—":>10}')
    print()
    stuck = [r for r in rows if r['jfinal'] == 0]
    done = [r for r in rows if r['prog'] >= 0.999]
    print(f'完全推不动（j 始终 0）: {len(stuck)}/{n} = {len(stuck)/n:.0%}')
    print(f'队列走完（j = N）    : {len(done)}/{n} = {len(done)/n:.0%}')
    print(f'最终进度 j/N 中位     : {statistics.median([r["prog"] for r in rows]):.2f}')
    print()
    print('=== 卡在哪一类约束上（= 该通道的价值下界）===')
    c = collections.Counter(r['stall'] for r in rows)
    for k, v in c.most_common():
        print(f'  {str(k):12} {v:3d} 集 ({v/n:5.1%})')
    allk = collections.Counter()
    for r in rows:
        allk.update(r['kinds'])
    t = sum(allk.values())
    print(f'\n队列中各类约束占比（共 {t} 条）：')
    for k, v in allk.most_common():
        print(f'  {k:12} {v:4d}  {v/t:5.1%}')


if __name__ == '__main__':
    main()

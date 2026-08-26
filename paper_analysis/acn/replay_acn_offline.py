#!/usr/bin/env python3
"""ACN 离线重放：L0 + L1 + M2 + L4 在 100 集 trace 上的行为 —— 20260805

**导入线上真实模块**（不复制逻辑），用现有 trace 的位姿 / RAM 标签 / 航点几何逐步喂进去，
对照 L1 规格 §5 的预注册闸门，以及 ACN §5.5b 的退化度量基线。

对照基线（completion_estimation 现状，实测）：
  进度倒退 75 次 · 每集不同输出中位 2 · None 率 4% · 与真实进度 r=0.120
"""
import json, glob, os, sys, types, importlib.util, re, statistics, collections

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXT = os.path.join(REPO, 'vlnce_baselines/common/opennav_ext')
sys.path.insert(0, REPO)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _pkg in ('vlnce_baselines', 'vlnce_baselines.common',
             'vlnce_baselines.common.opennav_ext'):
    if _pkg not in sys.modules:
        m = types.ModuleType(_pkg); m.__path__ = []; sys.modules[_pkg] = m
_load('vlnce_baselines.common.opennav_ext.agent_state',
      os.path.join(EXT, 'agent_state.py'))
_lm = _load('vlnce_baselines.common.opennav_ext.landmark_matching',
            os.path.join(EXT, 'landmark_matching.py'))
build_anchor_chain = _load('vlnce_baselines.common.opennav_ext.anchor_chain',
                           os.path.join(EXT, 'anchor_chain.py')).build_anchor_chain
ConstraintQueueLocator = _load(
    'vlnce_baselines.common.opennav_ext.progress_locator',
    os.path.join(EXT, 'progress_locator.py')).ConstraintQueueLocator
LandmarkPool = _load('vlnce_baselines.common.opennav_ext.landmark_pool',
                     os.path.join(EXT, 'landmark_pool.py')).LandmarkPool
TerminalGate = _load('vlnce_baselines.common.opennav_ext.terminal_gate',
                     os.path.join(EXT, 'terminal_gate.py')).TerminalGate

RUN = os.path.join(
    REPO, 'logs/harness_traces/ep100/20260719/clean_baseline_v1/'
    'ep100_series_m420260719_130559_train_val_unseen_seed0_r0_w1_20260719_130623/'
    'val_unseen/rank_0')
if not os.path.isdir(RUN):
    RUN = glob.glob(os.path.join(
        REPO, 'logs/harness_traces/ep100/20260719/clean_baseline_v1/*/val_unseen/rank_0'
    ))[0]


def parse_episode(path):
    meta, steps = None, collections.defaultdict(dict)
    for line in open(path):
        e = json.loads(line)
        st, et, p = e.get('step_id'), e['event_type'], e['payload']
        if et == 'episode_metadata':
            meta = p
        elif et == 'step_start':
            md = p.get('metadata') or p
            q = (p.get('positions') or md.get('positions') or [None])[0]
            h = (p.get('headings') or md.get('headings') or [None])[0]
            if q is not None:
                steps[st]['pos'], steps[st]['head'] = q, h
        elif et == 'waypoint_candidates':
            steps[st]['geo'] = {
                str(c.get('direction_id', c['candidate_id'])): c for c in p['candidates']
            }
        elif et == 'observation':
            v = {}
            for s in (p.get('observation') or []):
                m = re.match(r'Direction (\d+) ', str(s))
                if m:
                    mm = re.search(r'scene objects:(.*)', str(s), re.I | re.S)
                    v[m.group(1)] = [t.strip().lower()
                                     for t in mm.group(1).split('|') if t.strip()] if mm else []
            steps[st]['tags'] = v
    return meta, steps


ENABLE_LOCATION = os.environ.get('ACN_LOCATION', '1') != '0'
DOMINANCE = float(os.environ.get('ACN_DOMINANCE', '0.5'))
RESTRICT = os.environ.get('ACN_M2_RESTRICT', '1') != '0'


def main():
    files = sorted(glob.glob(os.path.join(RUN, '*.jsonl')))
    print(f'location 判定器: {"ON" if ENABLE_LOCATION else "OFF"}  dominance={DOMINANCE}  '
          f'M2 词表限制: {"ON" if RESTRICT else "OFF"}')
    print(f'trace {os.path.basename(os.path.dirname(os.path.dirname(RUN)))}  集数 {len(files)}')
    print()

    cov, kinds = [], collections.Counter()
    regress, uniq, prog, abstain_rate, stall = [], [], [], [], collections.Counter()
    degenerate = 0
    gate_verdicts = collections.Counter()
    pool_sizes, arrived_any = [], 0

    for f in files:
        meta, steps = parse_episode(f)
        if meta is None:
            continue
        chain = build_anchor_chain(meta.get('actions', ''), meta.get('landmarks', ''))
        cov.append(chain['alignment_coverage'])
        kinds.update(chain['kind_counts'])
        if chain['degenerate']:
            degenerate += 1
            continue
        loc = ConstraintQueueLocator(enable_location=ENABLE_LOCATION,
                                     location_dominance=DOMINANCE)
        loc.reset_episode(chain['anchors'])
        # M2 收紧：只收锚点链问到的类别（见 landmark_pool 注释）
        vocab = [a['key'] for a in chain['anchors'] if a['key']] + \
                [a['landmark'] for a in chain['anchors'] if a.get('landmark')]
        pool = LandmarkPool(restrict_to_vocabulary=RESTRICT)
        pool.reset_episode(vocab)
        gate = TerminalGate()
        ks = sorted(k for k in steps if all(x in steps[k] for x in ('pos', 'head', 'geo', 'tags')))
        last = None
        for st in ks:
            s = steps[st]
            geo = {k: {'angle_rad': c.get('angle_rad'), 'distance': c.get('distance')}
                   for k, c in s['geo'].items()}
            last = loc.update(st, s['pos'], s['head'], s['tags'], geo)
            pool.update(st, s['pos'], s['head'], s['tags'], geo)
        if last is None:
            continue
        term = (_lm.final_landmark_terms(meta.get('landmarks', '')) or [None])[-1]
        gate_verdicts[gate.evaluate(last, pool, term)['verdict']] += 1
        regress.append(last['regressions'])
        uniq.append(last['distinct_j_values'])
        prog.append(last['progress'] or 0.0)
        abstain_rate.append(last['abstain_steps'] / max(1, len(ks)))
        stall.update(last['stall_kind_counts'])
        pool_sizes.append(len(pool.entries))
        arrived_any += 1 if any(e['arrived'] for e in pool.entries) else 0

    n = len(regress)
    print('=== L0 锚点链 ===')
    print(f'对齐覆盖率 中位 {statistics.median(cov):.0%}   建不出链的集 {degenerate}')
    tot = sum(kinds.values())
    print('  约束类型占比：', '  '.join(f'{k} {v/tot:.1%}' for k, v in kinds.most_common()))
    print()
    print('=== L1 约束队列（对照 = completion_estimation 现状）===')
    print(f'{"指标":30}{"本方案":>10}{"现状":>10}{"闸门":>10}')
    print(f'{"进度倒退次数（全体集合计）":30}{sum(regress):>10}{75:>10}{"≤5":>10}')
    print(f'{"每集不同进度取值数 中位":30}{statistics.median(uniq):>10.0f}{2:>10}{"≥4":>10}')
    print(f'{"最终进度 j/N 中位":30}{statistics.median(prog):>10.2f}{"—":>10}{"—":>10}')
    print(f'{"每集弃权步占比 中位":30}{statistics.median(abstain_rate):>9.0%}{"—":>10}{"—":>10}')
    print()
    st_tot = sum(stall.values())
    print('卡在哪一类约束上（= 该通道的价值下界）：')
    for k, v in stall.most_common():
        print(f'  {k:12} {v:5d} 步  {v/st_tot:5.1%}')
    print()
    print('=== M2 地标池 ===')
    print(f'每集池大小 中位 {statistics.median(pool_sizes):.0f}   '
          f'至少一条 arrived 的集 {arrived_any}/{n}')
    print()
    print('=== L4 三重合取终止（若在每集末步评估）===')
    for k, v in gate_verdicts.most_common():
        print(f'  {k:10} {v:3d} 集 ({v/max(1,sum(gate_verdicts.values())):5.1%})')
    print()
    print('⚠ 判读提醒：L1 的闸门是**必要条件**，明确不预测 SR。')
    print('  项目已三次栽在「每步中间量变好但不转化」上。')


if __name__ == '__main__':
    main()

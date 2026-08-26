#!/usr/bin/env python3
"""C5 离线重放：GTA 式 G_topo 在我们 100 集 trace 上的触发率 —— 20260805

目的：C0 有靶子表才允许跑，C5 同理。本脚本**导入线上真实的类**
（`VisualGraphMemoryDiagnostic`，不是复制一份逻辑），把 trace 里的位姿逐步喂进去，
量出 GTA 节点式计数在不同 tau_loop 下的触发面。

⚠ 与旧口径的区别（必须一起报）：
  · 旧 `loop_flag`  = 最近历史位置 ≤1m（含上一步）→ 实测 41.2% 的步，太密，不可用
  · 旧 `visit_count`= 半径内**历史步数**       → 原地不动 3 步就算 3 次
  · 新 `node_visit_count` = GTA 节点访问计数    → 原地不动只算同一个节点的多次到达，
    但因为节点会做 running-mean 吸附，连续停留仍会累加。两者数值不可互换。

用法：python paper_analysis/memory_c5/replay_topological_memory.py
"""
import json, glob, os, sys, types, importlib.util, statistics

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

EXT = os.path.join(REPO, 'vlnce_baselines/common/opennav_ext')


def _load(name, path):
    """按文件路径加载，绕开 vlnce_baselines/__init__.py（它会 import habitat）。
    这样本脚本在没装 habitat 的机器上也能跑，且加载的仍是**线上那份源码**。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _pkg in ('vlnce_baselines', 'vlnce_baselines.common',
             'vlnce_baselines.common.opennav_ext'):
    if _pkg not in sys.modules:
        m = types.ModuleType(_pkg)
        m.__path__ = []
        sys.modules[_pkg] = m
_load('vlnce_baselines.common.opennav_ext.agent_state',
      os.path.join(EXT, 'agent_state.py'))
VisualGraphMemoryDiagnostic = _load(
    'vlnce_baselines.common.opennav_ext.visual_graph_memory',
    os.path.join(EXT, 'visual_graph_memory.py'),
).VisualGraphMemoryDiagnostic

RUN = os.path.join(
    REPO,
    'logs/harness_traces/ep100/20260719/clean_baseline_v1/'
    'ep100_series_m420260719_001700_train_val_unseen_seed0_r0_w1_20260719_001741/'
    'val_unseen/rank_0',
)
THRESHOLDS = (2, 3, 4, 5)
MERGE_RADII = (0.8, 1.0, 1.5)


def episode_poses(path):
    """→ [(step_id, position)]，按 step 升序。位姿取 step_start 的 positions[0]。"""
    out = {}
    for line in open(path):
        e = json.loads(line)
        if e.get('event_type') != 'step_start':
            continue
        p = e['payload']
        md = p.get('metadata') or p
        pos = (p.get('positions') or md.get('positions') or [None])[0]
        if pos is not None and e.get('step_id') is not None:
            out[e['step_id']] = pos
    return sorted(out.items())


def run(merge_radius, threshold):
    steps = 0
    fired_steps = 0
    fired_eps = set()
    nodes_per_ep = []
    maxcount_per_ep = []
    first_fire_frac = []
    for f in glob.glob(os.path.join(RUN, '*.jsonl')):
        ep = os.path.basename(f)[:-6]
        poses = episode_poses(f)
        if not poses:
            continue
        mem = VisualGraphMemoryDiagnostic(
            merge_radius=merge_radius, loop_alert_threshold=threshold
        )
        mem.reset_episode()
        n_ep, fired_at = len(poses), None
        mx = 0
        for i, (sid, pos) in enumerate(poses):
            r = mem.update(sid, pos, 0.0, [])
            steps += 1
            mx = max(mx, r['node_visit_count'])
            if r['loop_alert']:
                fired_steps += 1
                fired_eps.add(ep)
                if fired_at is None:
                    fired_at = i
        nodes_per_ep.append(len(mem.nodes))
        maxcount_per_ep.append(mx)
        if fired_at is not None and n_ep > 1:
            first_fire_frac.append(fired_at / (n_ep - 1))
    return dict(
        steps=steps, fired_steps=fired_steps, fired_eps=len(fired_eps),
        nodes_med=statistics.median(nodes_per_ep) if nodes_per_ep else 0,
        maxcount_med=statistics.median(maxcount_per_ep) if maxcount_per_ep else 0,
        first_fire_frac=(
            statistics.median(first_fire_frac) if first_fire_frac else float('nan')
        ),
    )


def main():
    n_files = len(glob.glob(os.path.join(RUN, '*.jsonl')))
    print(f'trace: {os.path.basename(RUN)}   集数 {n_files}')
    print()
    print('=== GTA 节点式计数：触发面扫描 ===')
    hdr = (f'{"merge_r":>8}{"tau":>5}{"触发步":>8}{"占全部步":>10}'
           f'{"触发集数":>10}{"节点数中位":>11}{"最大计数中位":>13}{"首次触发位置":>13}')
    print(hdr)
    print('-' * len(hdr))
    for mr in MERGE_RADII:
        for t in THRESHOLDS:
            r = run(mr, t)
            print(f'{mr:>8.1f}{t:>5}{r["fired_steps"]:>8}'
                  f'{r["fired_steps"]/max(1,r["steps"]):>9.1%}'
                  f'{r["fired_eps"]:>10}{r["nodes_med"]:>11.0f}'
                  f'{r["maxcount_med"]:>13.0f}{r["first_fire_frac"]:>12.0%}')
        print()
    print('判读：')
    print('  · 触发占比过高（>30%）= 告警变噪声，模型会学会忽略它')
    print('  · 触发集数过低（<20）= 靶子太小，ep100 上测不出量级')
    print('  · 首次触发位置越靠前，越有机会改变轨迹（靠后=木已成舟）')


if __name__ == '__main__':
    main()

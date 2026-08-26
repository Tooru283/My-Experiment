#!/usr/bin/env python3
"""C0/C2 冒烟核对 —— 20260805

**跑 ep100 要 ~9 小时。这个脚本的唯一目的是：在烧掉那 9 小时之前，用 ep5 确认开关真的生效。**

项目已经栽过一次：20260718 那轮因为 `DEPTH_STOP_VETO` 没被转发，
**逐字节复现了 clean baseline**，9 小时白跑；20260727 两轮日志里连 `[HARNESS SWITCHES]`
横幅都没有，防线整个失效。

用法（在 GPU 机器上）：
    # 1) 先跑 ep5，把 EVAL.EPISODE_COUNT 改成 5
    bash run_OpenNav.bash                       # 或你惯用的启动方式
    # 2) 再跑本脚本核对
    python paper_analysis/c0_c2/smoke_check.py <trace_dir> --min-steps 4 --short 15 --long 17

判据全部是**结构性**的，不看 SR —— ep5 上 SR 没有意义。
"""
import argparse, collections, glob, json, os, sys


def load_events(trace_dir):
    files = glob.glob(os.path.join(trace_dir, '**', '*.jsonl'), recursive=True)
    if not files:
        sys.exit(f'FAIL: {trace_dir} 下没有 jsonl，路径是不是写错了？')
    per_ep = collections.defaultdict(list)
    for f in files:
        ep = os.path.basename(f)[:-6]
        for line in open(f):
            try:
                per_ep[ep].append(json.loads(line))
            except Exception:
                pass
    return per_ep, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace_dir')
    ap.add_argument('--min-steps', type=int, default=4, help='C0: MIN_STEPS_BEFORE_ALLOW')
    ap.add_argument('--short', type=int, default=15, help='C2: SHORT_ACTION_STEP_LIMIT')
    ap.add_argument('--long', type=int, default=17, help='C2: LONG_ACTION_STEP_LIMIT')
    a = ap.parse_args()

    per_ep, files = load_events(a.trace_dir)
    print(f'trace: {a.trace_dir}')
    print(f'集数 {len(per_ep)}   文件 {len(files)}')
    print()

    results = []

    # ---- 检查 1：C0 的 blocker 真的触发过吗 ----
    # visual_target_verifier.py:669 在被拦时写 reason "before_min_steps:{step}<{min}"
    hits, max_step_seen = 0, 0
    for ev in per_ep.values():
        for e in ev:
            s = json.dumps(e.get('payload', {}), ensure_ascii=False)
            hits += s.count('before_min_steps')
            if e.get('step_id') is not None:
                max_step_seen = max(max_step_seen, int(e['step_id']))
    ok1 = hits > 0
    results.append(('C0 blocker 触发 (before_min_steps)', ok1,
                    f'{hits} 次' + ('' if ok1 else '  ← 0 次！停止走的是别的路径，'
                                                   '先找到那条路径，别直接烧 9 小时')))

    # ---- 检查 2：没有任何一集在 min_steps 之前停下 ----
    early = []
    for ep, ev in per_ep.items():
        steps = [e['step_id'] for e in ev if e.get('step_id') is not None]
        if steps and max(steps) + 1 < a.min_steps:
            early.append((ep, max(steps) + 1))
    ok2 = not early
    results.append((f'无早于 {a.min_steps} 步的终止', ok2,
                    'OK' if ok2 else f'{len(early)} 集: {early[:5]}'))

    # ---- 检查 3：C2 的预算真的放开了（至少有一集走过旧上限 10 步）----
    lens = sorted(max((e['step_id'] for e in ev if e.get('step_id') is not None),
                      default=-1) + 1 for ev in per_ep.values())
    beyond_old = sum(1 for L in lens if L > 10)
    ok3 = max(lens, default=0) <= a.long + 1
    results.append((f'最长集不超过新上限 {a.long}', ok3, f'实际最长 {max(lens, default=0)} 步'))
    print(f'每集步数: {lens}   超过旧上限(10)的集数: {beyond_old}')
    if beyond_old == 0:
        print('  ⚠ 没有任何一集突破旧上限 → C2 这一轮**没有被检验到**，'
              'ep5 抽样太小或预算没生效；ep100 前建议再抽一组长集')
    print()

    # ---- 检查 4：配置横幅在日志里（本脚本只能检查 trace，横幅在 stdout）----
    print('⚠ 横幅需要人工确认：启动日志里必须同时出现')
    print('    [HARNESS SWITCHES] ...')
    print(f'    [HARNESS SWITCHES 2] C0.min_steps_before_allow={a.min_steps} | '
          f'C2.step_limit short/long={a.short}/{a.long} | ...')
    print('  两行都在、且数值与本次意图一致，才算开关转发成功。')
    print()

    print('=' * 64)
    for name, ok, detail in results:
        print(f'  [{"PASS" if ok else "FAIL"}] {name:34} {detail}')
    print('=' * 64)
    allok = all(ok for _, ok, _ in results)
    print('\n结论：' + ('✅ 可以上 ep100' if allok
                       else '🔴 先修上面 FAIL 的项，不要烧 9 小时'))
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()

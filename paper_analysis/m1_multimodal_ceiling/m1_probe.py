#!/usr/bin/env python3
"""M1 step 3/3 — ask the 9B which way the goal is, once per input modality.

Three model arms over the SAME points, the SAME prompt skeleton and the SAME model,
differing only in what each option carries. That factorisation is the whole point:

  GEO  option = (turn angle, step length)                      <- geometry only
  TXT  option = GEO + the SpatialBot description of that view  <- the deployed channel
  IMG  option = GEO + the rendered RGB of that view            <- the untested channel

  TXT - GEO = what the perception->text interface is worth
  IMG - GEO = what the pixels are worth

Plus three free baselines from the trace: the online 9B choice (full harness prompt),
"always take the longest step" (best trivial geometric heuristic, 35.7% globally), and
random (analytic 1/K).

Options are relabelled A/B/C/... in a per-point deterministic shuffle, so the model
cannot ride the direction id (id k ~ 30k degrees) instead of the evidence, and the
"Direction N ..." header is stripped from descriptions for the same reason.

Metrics: top-1 hit (picked == argmin distance-to-goal) and metric regret in m/step,
which is the project's acceptance metric since 20260720. Gate: IMG hit >= 50%.

Usage:
  python m1_probe.py --arms geo --limit 4 --dump 2   # cheap plumbing smoke
  python m1_probe.py                                 # all arms, all points
  python m1_probe.py --score-only                    # re-score saved answers, no GPU
"""
import argparse, base64, json, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = '/root/models/Qwen3.5-9B'
DASHSCOPE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
LABELS = 'ABCDEFGH'
DESC_HEADER = re.compile(r'^Direction \d+ Direction Viewpoint ID: \d+ in Step ID: \d+ '
                         r'Elevation: Eye Level\s*')

SYSTEM = ('You are a navigation assistant. You are given a natural-language route '
          'instruction and several waypoints you could move to next. You pick the one '
          'that makes progress along the instruction toward its destination.')


def turn_phrase(angle_deg):
    """Positive angle_rad is a LEFT turn (verified in dump_candidate_views.py)."""
    a = angle_deg % 360
    if a < 1e-6 or abs(a - 360) < 1e-6:
        return 'straight ahead'
    return f'{a:.0f}° to your left' if a <= 180 else f'{360 - a:.0f}° to your right'


def build_options(pt, seed):
    """Deterministic per-point shuffle -> [{label, cid, angle_deg, distance, desc}]."""
    cands = sorted(pt['candidates'], key=lambda c: int(c['cid']))
    random.Random(f'{seed}:{pt["ep"]}:{pt["step"]}').shuffle(cands)
    return [dict(label=LABELS[i], cid=c['cid'], angle_deg=c['angle_deg'],
                 distance=c['distance'],
                 desc=DESC_HEADER.sub('', (c['desc'] or '').strip()))
            for i, c in enumerate(cands)]


def build_messages(pt, opts, arm, viewdir, manifest):
    """Return chat messages; identical across arms except for the per-option payload."""
    head = (f'Route instruction: "{pt["instruction"]}"\n'
            f'You have already taken {pt["step"] - 1} step(s) along this route.\n\n'
            f'These are the waypoints you can move to next.')
    content = [{'type': 'text', 'text': head}]
    files = (manifest['items'][f'{pt["ep"]}_{pt["step"]}']['files']
             if arm in ('img', 'imgshuf') else {})
    # imgshuf: each option keeps its true angle/distance but gets a NEIGHBOUR's image
    # (a rotation by one, i.e. a derangement). It is the falsification control -- if
    # IMG scores no better than IMGSHUF, the model is not reading the pixels, which is
    # a different failure from "the pixels hold no direction information".
    shifted = {o['cid']: opts[(i + 1) % len(opts)]['cid'] for i, o in enumerate(opts)}
    for o in opts:
        line = (f'\n\nOption {o["label"]}: turn {turn_phrase(o["angle_deg"])}, '
                f'{o["distance"]:.2f} m away.')
        if arm == 'txt':
            line += f'\nView from that direction: {o["desc"]}'
        content.append({'type': 'text', 'text': line})
        if arm in ('img', 'imgshuf'):
            cid = o['cid'] if arm == 'img' else shifted[o['cid']]
            content.append({'type': 'text', 'text': '\nView from that direction:'})
            content.append({'type': 'image',
                            'url': os.path.join(viewdir, files[cid])})
    labels = ', '.join(o['label'] for o in opts)
    content.append({'type': 'text', 'text':
                    f'\n\nWhich option leads toward the destination described in the '
                    f'instruction? Consider only where each option would take you.\n'
                    f'Answer with exactly one letter ({labels}) and nothing else.'})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': content}]


def parse_letter(text, opts):
    valid = {o['label'] for o in opts}
    for m in re.finditer(r'[A-H]', text.upper()):
        if m.group(0) in valid:
            return m.group(0)
    return None


# ---------------------------------------------------------------- backends


def to_openai(msgs):
    """build_messages() output -> OpenAI chat format. Prompt text is untouched, so
    the API arms stay comparable to the local ones; only the transport differs."""
    out = []
    for m in msgs:
        texts = [p['text'] for p in m['content'] if p['type'] == 'text']
        if m['role'] == 'system':          # keep system a plain string: some VL
            out.append({'role': 'system',  # endpoints reject list-typed system content
                        'content': ''.join(texts)})
            continue
        content = []
        for p in m['content']:
            if p['type'] == 'text':
                content.append({'type': 'text', 'text': p['text']})
            else:
                b64 = base64.b64encode(open(p['url'], 'rb').read()).decode()
                content.append({'type': 'image_url',
                                'image_url': {'url': f'data:image/png;base64,{b64}'}})
        out.append({'role': m['role'], 'content': content})
    return out


def api_call(client, model, msgs, max_tokens, retries=8):
    """One completion, greedy, thinking off. Retries transient failures with backoff.
    enable_thinking is an extra_body param on Qwen endpoints; drop it if rejected.

    429 gets its own long, jittered backoff: per-minute quotas are not survivable with
    second-scale retries, and giving up produces a NON-RANDOM dropout (the busiest arms
    lose the most calls), which silently biases every rate computed downstream."""
    extra = {'enable_thinking': False}
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=msgs, temperature=0.0,
                max_tokens=max_tokens, extra_body=extra if extra else None)
            return r.choices[0].message.content or ''
        except Exception as e:
            s = str(e)
            if extra and ('enable_thinking' in s or 'extra_body' in s
                          or 'unsupported' in s.lower()):
                extra = None                      # retry immediately without it
                continue
            if attempt == retries - 1:
                return f'__ERROR__ {s[:200]}'
            throttled = '429' in s or 'request limit' in s or 'rate' in s.lower()
            time.sleep(random.uniform(20, 40) if throttled else 2 ** attempt)
    return '__ERROR__ unreachable'


def run_api(points, arms, args, manifest):
    """Same loop as the local backend, but over an OpenAI-compatible endpoint.
    Threaded because these calls are latency- not compute-bound; results are written
    back by key so ordering never depends on completion order."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    key = args.api_key or os.environ.get('DASHSCOPE_API_KEY', '')
    if not key and args.api_key_file:
        key = open(os.path.expanduser(args.api_key_file)).read().strip()
    if not key:
        sys.exit('缺 API key：--api-key / --api-key-file / $DASHSCOPE_API_KEY')
    client = OpenAI(api_key=key, base_url=args.base_url, timeout=180.0, max_retries=0)

    answers, raw = {a: {} for a in arms}, {a: {} for a in arms}
    t0 = time.time()
    for arm in arms:
        jobs = []
        for pt in points:
            k = f'{pt["ep"]}_{pt["step"]}'
            opts = build_options(pt, args.seed)
            jobs.append((k, opts, to_openai(
                build_messages(pt, opts, arm, args.views, manifest))))
        done = [0]

        def work(job):
            k, opts, msgs = job
            comp = api_call(client, args.model, msgs, args.max_new_tokens)
            done[0] += 1
            if done[0] % 20 == 0:
                print(f'  [{arm}] {done[0]}/{len(jobs)}  '
                      f'{(time.time()-t0)/60:.1f}min', flush=True)
            return k, opts, comp

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, (k, opts, comp) in enumerate(ex.map(work, jobs)):
                answers[arm][k] = None if comp.startswith('__ERROR__') \
                    else parse_letter(comp, opts)
                raw[arm][k] = comp
        errs = sum(1 for v in raw[arm].values() if v.startswith('__ERROR__'))
        nmiss = sum(1 for v in answers[arm].values() if v is None)
        print(f'[{arm}] done, 未解析 {nmiss}/{len(points)}（其中报错 {errs}）', flush=True)
        if args.dump:
            for k in list(raw[arm])[:args.dump]:
                print(f'  {arm} {k} -> {raw[arm][k]!r}')
    return answers, raw, t0


# ---------------------------------------------------------------- scoring


def mcnemar(pairs):
    """Exact two-sided binomial McNemar on (a_hit, b_hit) pairs -> (b01, b10, p)."""
    from math import comb
    b01 = sum(1 for a, b in pairs if a and not b)
    b10 = sum(1 for a, b in pairs if b and not a)
    n, k = b01 + b10, min(b01, b10)
    if n == 0:
        return b01, b10, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b01, b10, p


def score(points, answers, seed):
    """answers: {arm: {key: letter|None}} -> per-arm hit/regret + free baselines."""
    rows = []
    for pt in points:
        key = f'{pt["ep"]}_{pt["step"]}'
        s = pt['_scoring']
        opts = build_options(pt, seed)
        d = s['dist2goal']
        best = s['best_cid']
        r = dict(key=key, K=len(opts), best=best,
                 logged=s['logged_sel'], logged_reg=s['logged_regret'],
                 hard=s['logged_regret'] > 1.0)
        longest = max(pt['candidates'], key=lambda c: (c['distance'], -int(c['cid'])))['cid']
        r['picks'] = {'logged': s['logged_sel'], 'longest': longest}
        for arm, per in answers.items():
            lab = per.get(key)
            r['picks'][arm] = next((o['cid'] for o in opts if o['label'] == lab), None)
        r['reg'] = {a: (d[c] - d[best]) if c is not None else None
                    for a, c in r['picks'].items()}
        r['hit'] = {a: (c == best) if c is not None else None
                    for a, c in r['picks'].items()}
        r['rand_hit'] = 1.0 / len(opts)
        r['rand_reg'] = sum(d[o['cid']] for o in opts) / len(opts) - d[best]
        rows.append(r)
    return rows


def report(rows, arms, title):
    n = len(rows)
    if not n:
        return
    print(f'\n=== {title} (n={n}) ===')
    print(f'{"arm":<26}{"命中最优候选":>14}{"米制regret":>14}{"未解析":>8}')
    order = [a for a in ('img', 'imgshuf', 'txt', 'geo') if a in arms] + ['logged', 'longest']
    for a in order:
        hits = [r['hit'][a] for r in rows if r['hit'].get(a) is not None]
        regs = [r['reg'][a] for r in rows if r['reg'].get(a) is not None]
        if not hits:
            continue
        miss = n - len(hits)
        name = {'img': 'IMG 图像臂', 'imgshuf': 'IMGSHUF 错配图对照', 'txt': 'TXT 描述臂',
                'geo': 'GEO 纯几何臂', 'logged': '9B在线(完整harness)',
                'longest': '最长步启发式'}[a]
        print(f'{name:<24}{sum(hits)/len(hits):>13.1%}{sum(regs)/len(regs):>13.3f}'
              f'{miss:>9}')
    print(f'{"随机基线":<24}{sum(r["rand_hit"] for r in rows)/n:>13.1%}'
          f'{sum(r["rand_reg"] for r in rows)/n:>13.3f}{0:>9}')
    for a, b in (('img', 'txt'), ('img', 'geo'), ('txt', 'geo'), ('img', 'imgshuf')):
        if a in arms and b in arms:
            pairs = [(r['hit'][b], r['hit'][a]) for r in rows
                     if r['hit'].get(a) is not None and r['hit'].get(b) is not None]
            b01, b10, p = mcnemar(pairs)
            print(f'  配对McNemar {a.upper()} vs {b.upper()}: '
                  f'{a.upper()}独对 {b10} / {b.upper()}独对 {b01}  p={p:.3f}')


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', default=os.path.join(HERE, 'm1_points.json'))
    ap.add_argument('--views', default=os.path.join(HERE, 'views'))
    ap.add_argument('--out', default=os.path.join(HERE, 'm1_answers.json'))
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--backend', choices=('local', 'api'), default='local')
    ap.add_argument('--base-url', default=DASHSCOPE_URL, help='OpenAI-compatible endpoint')
    ap.add_argument('--api-key', default='', help='优先级最高；否则读 --api-key-file / $DASHSCOPE_API_KEY')
    ap.add_argument('--api-key-file', default='')
    ap.add_argument('--workers', type=int, default=8, help='api backend 并发数')
    ap.add_argument('--arms', default='geo,txt,img,imgshuf')
    ap.add_argument('--seed', type=int, default=0, help='option-shuffle seed')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-new-tokens', type=int, default=48)
    ap.add_argument('--dump', type=int, default=0, help='print the first N prompts/answers')
    ap.add_argument('--score-only', action='store_true', help='re-score m1_answers.json, no GPU')
    args = ap.parse_args()

    blob = json.load(open(args.points))
    points = blob['points'][:args.limit] if args.limit else blob['points']
    manifest = json.load(open(os.path.join(args.views, 'manifest.json')))
    arms = [a for a in args.arms.split(',') if a]

    if args.score_only:
        saved = json.load(open(args.out))
        answers = saved['answers']
        arms = list(answers)
    else:
        missing = [p for p in points
                   if f'{p["ep"]}_{p["step"]}' not in manifest['items']]
        if ({'img', 'imgshuf'} & set(arms)) and missing:
            sys.exit(f'{len(missing)} 点缺渲染图，先跑 dump_candidate_views.py')

        if args.backend == 'api':
            answers, raw, t0 = run_api(points, arms, args, manifest)
        else:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            print(f'loading {args.model} ...', flush=True)
            proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                args.model, dtype=torch.bfloat16, device_map='auto',
                trust_remote_code=True).eval()

            answers, raw = {a: {} for a in arms}, {a: {} for a in arms}
            t0 = time.time()
            for arm in arms:
                for i, pt in enumerate(points):
                    key = f'{pt["ep"]}_{pt["step"]}'
                    opts = build_options(pt, args.seed)
                    msgs = build_messages(pt, opts, arm, args.views, manifest)
                    inputs = proc.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors='pt',
                        enable_thinking=False).to(model.device)
                    with torch.no_grad():
                        gen = model.generate(**inputs, do_sample=False,
                                             max_new_tokens=args.max_new_tokens)
                    comp = proc.decode(gen[0, inputs['input_ids'].shape[1]:],
                                       skip_special_tokens=True)
                    answers[arm][key] = parse_letter(comp, opts)
                    raw[arm][key] = comp
                    if i < args.dump:
                        print(f'--- {arm} {key} opts='
                              f'{ {o["label"]: o["cid"] for o in opts} } ---')
                        print(proc.apply_chat_template(msgs, add_generation_prompt=True,
                                                       tokenize=False)[-1200:])
                        print(f'  -> {comp!r} => {answers[arm][key]}')
                    if (i + 1) % 20 == 0:
                        el = time.time() - t0
                        print(f'  [{arm}] {i+1}/{len(points)}  {el/60:.1f}min elapsed',
                              flush=True)
                nmiss = sum(1 for v in answers[arm].values() if v is None)
                print(f'[{arm}] done, 未解析 {nmiss}/{len(points)}', flush=True)

        with open(args.out, 'w') as f:
            json.dump(dict(model=args.model, backend=args.backend,
                           base_url=args.base_url if args.backend == 'api' else None,
                           seed=args.seed, arms=arms,
                           answers=answers, raw=raw), f, indent=1)
        print(f'saved -> {args.out}  ({(time.time()-t0)/60:.1f} min)')

    rows = score(points, {a: answers[a] for a in arms}, args.seed)
    report(rows, arms, '全部抽样点（主判据）')
    report([r for r in rows if r['hard']], arms,
           '嵌套子集: 在线9B regret>1m 的点（预注册子集）')

    img = [r['hit']['img'] for r in rows if r['hit'].get('img') is not None]
    if img:
        rate = sum(img) / len(img)
        print(f'\n=== M1 闸门 ===\n图像选路命中 {rate:.1%}  闸门 >=50%  -> '
              f'{"过：selector 换图像输入" if rate >= 0.50 else "不过：换赛道/收口（见 current_task §一septem 分叉表）"}')


if __name__ == '__main__':
    main()

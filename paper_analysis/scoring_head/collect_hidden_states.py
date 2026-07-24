#!/usr/bin/env python3
"""Offline re-forward: dump the frozen 9B's candidate-token hidden states.

Why offline (not an online serving hook): the selector prompt is already logged
(`navigator_prompt` event, 100/100 episodes), the 9B runs behind an lmdeploy HTTP
server that returns text only, and greedy decisions are reproducible. So we replay
each logged prompt through an in-process HF-transformers forward with
output_hidden_states=True and read the per-candidate representations. Zero harness
edits, no fresh collect.

Fidelity caveat: the online decision was made by lmdeploy; this forward uses HF
kernels. They differ numerically, so we self-check that HF greedy reproduces the
logged `selector_final.selected_candidate`. Verdict (p1_design §9.5b三段裁决):
  >=95% PASS  |  85-95% GRAY (keep matched subset)  |  <85% RERUN (reconsider online hook)

Output (np.save, allow_pickle): a list of per-decision-point records:
  {ep, step, scene?, candidate_ids:[K], layers:[L], hs:[K,L,H] float16,
   fid_pred, fid_gold, fid_match}
The G1 probe (g1_hidden.py) joins hs to regret labels by (ep, step, candidate_id).

Usage:
  # CPU plumbing check, no model forward:
  python collect_hidden_states.py --dry --smoke 3
  # GPU fidelity smoke (first N points, generate+compare, no full dump):
  python collect_hidden_states.py --smoke 40 --fidelity-only
  # GPU full dump:
  python collect_hidden_states.py --out hidden_states.npy
"""
import argparse, glob, json, os, re, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'vlnce_baselines', 'common', 'navigator'))
from prompts import NAVIGATOR  # noqa: E402  (system/user selector prompt template)

DEFAULT_RUN = os.path.join(
    REPO, 'logs/harness_traces/ep100/20260719/clean_baseline_v1/'
    'ep100_series_m420260719_130559_train_val_unseen_seed0_r0_w1_20260719_130623')
DEFAULT_MODEL = '/root/models/Qwen3.5-9B'

BLOCK_RE = re.compile(r'Direction (\d+) Direction Viewpoint ID:')


def load_decision_points(run):
    """Yield (ep, step, prompt, candidate_ids, gold_selected) from logged traces."""
    dps = []
    for f in sorted(glob.glob(os.path.join(run, 'val_unseen/rank_0/*.jsonl'))):
        ep = os.path.basename(f)[:-6]
        prompts, gold_raw, gold_final = {}, {}, {}
        for line in open(f):
            e = json.loads(line)
            et, st, p = e['event_type'], e.get('step_id'), e['payload']
            if et == 'navigator_prompt':
                prompts[st] = (p['prompt'], [str(c) for c in p['candidate_ids']])
            elif et == 'selector_raw':
                preds = p.get('predictions') or []
                if preds:
                    gold_raw[st] = str(preds[0])  # rawest greedy LLM output
            elif et == 'selector_final':
                gold_final[st] = str(p.get('selected_candidate'))
        for st, (prompt, cids) in prompts.items():
            dps.append(dict(ep=ep, step=st, prompt=prompt, candidate_ids=cids,
                            gold=gold_raw.get(st, gold_final.get(st))))
    return dps


def candidate_char_spans(prompt, candidate_ids):
    """Map each candidate_id -> (start,end) char span of its 'Direction {id} ...' block
    inside the prompt. Block runs to the next Direction block or end of observation."""
    marks = [(m.group(1), m.start()) for m in BLOCK_RE.finditer(prompt)]
    spans = {}
    for i, (cid, s) in enumerate(marks):
        e = marks[i + 1][1] if i + 1 < len(marks) else len(prompt)
        # keep first occurrence per id (blocks are unique per candidate)
        spans.setdefault(cid, (s, e))
    # only return spans for the actual candidate ids we care about
    return {cid: spans[cid] for cid in candidate_ids if cid in spans}


def render_and_tokenize(tokenizer, user_prompt):
    """Return (render_text, input_ids[1,T], offsets[T,2]) with the exact chat template
    the lmdeploy server applies to [system, user]."""
    msgs = [{'role': 'system', 'content': NAVIGATOR['system']},
            {'role': 'user', 'content': user_prompt}]
    # enable_thinking=False mirrors the online server's `--reasoning off` (transformers
    # serve): it injects an empty <think></think> so the model answers concisely as
    # "Thought: ... Prediction: N" instead of a long reasoning trace. Candidate tokens
    # sit before this marker, so their hidden states are unaffected either way.
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    # user_prompt is verbatim inside `text`; find its offset to translate spans.
    base = text.find(user_prompt)
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
    return text, base, enc


def span_to_token_idx(offsets, base, char_span):
    """Char span (relative to user_prompt) -> list of token indices covering it."""
    lo, hi = base + char_span[0], base + char_span[1]
    idx = [t for t, (a, b) in enumerate(offsets) if b > lo and a < hi and b > a]
    return idx


def parse_prediction(text, candidate_ids):
    """Minimal replica of navigator._parse_prediction: last candidate id (or STOP)
    appearing after the final 'Prediction:'; fall back to any id in text."""
    tail = text.rsplit('Prediction:', 1)[-1] if 'Prediction:' in text else text
    for tok in re.findall(r'\d+|STOP', tail):
        if tok in candidate_ids or tok == 'STOP':
            return tok
    for tok in re.findall(r'\d+|STOP', text):
        if tok in candidate_ids or tok == 'STOP':
            return tok
    return None


def resolve_layers(spec, n_layers):
    """spec: 'all' | comma fractions in [0,1] | comma ints -> sorted layer indices
    into hidden_states tuple (len n_layers+1; 0=embeddings)."""
    if spec == 'all':
        return list(range(n_layers + 1))
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        out.append(int(round(v * n_layers)) if 0.0 <= v <= 1.0 and '.' in tok
                   else int(v))
    return sorted(set(max(0, min(n_layers, i)) for i in out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=DEFAULT_RUN)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'hidden_states.npy'))
    ap.add_argument('--layers', default='0.5,0.625,0.75,0.875,1.0',
                    help="'all' | fractions of depth | explicit indices")
    ap.add_argument('--smoke', type=int, default=0, help='only first N decision points')
    ap.add_argument('--fidelity-only', action='store_true',
                    help='generate+compare for the fidelity gate, skip hidden-state dump')
    ap.add_argument('--max-new-tokens', type=int, default=2048,
                    help='match online NAVIGATOR_MAX_TOKENS=2048; selector writes a long '
                         'Thought before "Prediction: N", so a low cap truncates -> random parse')
    ap.add_argument('--dump-completions', type=int, default=0,
                    help='print the first K generated completions for fidelity diagnosis')
    ap.add_argument('--dump-mismatch', action='store_true',
                    help='print every fidelity mismatch (offline pred != online gold) for audit')
    ap.add_argument('--fidelity-first', type=int, default=10**9,
                    help='cap fidelity generate to the first N points (default: all)')
    ap.add_argument('--dry', action='store_true',
                    help='CPU: validate parsing/offset/span plumbing, no model forward')
    args = ap.parse_args()

    dps = load_decision_points(args.run)
    if args.smoke:
        dps = dps[:args.smoke]
    print(f'decision points: {len(dps)}  (episodes={len(set(d["ep"] for d in dps))})')

    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(args.model)
    # Qwen3.5-9B is a VLM (Qwen3_5ForConditionalGeneration): text dims live in text_config.
    tcfg = getattr(cfg, 'text_config', None) or cfg
    n_layers = getattr(tcfg, 'num_hidden_layers', None)
    hidden = getattr(tcfg, 'hidden_size', None)
    layers = resolve_layers(args.layers, n_layers)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    print(f'model n_layers={n_layers} hidden={hidden}  extracting layers={layers}  '
          f'im_end_id={im_end_id}')

    if args.dry:
        ok = 0
        for d in dps:
            text, base, enc = render_and_tokenize(tokenizer, d['prompt'])
            offs = enc['offset_mapping'][0].tolist()
            spans = candidate_char_spans(d['prompt'], d['candidate_ids'])
            missing = [c for c in d['candidate_ids'] if c not in spans]
            tok_counts = {c: len(span_to_token_idx(offs, base, spans[c]))
                          for c in spans}
            empty = [c for c, n in tok_counts.items() if n == 0]
            good = not missing and not empty
            ok += good
            print(f'  ep{d["ep"]} step{d["step"]} T={enc["input_ids"].shape[1]} '
                  f'cids={d["candidate_ids"]} tok/cand={tok_counts} '
                  f'{"OK" if good else f"MISS={missing} EMPTY={empty}"}')
        print(f'\n[DRY] {ok}/{len(dps)} decision points map all candidates to non-empty '
              f'token spans.')
        return

    import torch
    # Qwen3.5-9B is Qwen3_5ForConditionalGeneration (VLM). Load the image-text-to-text
    # class; text-only forward skips the vision tower. Fall back to CausalLM.
    try:
        from transformers import AutoModelForImageTextToText as _AutoModel
    except ImportError:
        from transformers import AutoModelForCausalLM as _AutoModel
    model = _AutoModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True).eval()

    records, fid_match, fid_total = [], 0, 0
    for i, d in enumerate(dps):
        text, base, enc = render_and_tokenize(tokenizer, d['prompt'])
        input_ids = enc['input_ids'].to(model.device)
        attn_mask = enc['attention_mask'].to(model.device)
        offs = enc['offset_mapping'][0].tolist()
        spans = candidate_char_spans(d['prompt'], d['candidate_ids'])

        rec = dict(ep=d['ep'], step=d['step'], candidate_ids=d['candidate_ids'],
                   layers=layers, gold=d['gold'])

        if not args.fidelity_only:
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            hs_all = out.hidden_states  # tuple(n_layers+1) of [1,T,H]
            K, L, H = len(d['candidate_ids']), len(layers), hs_all[0].shape[-1]
            # two pooling variants per candidate block (single forward, pick later):
            #   mean = average over the block span; last = final token of the block
            #   (causal LM: it has attended over the whole block + all prior context)
            hs_mean = np.zeros((K, L, H), dtype=np.float16)
            hs_last = np.zeros((K, L, H), dtype=np.float16)
            for ci, cid in enumerate(d['candidate_ids']):
                if cid not in spans:
                    continue
                tok = span_to_token_idx(offs, base, spans[cid])
                if not tok:
                    continue
                for li, layer in enumerate(layers):
                    row = hs_all[layer][0]  # [T,H]
                    hs_mean[ci, li] = row[tok, :].mean(0).float().cpu().numpy().astype(np.float16)
                    hs_last[ci, li] = row[tok[-1], :].float().cpu().numpy().astype(np.float16)
            rec['hs_mean'], rec['hs_last'] = hs_mean, hs_last

        # fidelity: greedy generate + parse, compare to logged selector decision.
        # Computed for EVERY point (not just the first 40) so a full dump carries a
        # per-record fid_match mask -> g1_hidden.py can report matched-subset (primary)
        # and full-set (sensitivity) side by side. --fidelity-first N caps it if needed.
        if args.fidelity_only or i < args.fidelity_first:
            with torch.no_grad():
                # no generation_config.json ships with the checkpoint, so we must
                # pass the turn terminator explicitly or greedy runs to max_new_tokens.
                gen = model.generate(input_ids, attention_mask=attn_mask,
                                     do_sample=False,
                                     max_new_tokens=args.max_new_tokens,
                                     eos_token_id=im_end_id,
                                     pad_token_id=im_end_id)
            comp = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
            pred = parse_prediction(comp, d['candidate_ids'])
            rec['fid_pred'], rec['fid_gold'] = pred, d['gold']
            rec['fid_match'] = (pred is not None and pred == d['gold'])
            fid_total += 1
            fid_match += rec['fid_match']
            if i < args.dump_completions or (args.dump_mismatch and not rec['fid_match']):
                tag = 'MISMATCH' if not rec['fid_match'] else f'dump #{i}'
                print(f'--- {tag} ep{d["ep"]} step{d["step"]} '
                      f'gold={d["gold"]} pred={pred} match={rec["fid_match"]} '
                      f'gen_tokens={gen.shape[1]-input_ids.shape[1]} '
                      f'cids={d["candidate_ids"]} ---')
                slc = 1500 if not rec['fid_match'] else 400
                print(f'    completion: {comp[:slc]!r}')

        records.append(rec)
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(dps)} done'
                  + (f'  fidelity {fid_match}/{fid_total}' if fid_total else ''))

    if fid_total:
        r = fid_match / fid_total
        verdict = 'PASS' if r >= 0.95 else ('GRAY' if r >= 0.85 else 'RERUN')
        print(f'\n=== 保真自检 (HF greedy vs lmdeploy selector) ===')
        print(f'match {fid_match}/{fid_total} = {r:.1%}  ->  {verdict}')
        print('三段裁决: >=95% PASS | 85-95% GRAY(保留匹配子集) | <85% RERUN(重新考虑在线hook)')

    if not args.fidelity_only:
        np.save(args.out, np.array(records, dtype=object), allow_pickle=True)
        print(f'\nsaved {len(records)} records -> {args.out}')


if __name__ == '__main__':
    main()

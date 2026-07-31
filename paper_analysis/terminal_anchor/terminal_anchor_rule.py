#!/usr/bin/env python3
"""终点锚点抽取规则 v1（确定性，无 LLM）—— 20260730

用途：为「终点锚点注册检验」提供一个高精度、可弃权的终点定义。
设计依据（100 集实测，见 docs/ 对应记录）：
  · 89/100 指令含显式停止动词；停止动词后首词 in21/at19/by12/next7/near6/on4
  · 12 集停止动词后是 there/when/once（回指）→ 结构不同，必须弃权
  · 11 集无停止动词 → 回退到末尾移动动词目标
  · ★ 「取 landmarks 列表最后一个」系统性错误：停止子句常含
    目的地 NP（主方位介词支配）+ 空间参照 NP（次级介词支配），
    最后一个抓到的是参照。故取**主介词后第一个 NP**。
  · ★ 不对齐 landmarks 列表（LLM 产物，ep643 已见合并错误），直接解析原始指令。

输出：每集 (terminal_np | ABSTAIN, reason)，并导出待人工标注表。
验收（跑注册检验之前）：人工标注子集上，**触发子集精度 ≥0.90**；覆盖率只报告不设闸门。
"""
import json, glob, re, os, csv, collections

TRACE_GLOB = 'logs/harness_traces/ep100/20260719/clean_baseline_v1/*/val_unseen/rank_0/*.jsonl'
OUT_DIR = 'paper_analysis/terminal_anchor'

STOP_VERB = r'(?:stop|stops|stopping|wait|waits|waiting|stand|stands|standing|stay|stays|remain|remains|halt|halts)'
# 主方位介词（多词的放前面，正则按顺序匹配）
LOC_PREPS = [
    'in front of', 'next to', 'on top of', 'close to', 'across from',
    'at the end of', 'to the left of', 'to the right of',
    'outside of', 'inside of', 'out of',            # v2 修复：多词优先，否则残留 "of the ..."
    'in', 'at', 'by', 'on', 'near', 'beside', 'outside', 'inside',
    'under', 'underneath', 'above', 'behind', 'between', 'alongside', 'opposite',
]
# 回指/延迟型 —— 必须弃权
ANAPHORIC = {'there', 'when', 'once', 'until', 'after', 'as', 'where', 'here'}
# 部分词：「the <part> of the <X>」时真正的锚点是 X
PART_WORDS = {'edge', 'side', 'end', 'top', 'bottom', 'front', 'back',
              'corner', 'middle', 'center', 'centre', 'base', 'foot'}
# NP 终止标记
NP_STOP = re.compile(
    r'\b(?:' + '|'.join([
        'in', 'at', 'by', 'on', 'near', 'beside', 'outside', 'inside', 'under',
        'above', 'behind', 'between', 'with', 'from', 'and', 'or', 'but',
        'before', 'after', 'while', 'when', 'that', 'which', 'who', 'until',
        'facing', 'towards', 'toward', 'directly', 'straight',
        'to',   # v4：'to' 恢复为终止符；限定语由下方 extend 逻辑择机接回
    ]) + r')\b'
    r'|\bnext(?=\s+to\b)'   # v3：只有 "next to" 终止 NP；"next room" 不能被切
    r'|[.,;:]'
)
# v4 新增模式
REACH_CLAUSE = re.compile(
    r'\b(?:once|when|until|after|as\s+soon\s+as)\s+(?:you|we)\s+'
    r'(?:reach|reaches|enter|enters|exit|exits|get\s+to|get\s+into|arrive\s+at|pass|see)\s+'
    r'(?P<np>.+)$', re.I)
PRE_STOP_GOAL = re.compile(
    r'\b(?:into|onto|to|enter|entering|reach|reaching|take|takes|taking|through)\s+(?P<np>[^.,;]+?)\s*(?:,|\band\b|$)', re.I)
MOTION_VERB = re.compile(r'\b(?:go|goes|walk|walks|turn|turns|enter|enters|head|heads|move|moves|proceed|continue|exit|exits|take|takes)\b', re.I)
LEAD_VERB = re.compile(r'^(?:enter|go|walk|turn|reach|exit|get|make|take|head|move)\s+', re.I)
# v2：'to' 不再无条件终止 —— "the doorway to the bedroom" 里的限定语要保留
# （它正是消歧信息），但 "the table to your left" 里的方位参照要切掉。
TO_REFERENCE = re.compile(
    r'^\s*to\s+(?:'
    r'(?:your|you|his|her|its|their|the)\s+(?:left|right|front|back|side)'   # "to you(r) left"（含原文拼写错漏）
    r'|(?:left|right)\b'
    r'|(?:your|you|his|her|its|their)\b'
    r')', re.I)
MOTION_GOAL = re.compile(
    r'\b(?:towards|toward|to|into|onto|enter|enters|entering)\s+(?P<np>.+?)(?:[.,;]|$)', re.I)


def split_clauses(s):
    """按句号/分号/then/and then 切子句。"""
    parts = re.split(r'(?<=[.;])\s+|\s+(?:and then|then)\s+', s)
    return [p.strip() for p in parts if p.strip()]


def clean_np(np_raw):
    """去限定词、去序数修饰、处理 '<part> of the <X>'。"""
    np = np_raw.strip().strip('.,;:').lower()
    np = re.sub(r'^(?:the|a|an|your|his|her|its|their)\s+', '', np)
    np = re.sub(r'^(?:first|second|third|last|next|final|other)\s+(?:few\s+)?', '', np)
    np = re.sub(r'^(?:the|a|an)\s+', '', np)
    m = re.match(r'^(\w+)\s+of\s+(?:the\s+|a\s+)?(.+)$', np)
    if m and m.group(1) in PART_WORDS:
        np = m.group(2)
        np = re.sub(r'^(?:the|a|an)\s+', '', np)
    np = np.strip()
    # v3：切完只剩限定词/虚词 → 视为抽取失败
    if np in {'the', 'a', 'an', 'your', 'you', 'his', 'her', 'its', 'their',
              'first', 'second', 'next', 'last', 'other', ''}:
        return ''
    return np


def _motion_goal(text):
    """MOTION_GOAL 分支的共用抽取（v4：剥掉被 'to' 吞进来的动词）。"""
    m = MOTION_GOAL.search(text)
    if not m:
        return None
    raw = LEAD_VERB.sub('', m.group('np').strip())   # v4：'to enter another bathroom' → 'another bathroom'
    seg = NP_STOP.split(raw)[0]
    np = clean_np(seg)
    if np and not re.match(r'^(?:the\s+)?(?:left|right|end|front|back|side)$', np):
        return np
    return None


def _pre_goal(text):
    """停止动词之前的移动目标；串联多个时取**最后一个**（v6，ep715）。"""
    best = None
    for mm in PRE_STOP_GOAL.finditer(text or ''):
        np = clean_np(NP_STOP.split(mm.group('np'))[0])
        if np and not re.match(r'^(?:the\s+)?(?:left|right|end|front|back|side)$', np):
            best = np
    return best


def extract(instruction):
    """→ (np|None, reason, stop_clause|None)"""
    clauses = split_clauses(instruction)
    stop_i = None
    for i in range(len(clauses) - 1, -1, -1):
        if re.search(r'\b' + STOP_VERB + r'\b', clauses[i], re.I):
            stop_i = i
            break
    stop_c = clauses[stop_i] if stop_i is not None else None

    # v4：停止子句之后仍有位移 → 该 stop 非终末（ep439），改用其后的移动目标
    if stop_i is not None and stop_i < len(clauses) - 1:
        tail_txt = ' '.join(clauses[stop_i + 1:])
        if MOTION_VERB.search(tail_txt):
            np = _motion_goal(tail_txt)
            if np:
                return np, 'POST_STOP_MOTION', tail_txt

    if stop_c is None:
        # R3 回退：末尾移动动词目标
        tail = clauses[-1] if clauses else instruction
        np = _motion_goal(tail)
        if np:
            return np, 'MOTION_GOAL', tail
        return None, 'ABSTAIN_no_stop_clause', tail

    # 停止动词之后的残段
    m = re.search(r'\b' + STOP_VERB + r'\b(.*)$', stop_c, re.I | re.S)
    rest = (m.group(1) if m else '').strip()

    first_tok = re.match(r'^\W*(\w+)', rest)
    # v5：停止动词后无实质内容（"...and wait." / "That's where you will wait."）等同回指
    if (first_tok is None) or (first_tok.group(1).lower() in ANAPHORIC):
        # v4-a：「once/when you reach|enter|exit|get to X」是可抽的，不是纯回指
        mr = REACH_CLAUSE.search(rest)
        if mr:
            np = clean_np(NP_STOP.split(mr.group('np'))[0])
            if np:
                return np, 'REACH_CLAUSE', stop_c
        # v4-b：纯回指（wait there）→ 回看更早的子句 / 本子句停止动词之前
        for earlier in reversed(clauses[:stop_i]):
            if re.search(r'\b' + STOP_VERB + r'\b', earlier, re.I):
                sub, reason, _ = extract(earlier)
                if sub:
                    return sub, 'ANAPHORA_PREV_' + reason, earlier
        pre = stop_c[:m.start()] if m else ''
        np = _pre_goal(pre)
        if np:
            return np, 'ANAPHORA_SAME_CLAUSE', stop_c
        # v6：本子句无内容（"wait." 独立成句）→ 退到紧邻的上一子句
        if stop_i and not pre.strip():
            prev = clauses[stop_i - 1]
            np = _pre_goal(prev) or _motion_goal(prev)
            if np:
                return np, 'PREV_CLAUSE_GOAL', prev
        tag = first_tok.group(1).lower() if first_tok else 'empty'
        return None, 'ABSTAIN_anaphoric_' + tag, stop_c

    # R2：主方位介词后的第一个 NP（v4：容忍 right/just/directly 修饰词）
    for prep in LOC_PREPS:
        mm = re.search(r'^\W*(?:right|just|directly)?\s*' + re.escape(prep) + r'\s+(.+)$',
                       rest, re.I | re.S)
        if not mm:
            continue
        tail_full = mm.group(1)
        seg = NP_STOP.split(tail_full)[0]
        # v2：并列检测——seg 之后紧跟 "and <NP>" 则无主次可判，弃权
        after_seg = tail_full[len(seg):]
        mand = re.match(r'\s*and\s+((?:the\s+|a\s+)?\w+)', after_seg, re.I)
        if mand:
            return None, 'ABSTAIN_coordinated_' + clean_np(mand.group(1)), stop_c
        # v2：保留 "X to the Y" 限定语（消歧信息），但切掉 "to your left" 类方位参照
        mto = re.match(r'\s*to\s+', after_seg, re.I)
        if mto and not TO_REFERENCE.match(after_seg):
            ext = NP_STOP.split(after_seg[mto.end():])[0]
            if ext.strip():
                seg = seg + ' to ' + ext
        np = clean_np(seg)
        if not np:
            return None, 'ABSTAIN_empty_np', stop_c
        if len(np.split()) > 5:
            return None, 'ABSTAIN_np_too_long', stop_c
        return np, 'LOC_' + prep.replace(' ', '_'), stop_c

    # v4-c：停止动词后无方位介词 → 回看本子句停止动词**之前**的移动目标
    #        （"Go into the room with the tile floor and wait." / "Enter the bedroom then wait."）
    pre = stop_c[:m.start()] if m else ''
    np = _pre_goal(pre)
    if np:
        return np, 'PRE_STOP_GOAL', stop_c
    # v5：原文漏写介词（"wait the archway"）—— 裸限定词+名词兜底，单独打标便于审计
    mb = re.match(r'^\W*((?:the|a|an)\s+\w+(?:\s+\w+)?)\s*[.,;]?\s*$', rest, re.I)
    if mb:
        np = clean_np(mb.group(1))
        if np and not re.match(r'^(?:the\s+)?(?:left|right|end|front|back|side)$', np):
            return np, 'BARE_NP_missing_prep', stop_c
    return None, 'ABSTAIN_no_locative_prep', stop_c


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = {}
    for f in glob.glob(TRACE_GLOB):
        instr = None
        for line in open(f):
            e = json.loads(line)
            if e['event_type'] == 'episode_start':
                instr = e['payload'].get('instruction', '')
            if e['event_type'] == 'episode_metadata':
                ep = e['episode_id']
                if ep in seen:
                    break
                lms = [x.strip().lower() for x in
                       re.split(r'[\n,]', e['payload'].get('landmarks') or '') if x.strip()]
                seen[ep] = (instr or '', lms)
                break

    rows = []
    for ep in sorted(seen, key=int):
        ins, lms = seen[ep]
        np, reason, sc = extract(ins)
        naive = lms[-1] if lms else ''
        rows.append(dict(
            episode_id=ep,
            instruction=ins.strip(),
            stop_clause=(sc or '').strip(),
            rule_output=np or '',
            reason=reason,
            fired=int(np is not None),
            naive_last_landmark=naive,
            agrees_with_naive=int(bool(np) and np == naive),
            human_label='',
            correct='',
        ))

    fired = [r for r in rows if r['fired']]
    print(f'集数 {len(rows)}')
    print(f'规则触发 {len(fired)} ({len(fired)/len(rows):.0%})  弃权 {len(rows)-len(fired)}')
    print('\n=== 触发路径分布 ===')
    for k, v in collections.Counter(r['reason'] for r in fired).most_common():
        print(f'  {v:3d}  {k}')
    print('\n=== 弃权原因分布 ===')
    for k, v in collections.Counter(r['reason'] for r in rows if not r['fired']).most_common():
        print(f'  {v:3d}  {k}')
    agree = sum(r['agrees_with_naive'] for r in fired)
    print(f'\n与「landmarks 列表最后一个」一致: {agree}/{len(fired)} = {agree/len(fired):.0%}')
    print('  → 不一致的那些正是需要人工裁决的重点')

    out = os.path.join(OUT_DIR, 'terminal_anchor_annotation.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n标注表已写出: {out}')
    return rows


if __name__ == '__main__':
    main()

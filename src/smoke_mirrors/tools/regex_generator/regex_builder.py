try:
    from . import regex_sre_parse
    from . import regex_sre_constants
except Exception:
    # running as script
    import regex_sre_parse  # type: ignore
    import regex_sre_constants  # type: ignore
import string
import random
from collections import deque

DEFAULT_ALPHABET = "".join(chr(i) for i in range(32, 127))

CATEGORY_TO_EXPR = {
    regex_sre_constants.CATEGORY_DIGIT: "string.digits",
    regex_sre_constants.CATEGORY_SPACE: "string.whitespace",
    regex_sre_constants.CATEGORY_WORD: "string.ascii_letters + string.digits + '_'",
}


def _expand_in_child_static(arg, flags, alphabet):
    negate = False
    candidates = []
    for tok, val in arg:
        if tok is regex_sre_constants.NEGATE:
            negate = True
            continue
        if tok is regex_sre_constants.LITERAL:
            candidates.append(chr(val))
        elif tok is regex_sre_constants.RANGE:
            a, b = val
            candidates.extend(chr(c) for c in range(a, b + 1))
        elif tok is regex_sre_constants.CATEGORY:
            if val == regex_sre_constants.CATEGORY_DIGIT:
                candidates.extend(list(string.digits))
            elif val == regex_sre_constants.CATEGORY_SPACE:
                candidates.extend(list(string.whitespace))
            elif val == regex_sre_constants.CATEGORY_WORD:
                candidates.extend(list(string.ascii_letters + string.digits + "_"))
            else:
                candidates.extend(list(alphabet))
        elif tok is regex_sre_constants.IN:
            candidates.extend(_expand_in_child_static(val, flags, alphabet))
        else:
            candidates.extend(list(alphabet))
    if negate:
        candidates = [c for c in alphabet if c not in set(candidates)]
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    if not out:
        out = list(alphabet)
    return out


def expand_in_child(arg, alphabet=DEFAULT_ALPHABET):
    candidates = set()
    for tok, val in arg:
        if tok == regex_sre_constants.LITERAL:
            candidates.add(chr(val))
        elif tok == regex_sre_constants.RANGE:
            a, b = val
            candidates.update(chr(c) for c in range(a, b + 1))
        elif tok == regex_sre_constants.CATEGORY:
            if val == regex_sre_constants.CATEGORY_DIGIT:
                candidates.update(string.digits)
            elif val == regex_sre_constants.CATEGORY_WORD:
                candidates.update(string.ascii_letters + string.digits + "_")
            elif val == regex_sre_constants.CATEGORY_SPACE:
                candidates.update(string.whitespace)
            else:
                candidates.update(alphabet)
        elif tok == regex_sre_constants.IN:
            candidates.update(expand_in_child(val, alphabet))
        else:
            candidates.update(alphabet)
    return candidates


def enumerate_sequences_for_subpattern(sub, max_repeat, alphabet):
    """
    Enumerate possible sequences (each sequence is a list of sets of characters)
    for the subpattern, bounded by max_repeat.
    """
    from functools import lru_cache

    def walk(pat):
        sequences = [[]]
        for tok, arg in pat:
            if tok == regex_sre_constants.LITERAL:
                ch = chr(arg)
                sequences = [seq + [{ch}] for seq in sequences]
            elif tok == regex_sre_constants.IN:
                chars = set(_expand_in_child_static(arg, 0, alphabet))
                sequences = [seq + [chars] for seq in sequences]
            elif tok == regex_sre_constants.CATEGORY:
                if arg == regex_sre_constants.CATEGORY_DIGIT:
                    chars = set(string.digits)
                elif arg == regex_sre_constants.CATEGORY_SPACE:
                    chars = set(string.whitespace)
                elif arg == regex_sre_constants.CATEGORY_WORD:
                    chars = set(string.ascii_letters + string.digits + "_")
                else:
                    chars = set(alphabet)
                sequences = [seq + [chars] for seq in sequences]
            elif tok in (regex_sre_constants.MAX_REPEAT, regex_sre_constants.MIN_REPEAT):
                lo, hi, inner = arg
                if hi == regex_sre_constants.MAXREPEAT or hi is None:
                    hi_eff = lo + max_repeat
                else:
                    hi_eff = min(hi, lo + max_repeat)
                inner_sequences = walk(inner)

                @lru_cache(None)
                def repeat_concat(count):
                    if count == 0:
                        return [[]]
                    prev = repeat_concat(count - 1)
                    out = []
                    for p in prev:
                        for s in inner_sequences:
                            out.append(p + s)
                    return out

                new_sequences = []
                for seq in sequences:
                    for count in range(lo, hi_eff + 1):
                        if count == 0:
                            if lo == 0:
                                new_sequences.append(seq.copy())
                            continue
                        for rseq in repeat_concat(count):
                            new_sequences.append(seq + rseq)
                sequences = new_sequences
            elif tok == regex_sre_constants.SUBPATTERN:
                if isinstance(arg, tuple):
                    sub_inner = arg[-1]
                else:
                    sub_inner = arg
                sub_sequences = walk(sub_inner)
                sequences = [seq + sseq for seq in sequences for sseq in sub_sequences]
            elif tok == regex_sre_constants.BRANCH:
                _, branches = arg
                branch_sequences = []
                for branch in branches:
                    bseqs = walk(branch)
                    branch_sequences.extend(bseqs)
                sequences = [seq + bseq for seq in sequences for bseq in branch_sequences]
            elif tok in (regex_sre_constants.AT,):
                continue
            else:
                sequences = [seq + [set(alphabet)] for seq in sequences]
        return sequences

    seqs = walk(sub)
    if not seqs:
        seqs = [[]]
    lengths = [len(s) for s in seqs]
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    return seqs, min_len, max_len


def merged_forbidden_from_sequences(seqs, max_len=None, alphabet=DEFAULT_ALPHABET):
    if not seqs:
        return [], []
    if max_len is None:
        max_len = max((len(s) for s in seqs), default=0)
    merged = [set() for _ in range(max_len)]
    epsilon_by_offset = [False for _ in range(max_len)]
    for s in seqs:
        L = len(s)
        for i in range(max_len):
            if i < L:
                merged[i].update(s[i])
            else:
                epsilon_by_offset[i] = True
    return merged, epsilon_by_offset


def _collect_atomic_requirements(subp, flags, alphabet, repeat_multiplier=1):
    reqs = []

    def walk(pat, mult):
        for tok, arg in pat:
            if tok in (regex_sre_constants.LITERAL,):
                ch = chr(arg)
                reqs.append((mult, [ch]))
            elif tok is regex_sre_constants.IN:
                choices = _expand_in_child_static(arg, flags, alphabet)
                reqs.append((mult, choices))
            elif tok is regex_sre_constants.CATEGORY:
                if arg == regex_sre_constants.CATEGORY_DIGIT:
                    choices = list(string.digits)
                elif arg == regex_sre_constants.CATEGORY_SPACE:
                    choices = list(string.whitespace)
                elif arg == regex_sre_constants.CATEGORY_WORD:
                    choices = list(string.ascii_letters + string.digits + "_")
                else:
                    choices = list(alphabet)
                reqs.append((mult, choices))
            elif tok in (regex_sre_constants.MAX_REPEAT, regex_sre_constants.MIN_REPEAT):
                lo, hi, inner = arg
                inner_mult = mult * max(1, lo)
                walk(inner, inner_mult)
            elif tok is regex_sre_constants.SUBPATTERN:
                if isinstance(arg, tuple):
                    sub = arg[-1]
                else:
                    sub = arg
                walk(sub, mult)
            elif tok is regex_sre_constants.BRANCH:
                _, branches = arg
                for branch in branches:
                    walk(branch, mult)
            else:
                pass

    walk(subp, repeat_multiplier)
    combined = []
    from collections import defaultdict
    agg = defaultdict(int)
    choice_map = {}
    for cnt, choices in reqs:
        key = tuple(sorted(set(choices)))
        agg[key] += cnt
        choice_map[key] = choices
    for k, cnt in agg.items():
        combined.append({'count': cnt, 'choices': choice_map[k]})
    return combined


def extract_negative_lookarounds(parsed):
    results = []

    def walk(subpat):
        for tok, arg in subpat:
            if tok == regex_sre_constants.ASSERT_NOT:
                dirn, sub = arg
                results.append(sub)
            elif tok in (regex_sre_constants.SUBPATTERN, regex_sre_constants.BRANCH,
                         regex_sre_constants.MAX_REPEAT, regex_sre_constants.MIN_REPEAT):
                if tok in (regex_sre_constants.SUBPATTERN, regex_sre_constants.MAX_REPEAT, regex_sre_constants.MIN_REPEAT):
                    if isinstance(arg, tuple):
                        sub = arg[-1]
                    else:
                        sub = arg
                    walk(sub)
                elif tok == regex_sre_constants.BRANCH:
                    _, branches = arg
                    for branch in branches:
                        walk(branch)
            elif tok == regex_sre_constants.IN:
                walk(arg)
    walk(parsed)
    return results


def extract_positive_lookahead_literals(parsed, max_repeat, alphabet):
    """
    Best-effort extraction of literal substrings that are required by positive lookaheads.
    Looks for ASSERT tokens with dirn>0 and heuristically collects contiguous literal runs
    from enumerated sequences of the subpattern (bounded by max_repeat).
    Returns list of unique substrings (sorted with longest first).
    """
    found = []

    def walk(subpat):
        for tok, arg in subpat:
            if tok == regex_sre_constants.ASSERT:
                dirn, sub = arg
                if dirn > 0:
                    seqs, min_len, max_len = enumerate_sequences_for_subpattern(sub, min(max_repeat, 6), alphabet)
                    for seq in seqs:
                        i = 0
                        while i < len(seq):
                            if len(seq[i]) == 1:
                                j = i
                                s_chars = []
                                while j < len(seq) and len(seq[j]) == 1:
                                    s_chars.append(next(iter(seq[j])))
                                    j += 1
                                s_str = ''.join(s_chars)
                                if s_str:
                                    found.append(s_str)
                                i = j
                            else:
                                i += 1
            elif tok in (regex_sre_constants.SUBPATTERN, regex_sre_constants.MAX_REPEAT, regex_sre_constants.MIN_REPEAT):
                if isinstance(arg, tuple):
                    sub = arg[-1]
                else:
                    sub = arg
                walk(sub)
            elif tok == regex_sre_constants.BRANCH:
                _, branches = arg
                for branch in branches:
                    walk(branch)
            elif tok == regex_sre_constants.IN:
                continue

    walk(parsed)
    uniq = sorted(set(found), key=lambda x: (-len(x), x))
    return uniq


def compute_min_max_length(parsed, max_repeat, alphabet):
    """
    Compute approximate min and max length of the entire pattern (conservative).
    Uses max_repeat to bound open-ended repeats.
    """
    def walk(pat):
        total_min = 0
        total_max = 0
        for tok, arg in pat:
            if tok == regex_sre_constants.LITERAL:
                total_min += 1
                total_max += 1
            elif tok == regex_sre_constants.IN:
                total_min += 1
                total_max += 1
            elif tok == regex_sre_constants.CATEGORY:
                total_min += 1
                total_max += 1
            elif tok in (regex_sre_constants.MAX_REPEAT, regex_sre_constants.MIN_REPEAT):
                lo, hi, inner = arg
                if hi == regex_sre_constants.MAXREPEAT or hi is None:
                    hi_eff = lo + max_repeat
                else:
                    hi_eff = min(hi, lo + max_repeat)
                sub_min, sub_max = walk(inner)
                total_min += lo * sub_min
                total_max += hi_eff * sub_max
            elif tok == regex_sre_constants.SUBPATTERN:
                if isinstance(arg, tuple):
                    sub = arg[-1]
                else:
                    sub = arg
                sub_min, sub_max = walk(sub)
                total_min += sub_min
                total_max += sub_max
            elif tok == regex_sre_constants.BRANCH:
                _, branches = arg
                mins = []
                maxs = []
                for br in branches:
                    bmin, bmax = walk(br)
                    mins.append(bmin)
                    maxs.append(bmax)
                total_min += min(mins) if mins else 0
                total_max += max(maxs) if maxs else 0
            elif tok in (regex_sre_constants.AT,):
                continue
            else:
                total_min += 1
                total_max += 1
        return total_min, total_max

    mn, mx = walk(parsed)
    return mn, mx


def subpattern_is_anchored_to_whole(sub):
    """
    Heuristic to detect ^ ... $ inside lookaheads used earlier.
    """
    if not sub:
        return False
    try:
        first_tok = sub[0][0]
        last_tok = sub[-1][0]
        return (first_tok == regex_sre_constants.AT) and (last_tok == regex_sre_constants.AT)
    except Exception:
        return False


def is_pattern_anchored(parsed):
    """
    Simple heuristic: if top-level token list starts with AT and ends with AT, we treat as anchored.
    This is used by the test harness to decide whether to use re.fullmatch or re.search.
    """
    if not parsed:
        return False
    try:
        first_tok = parsed[0][0]
        last_tok = parsed[-1][0]
        return (first_tok == regex_sre_constants.AT) and (last_tok == regex_sre_constants.AT)
    except Exception:
        return False


def compile_regex_to_function_source(
    pattern: str,
    flags: int = 0,
    max_repeat: int = 65535,
    alphabet: str = None,
    func_name: str = "gen",
    max_attempts: int = 2000,
):
    if alphabet is None:
        alphabet = DEFAULT_ALPHABET

    parsed = list(regex_sre_parse.parse(pattern, flags))
    if parsed:
        first_tok, first_arg = parsed[0]
        if first_tok is regex_sre_constants.ASSERT:
            dirn, sub = first_arg
            if dirn < 0:
                def fixed_width(subpat):
                    width = 0
                    for tok, arg in subpat:
                        if tok is regex_sre_constants.LITERAL:
                            width += 1
                        elif tok is regex_sre_constants.IN:
                            width += 1
                        elif tok is regex_sre_constants.CATEGORY:
                            width += 1
                        elif tok is regex_sre_constants.SUBPATTERN:
                            if isinstance(arg, tuple):
                                inner = arg[-1]
                            else:
                                inner = arg
                            w = fixed_width(inner)
                            if w is None:
                                return None
                            width += w
                        else:
                            return None
                    return width
                w = fixed_width(sub)
                if w is not None and w > 0:
                    raise ValueError(
                        f"Pattern: '{pattern}' :starts with a lookbehind of fixed width {w}. "
                        "Such a pattern can never match the entire string with re.fullmatch() "
                        "(there is no text before position 0). Rewrite the pattern so the lookbehind's "
                        "context is part of the matched text."
                    )

    required_substrings = extract_positive_lookahead_literals(parsed, min(max_repeat, 6), alphabet)

    min_len_approx, max_len_approx = compute_min_max_length(parsed, min(max_repeat, 6), alphabet)
    if max_len_approx > 10**6:
        max_len_approx = None

    blocks_src = []
    block_counter = 0
    choices_pool = {}
    choices_list_by_name = {}
    choices_counter = 0

    def get_choice_name_for(choices):
        nonlocal choices_counter
        key = tuple(choices)
        if key in choices_pool:
            return choices_pool[key]
        name = f"c{choices_counter}"
        choices_counter += 1
        choices_pool[key] = name
        choices_list_by_name[name] = tuple(choices)
        return name

    def new_block_id():
        nonlocal block_counter
        i = block_counter
        block_counter += 1
        return i

    def _is_atomic_single_char(sub):
        if not sub:
            return None, None
        if len(sub) != 1:
            return None, None
        tok, arg = sub[0]
        if tok is regex_sre_constants.LITERAL:
            ch = chr(arg)
            return [ch], get_choice_name_for([ch])
        if tok is regex_sre_constants.NOT_LITERAL:
            forbidden = chr(arg)
            choices = [c for c in alphabet if c != forbidden]
            if not choices:
                choices = list(alphabet)
            return choices, get_choice_name_for(choices)
        if tok is regex_sre_constants.ANY:
            choices = [c for c in alphabet if c != "\n"]
            return choices, get_choice_name_for(choices)
        if tok is regex_sre_constants.IN:
            choices = _expand_in_child_static(arg, flags, alphabet)
            return choices, get_choice_name_for(choices)
        if tok is regex_sre_constants.CATEGORY:
            if arg == regex_sre_constants.CATEGORY_DIGIT:
                choices = list(string.digits)
            elif arg == regex_sre_constants.CATEGORY_SPACE:
                choices = list(string.whitespace)
            elif arg == regex_sre_constants.CATEGORY_WORD:
                choices = list(string.ascii_letters + string.digits + "_")
            else:
                choices = list(alphabet)
            return choices, get_choice_name_for(choices)
        if tok is regex_sre_constants.RANGE:
            a, b = arg
            choices = [chr(c) for c in range(a, b + 1)]
            return choices, get_choice_name_for(choices)
        return None, None

    def build_block_for_subpattern(subp) -> int:
        bid = new_block_id()
        lines = []
        lines.append(f"    def _b{bid}(out):")
        lines.append("        local_groups = {}")
        for token, arg in subp:
            if token is regex_sre_constants.LITERAL:
                ch = chr(arg)
                cname = get_choice_name_for([ch])
                lines.append(f"        out.append(_choose_from_list(_CHOICES['{cname}'], out))")
            elif token is regex_sre_constants.NOT_LITERAL:
                forbidden = chr(arg)
                choices = [c for c in alphabet if c != forbidden]
                if not choices:
                    choices = list(alphabet)
                cname = get_choice_name_for(choices)
                lines.append(f"        out.append(_choose_from_list(_CHOICES['{cname}'], out))")
            elif token is regex_sre_constants.ANY:
                choices = [c for c in alphabet if c != "\n"]
                cname = get_choice_name_for(choices)
                lines.append(f"        out.append(_choose_from_list(_CHOICES['{cname}'], out))")
            elif token is regex_sre_constants.IN:
                choices = _expand_in_child_static(arg, flags, alphabet)
                cname = get_choice_name_for(choices)
                lines.append(f"        out.append(_choose_from_list(_CHOICES['{cname}'], out))")
            elif token is regex_sre_constants.CATEGORY:
                if arg in CATEGORY_TO_EXPR:
                    if arg == regex_sre_constants.CATEGORY_DIGIT:
                        choices = list(string.digits)
                    elif arg == regex_sre_constants.CATEGORY_SPACE:
                        choices = list(string.whitespace)
                    elif arg == regex_sre_constants.CATEGORY_WORD:
                        choices = list(string.ascii_letters + string.digits + "_")
                    else:
                        choices = list(alphabet)
                else:
                    if arg == regex_sre_constants.CATEGORY_NOT_DIGIT:
                        choices = [ch for ch in alphabet if ch not in string.digits]
                    elif arg == regex_sre_constants.CATEGORY_NOT_SPACE:
                        choices = [ch for ch in alphabet if ch not in string.whitespace]
                    elif arg == regex_sre_constants.CATEGORY_NOT_WORD:
                        word_chars = set(string.ascii_letters + string.digits + "_")
                        choices = [ch for ch in alphabet if ch not in word_chars]
                    else:
                        choices = list(alphabet)
                cname = get_choice_name_for(choices)
                lines.append(f"        out.append(_choose_from_list(_CHOICES['{cname}'], out))")
            elif token is regex_sre_constants.BRANCH:
                _, branches = arg
                n = len(branches)
                branch_ids = [build_block_for_subpattern(branch) for branch in branches]
                lines.append(f"        i = random.randint(0, {n - 1})")
                for idx, bid2 in enumerate(branch_ids):
                    prefix = "        if" if idx == 0 else "        elif"
                    lines.append(f"{prefix} i == {idx}:")
                    lines.append(f"            lg = _b{bid2}(out)")
                    lines.append("            local_groups.update(lg)")
            elif token is regex_sre_constants.SUBPATTERN:
                if isinstance(arg, tuple):
                    if len(arg) == 4:
                        groupnum, _, _, sub = arg
                    elif len(arg) == 2:
                        groupnum, sub = arg
                    else:
                        groupnum = arg[0] if arg else None
                        sub = arg[-1] if arg else []
                else:
                    groupnum = None
                    sub = arg
                sub_bid = build_block_for_subpattern(sub)
                if groupnum and groupnum > 0:
                    lines.append(f"        start_idx = len(out)")
                    lines.append(f"        lg = _b{sub_bid}(out)")
                    lines.append("        local_groups.update(lg)")
                    lines.append("        groups.update(lg)")
                    lines.append(f"        local_groups[{groupnum}] = ''.join(out[start_idx:])")
                    lines.append(f"        groups[{groupnum}] = local_groups[{groupnum}]")
                else:
                    lines.append(f"        lg = _b{sub_bid}(out)")
                    lines.append("        local_groups.update(lg)")
            elif token in (regex_sre_constants.MAX_REPEAT, regex_sre_constants.MIN_REPEAT):
                lo, hi, sub = arg
                sub_bid = build_block_for_subpattern(sub)
                if hi == regex_sre_constants.MAXREPEAT or hi is None:
                    hi_eff = max(lo + max_repeat, max_repeat)
                else:
                    hi_eff = min(hi, lo + max_repeat)
                if hi_eff < lo:
                    raise ValueError(f"Maximum value {hi_eff} is smaller than minimum value {lo}")
                choices, cname = _is_atomic_single_char(sub)
                if choices is not None:
                    lines.append(f"        count = random.randint({lo}, {hi_eff})")
                    lines.append("        if lookahead_stack or pending_requirements or forbidden_literals:")
                    lines.append("            for _ in range(count):")
                    lines.append(f"                lg = _b{sub_bid}(out)")
                    lines.append("                local_groups.update(lg)")
                    lines.append("        else:")
                    if len(choices) == 1:
                        ch = choices[0]
                        lines.append("            if count:")
                        lines.append(f"                out.extend([{repr(ch)}] * count)")
                    else:
                        lines.append("            if count:")
                        lines.append(f"                picks = random.choices(_CHOICES['{cname}'], k=count)")
                        lines.append("                out.extend(picks)")
                else:
                    lines.append(f"        count = random.randint({lo}, {hi_eff})")
                    lines.append("        for _ in range(count):")
                    lines.append(f"            lg = _b{sub_bid}(out)")
                    lines.append("            local_groups.update(lg)")
            elif token is regex_sre_constants.GROUPREF:
                groupnum = arg
                lines.append(f"        out.append(groups.get({groupnum}, ''))")
            elif token is regex_sre_constants.GROUPREF_IGNORE:
                groupnum = arg
                lines.append(f"        out.append(groups.get({groupnum}, ''))")
            elif token is regex_sre_constants.AT:
                continue
            elif token is regex_sre_constants.RANGE:
                a, b = arg
                choices = [chr(c) for c in range(a, b + 1)]
                cname = get_choice_name_for(choices)
                lines.append(f"        out.append(_choose_from_list(_CHOICES['{cname}'], out))")
            elif token in (regex_sre_constants.ASSERT, regex_sre_constants.ASSERT_NOT):
                dirn, sub = arg
                if token is regex_sre_constants.ASSERT and dirn > 0:
                    reqs = _collect_atomic_requirements(sub, flags, alphabet)
                    if reqs:
                        is_positional = all(
                            isinstance(r, dict) and int(r.get('count', 1)) == 1 and len(tuple(r.get('choices', ()))) == 1
                            for r in reqs
                        )
                        if is_positional:
                            forced_seq = [tuple(r['choices'])[0] for r in reqs]
                            lines.append("        if %d > 0:" % (dirn))
                            lines.append(f"            lookahead_stack.append({{'forced': list({repr(forced_seq)}), 'pos': 0, 'start_pos': len(out)}})")
                        else:
                            lines.append(f"        _register_requirements({repr(reqs)})")
                    else:
                        sub_bid = build_block_for_subpattern(sub)
                        lines.append("        # fallback: couldn't collect atomic requirements; use generated assertion string")
                        lines.append("        old_stack = list(lookahead_stack)")
                        lines.append("        lookahead_stack.clear()")
                        lines.append("        tmp_out = []")
                        lines.append(f"        lg_assert = _b{sub_bid}(tmp_out)")
                        lines.append("        t_assert = ''.join(tmp_out)")
                        lines.append("        lookahead_stack[:] = old_stack")
                        lines.append("        if %d > 0:" % (dirn))
                        lines.append("            lookahead_stack.append({'forced': list(t_assert), 'pos': 0, 'start_pos': len(out)})")
                        lines.append("        else:")
                        lines.append("            total = ''.join(out)")
                        lines.append("            if len(total) < len(t_assert):")
                        lines.append("                if not t_assert.endswith(total):")
                        lines.append("                    raise AssertionError('lookbehind failed - mismatch')")
                        lines.append("                missing = len(t_assert) - len(total)")
                        lines.append("                out[0:0] = list(t_assert[:missing])")
                        lines.append("            else:")
                        lines.append("                if total[-len(t_assert):] != t_assert:")
                        lines.append("                    raise AssertionError('lookbehind assertion failed')")
                elif token is regex_sre_constants.ASSERT and dirn < 0:
                    sub_bid = build_block_for_subpattern(sub)
                    lines.append("        # ASSERT (lookbehind) - fallback")
                    lines.append("        old_stack = list(lookahead_stack)")
                    lines.append("        lookahead_stack.clear()")
                    lines.append("        tmp_out = []")
                    lines.append(f"        lg_assert = _b{sub_bid}(tmp_out)")
                    lines.append("        t_assert = ''.join(tmp_out)")
                    lines.append("        lookahead_stack[:] = old_stack")
                    lines.append("        if %d > 0:" % (dirn))
                    lines.append("            lookahead_stack.append({'forced': list(t_assert), 'pos': 0, 'start_pos': len(out)})")
                    lines.append("        else:")
                    lines.append("            total = ''.join(out)")
                    lines.append("            if len(total) < len(t_assert):")
                    lines.append("                if not t_assert.endswith(total):")
                    lines.append("                    raise AssertionError('lookbehind failed - mismatch')")
                    lines.append("                missing = len(t_assert) - len(total)")
                    lines.append("                out[0:0] = list(t_assert[:missing])")
                    lines.append("            else:")
                    lines.append("                if total[-len(t_assert):] != t_assert:")
                    lines.append("                    raise AssertionError('lookbehind assertion failed')")
                elif token is regex_sre_constants.ASSERT_NOT:
                    seqs, min_len, max_len = enumerate_sequences_for_subpattern(sub, max_repeat, alphabet)
                    merged, epsilon = merged_forbidden_from_sequences(seqs, max_len, alphabet)
                    seqs_as_lists = [[tuple(sorted(s)) for s in seq] for seq in seqs]
                    merged_as_lists = [tuple(sorted(s)) for s in merged]
                    epsilon_as_list = list(epsilon)
                    anchored_whole = subpattern_is_anchored_to_whole(sub)
                    lines.append("        # ASSERT_NOT (negative lookahead)")
                    lines.append(f"        _NEG_LOOKS.append({{")
                    lines.append(f"            'start_pos': len(out),")
                    lines.append(f"            'seqs': {repr(seqs_as_lists)},")
                    lines.append(f"            'merged': {repr(merged_as_lists)},")
                    lines.append(f"            'epsilon': {repr(epsilon_as_list)},")
                    lines.append(f"            'min_len': {min_len},")
                    lines.append(f"            'max_len': {max_len},")
                    lines.append(f"            'anchored_to_whole': {repr(bool(anchored_whole))}")
                    lines.append("        })")
                else:
                    sub_bid = build_block_for_subpattern(sub)
                    lines.append("        # ASSERT unexpected form - fallback")
                    lines.append("        old_stack = list(lookahead_stack)")
                    lines.append("        lookahead_stack.clear()")
                    lines.append("        tmp_out = []")
                    lines.append(f"        lg_assert = _b{sub_bid}(tmp_out)")
                    lines.append("        t_assert = ''.join(tmp_out)")
                    lines.append("        lookahead_stack[:] = old_stack")
                    lines.append("        if %d > 0:" % (dirn))
                    lines.append("            lookahead_stack.append({'forced': list(t_assert), 'pos': 0, 'start_pos': len(out)})")
                    lines.append("        else:")
                    lines.append("            total = ''.join(out)")
                    lines.append("            if len(total) < len(t_assert):")
                    lines.append("                if not t_assert.endswith(total):")
                    lines.append("                    raise AssertionError('lookbehind failed - mismatch')")
                    lines.append("                missing = len(t_assert) - len(total)")
                    lines.append("                out[0:0] = list(t_assert[:missing])")
                    lines.append("            else:")
                    lines.append("                if total[-len(t_assert):] != t_assert:")
                    lines.append("                    raise AssertionError('lookbehind assertion failed')")
            else:
                raise NotImplementedError(f"Token not implemented in compiler: {token} ({arg})")
        lines.append("        return local_groups")
        block_src = "\n".join(lines)
        blocks_src.append(block_src)
        return bid

    top_block_id = build_block_for_subpattern(parsed)

    func_lines = []
    func_lines.append(f"def {func_name}():")
    func_lines.append("    groups = {}")
    func_lines.append("    lookahead_stack = []")
    func_lines.append("    pending_requirements = deque()")
    func_lines.append("    forbidden_literals = set()")
    func_lines.append("    LFORB_MAX = 0")
    func_lines.append("    _NEG_LOOKS = []")
    func_lines.append("    flagged_positions = []")
    func_lines.append(f"    _REQUIRED_SUBSTRINGS = {repr(tuple(required_substrings))}")
    func_lines.append(f"    _MIN_LEN = {repr(min_len_approx)}")
    func_lines.append(f"    _MAX_LEN = {repr(max_len_approx)}")

    func_lines.append("")
    func_lines.append("    def _register_requirements(reqs):")
    func_lines.append("        for r in reqs:")
    func_lines.append("            tpl = tuple(r['choices'])")
    func_lines.append("            pending_requirements.append({'count': int(r['count']), 'choices': tpl})")
    func_lines.append("")
    func_lines.append("    def _register_forbidden_literal(seq):")
    func_lines.append("        forbidden_literals.add(seq)")
    func_lines.append("        nonlocal LFORB_MAX")
    func_lines.append("        LFORB_MAX = max(LFORB_MAX, len(seq))")
    func_lines.append("")
    func_lines.append("    def _rand_choice(tpl):")
    func_lines.append("        if len(tpl) == 1:")
    func_lines.append("            return tpl[0]")
    func_lines.append("        return tpl[random.randrange(len(tpl))]")
    func_lines.append("")
    func_lines.append("    def _choose_from_list(choices_tpl, out):")
    func_lines.append("        if lookahead_stack:")
    func_lines.append("            top = lookahead_stack[-1]")
    func_lines.append("            pos = top.get('pos', 0)")
    func_lines.append("            if 'forced' in top and top['forced'] is not None and pos < len(top['forced']):")
    func_lines.append("                required = top['forced'][pos]")
    func_lines.append("                if required not in choices_tpl:")
    func_lines.append("                    raise AssertionError('lookahead forced char not available')")
    func_lines.append("                ch = required")
    func_lines.append("                top['pos'] = pos + 1")
    func_lines.append("                if top['pos'] >= len(top['forced']):")
    func_lines.append("                    lookahead_stack.pop()")
    func_lines.append("                for _ in range(len(pending_requirements)):")
    func_lines.append("                    r = pending_requirements[0]")
    func_lines.append("                    if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                        r['count'] -= 1")
    func_lines.append("                        if r['count'] <= 0:")
    func_lines.append("                            pending_requirements.popleft()")
    func_lines.append("                        break")
    func_lines.append("                return ch")
    func_lines.append("            if 'forbidden' in top and top['forbidden'] is not None and pos < len(top['forbidden']):")
    func_lines.append("                forb = top['forbidden'][pos]")
    func_lines.append("                attempts_inner = 3")
    func_lines.append("                while attempts_inner > 0:")
    func_lines.append("                    ch = _rand_choice(choices_tpl)")
    func_lines.append("                    if ch != forb:")
    func_lines.append("                        break")
    func_lines.append("                    attempts_inner -= 1")
    func_lines.append("                if attempts_inner <= 0 and len(choices_tpl) == 1 and choices_tpl[0] == forb:")
    func_lines.append("                    raise AssertionError('lookahead forbidden removed all choices')")
    func_lines.append("                top['pos'] = pos + 1")
    func_lines.append("                if top['pos'] >= len(top['forbidden']):")
    func_lines.append("                    lookahead_stack.pop()")
    func_lines.append("                for _ in range(len(pending_requirements)):")
    func_lines.append("                    r = pending_requirements[0]")
    func_lines.append("                    if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                        r['count'] -= 1")
    func_lines.append("                        if r['count'] <= 0:")
    func_lines.append("                            pending_requirements.popleft()")
    func_lines.append("                        break")
    func_lines.append("                return ch")
    func_lines.append("")
    func_lines.append("        filtered = list(choices_tpl)")
    func_lines.append("        if _NEG_LOOKS:")
    func_lines.append("            cur_pos = len(out)")
    func_lines.append("            for ni, nl in enumerate(_NEG_LOOKS):")
    func_lines.append("                sp = nl['start_pos']")
    func_lines.append("                o = cur_pos - sp")
    func_lines.append("                if o >= 0 and o < nl['max_len']:")
    func_lines.append("                    forb_tuple = nl.get('merged')")
    func_lines.append("                    if forb_tuple and o < len(forb_tuple):")
    func_lines.append("                        forb = set(forb_tuple[o])")
    func_lines.append("                        if forb:")
    func_lines.append("                            filtered = [c for c in filtered if c not in forb]")
    func_lines.append("")
    func_lines.append("        if pending_requirements and filtered:")
    func_lines.append("            req_choices_union = set()")
    func_lines.append("            for r in pending_requirements:")
    func_lines.append("                if r['count'] > 0:")
    func_lines.append("                    req_choices_union.update(r['choices'])")
    func_lines.append("            valid_candidates = [c for c in filtered if c in req_choices_union]")
    func_lines.append("            if valid_candidates:")
    func_lines.append("                ch = random.choice(valid_candidates)")
    func_lines.append("            else:")
    func_lines.append("                ch = _rand_choice(tuple(filtered))")
    func_lines.append("            to_remove = []")
    func_lines.append("            for idx, r in enumerate(pending_requirements):")
    func_lines.append("                if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                    r['count'] -= 1")
    func_lines.append("                    if r['count'] <= 0:")
    func_lines.append("                        to_remove.append(idx)")
    func_lines.append("            for idx in reversed(to_remove):")
    func_lines.append("                del pending_requirements[idx]")
    func_lines.append("            return ch")
    func_lines.append("")
    func_lines.append("        if not filtered:")
    func_lines.append("            temp_ch = _rand_choice(_CHOICES['__alphabet__'])")
    func_lines.append("            flagged_positions.append({'idx': len(out), 'temp': temp_ch, 'token_choices': tuple(choices_tpl), 'causes': None})")
    func_lines.append("            return temp_ch")
    func_lines.append("")
    func_lines.append("        if forbidden_literals and LFORB_MAX > 0:")
    func_lines.append("            tail_len = LFORB_MAX - 1")
    func_lines.append("            tail = ''.join(out[-tail_len:]) if tail_len > 0 else ''")
    func_lines.append("            attempts_inner = 5")
    func_lines.append("            while attempts_inner > 0:")
    func_lines.append("                ch = _rand_choice(tuple(filtered))")
    func_lines.append("                would = tail + ch")
    func_lines.append("                bad = False")
    func_lines.append("                for forb in forbidden_literals:")
    func_lines.append("                    if would.endswith(forb):")
    func_lines.append("                        bad = True")
    func_lines.append("                        break")
    func_lines.append("                if not bad:")
    func_lines.append("                    for _ in range(len(pending_requirements)):")
    func_lines.append("                        r = pending_requirements[0]")
    func_lines.append("                        if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                            r['count'] -= 1")
    func_lines.append("                            if r['count'] <= 0:")
    func_lines.append("                                pending_requirements.popleft()")
    func_lines.append("                            break")
    func_lines.append("                    return ch")
    func_lines.append("                attempts_inner -= 1")
    func_lines.append("")
    func_lines.append("        ch = _rand_choice(tuple(filtered))")
    func_lines.append("        for _ in range(len(pending_requirements)):")
    func_lines.append("            r = pending_requirements[0]")
    func_lines.append("            if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                r['count'] -= 1")
    func_lines.append("                if r['count'] <= 0:")
    func_lines.append("                    pending_requirements.popleft()")
    func_lines.append("                break")
    func_lines.append("        return ch")
    func_lines.append("")

    func_lines.append("    _CHOICES = {}")
    func_lines.append(f"    _CHOICES['__alphabet__'] = {repr(tuple(alphabet))}")
    for name, tpl in choices_list_by_name.items():
        func_lines.append(f"    _CHOICES['{name}'] = {repr(tpl)}")

    func_lines.extend(blocks_src)

    func_lines.append("    def _neg_match_on_string(nl, s):")
    func_lines.append("        start = nl['start_pos']")
    func_lines.append("        anchored_whole = nl.get('anchored_to_whole', False)")
    func_lines.append("        if nl.get('seqs'):")
    func_lines.append("            for seq in nl['seqs']:")
    func_lines.append("                seq_len = len(seq)")
    func_lines.append("                if anchored_whole and (start + seq_len != len(s)):")
    func_lines.append("                    continue")
    func_lines.append("                if start + seq_len > len(s):")
    func_lines.append("                    continue")
    func_lines.append("                ok = True")
    func_lines.append("                for i, charset in enumerate(seq):")
    func_lines.append("                    ch = s[start + i]")
    func_lines.append("                    if ch not in set(charset):")
    func_lines.append("                        ok = False")
    func_lines.append("                        break")
    func_lines.append("                if ok:")
    func_lines.append("                    return True")
    func_lines.append("            return False")
    func_lines.append("        merged = nl.get('merged', [])")
    func_lines.append("        max_len = nl.get('max_len', 0)")
    func_lines.append("        if anchored_whole:")
    func_lines.append("            wanted_len = len(s) - start")
    func_lines.append("            if wanted_len < 0 or wanted_len > max_len:")
    func_lines.append("                return False")
    func_lines.append("            ok = True")
    func_lines.append("            for i in range(wanted_len):")
    func_lines.append("                if i < len(merged):")
    func_lines.append("                    ch = s[start + i]")
    func_lines.append("                    if ch not in set(merged[i]):")
    func_lines.append("                        ok = False")
    func_lines.append("                        break")
    func_lines.append("            return ok")
    func_lines.append("        for length in range(0, max_len + 1):")
    func_lines.append("            if start + length > len(s):")
    func_lines.append("                continue")
    func_lines.append("            ok = True")
    func_lines.append("            for i in range(length):")
    func_lines.append("                if i < len(merged):")
    func_lines.append("                    ch = s[start + i]")
    func_lines.append("                    if ch not in set(merged[i]):")
    func_lines.append("                        ok = False")
    func_lines.append("                        break")
    func_lines.append("            if ok:")
    func_lines.append("                return True")
    func_lines.append("        return False")
    func_lines.append("")

    func_lines.append("    def _try_fix_flagged_positions(out, flagged_positions, _NEG_LOOKS, max_attempts_per_flag=20, alphabet_tuple=None):")
    func_lines.append("        if alphabet_tuple is None:")
    func_lines.append("            alphabet_tuple = tuple(_CHOICES['__alphabet__'])")
    func_lines.append("        flags = list(flagged_positions)")
    func_lines.append("        total_pass_attempts = 3")
    func_lines.append("        for _ in range(total_pass_attempts):")
    func_lines.append("            progress = False")
    func_lines.append("            random.shuffle(flags)")
    func_lines.append("            for f in flags[:]:")
    func_lines.append("                idx = f['idx']")
    func_lines.append("                token_choices = f.get('token_choices')")
    func_lines.append("                pool = list(token_choices) if token_choices else list(alphabet_tuple)")
    func_lines.append("                random.shuffle(pool)")
    func_lines.append("                tried = set()")
    func_lines.append("                attempts = 0")
    func_lines.append("                while attempts < max_attempts_per_flag and pool:")
    func_lines.append("                    ch = pool.pop()")
    func_lines.append("                    if ch in tried:")
    func_lines.append("                        continue")
    func_lines.append("                    tried.add(ch)")
    func_lines.append("                    old = out[idx]")
    func_lines.append("                    out[idx] = ch")
    func_lines.append("                    affected = []")
    func_lines.append("                    for ni, nl in enumerate(_NEG_LOOKS):")
    func_lines.append("                        sp = nl['start_pos']")
    func_lines.append("                        if sp <= idx < sp + nl['max_len']:")
    func_lines.append("                            affected.append(ni)")
    func_lines.append("                    s = ''.join(out)")
    func_lines.append("                    ok = True")
    func_lines.append("                    for ai in affected:")
    func_lines.append("                        if _neg_match_on_string(_NEG_LOOKS[ai], s):")
    func_lines.append("                            ok = False")
    func_lines.append("                            break")
    func_lines.append("                    if ok:")
    func_lines.append("                        progress = True")
    func_lines.append("                        flags.remove(f)")
    func_lines.append("                        break")
    func_lines.append("                    else:")
    func_lines.append("                        out[idx] = old")
    func_lines.append("                        attempts += 1")
    func_lines.append("            if not progress:")
    func_lines.append("                break")
    func_lines.append("        final_s = ''.join(out)")
    func_lines.append("        for nl in _NEG_LOOKS:")
    func_lines.append("            if _neg_match_on_string(nl, final_s):")
    func_lines.append("                return False")
    func_lines.append("        flagged_positions[:] = flags")
    func_lines.append("        return True")
    func_lines.append("")

    func_lines.append("    def _enforce_required_substrings(out, required_list, max_attempts_overall=50, max_positions_try=50):")
    func_lines.append("        s0 = ''.join(out)")
    func_lines.append("        def candidate_positions(req, s):")
    func_lines.append("            L = len(s)")
    func_lines.append("            max_base = max(0, L - len(req))")
    func_lines.append("            positions = list(range(0, max_base + 1))")
    func_lines.append("            positions.append(L)")
    func_lines.append("            random.shuffle(positions)")
    func_lines.append("            return positions[:max_positions_try]")
    func_lines.append("        def violates_neg_looks(assembled):")
    func_lines.append("            for nl in _NEG_LOOKS:")
    func_lines.append("                if _neg_match_on_string(nl, assembled):")
    func_lines.append("                    return True")
    func_lines.append("            return False")
    func_lines.append("        reqs = [r for r in required_list if r]")
    func_lines.append("        if not reqs:")
    func_lines.append("            return True")
    func_lines.append("        if _MAX_LEN is not None and sum(len(r) for r in reqs) > _MAX_LEN:")
    func_lines.append("            return False")
    func_lines.append("        attempts = 0")
    func_lines.append("        while attempts < max_attempts_overall:")
    func_lines.append("            attempts += 1")
    func_lines.append("            reserved = {}")
    func_lines.append("            placements = [None] * len(reqs)")
    func_lines.append("            order = sorted(range(len(reqs)), key=lambda i: -len(reqs[i]))")
    func_lines.append("            def backtrack(idx):")
    func_lines.append("                nonlocal reserved, placements")
    func_lines.append("                if idx >= len(order):")
    func_lines.append("                    base = list(s0)")
    func_lines.append("                    needed_len = len(base)")
    func_lines.append("                    for pi, pos in enumerate(placements):")
    func_lines.append("                        req = reqs[pi]")
    func_lines.append("                        if pos is None:")
    func_lines.append("                            return False")
    func_lines.append("                        endpos = pos + len(req)")
    func_lines.append("                        if endpos > needed_len:")
    func_lines.append("                            needed_len = endpos")
    func_lines.append("                    while len(base) < needed_len:")
    func_lines.append("                        base.append(_rand_choice(_CHOICES['__alphabet__']))")
    func_lines.append("                    for pi, pos in enumerate(placements):")
    func_lines.append("                        req = reqs[pi]")
    func_lines.append("                        for j, ch in enumerate(req):")
    func_lines.append("                            base[pos + j] = ch")
    func_lines.append("                    assembled = ''.join(base)")
    func_lines.append("                    if _MAX_LEN is not None and len(assembled) > _MAX_LEN:")
    func_lines.append("                        return False")
    func_lines.append("                    if violates_neg_looks(assembled):")
    func_lines.append("                        return False")
    func_lines.append("                    out[:] = list(assembled)")
    func_lines.append("                    return True")
    func_lines.append("                i = order[idx]")
    func_lines.append("                req = reqs[i]")
    func_lines.append("                positions = candidate_positions(req, s0)")
    func_lines.append("                for p in range(0, max(0, len(s0) - len(req) + 1)):")
    func_lines.append("                    if s0[p:p+len(req)] == req:")
    func_lines.append("                        positions.insert(0, p)")
    func_lines.append("                random.shuffle(positions)")
    func_lines.append("                for pos in positions:")
    func_lines.append("                    conflict = False")
    func_lines.append("                    to_add = []")
    func_lines.append("                    for j, ch in enumerate(req):")
    func_lines.append("                        target = pos + j")
    func_lines.append("                        if target in reserved and reserved[target] != ch:")
    func_lines.append("                            conflict = True")
    func_lines.append("                            break")
    func_lines.append("                        to_add.append((target, ch))")
    func_lines.append("                    if conflict:")
    func_lines.append("                        continue")
    func_lines.append("                    for (t, ch) in to_add:")
    func_lines.append("                        reserved[t] = ch")
    func_lines.append("                    placements[i] = pos")
    func_lines.append("                    ok = backtrack(idx + 1)")
    func_lines.append("                    if ok:")
    func_lines.append("                        return True")
    func_lines.append("                    placements[i] = None")
    func_lines.append("                    for (t, ch) in to_add:")
    func_lines.append("                        if reserved.get(t) == ch:")
    func_lines.append("                            del reserved[t]")
    func_lines.append("                return False")
    func_lines.append("            if backtrack(0):")
    func_lines.append("                return True")
    func_lines.append("        return False")
    func_lines.append("")

    func_lines.append(f"    attempts = {max_attempts}")
    func_lines.append("    while True:")
    func_lines.append("        groups = {}")
    func_lines.append("        lookahead_stack = []")
    func_lines.append("        pending_requirements = deque()")
    func_lines.append("        forbidden_literals = set()")
    func_lines.append("        flagged_positions = []")
    func_lines.append("        LFORB_MAX = 0")
    func_lines.append("        try:")
    func_lines.append("            out = []")
    func_lines.append(f"            lg = _b{top_block_id}(out)")
    func_lines.append("            groups.update(lg)")
    func_lines.append("")
    func_lines.append("            s_current = ''.join(out)")
    func_lines.append("            missing = []")
    func_lines.append("            while pending_requirements:")
    func_lines.append("                r = pending_requirements.popleft()")
    func_lines.append("                needed = r['count']")
    func_lines.append("                satisfied = sum(1 for choice in r['choices'] if choice in s_current)")
    func_lines.append("                still_needed = max(0, needed - satisfied)")
    func_lines.append("                for _ in range(still_needed):")
    func_lines.append("                    missing.append(_rand_choice(r['choices']))")
    func_lines.append("")
    func_lines.append("            if missing:")
    func_lines.append("                for ch in missing:")
    func_lines.append("                    safe = False")
    func_lines.append("                    attempts_inner = 5")
    func_lines.append("                    while attempts_inner > 0:")
    func_lines.append("                        tail_len = LFORB_MAX - 1")
    func_lines.append("                        tail = ''.join(out[-tail_len:]) if tail_len > 0 else ''")
    func_lines.append("                        would = tail + ch")
    func_lines.append("                        bad = False")
    func_lines.append("                        for forb in forbidden_literals:")
    func_lines.append("                            if would.endswith(forb):")
    func_lines.append("                                bad = True")
    func_lines.append("                                break")
    func_lines.append("                        if not bad:")
    func_lines.append("                            out.append(ch)")
    func_lines.append("                            safe = True")
    func_lines.append("                            break")
    func_lines.append("                        attempts_inner -= 1")
    func_lines.append("                    if not safe:")
    func_lines.append("                        raise AssertionError('forbidden literal produced during missing append')")
    func_lines.append("            s = ''.join(out)")
    func_lines.append("            for forb in list(forbidden_literals):")
    func_lines.append("                if forb and forb in s:")
    func_lines.append("                    raise AssertionError('forbidden literal produced')")
    func_lines.append("")
    func_lines.append("            if flagged_positions:")
    func_lines.append("                ok = _try_fix_flagged_positions(out, flagged_positions, _NEG_LOOKS)")
    func_lines.append("                if ok:")
    func_lines.append("                    if _REQUIRED_SUBSTRINGS:")
    func_lines.append("                        if not _enforce_required_substrings(out, _REQUIRED_SUBSTRINGS):")
    func_lines.append("                            raise AssertionError('required substrings insertion failed after flag repair')")
    func_lines.append("                    return ''.join(out)")
    func_lines.append("                else:")
    func_lines.append("                    final_s = ''.join(out)")
    func_lines.append("                    any_neg_match = False")
    func_lines.append("                    for nl in _NEG_LOOKS:")
    func_lines.append("                        if _neg_match_on_string(nl, final_s):")
    func_lines.append("                            any_neg_match = True")
    func_lines.append("                            break")
    func_lines.append("                    if any_neg_match:")
    func_lines.append("                        raise AssertionError('negative lookahead produced after repairs')")
    func_lines.append("                    else:")
    func_lines.append("                        if _REQUIRED_SUBSTRINGS:")
    func_lines.append("                            if not _enforce_required_substrings(out, _REQUIRED_SUBSTRINGS):")
    func_lines.append("                                raise AssertionError('required substrings insertion failed after unsuccessful flag repair')")
    func_lines.append("                        return ''.join(out)")
    func_lines.append("            else:")
    func_lines.append("                final_s = ''.join(out)")
    func_lines.append("                any_neg_match = False")
    func_lines.append("                for nl in _NEG_LOOKS:")
    func_lines.append("                    if _neg_match_on_string(nl, final_s):")
    func_lines.append("                        any_neg_match = True")
    func_lines.append("                        break")
    func_lines.append("                if any_neg_match:")
    func_lines.append("                    raise AssertionError('negative lookahead produced')")
    func_lines.append("                if _REQUIRED_SUBSTRINGS:")
    func_lines.append("                    if not _enforce_required_substrings(out, _REQUIRED_SUBSTRINGS):")
    func_lines.append("                        raise AssertionError('required substrings insertion failed (no flags)')")
    func_lines.append("                return ''.join(out)")
    func_lines.append("        except AssertionError as e:")
    func_lines.append("            attempts -= 1")
    func_lines.append("            if attempts <= 0:")
    func_lines.append("                raise RuntimeError('Failed to generate matching string: ' + str(e))")
    func_lines.append("            # else retry by looping back")
    func_lines.append("")

    return "\n".join(func_lines)



if __name__ == "__main__":
    patterns = [
        r"^(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9])(?=.*[!@#\$%\^&\*]).{12,20}$",
        r"^(a|b|c)\1{3,}$",
        r"^(?=.*foo)(?=.*bar)(?=.*baz).{10,50}$",
        r"^(a|aa)+$",
        r"(?<=foo|bar)baz",
        r"^(?!.*(.)\1).*$",
        r"^(?![A-Z]{2}\d)(?=.*[a-z])foo\d{2}$",
        r"(?=[A-Za-z])[a-z]",
        r"(?![A-Z]{1,4}[1-9]{1,4})",
        r"^(?!^[-+.]*$)[+-]?0*\d*\.?\d{0,2}0*$",
        r"([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?",
        r"(?:https?:\/\/)?(?:(?:(?:www\.?)?youtube\.com(?:\/(?:(?:watch\?.*?(v=[^&\s]+).*)|(?:v(\/.*))|(channel\/.+)|(?:user\/(.+))|(?:results\?(search_query=.+))))?)|(?:youtu\.be(\/.*)?))",
        r"^(0?[1-9]|1[0-2])[\/](0?[1-9]|[12]\d|3[01])[\/](19|20)\d{2}$",
        r"^\s*(?:\+?(\d{1,3}))?([-. (]*(\d{3})[-. )]*)?((\d{3})[-. ]*(\d{2,4})(?:[-.x ]*(\d+))?)\s*$",
        r"\b(?:A[cglmr-u]|B[aehikr]?|C[adefl-orsu]?|D[bsy]|E[rsu]|F[elmr]?|G[ade]|H[efgos]?|I[nr]?|Kr?|L[airuv]|M[dgont]|N[abdeiop]?|Os?|P[abdmortu]?|R[abe-hnu]|S[bcegimnr]?|T[abcehilm]|U(?:u[opst])?|V|W|Xe|Yb?|Z[nr])\b",
        r"[-a-zA-Z0-9@:%_\+.~#?&//=]{2,256}\.[a-z]{2,4}\b(\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?",
        r"<.*?script.*\/?>",
        r"\b(?:(?:2(?:[0-4][0-9]|5[0-5])|[0-1]?[0-9]?[0-9])\.){3}(?:(?:2([0-4][0-9]|5[0-5])|[0-1]?[0-9]?[0-9]))\b",
        r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[a-zA-Z]).{8,}$",
        r"[a-z0-9!#$%&'*+/=?^_{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
        r"^\d{5}(-\d{4})?$",
        r"^(?=(?:.*[A-Z]){2})(?=(?:.*[a-z]){2})(?=.*\d).{5,15}$",
        
        r"foo(?=bar)bar",
        r"pre(?<=pre)mid",
        r"start(?!bad)good",
        r"(?<!no)yes",
        r"^[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}$",
        r"^\d{4}(-\d{5})?$",
        r"^hello$",
        r"^[A-Za-z0-9]{20}$",
        r"^(a|b|c|d|e|f|g|h){10}$",
        r"^a{0,}$",
        r"^(a|b)(c|d)\1\2$",
        r"[ -~]{100}",
        r"^[A-Z0-9_-]{5,10}$",
        r"^---$",
        r"^[\w.-]+$",
        r"^[^-]+$",
        r"^[.-]+$",
        r"^[a-zA-Z]{3}\d{2}$",
        r"^a{0,}$",
        r"^b{2,5}$",
        r"^c{,3}$",
        r"^.{1,10}$",
        r"^(?:ab){2,4}$",
        r"^(foo|bar|baz)$",
        r"^(x|y|z){5}$",
        r"^(a|bc|def)$",
        r"^(a|b)\1$",
        r"^(.)(.)\2\1$",
        r"^(hello)\1$",
        r"^start.*end$",
        r"^\d+$",
        r"^\s+\w+\s*$",
        r"^[\t\n\r\f\v]+$",
        r"^[\d\D]+$",
        r"^[\w\W]+$",
        r"^[\s\S]+$",
        r"^[01]{32}$",
        r"^[A-F0-9]{8}$",
        r"^[ -~]{50}$",
        r"^[a-z]{1000}$",
    ]

    amount = 100000
    for pat in patterns:
        print("=== pattern:", pat, "=== amount:", amount)
        import time
        import re
        start = time.time()
        max_repeat = 6
        try:
            parsed_local = list(regex_sre_parse.parse(pat))
            anchored = is_pattern_anchored(parsed_local)
            src = compile_regex_to_function_source(pat, flags=0, max_repeat=max_repeat, func_name="gen", max_attempts=500)
            safe_builtins = {
                "len": len, "range": range, "min": min, "max": max, "sum":sum, "sorted":sorted,
                "list": list, "tuple": tuple, "chr": chr, "ord": ord,
                "set": set, "map": map, "int": int, "AssertionError": AssertionError,
                "enumerate": enumerate, "reversed": reversed,
                "str": str, "RuntimeError": RuntimeError, "ValueError": ValueError,
                "TypeError": TypeError, "print": print, "bool": bool,
            }
            env = {
                "__builtins__": safe_builtins,
                "random": random,
                "string": string,
                "deque": deque,
            }
            exec(src, env)
            gen = env["gen"]
            samples = [gen() for _ in range(amount)]
            print("Samples example:", samples[0:3], len(samples))
            end = time.time()
            print("time:", end - start)
            for s in samples:
                if anchored:
                    ok = bool(re.fullmatch(pat, s))
                else:
                    ok = bool(re.search(pat, s))
                if not ok:
                    print("FAILED: pattern", pat, "string", repr(s))
                    raise AssertionError("Generated string does not match pattern")
        except Exception as e:
            print("ERROR generating for pattern:", pat, " ->", e)

try:
    from . import regex_sre_parse
    from . import regex_sre_constants
except:
    print("running from main")
    import regex_sre_parse
    import regex_sre_constants
import string
import random
from collections import deque

DEFAULT_ALPHABET = "".join(chr(i) for i in range(32, 127))

CATEGORY_TO_EXPR = {
    regex_sre_constants.CATEGORY_DIGIT: "string.digits",
    regex_sre_constants.CATEGORY_SPACE: "string.whitespace",
    regex_sre_constants.CATEGORY_WORD: "string.ascii_letters + string.digits + '_'",
}

# ---------- Helper static utilities (kept at top for clarity) ----------

def _expand_in_child_static(arg, flags, alphabet):
    """
    Static version used at compile-time to compute the concrete choices set for an IN token.
    """
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

    blocks_src = []
    block_counter = 0

    # choices pool deduplication (compile-time)
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
        """
        Compile-time check whether `sub` represents a single atomic token that emits exactly
        one character each repetition, and can therefore be optimized into a random.choices() call.
        Returns (choices_list, cname) or (None, None) if not simple.
        """
        # Accept only a single-token sequence which is one of the atomic emitters.
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
                # Always precompute concrete choices at compile time and reference via _CHOICES
                if arg in CATEGORY_TO_EXPR:
                    # compute choices now
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
                # record start index, call sub-block which appends into out
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
                    hi_eff = min(lo + max_repeat, max_repeat)
                else:
                    hi_eff = min(hi, lo + max_repeat)
                if hi_eff < lo:
                    raise ValueError(f"Maximum value {hi_eff} is smaller than minimum value {lo}")

                # Try compile-time detection whether `sub` is a single atomic emitter that can be
                # optimized using random.choices(k=count). If so, use a fast path at runtime but
                # only when there are no active lookahead/pending/forbidden constraints (checked
                # at runtime). Otherwise, fall back to the previous per-iteration sub-block call.
                choices, cname = _is_atomic_single_char(sub)
                if choices is not None:
                    # optimized fast path guarded by runtime checks
                    lines.append(f"        count = random.randint({lo}, {hi_eff})")
                    lines.append("        if lookahead_stack or pending_requirements or forbidden_literals:")
                    lines.append("            for _ in range(count):")
                    lines.append(f"                lg = _b{sub_bid}(out)")
                    lines.append("                local_groups.update(lg)")
                    lines.append("        else:")
                    # For a single literal it's often faster to replicate; for choices use random.choices
                    if len(choices) == 1:
                        ch = choices[0]
                        # repeat fixed char
                        lines.append("            if count:")
                        lines.append(f"                out.extend([{repr(ch)}] * count)")
                    else:
                        lines.append("            if count:")
                        lines.append(f"                picks = random.choices(_CHOICES['{cname}'], k=count)")
                        lines.append("                for ch in picks:")
                        lines.append("                    out.append(ch)")
                else:
                    # fallback: previous behavior (per-iteration call)
                    lines.append(f"        count = random.randint({lo}, {hi_eff})")
                    lines.append("        for _ in range(count):")
                    lines.append(f"            lg = _b{sub_bid}(out)")
                    lines.append("            local_groups.update(lg)")
            elif token is regex_sre_constants.GROUPREF:
                groupnum = arg
                # insert previously captured group content if available (groups dict), else empty
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
                # Use improved ASSERT handling, but keep old fallback behavior via temporary buffers
                dirn, sub = arg
                if token is regex_sre_constants.ASSERT and dirn > 0:
                    reqs = _collect_atomic_requirements(sub, flags, alphabet)
                    lines.append("        # ASSERT (positive lookahead) - register minimal atomic requirements")
                    if reqs:
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
                        lines.append("            lookahead_stack.append({'forced': list(t_assert), 'pos': 0})")
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
                    lines.append("        # ASSERT (lookbehind) - fallback to previous behavior")
                    lines.append("        old_stack = list(lookahead_stack)")
                    lines.append("        lookahead_stack.clear()")
                    lines.append("        tmp_out = []")
                    lines.append(f"        lg_assert = _b{sub_bid}(tmp_out)")
                    lines.append("        t_assert = ''.join(tmp_out)")
                    lines.append("        lookahead_stack[:] = old_stack")
                    lines.append("        if %d > 0:" % (dirn))
                    lines.append("            lookahead_stack.append({'forced': list(t_assert), 'pos': 0})")
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
                    can_extract_literal = True
                    literal_chars = []
                    def try_extract_literal(pat):
                        nonlocal can_extract_literal, literal_chars
                        for tok2, arg2 in pat:
                            if tok2 is regex_sre_constants.LITERAL:
                                literal_chars.append(chr(arg2))
                            elif tok2 is regex_sre_constants.SUBPATTERN:
                                if isinstance(arg2, tuple):
                                    sub2 = arg2[-1]
                                else:
                                    sub2 = arg2
                                try_extract_literal(sub2)
                            elif tok2 is regex_sre_constants.BRANCH:
                                can_extract_literal = False
                            elif tok2 in (regex_sre_constants.IN, regex_sre_constants.CATEGORY,
                                          regex_sre_constants.MAX_REPEAT, regex_sre_constants.MIN_REPEAT):
                                can_extract_literal = False
                            else:
                                can_extract_literal = False
                    try_extract_literal(sub)
                    if can_extract_literal and literal_chars:
                        seq = ''.join(literal_chars)
                        lines.append("        # ASSERT_NOT (negative lookahead) - register forbidden literal sequence")
                        lines.append(f"        _register_forbidden_literal({repr(seq)})")
                    else:
                        sub_bid = build_block_for_subpattern(sub)
                        lines.append("        # ASSERT_NOT fallback (complex): use generated forbidden sequence as before")
                        lines.append("        old_stack = list(lookahead_stack)")
                        lines.append("        lookahead_stack.clear()")
                        lines.append("        tmp_out = []")
                        lines.append(f"        lg_assert = _b{sub_bid}(tmp_out)")
                        lines.append("        t_assert = ''.join(tmp_out)")
                        lines.append("        lookahead_stack[:] = old_stack")
                        lines.append("        if %d > 0:" % (dirn))
                        lines.append("            lookahead_stack.append({'forbidden': list(t_assert), 'pos': 0})")
                        lines.append("        else:")
                        lines.append("            total = ''.join(out)")
                        lines.append("            if len(total) < len(t_assert):")
                        lines.append("                pass")
                        lines.append("            else:")
                        lines.append("                if total[-len(t_assert):] == t_assert:")
                        lines.append("                    raise AssertionError('negative lookbehind failed')")
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
                    lines.append("            lookahead_stack.append({'forced': list(t_assert), 'pos': 0})")
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

    # --- generate final function lines ---
    func_lines = []
    func_lines.append(f"def {func_name}():")
    func_lines.append("    groups = {}")
    func_lines.append("    lookahead_stack = []")
    func_lines.append("    # pending_requirements: deque of {'count': int, 'choices': tuple}")
    func_lines.append("    pending_requirements = deque()")
    func_lines.append("    # forbidden_literals is a set of literal sequences we should avoid emitting later")
    func_lines.append("    forbidden_literals = set()")
    func_lines.append("    LFORB_MAX = 0  # maximum length of any forbidden literal (set below if any)")

    func_lines.append("    def _register_requirements(reqs):")
    func_lines.append("        for r in reqs:")
    func_lines.append("            tpl = tuple(r['choices'])")
    func_lines.append("            pending_requirements.append({'count': int(r['count']), 'choices': tpl})")
    
    func_lines.append("    def _register_forbidden_literal(seq):")
    func_lines.append("        forbidden_literals.add(seq)")
    func_lines.append("        nonlocal LFORB_MAX")
    func_lines.append("        LFORB_MAX = max(LFORB_MAX, len(seq))")

    func_lines.append("    def _rand_choice(tpl):")
    func_lines.append("        # fast random choice without constructing a new list")
    func_lines.append("        if len(tpl) == 1:")
    func_lines.append("            return tpl[0]")
    func_lines.append("        return tpl[random.randrange(len(tpl))]")

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
    func_lines.append("                # consume pending requirement if matched")
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
    func_lines.append("                # pick a random char not equal to the forbidden one")
    func_lines.append("                # attempt limited times to avoid pathological infinite loops")
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
    
    func_lines.append("        if pending_requirements:")
    func_lines.append("            r = pending_requirements[0]")
    func_lines.append("            if r['count'] > 0:")
    func_lines.append("                # find any char in choices_tpl that matches r['choices']")
    func_lines.append("                # def _return_ch(choices_tpl,pending_requirements):")
    func_lines.append("                if pending_requirements:")
    func_lines.append("                    # union of all currently-required characters")
    func_lines.append("                    req_choices_union = set()")
    func_lines.append("                    for r in pending_requirements:")
    func_lines.append("                        if r['count'] > 0:")
    func_lines.append("                            req_choices_union.update(r['choices'])")
    func_lines.append("                    valid_candidates = [c for c in choices_tpl if c in req_choices_union]")
    func_lines.append("                else:")
    func_lines.append("                    valid_candidates = []")
    func_lines.append("                if valid_candidates:")
    func_lines.append("                    ch = random.choice(valid_candidates)")
    func_lines.append("                else:")
    func_lines.append("                    ch = _rand_choice(choices_tpl)")
    func_lines.append("                # consume from all requirements that match this char")
    func_lines.append("                to_remove = []")
    func_lines.append("                for idx, r in enumerate(pending_requirements):")
    func_lines.append("                    if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                        r['count'] -= 1")
    func_lines.append("                        if r['count'] <= 0:")
    func_lines.append("                            to_remove.append(idx)")
    func_lines.append("                for idx in reversed(to_remove):")
    func_lines.append("                    del pending_requirements[idx]")
    func_lines.append("                return ch")

    func_lines.append("        # If forbidden_literals exist, prefer characters that don't create them when appended")
    func_lines.append("        if forbidden_literals and LFORB_MAX > 0:")
    func_lines.append("            tail_len = LFORB_MAX - 1")
    func_lines.append("            tail = ''.join(out[-tail_len:]) if tail_len > 0 else ''")
    func_lines.append("            attempts_inner = 5")
    func_lines.append("            while attempts_inner > 0:")
    func_lines.append("                ch = _rand_choice(choices_tpl)")
    func_lines.append("                would = tail + ch")
    func_lines.append("                bad = False")
    func_lines.append("                for forb in forbidden_literals:")
    func_lines.append("                    if would.endswith(forb):")
    func_lines.append("                        bad = True")
    func_lines.append("                        break")
    func_lines.append("                if not bad:")
    func_lines.append("                    # consume requirement if matched")
    func_lines.append("                    for _ in range(len(pending_requirements)):")
    func_lines.append("                        r = pending_requirements[r_i]")
    func_lines.append("                        if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                            r['count'] -= 1")
    func_lines.append("                            if r['count'] <= 0:")
    func_lines.append("                                pending_requirements.popleft()")
    func_lines.append("                            break")
    func_lines.append("                    return ch")
    func_lines.append("                attempts_inner -= 1")
    func_lines.append("            # if we didn't find a safe char, fall back to random pick (may trigger retry upstream)")

    func_lines.append("        # final fallback: random pick")
    func_lines.append("        ch = _rand_choice(choices_tpl)")
    func_lines.append("        for _ in range(len(pending_requirements)):")
    func_lines.append("            r = pending_requirements[0]")
    func_lines.append("            if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                r['count'] -= 1")
    func_lines.append("                if r['count'] <= 0:")
    func_lines.append("                    pending_requirements.popleft()")
    func_lines.append("                break")
    func_lines.append("        return ch")

    # Inject _CHOICES dictionary (tuples) at top of generated function
    func_lines.append("    # Precomputed concrete choice-lists (deduplicated) used by nested blocks")
    func_lines.append("    _CHOICES = {}")
    func_lines.append("    # Also create a frozenset mapping for quick membership tests if useful")
    for name, tpl in choices_list_by_name.items():
        func_lines.append(f"    _CHOICES['{name}'] = ({repr(tpl)})")

    # Append generated block functions
    func_lines.extend(blocks_src)

    func_lines.append(f"    attempts = {max_attempts}")
    func_lines.append("    while True:")
    func_lines.append("        groups = {}")
    func_lines.append("        lookahead_stack = []")
    func_lines.append("        pending_requirements = deque()")
    func_lines.append("        forbidden_literals = set()")
    func_lines.append("        LFORB_MAX = 0")
    func_lines.append("        try:")
    func_lines.append("            out = []")
    func_lines.append(f"            lg = _b{top_block_id}(out)")
    func_lines.append("            groups.update(lg)")
    func_lines.append("            # After generation, ensure pending requirements are satisfied by appending missing chars.")
    func_lines.append("            missing = []")
    func_lines.append("            while pending_requirements:")
    func_lines.append("                r = pending_requirements.popleft()")
    func_lines.append("                while r['count'] > 0:")
    func_lines.append("                    # pick one of the allowed choices for this requirement (no shuffle)")
    func_lines.append("                    missing.append(_rand_choice(r['choices']))")
    func_lines.append("                    r['count'] -= 1")
    func_lines.append("            # Append missing characters while trying to avoid forbidden sequences")
    func_lines.append("            if missing:")
    func_lines.append("                for ch in missing:")
    func_lines.append("                    # try to place ch safely; if not safe after a few tries, fail to retry whole generation")
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
    func_lines.append("            # final scan for forbidden strings (defensive)")
    func_lines.append("            for forb in list(forbidden_literals):")
    func_lines.append("                if forb and forb in s:")
    func_lines.append("                    raise AssertionError('forbidden literal produced')")
    func_lines.append("            return s")
    func_lines.append("        except AssertionError as e:")
    func_lines.append("            attempts -= 1")
    func_lines.append("            if attempts <= 0:")
    func_lines.append("                raise RuntimeError('Failed to generate matching string: ' + str(e))")

    return "\n".join(func_lines)



if __name__ == "__main__":
    patterns = [
        r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
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
        print("=== pattern:", pat, "===amount:", amount)
        import time
        import random
        import re
        start = time.time()
        src = compile_regex_to_function_source(
            pat, flags=0, max_repeat=6, func_name="gen"
        )
        #print(src)
        safe_builtins = {
            "len": len, "range": range, "min": min, "max": max,
            "list": list, "tuple": tuple, "deque": deque, "chr": chr, "ord": ord, "set": set, "map": map,
            "int": int, "AssertionError": AssertionError, "enumerate":enumerate, "reversed":reversed
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
        print("Samples example:", samples[0:2], len(samples))
        for s in samples:
            if not re.fullmatch(pat, s):
                raise AssertionError("Generated string does not match pattern", pat, s)
        end = time.time()
        print("time:", end - start)
        print()

try:
    from . import regex_sre_parse
    from . import regex_sre_constants
except:
    print("running from main")
    import regex_sre_parse
    import regex_sre_constants
import string
import random

DEFAULT_ALPHABET = "".join(chr(i) for i in range(32, 127))

CATEGORY_TO_EXPR = {
    regex_sre_constants.CATEGORY_DIGIT: "string.digits",
    regex_sre_constants.CATEGORY_SPACE: "string.whitespace",
    regex_sre_constants.CATEGORY_WORD: "string.ascii_letters + string.digits + '_'",
}


def _expand_in_child_static(arg, flags, alphabet):
    """
    Static version used at compile-time to compute the concrete choices set for an IN token.
    Mirrors your original _expand_in_child but works on the parsed tokens directly.
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
            # For unsupported tokens inside IN, conservatively include whole alphabet.
            candidates.extend(list(alphabet))
    if negate:
        candidates = [c for c in alphabet if c not in set(candidates)]
    # deduplicate preserving order
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
    """
    Walk a parsed subpattern and collect a list of (count, choices_list) requirements.
    We look for atomic tokens: LITERAL, IN, CATEGORY and use the repeat's minimum
    repetition as the count contribution.
    For MIN_REPEAT/MAX_REPEAT we use the lower bound (lo) since lookaheads impose 'at least' semantics.
    This is intentionally conservative and focused on common lookahead shapes (.*[A-Z]){2}, (?=.*), etc.
    """
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
                    # convert to concrete choices set
                    if arg == regex_sre_constants.CATEGORY_DIGIT:
                        choices = list(string.digits)
                    elif arg == regex_sre_constants.CATEGORY_SPACE:
                        choices = list(string.whitespace)
                    elif arg == regex_sre_constants.CATEGORY_WORD:
                        choices = list(string.ascii_letters + string.digits + "_")
                    else:
                        choices = list(alphabet)
                else:
                    # conservative fallback
                    choices = list(alphabet)
                reqs.append((mult, choices))
            elif tok in (
                regex_sre_constants.MAX_REPEAT,
                regex_sre_constants.MIN_REPEAT,
            ):
                lo, hi, inner = arg
                # use lo as minimum number of occurrences
                inner_mult = mult * max(1, lo)
                walk(inner, inner_mult)
            elif tok is regex_sre_constants.SUBPATTERN:
                if isinstance(arg, tuple):
                    sub = arg[-1]
                else:
                    sub = arg
                walk(sub, mult)
            elif tok is regex_sre_constants.BRANCH:
                # branches: treat each branch conservatively, take minimal requirements across branches.
                # For simplicity, walk all branches and accumulate (this is conservative - may over-constrain)
                _, branches = arg
                for branch in branches:
                    walk(branch, mult)
            else:
                # ignore DOT/ANY/AT and other structural tokens; they do not impose atomic requirements directly
                # If pattern is more complex, we intentionally remain conservative and may not extract requirements.
                pass

    walk(subp, repeat_multiplier)
    # Combine same-choice requirement entries by summing their counts for tidy runtime registration.
    combined = []
    # Use canonical tuple of sorted choices as key
    from collections import defaultdict

    agg = defaultdict(int)
    choice_map = {}
    for cnt, choices in reqs:
        key = tuple(sorted(choices))
        agg[key] += cnt
        choice_map[key] = choices
    for k, cnt in agg.items():
        combined.append({"count": cnt, "choices": choice_map[k]})
    return combined


def compile_regex_to_function_source(
    pattern: str,
    flags: int = 0,
    max_repeat: int = 65535,
    alphabet: str = None,
    func_name: str = "gen",
    max_attempts: int = 2000,
):
    """
    Compiler that generates a generator function source which implements lookahead
    constraints as 'requirements' (counts of characters from choice sets to appear later).
    This variant precomputes concrete choice-lists once in the top block (as _CHOICES)
    and reuses them from nested blocks to avoid rebuilding identical lists repeatedly.
    """
    if alphabet is None:
        alphabet = DEFAULT_ALPHABET

    parsed = list(regex_sre_parse.parse(pattern, flags))

    # --- quick check: pattern starts with a positive lookbehind that requires preceding chars
    if parsed:
        first_tok, first_arg = parsed[0]
        if first_tok is regex_sre_constants.ASSERT:
            dirn, sub = first_arg  # dirn < 0 -> lookbehind; dirn > 0 -> lookahead
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
                        f"Pattern: '{pattern}' :starts with a positive lookbehind of fixed width {w}. "
                        "Such a pattern can never match the entire string with re.fullmatch() "
                        "(there is no text before position 0). Rewrite the pattern so the lookbehind's "
                        "context is part of the matched text."
                    )

    blocks_src = []
    block_counter = 0

    # --- NEW: a pool to deduplicate concrete choice-lists (compiled-time)
    choices_pool = {}  # key: tuple(choices) -> name (e.g. "c0")
    choices_list_by_name = {}  # name -> list(chars)
    choices_counter = 0

    def get_choice_name_for(choices):
        """Return a deduplicated choice-name for this concrete choices list."""
        nonlocal choices_counter
        key = tuple(choices)
        if key in choices_pool:
            return choices_pool[key]
        name = f"c{choices_counter}"
        choices_counter += 1
        choices_pool[key] = name
        choices_list_by_name[name] = list(choices)
        return name

    def new_block_id():
        nonlocal block_counter
        i = block_counter
        block_counter += 1
        return i

    regex_module_name = "regex_sre_parse"

    def build_block_for_subpattern(subp) -> int:
        bid = new_block_id()
        lines = []
        lines.append(f"    def _b{bid}():")
        lines.append("        local_groups = {}")
        lines.append("        parts = []")
        for token, arg in subp:
            if token is regex_sre_constants.LITERAL:
                ch = chr(arg)
                # small concrete list with one element -> dedupe it too
                cname = get_choice_name_for([ch])
                lines.append(
                    f"        parts.append(_choose_from_list(_CHOICES['{cname}']))"
                )
            elif token is regex_sre_constants.NOT_LITERAL:
                forbidden = chr(arg)
                choices = [c for c in alphabet if c != forbidden]
                if not choices:
                    choices = list(alphabet)
                cname = get_choice_name_for(choices)
                lines.append(
                    f"        parts.append(_choose_from_list(_CHOICES['{cname}']))"
                )
            elif token is regex_sre_constants.ANY:
                choices = [c for c in alphabet if c != "\n"]
                cname = get_choice_name_for(choices)
                lines.append(
                    f"        parts.append(_choose_from_list(_CHOICES['{cname}']))"
                )
            elif token is regex_sre_constants.IN:
                choices = _expand_in_child_static(arg, flags, alphabet)
                cname = get_choice_name_for(choices)
                lines.append(
                    f"        parts.append(_choose_from_list(_CHOICES['{cname}']))"
                )
            elif token is regex_sre_constants.CATEGORY:
                if arg in CATEGORY_TO_EXPR:
                    expr = CATEGORY_TO_EXPR[arg]
                    lines.append(f"        choices = list({expr})")
                    lines.append("        parts.append(_choose_from_list(choices))")
                else:
                    if arg == regex_sre_constants.CATEGORY_NOT_DIGIT:
                        choices = "".join(
                            ch for ch in alphabet if ch not in string.digits
                        )
                    elif arg == regex_sre_constants.CATEGORY_NOT_SPACE:
                        choices = "".join(
                            ch for ch in alphabet if ch not in string.whitespace
                        )
                    elif arg == regex_sre_constants.CATEGORY_NOT_WORD:
                        word_chars = set(string.ascii_letters + string.digits + "_")
                        choices = "".join(ch for ch in alphabet if ch not in word_chars)
                    else:
                        choices = alphabet
                    # These are concrete at compile-time; dedupe them too
                    cname = get_choice_name_for(list(choices))
                    lines.append(
                        f"        parts.append(_choose_from_list(_CHOICES['{cname}']))"
                    )
            elif token is regex_sre_constants.BRANCH:
                _, branches = arg
                n = len(branches)
                branch_ids = [build_block_for_subpattern(branch) for branch in branches]
                lines.append(f"        i = random.randint(0, {n - 1})")
                for idx, bid2 in enumerate(branch_ids):
                    prefix = "        if" if idx == 0 else "        elif"
                    lines.append(f"{prefix} i == {idx}:")
                    lines.append(f"            t, lg = _b{bid2}()")
                    lines.append("            local_groups.update(lg)")
                    lines.append("            groups.update(lg)")
                    lines.append("            parts.append(t)")
                lines.append("        else:")
                lines.append(f"            t, lg = _b{branch_ids[0]}()")
                lines.append("            local_groups.update(lg)")
                lines.append("            groups.update(lg)")
                lines.append("            parts.append(t)")
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
                lines.append(f"        t, lg = _b{sub_bid}()")
                lines.append("        local_groups.update(lg)")
                lines.append("        groups.update(lg)")
                if groupnum and groupnum > 0:
                    lines.append(f"        local_groups[{groupnum}] = t")
                    lines.append(f"        groups[{groupnum}] = t")
                lines.append("        parts.append(t)")
            elif token in (
                regex_sre_constants.MAX_REPEAT,
                regex_sre_constants.MIN_REPEAT,
            ):
                lo, hi, sub = arg
                sub_bid = build_block_for_subpattern(sub)
                if hi == regex_sre_constants.MAXREPEAT or hi is None:
                    hi_eff = min(lo + max_repeat, max_repeat)
                else:
                    hi_eff = min(hi, lo + max_repeat)
                if hi_eff < lo:
                    raise ValueError(
                        f"Maximum value {hi_eff} is smaller than minimum value {lo}"
                    )
                lines.append(f"        count = random.randint({lo}, {hi_eff})")
                lines.append("        for _ in range(count):")
                lines.append(f"            t, lg = _b{sub_bid}()")
                lines.append("            local_groups.update(lg)")
                lines.append("            groups.update(lg)")
                lines.append("            parts.append(t)")
            elif token is regex_sre_constants.GROUPREF:
                groupnum = arg
                lines.append(f"        parts.append(groups.get({groupnum}, ''))")
            elif token is regex_sre_constants.GROUPREF_IGNORE:
                groupnum = arg
                lines.append(f"        parts.append(groups.get({groupnum}, ''))")
            elif token is regex_sre_constants.AT:
                continue
            elif token is regex_sre_constants.RANGE:
                a, b = arg
                choices = [chr(c) for c in range(a, b + 1)]
                cname = get_choice_name_for(choices)
                lines.append(
                    f"        parts.append(_choose_from_list(_CHOICES['{cname}']))"
                )
            elif token in (regex_sre_constants.ASSERT, regex_sre_constants.ASSERT_NOT):
                # (unchanged ASSERT/ASSERT_NOT behavior) ...
                # For brevity in this snippet, reuse your existing ASSERT handling code as-is.
                # (Copy the same ASSERT handling branches you had originally.)
                # INSERT your ASSERT/ASSERT_NOT handling lines here exactly like before.
                # --- BEGIN (unchanged ASSERT/ASSERT_NOT handling) ---
                dirn, sub = arg
                if token is regex_sre_constants.ASSERT and dirn > 0:
                    reqs = _collect_atomic_requirements(sub, flags, alphabet)
                    lines.append(
                        "        # ASSERT (positive lookahead) - register minimal atomic requirements"
                    )
                    if reqs:
                        lines.append(f"        _register_requirements({repr(reqs)})")
                    else:
                        sub_bid = build_block_for_subpattern(sub)
                        lines.append(
                            "        # fallback: couldn't collect atomic requirements; use generated assertion string"
                        )
                        lines.append("        old_stack = list(lookahead_stack)")
                        lines.append("        lookahead_stack.clear()")
                        lines.append(f"        t_assert, lg_assert = _b{sub_bid}()")
                        lines.append("        lookahead_stack[:] = old_stack")
                        lines.append("        if %d > 0:" % (dirn))
                        lines.append(
                            "            lookahead_stack.append({'forced': list(t_assert), 'pos': 0})"
                        )
                        lines.append("        else:")
                        lines.append("            total = ''.join(parts)")
                        lines.append("            if len(total) < len(t_assert):")
                        lines.append("                if not t_assert.endswith(total):")
                        lines.append(
                            "                    raise AssertionError('lookbehind failed - mismatch')"
                        )
                        lines.append(
                            "                missing = len(t_assert) - len(total)"
                        )
                        lines.append(
                            "                parts.insert(0, t_assert[:missing])"
                        )
                        lines.append("            else:")
                        lines.append(
                            "                if total[-len(t_assert):] != t_assert:"
                        )
                        lines.append(
                            "                    raise AssertionError('lookbehind assertion failed')"
                        )
                elif token is regex_sre_constants.ASSERT and dirn < 0:
                    sub_bid = build_block_for_subpattern(sub)
                    lines.append(
                        "        # ASSERT (lookbehind) - fallback to previous behavior"
                    )
                    lines.append("        old_stack = list(lookahead_stack)")
                    lines.append("        lookahead_stack.clear()")
                    lines.append(f"        t_assert, lg_assert = _b{sub_bid}()")
                    lines.append("        lookahead_stack[:] = old_stack")
                    lines.append("        if %d > 0:" % (dirn))
                    lines.append(
                        "            lookahead_stack.append({'forced': list(t_assert), 'pos': 0})"
                    )
                    lines.append("        else:")
                    lines.append("            total = ''.join(parts)")
                    lines.append("            if len(total) < len(t_assert):")
                    lines.append("                if not t_assert.endswith(total):")
                    lines.append(
                        "                    raise AssertionError('lookbehind failed - mismatch')"
                    )
                    lines.append("                missing = len(t_assert) - len(total)")
                    lines.append("                parts.insert(0, t_assert[:missing])")
                    lines.append("            else:")
                    lines.append(
                        "                if total[-len(t_assert):] != t_assert:"
                    )
                    lines.append(
                        "                    raise AssertionError('lookbehind assertion failed')"
                    )
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
                            elif tok2 in (
                                regex_sre_constants.IN,
                                regex_sre_constants.CATEGORY,
                                regex_sre_constants.MAX_REPEAT,
                                regex_sre_constants.MIN_REPEAT,
                            ):
                                can_extract_literal = False
                            else:
                                can_extract_literal = False

                    try_extract_literal(sub)
                    if can_extract_literal and literal_chars:
                        seq = "".join(literal_chars)
                        lines.append(
                            "        # ASSERT_NOT (negative lookahead) - register forbidden literal sequence"
                        )
                        lines.append(
                            f"        _register_forbidden_literal({repr(seq)})"
                        )
                    else:
                        sub_bid = build_block_for_subpattern(sub)
                        lines.append(
                            "        # ASSERT_NOT fallback (complex): use generated forbidden sequence as before"
                        )
                        lines.append("        old_stack = list(lookahead_stack)")
                        lines.append("        lookahead_stack.clear()")
                        lines.append(f"        t_assert, lg_assert = _b{sub_bid}()")
                        lines.append("        lookahead_stack[:] = old_stack")
                        lines.append("        if %d > 0:" % (dirn))
                        lines.append(
                            "            lookahead_stack.append({'forbidden': list(t_assert), 'pos': 0})"
                        )
                        lines.append("        else:")
                        lines.append("            total = ''.join(parts)")
                        lines.append("            if len(total) < len(t_assert):")
                        lines.append("                pass")
                        lines.append("            else:")
                        lines.append(
                            "                if total[-len(t_assert):] == t_assert:"
                        )
                        lines.append(
                            "                    raise AssertionError('negative lookbehind failed')"
                        )
                else:
                    sub_bid = build_block_for_subpattern(sub)
                    lines.append("        # ASSERT unexpected form - fallback")
                    lines.append("        old_stack = list(lookahead_stack)")
                    lines.append("        lookahead_stack.clear()")
                    lines.append(f"        t_assert, lg_assert = _b{sub_bid}()")
                    lines.append("        lookahead_stack[:] = old_stack")
                    lines.append("        if %d > 0:" % (dirn))
                    lines.append(
                        "            lookahead_stack.append({'forced': list(t_assert), 'pos': 0})"
                    )
                    lines.append("        else:")
                    lines.append("            total = ''.join(parts)")
                    lines.append("            if len(total) < len(t_assert):")
                    lines.append("                if not t_assert.endswith(total):")
                    lines.append(
                        "                    raise AssertionError('lookbehind failed - mismatch')"
                    )
                    lines.append("                missing = len(t_assert) - len(total)")
                    lines.append("                parts.insert(0, t_assert[:missing])")
                    lines.append("            else:")
                    lines.append(
                        "                if total[-len(t_assert):] != t_assert:"
                    )
                    lines.append(
                        "                    raise AssertionError('lookbehind assertion failed')"
                    )
                # --- END ASSERT/ASSERT_NOT handling ---
            else:
                raise NotImplementedError(
                    f"Token not implemented in compiler: {token} ({arg})"
                )
        lines.append("        return (''.join(parts), local_groups)")
        block_src = "\n".join(lines)
        blocks_src.append(block_src)
        return bid

    top_block_id = build_block_for_subpattern(parsed)

    func_lines = []
    func_lines.append(f"def {func_name}():")
    func_lines.append("    groups = {}")
    func_lines.append("    lookahead_stack = []")
    func_lines.append(
        "    # pending_requirements holds dicts: {'count': int, 'choices': [chars]}"
    )
    func_lines.append("    pending_requirements = []")
    func_lines.append(
        "    # forbidden_literals is a set of literal sequences we should avoid emitting later"
    )
    func_lines.append("    forbidden_literals = set()")
    func_lines.append("    def _register_requirements(reqs):")
    func_lines.append(
        "        # expect reqs as list of {'count': int, 'choices': [chars]}"
    )
    func_lines.append("        for r in reqs:")
    func_lines.append("            # shallow copy to keep local lists independent")
    func_lines.append(
        "            pending_requirements.append({'count': int(r['count']), 'choices': list(r['choices'])})"
    )
    func_lines.append("    def _register_forbidden_literal(seq):")
    func_lines.append("        forbidden_literals.add(seq)")
    func_lines.append("    def _choose_from_list(choices):")
    func_lines.append(
        "        # Prefer to consume a pending requirement if possible (FIFO)."
    )
    func_lines.append(
        "        # If the top of lookahead_stack contains a forced/forbidden sequence (legacy fallback), honor it first."
    )
    func_lines.append("        if lookahead_stack:")
    func_lines.append("            top = lookahead_stack[-1]")
    func_lines.append("            pos = top.get('pos', 0)")
    func_lines.append(
        "            if 'forced' in top and top['forced'] is not None and pos < len(top['forced']):"
    )
    func_lines.append("                required = top['forced'][pos]")
    func_lines.append("                if required not in choices:")
    func_lines.append(
        "                    raise AssertionError('lookahead forced char not available')"
    )
    func_lines.append("                ch = required")
    func_lines.append("                top['pos'] = pos + 1")
    func_lines.append("                if top['pos'] >= len(top['forced']):")
    func_lines.append("                    lookahead_stack.pop()")
    func_lines.append(
        "                # consume any pending requirement matched by this char"
    )
    func_lines.append("                for r in pending_requirements:")
    func_lines.append("                    if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                        r['count'] -= 1")
    func_lines.append("                        break")
    func_lines.append("                return ch")
    func_lines.append(
        "            if 'forbidden' in top and top['forbidden'] is not None and pos < len(top['forbidden']):"
    )
    func_lines.append("                forb = top['forbidden'][pos]")
    func_lines.append("                filtered = [c for c in choices if c != forb]")
    func_lines.append("                if not filtered:")
    func_lines.append(
        "                    raise AssertionError('lookahead forbidden removed all choices')"
    )
    func_lines.append("                ch = random.choice(filtered)")
    func_lines.append("                top['pos'] = pos + 1")
    func_lines.append("                if top['pos'] >= len(top['forbidden']):")
    func_lines.append("                    lookahead_stack.pop()")
    func_lines.append(
        "                # consume any pending requirement matched by this char"
    )
    func_lines.append("                for r in pending_requirements:")
    func_lines.append("                    if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                        r['count'] -= 1")
    func_lines.append("                        break")
    func_lines.append("                return ch")
    func_lines.append(
        "        # Try to satisfy the earliest pending requirement if possible"
    )
    func_lines.append("        if pending_requirements:")
    func_lines.append("            for r in pending_requirements:")
    func_lines.append("                if r['count'] > 0:")
    func_lines.append(
        "                    # if any choice intersects r['choices'], pick from intersection"
    )
    func_lines.append(
        "                    inter = [c for c in choices if c in r['choices']]"
    )
    func_lines.append("                    if inter:")
    func_lines.append("                        ch = random.choice(inter)")
    func_lines.append("                        r['count'] -= 1")
    func_lines.append("                        return ch")
    func_lines.append(
        "                    # otherwise, fallback to random choice; we will append missing required chars later"
    )
    func_lines.append("                    break")
    func_lines.append("        if not choices:")
    func_lines.append("            raise AssertionError('no choices available')")
    func_lines.append("        ch = random.choice(choices)")
    func_lines.append("        # maybe this ch satisfies some pending requirement")
    func_lines.append("        for r in pending_requirements:")
    func_lines.append("            if r['count'] > 0 and ch in r['choices']:")
    func_lines.append("                r['count'] -= 1")
    func_lines.append("                break")
    func_lines.append("        return ch")

    # --- NEW: inject _CHOICES dict with all deduplicated concrete lists.
    func_lines.append(
        "    # Precomputed concrete choice-lists (deduplicated) used by nested blocks"
    )
    func_lines.append("    _CHOICES = {}")
    for name, lst in choices_list_by_name.items():
        # safe literal representation of list
        items_repr = ", ".join(repr(c) for c in lst)
        func_lines.append(f"    _CHOICES['{name}'] = [{items_repr}]")

    # append all generated block functions (which reference _CHOICES['cN'] for concrete lists)
    func_lines.extend(blocks_src)

    func_lines.append(f"    attempts = {(max_attempts)}")
    func_lines.append("    while True:")
    func_lines.append("        try:")
    func_lines.append(f"            t, lg = _b{top_block_id}()")
    func_lines.append("            groups.update(lg)")
    func_lines.append(
        "            # After generation, ensure pending requirements are satisfied by appending missing chars."
    )
    func_lines.append("            missing = []")
    func_lines.append("            for r in pending_requirements:")
    func_lines.append("                while r['count'] > 0:")
    func_lines.append(
        "                    # pick one of the allowed choices for this requirement"
    )
    func_lines.append("                    missing.append(random.choice(r['choices']))")
    func_lines.append("                    r['count'] -= 1")
    func_lines.append(
        "            # Try to avoid forbidden literal sequences by inserting suffix carefully."
    )
    func_lines.append("            s = t")
    func_lines.append("            if missing:")
    func_lines.append(
        "                # naive strategy: append the missing characters in random order."
    )
    func_lines.append("                random.shuffle(missing)")
    func_lines.append("                s = s + ''.join(missing)")
    func_lines.append(
        "            # If any registered forbidden literal sequence is now present, fail this attempt to let a retry happen"
    )
    func_lines.append("            for forb in list(forbidden_literals):")
    func_lines.append("                if forb and forb in s:")
    func_lines.append(
        "                    raise AssertionError('forbidden literal produced') "
    )
    func_lines.append("            return s")
    func_lines.append("        except AssertionError as e:")
    func_lines.append("            attempts -= 1")
    func_lines.append("            if attempts <= 0:")
    func_lines.append(
        "                raise RuntimeError('Failed to generate matching string: ' + str(e))"
    )
    return "\n".join(func_lines)


# ------------------------------
# Example usage and testing
# ------------------------------


if __name__ == "__main__":
    patterns = [
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

    amount = 100_000
    for pat in patterns:
        print("=== pattern:", pat, "===amount:", amount)
        import time
        import random
        import re

        start = time.time()
        src = compile_regex_to_function_source(
            pat, flags=0, max_repeat=6, func_name="gen"
        )
        # print(src)
        # execute generated source to get a real function
        safe_builtins = {
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "list": list,
            "chr": chr,
            "ord": ord,
            "set": set,
            "map": map,
            "int": int,
        }
        env = {
            "__builtins__": safe_builtins,  # dissallow anything except random and string
            "random": random,
            "string": string,
        }
        exec(src, env)
        gen = env["gen"]
        samples = [gen() for _ in range(amount)]
        print("=" * 30)
        print("Samples:", samples[0:2], len(samples))
        for s in samples:
            if not re.search(pat, s):
                raise AssertionError("Generated string does not match pattern", pat, s)
        end = time.time()
        print(end - start)
        print()

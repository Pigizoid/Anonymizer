try:
    from . import regex_sre_parse
    from . import regex_sre_constants
except:
    print("running from main")
    import regex_sre_parse
    import regex_sre_constants
import string

DEFAULT_ALPHABET = "".join(chr(i) for i in range(32, 127))

CATEGORY_TO_EXPR = {
    regex_sre_constants.CATEGORY_DIGIT: "string.digits",
    regex_sre_constants.CATEGORY_SPACE: "string.whitespace",
    regex_sre_constants.CATEGORY_WORD: "string.ascii_letters + string.digits + '_'",
}
# inverse categories (not_*) will be computed from DEFAULT_ALPHABET at compile-time


def _expand_in_child(arg, flags, alphabet):
    """Return list of candidate chars from an IN token's children (compile-time)."""
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
            if val in CATEGORY_TO_EXPR:
                # compile-time expand using DEFAULT_ALPHABET
                if val == regex_sre_constants.CATEGORY_DIGIT:
                    candidates.extend(list(string.digits))
                elif val == regex_sre_constants.CATEGORY_SPACE:
                    candidates.extend(list(string.whitespace))
                elif val == regex_sre_constants.CATEGORY_WORD:
                    candidates.extend(list(string.ascii_letters + string.digits + "_"))
                else:
                    candidates.extend(list(alphabet))
            else:
                # unknown category: fall back to alphabet
                candidates.extend(list(alphabet))
        elif tok is regex_sre_constants.IN:
            candidates.extend(_expand_in_child(val, flags, alphabet))
        else:
            raise NotImplementedError(f"IN child token not supported: {tok} ({val})")
    if negate:
        candidates = [c for c in alphabet if c not in set(candidates)]
    # dedupe while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    if not out:
        out = list(alphabet)
    return out


def compile_regex_to_function_source(
    pattern: str,
    flags: int = 0,
    max_repeat: int = 10,
    alphabet: str = None,
    func_name: str = "gen",
):
    """
    Compile a Python `re` regex into a Python source string that defines
    a function `func_name()` returning a random string matching the regex.
    """
    if alphabet is None:
        alphabet = DEFAULT_ALPHABET

    parsed = list(regex_sre_parse.parse(pattern, flags))

    blocks_src = []  # list of block function source strings
    block_counter = 0

    def new_block_id():
        nonlocal block_counter
        i = block_counter
        block_counter += 1
        return i

    # map from tuple(subpattern) to block id to reuse blocks if identical nodes encountered (optional)
    # We won't attempt deep reuse here for simplicity.

    def build_block_for_subpattern(subp) -> int:
        """
        Create a helper block function for the subpattern `subp` (list of tokens).
        Returns block id (integer). The block will be named _b{ID} and will
        return (text, local_groups).
        """
        bid = new_block_id()
        lines = []
        lines.append(f"    def _b{bid}():")
        lines.append("        local_groups = {}")
        lines.append("        parts = []")

        # walk tokens in sequence
        for token, arg in subp:
            if token is regex_sre_constants.LITERAL:
                ch = chr(arg)
                lines.append(f"        parts.append({repr(ch)})")
            elif token is regex_sre_constants.NOT_LITERAL:
                forbidden = chr(arg)
                choices = [c for c in alphabet if c != forbidden]
                if not choices:
                    choices = list(alphabet)
                choices_repr = ", ".join(repr(c) for c in choices)
                lines.append(f"        parts.append(random.choice([{choices_repr}]))")
            elif token is regex_sre_constants.ANY:
                # dot: exclude newline unless DOTALL set at compile call time (we don't have access to that here,
                # but re.DOTALL passed into parse influences AST tokens - still treat newline as excluded unless alphabet contains it)
                # Simpler: use alphabet but ensure '\n' excluded unless present in alphabet
                choices = [c for c in alphabet if c != "\n"]
                choices_repr = ", ".join(repr(c) for c in choices)
                lines.append(f"        parts.append(random.choice([{choices_repr}]))")
            elif token is regex_sre_constants.IN:
                choices = _expand_in_child(arg, flags, alphabet)
                choices_repr = ", ".join(repr(c) for c in choices)
                lines.append(f"        parts.append(random.choice([{choices_repr}]))")
            elif token is regex_sre_constants.CATEGORY:
                if arg in CATEGORY_TO_EXPR:
                    lines.append(
                        f"        parts.append(random.choice(list({CATEGORY_TO_EXPR[arg]})))"
                    )
                else:
                    # not_word, not_digit, not_space -> compile-time complement of default alphabet
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
                    choices_repr = ", ".join(repr(c) for c in choices)
                    lines.append(
                        f"        parts.append(random.choice([{choices_repr}]))"
                    )
            elif token is regex_sre_constants.BRANCH:
                # arg = (None, [list_of_subpatterns])
                _, branches = arg
                n = len(branches)
                # create blocks for each branch
                branch_ids = [build_block_for_subpattern(branch) for branch in branches]
                # compile runtime choice and dispatch
                lines.append(f"        i = random.randint(0, {n - 1})")
                for idx, bid2 in enumerate(branch_ids):
                    prefix = "        if" if idx == 0 else "        elif"
                    lines.append(f"{prefix} i == {idx}:")
                    lines.append(f"            t, lg = _b{bid2}()")
                    lines.append("            local_groups.update(lg)")
                    lines.append("            parts.append(t)")
                # safety fallback (shouldn't be needed)
                lines.append("        else:")
                lines.append(f"            t, lg = _b{branch_ids[0]}()")
                lines.append("            local_groups.update(lg)")
                lines.append("            parts.append(t)")
            elif token is regex_sre_constants.SUBPATTERN:
                # arg can be (groupnum, add_flags, del_flags, sublist) or (groupnum, sublist)
                if isinstance(arg, tuple):
                    if len(arg) == 4:
                        groupnum, _, _, sub = arg
                    elif len(arg) == 2:
                        groupnum, sub = arg
                    else:
                        # fallback: expect groupnum then sub
                        groupnum = arg[0] if arg else None
                        sub = arg[-1] if arg else []
                else:
                    groupnum = None
                    sub = arg
                sub_bid = build_block_for_subpattern(sub)
                lines.append(f"        t, lg = _b{sub_bid}()")
                lines.append("        local_groups.update(lg)")
                if groupnum and groupnum > 0:
                    lines.append(f"        local_groups[{groupnum}] = t")
                lines.append("        parts.append(t)")
            elif token in (
                regex_sre_constants.MAX_REPEAT,
                regex_sre_constants.MIN_REPEAT,
            ):
                lo, hi, sub = arg
                sub_bid = build_block_for_subpattern(sub)
                # handle unlimited
                if hi == regex_sre_constants.MAXREPEAT or hi is None:
                    hi_eff = max_repeat
                else:
                    hi_eff = min(hi, max_repeat)
                # runtime choose count between lo and hi_eff
                lines.append(f"        count = random.randint({lo}, {hi_eff})")
                lines.append("        for _ in range(count):")
                lines.append(f"            t, lg = _b{sub_bid}()")
                lines.append("            local_groups.update(lg)")
                lines.append("            parts.append(t)")
            elif token is regex_sre_constants.GROUPREF:
                groupnum = arg
                # read from outer groups (outer variable 'groups' exists in the gen() function)
                lines.append(f"        parts.append(groups.get({groupnum}, ''))")
            elif token is regex_sre_constants.GROUPREF_IGNORE:
                groupnum = arg
                lines.append(f"        parts.append(groups.get({groupnum}, ''))")
            elif token is regex_sre_constants.AT:
                # anchors - ignore
                continue
            elif token is regex_sre_constants.RANGE:
                a, b = arg
                lines.append(f"        parts.append(chr(random.randint({a}, {b})))")
            elif token in (regex_sre_constants.ASSERT, regex_sre_constants.ASSERT_NOT):
                raise NotImplementedError(
                    "Lookaround assertions (ASSERT/ASSERT_NOT) are not supported by this compiler."
                )
            else:
                raise NotImplementedError(
                    f"Token not implemented in compiler: {token} ({arg})"
                )

        lines.append("        return (''.join(parts), local_groups)")
        block_src = "\n".join(lines)
        blocks_src.append(block_src)
        return bid

    # Build only one top-level block for the whole parsed pattern
    top_block_id = build_block_for_subpattern(parsed)

    # assemble final function source
    func_lines = []
    func_lines.append(f"def {func_name}():")
    func_lines.append("    groups = {}")
    func_lines.append("    # helper block functions")
    # insert all block function defs
    func_lines.extend(blocks_src)
    func_lines.append("    # run top block")
    func_lines.append(f"    t, lg = _b{top_block_id}()")
    func_lines.append("    groups.update(lg)")
    func_lines.append("    return t")

    return "\n".join(func_lines)


# ------------------------------
# Example usage and testing
# ------------------------------


if __name__ == "__main__":
    patterns = [
        r"^\d{3}(-\d{6})?$",
    ]

    for pat in patterns:
        print("=== pattern:", pat)
        import time

        start = time.time()
        src = compile_regex_to_function_source(
            pat, flags=0, max_repeat=6, func_name="gen"
        )
        print(src)
        # execute generated source to get a real function
        env = {}
        exec(src, env)
        gen = env["gen"]
        samples = [gen() for _ in range(1_250_000)]
        print("Samples:", samples[0], len(samples))
        end = time.time()
        print(end - start)
        print()

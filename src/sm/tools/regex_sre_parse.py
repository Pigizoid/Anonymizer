#
# Secret Labs' Regular Expression Engine (modernized)
#
# convert re-style regular expression to sre pattern
#
# Copyright (c) 1998-2001 by Secret Labs AB.  All rights reserved.
# :
# See the sre.py file for information on usage and redistribution.
#

"""Internal support module for sre (modernized for Python 3)"""

# XXX: show string offset and offending character for all errors

import sys
from typing import Optional, Tuple

from .regex_sre_constants import (
    LITERAL,
    AT,
    AT_BEGINNING_STRING,
    AT_BOUNDARY,
    AT_NON_BOUNDARY,
    IN,
    CATEGORY,
    CATEGORY_DIGIT,
    CATEGORY_NOT_DIGIT,
    CATEGORY_SPACE,
    CATEGORY_NOT_SPACE,
    CATEGORY_WORD,
    CATEGORY_NOT_WORD,
    AT_END_STRING,
    SRE_FLAG_IGNORECASE,
    SRE_FLAG_LOCALE,
    SRE_FLAG_MULTILINE,
    SRE_FLAG_DOTALL,
    SRE_FLAG_VERBOSE,
    SRE_FLAG_TEMPLATE,
    SRE_FLAG_UNICODE,
    ANY,
    RANGE,
    NOT_LITERAL,
    MIN_REPEAT,
    MAX_REPEAT,
    BRANCH,
    CALL,
    SUBPATTERN,
    MAXREPEAT,
    SUCCESS,
    GROUPREF,
    GROUPREF_EXISTS,
    NEGATE,
    ASSERT,
    ASSERT_NOT,
    AT_BEGINNING,
    AT_END,
    SRE_FLAG_DEBUG,
    MARK,
)  # keep existing opcode constants

SPECIAL_CHARS = ".\\[{()*+?^$|"
REPEAT_CHARS = "*+?{"

DIGITS = set("0123456789")
OCTDIGITS = set("01234567")
HEXDIGITS = set("0123456789abcdefABCDEF")
WHITESPACE = set(" \t\n\r\v\f")

ESCAPES = {
    r"\a": (LITERAL, ord("\a")),
    r"\b": (LITERAL, ord("\b")),
    r"\f": (LITERAL, ord("\f")),
    r"\n": (LITERAL, ord("\n")),
    r"\r": (LITERAL, ord("\r")),
    r"\t": (LITERAL, ord("\t")),
    r"\v": (LITERAL, ord("\v")),
    r"\\": (LITERAL, ord("\\")),
}

CATEGORIES = {
    r"\A": (AT, AT_BEGINNING_STRING),  # start of string
    r"\b": (AT, AT_BOUNDARY),
    r"\B": (AT, AT_NON_BOUNDARY),
    r"\d": (IN, [(CATEGORY, CATEGORY_DIGIT)]),
    r"\D": (IN, [(CATEGORY, CATEGORY_NOT_DIGIT)]),
    r"\s": (IN, [(CATEGORY, CATEGORY_SPACE)]),
    r"\S": (IN, [(CATEGORY, CATEGORY_NOT_SPACE)]),
    r"\w": (IN, [(CATEGORY, CATEGORY_WORD)]),
    r"\W": (IN, [(CATEGORY, CATEGORY_NOT_WORD)]),
    r"\Z": (AT, AT_END_STRING),  # end of string
}

FLAGS = {
    # standard flags
    "i": SRE_FLAG_IGNORECASE,
    "L": SRE_FLAG_LOCALE,
    "m": SRE_FLAG_MULTILINE,
    "s": SRE_FLAG_DOTALL,
    "x": SRE_FLAG_VERBOSE,
    # extensions
    "t": SRE_FLAG_TEMPLATE,
    "u": SRE_FLAG_UNICODE,
}


class Pattern:
    # master pattern object. keeps track of global attributes
    def __init__(self):
        self.flags = 0
        self.open = []
        self.groups = 1
        self.groupdict = {}

    def opengroup(self, name: Optional[str] = None) -> int:
        gid = self.groups
        self.groups = gid + 1
        if name is not None:
            ogid = self.groupdict.get(name, None)
            if ogid is not None:
                raise Exception(
                    f"redefinition of group name {repr(name)} as group {gid}\nwas group {ogid}"
                )
            self.groupdict[name] = gid
        self.open.append(gid)
        return gid

    def closegroup(self, gid: int) -> None:
        try:
            self.open.remove(gid)
        except ValueError:
            # closing non-open group: keep previous semantics (no-op or error?)
            raise Exception(f"attempt to close non-open group {gid}")

    def checkgroup(self, gid: int) -> bool:
        return gid < self.groups and gid not in self.open


class SubPattern:
    # a subpattern, in intermediate form
    def __init__(self, pattern: Pattern, data=None):
        self.pattern = pattern
        self.data = [] if data is None else data
        self.width: Optional[Tuple[int, int]] = None

    def __repr__(self):
        return repr(self.data)

    def __len__(self):
        return len(self.data)

    def __delitem__(self, index):
        del self.data[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SubPattern(self.pattern, self.data[index])
        return self.data[index]

    def __setitem__(self, index, code):
        self.data[index] = code

    def insert(self, index, code):
        self.data.insert(index, code)

    def append(self, code):
        self.data.append(code)

    def getwidth(self) -> Tuple[int, int]:
        # determine the width (min, max) for this subpattern
        if self.width is not None:
            return self.width

        lo = hi = 0
        UNITCODES = (ANY, RANGE, IN, LITERAL, NOT_LITERAL, CATEGORY)
        REPEATCODES = (MIN_REPEAT, MAX_REPEAT)

        for op, av in self.data:
            if op == BRANCH:
                i = sys.maxsize
                j = 0
                for sub in av[1]:
                    lv, hv = sub.getwidth()
                    i = min(i, lv)
                    j = max(j, hv)
                lo += i
                hi += j
            elif op == CALL:
                i, j = av.getwidth()
                lo += i
                hi += j
            elif op == SUBPATTERN:
                # av is (group, subpattern)
                _, sp = av
                i, j = sp.getwidth()
                lo += i
                hi += j
            elif op in REPEATCODES:
                minrep, maxrep, rep_sub = av
                i, j = rep_sub.getwidth()
                lo += i * minrep
                # if j could be infinite (MAXREPEAT), cap at sys.maxsize
                if maxrep == MAXREPEAT:
                    hi = hi + (j * sys.maxsize if j != MAXREPEAT else sys.maxsize)
                else:
                    hi += j * maxrep
            elif op in UNITCODES:
                lo += 1
                hi += 1
            elif op == SUCCESS:
                break

        # guard against overflow
        lo = int(min(lo, sys.maxsize))
        hi = int(min(hi, sys.maxsize))
        self.width = lo, hi
        return self.width


class Tokenizer:
    def __init__(self, string: str):
        self.string = string
        self.index = 0
        self.peek: Optional[str] = None
        self._advance()

    def _advance(self) -> None:
        """Advance and update self.peek based on current index."""
        if self.index >= len(self.string):
            self.peek = None
            return
        ch = self.string[self.index]
        if ch == "\\":
            # escape: combine next char if present
            if self.index + 1 >= len(self.string):
                raise Exception("bogus escape (end of line)")
            ch = ch + self.string[self.index + 1]
        # advance index by length of char sequence
        self.index += len(ch)
        self.peek = ch

    def match(self, char: str, skip: bool = True) -> bool:
        """Return True if peek equals char. Advance if skip."""
        if char == self.peek:
            if skip:
                self._advance()
            return True
        return False

    def get(self) -> Optional[str]:
        this = self.peek
        self._advance()
        return this

    def tell(self) -> Tuple[int, Optional[str]]:
        return self.index, self.peek

    def seek(self, index_or_state) -> None:
        """
        Seek to index or state tuple returned by tell().
        Accept either an integer index (then re-advance to set peek) or a (index, peek) tuple.
        """
        if isinstance(index_or_state, tuple):
            idx, pk = index_or_state
            self.index = idx
            self.peek = pk
        else:
            self.index = int(index_or_state)
            # set peek consistent with index
            if self.index >= len(self.string):
                self.peek = None
            else:
                # recompute peek
                ch = self.string[self.index]
                if ch == "\\" and self.index + 1 < len(self.string):
                    self.peek = ch + self.string[self.index + 1]
                else:
                    self.peek = ch


def isident(char: str) -> bool:
    return char.isalpha() or char == "_"


def isdigit(char: str) -> bool:
    return char.isdigit()


def isname(name: str) -> bool:
    # check that group name is a valid identifier
    if not name:
        return False
    if not isident(name[0]):
        return False
    for char in name[1:]:
        if not (isident(char) or isdigit(char)):
            return False
    return True


def _class_escape(source: Tokenizer, escape: str):
    # handle escape code inside character class
    code = ESCAPES.get(escape)
    if code:
        return code
    code = CATEGORIES.get(escape)
    if code:
        return code
    try:
        c = escape[1:2]
        if c == "x":
            # hexadecimal escape (exactly two digits)
            while source.peek in HEXDIGITS and len(escape) < 4:
                escape = escape + source.get()
            escape = escape[2:]
            if len(escape) != 2:
                raise Exception("bogus escape: %s" % repr("\\" + escape))
            return LITERAL, int(escape, 16) & 0xFF
        elif c in OCTDIGITS:
            # octal escape (up to three digits)
            while source.peek in OCTDIGITS and len(escape) < 4:
                escape = escape + source.get()
            escape = escape[1:]
            return LITERAL, int(escape, 8) & 0xFF
        elif c in DIGITS:
            raise Exception("bogus escape: %s" % repr(escape))
        if len(escape) == 2:
            return LITERAL, ord(escape[1])
    except ValueError:
        pass
    raise Exception("bogus escape: %s" % repr(escape))


def _escape(source: Tokenizer, escape: str, state: Pattern):
    # handle escape code in expression
    code = CATEGORIES.get(escape)
    if code:
        return code
    code = ESCAPES.get(escape)
    if code:
        return code
    try:
        c = escape[1:2]
        if c == "x":
            # hexadecimal escape
            while source.peek in HEXDIGITS and len(escape) < 4:
                escape = escape + source.get()
            if len(escape) != 4:
                raise ValueError
            return LITERAL, int(escape[2:], 16) & 0xFF
        elif c == "0":
            # octal escape
            while source.peek in OCTDIGITS and len(escape) < 4:
                escape = escape + source.get()
            return LITERAL, int(escape[1:], 8) & 0xFF
        elif c in DIGITS:
            # octal escape *or* decimal group reference
            if source.peek in DIGITS:
                escape = escape + source.get()
                if (
                    escape[1] in OCTDIGITS
                    and escape[2] in OCTDIGITS
                    and source.peek in OCTDIGITS
                ):
                    # three octal digits: octal escape
                    escape = escape + source.get()
                    return LITERAL, int(escape[1:], 8) & 0xFF
            # group reference
            group = int(escape[1:])
            if group < state.groups:
                if not state.checkgroup(group):
                    raise Exception("cannot refer to open group")
                return GROUPREF, group
            raise ValueError
        if len(escape) == 2:
            return LITERAL, ord(escape[1])
    except ValueError:
        pass
    raise Exception("bogus escape: %s" % repr(escape))


def _parse_sub(source: Tokenizer, state: Pattern, nested: int = 1):
    # parse an alternation: a|b|c
    items = []
    itemsappend = items.append
    sourcematch = source.match

    while True:
        itemsappend(_parse(source, state))
        if sourcematch("|"):
            continue
        if not nested:
            break
        if not source.peek or sourcematch(")", False):
            break
        else:
            raise Exception("pattern not properly closed")

    if len(items) == 1:
        return items[0]

    subpattern = SubPattern(state)
    subpatternappend = subpattern.append

    # check if all items share a common prefix
    while True:
        prefix = None
        for item in items:
            if not item:
                break
            if prefix is None:
                prefix = item[0]
            elif item[0] != prefix:
                break
        else:
            # all subitems start with a common "prefix".
            # move it out of the branch
            for item in items:
                del item[0]
            subpatternappend(prefix)
            continue  # check next one
        break

    # check if the branch can be replaced by a character set
    for item in items:
        if len(item) != 1 or item[0][0] != LITERAL:
            break
    else:
        # we can store this as a character set instead of a branch
        set_ = []
        setappend = set_.append
        for item in items:
            setappend(item[0])
        subpatternappend((IN, set_))
        return subpattern

    subpattern.append((BRANCH, (None, items)))
    return subpattern


def _parse_sub_cond(source: Tokenizer, state: Pattern, condgroup):
    item_yes = _parse(source, state)
    if source.match("|"):
        item_no = _parse(source, state)
        if source.match("|"):
            raise Exception("conditional backref with more than two branches")
    else:
        item_no = None
    if source.peek and not source.match(")", False):
        raise Exception("pattern not properly closed")
    subpattern = SubPattern(state)
    subpattern.append((GROUPREF_EXISTS, (condgroup, item_yes, item_no)))
    return subpattern


_PATTERNENDERS = set("|)")
_ASSERTCHARS = set("=!<")
_LOOKBEHINDASSERTCHARS = set("=!")
_REPEATCODES = set([MIN_REPEAT, MAX_REPEAT])


def _parse(source: Tokenizer, state: Pattern):
    # parse a simple pattern
    subpattern = SubPattern(state)

    # precompute constants into local variables
    subpatternappend = subpattern.append
    sourceget = source.get
    sourcematch = source.match
    _len = len
    PATTERNENDERS = _PATTERNENDERS
    ASSERTCHARS = _ASSERTCHARS
    LOOKBEHINDASSERTCHARS = _LOOKBEHINDASSERTCHARS
    REPEATCODES = _REPEATCODES

    while True:
        if source.peek in PATTERNENDERS:
            break  # end of subpattern
        this = sourceget()
        if this is None:
            break  # end of pattern

        if state.flags & SRE_FLAG_VERBOSE:
            # skip whitespace and comments
            if this in WHITESPACE:
                continue
            if this == "#":
                while True:
                    this = sourceget()
                    if this in (None, "\n"):
                        break
                continue

        if this and this[0] not in SPECIAL_CHARS:
            subpatternappend((LITERAL, ord(this)))

        elif this == "[":
            # character set
            set_ = []
            setappend = set_.append
            if sourcematch("^"):
                setappend((NEGATE, None))
            start = set_[:]
            while True:
                this = sourceget()
                if this == "]" and set_ != start:
                    break
                elif this and this[0] == "\\":
                    code1 = _class_escape(source, this)
                elif this:
                    code1 = (LITERAL, ord(this))
                else:
                    raise Exception("unexpected end of regular expression")
                if sourcematch("-"):
                    # potential range
                    this = sourceget()
                    if this == "]":
                        # literal '-' at end
                        if code1[0] == IN:
                            code1 = code1[1][0]
                        setappend(code1)
                        setappend((LITERAL, ord("-")))
                        break
                    elif this:
                        if this[0] == "\\":
                            code2 = _class_escape(source, this)
                        else:
                            code2 = (LITERAL, ord(this))
                        if code1[0] != LITERAL or code2[0] != LITERAL:
                            raise Exception("bad character range")
                        lo = code1[1]
                        hi = code2[1]
                        if hi < lo:
                            raise Exception("bad character range")
                        setappend((RANGE, (lo, hi)))
                    else:
                        raise Exception("unexpected end of regular expression")
                else:
                    if code1[0] == IN:
                        code1 = code1[1][0]
                    setappend(code1)

            if _len(set_) == 1 and set_[0][0] == LITERAL:
                subpatternappend(set_[0])  # optimization
            elif _len(set_) == 2 and set_[0][0] == NEGATE and set_[1][0] == LITERAL:
                subpatternappend((NOT_LITERAL, set_[1][1]))  # optimization
            else:
                subpatternappend((IN, set_))

        elif this and this[0] in REPEAT_CHARS:
            # repeat previous item
            if this == "?":
                minrep, maxrep = 0, 1
            elif this == "*":
                minrep, maxrep = 0, MAXREPEAT
            elif this == "+":
                minrep, maxrep = 1, MAXREPEAT
            elif this == "{":
                if source.peek == "}":
                    subpatternappend((LITERAL, ord(this)))
                    continue
                here = source.tell()
                minrep, maxrep = 0, MAXREPEAT
                lo = hi = ""
                while source.peek in DIGITS:
                    lo = lo + source.get()
                if sourcematch(","):
                    while source.peek in DIGITS:
                        hi = hi + source.get()
                else:
                    hi = lo
                if not sourcematch("}"):
                    subpatternappend((LITERAL, ord(this)))
                    source.seek(here)
                    continue
                if lo:
                    minrep = int(lo)
                if hi:
                    maxrep = int(hi)
                if maxrep < minrep:
                    raise Exception("bad repeat interval")
            else:
                raise Exception("not supported")
            # figure out which item to repeat
            if subpattern:
                item = subpattern[-1:]
            else:
                item = None
            if not item or (_len(item) == 1 and item[0][0] == AT):
                raise Exception("nothing to repeat")
            if item[0][0] in REPEATCODES:
                raise Exception("multiple repeat")
            if sourcematch("?"):
                subpattern[-1] = (MIN_REPEAT, (minrep, maxrep, item))
            else:
                subpattern[-1] = (MAX_REPEAT, (minrep, maxrep, item))

        elif this == ".":
            subpatternappend((ANY, None))

        elif this == "(":
            group = 1
            name = None
            condgroup = None
            if sourcematch("?"):
                group = 0
                # options
                if sourcematch("P"):
                    # python extensions
                    if sourcematch("<"):
                        # named group: read name
                        name = ""
                        while True:
                            char = sourceget()
                            if char is None:
                                raise Exception("unterminated name")
                            if char == ">":
                                break
                            name = name + char
                        group = 1
                        if not isname(name):
                            raise Exception("bad character in group name")
                    elif sourcematch("="):
                        # named backreference
                        name = ""
                        while True:
                            char = sourceget()
                            if char is None:
                                raise Exception("unterminated name")
                            if char == ")":
                                break
                            name = name + char
                        if not isname(name):
                            raise Exception("bad character in group name")
                        gid = state.groupdict.get(name)
                        if gid is None:
                            raise Exception("unknown group name")
                        subpatternappend((GROUPREF, gid))
                        continue
                    else:
                        char = sourceget()
                        if char is None:
                            raise Exception("unexpected end of pattern")
                        raise Exception(f"unknown specifier: ?P{char}")
                elif sourcematch(":"):
                    # non-capturing group
                    group = 2
                elif sourcematch("#"):
                    # comment
                    while True:
                        if source.peek is None or source.peek == ")":
                            break
                        sourceget()
                    if not sourcematch(")"):
                        raise Exception("unbalanced parenthesis")
                    continue
                elif source.peek in ASSERTCHARS:
                    # lookahead assertions
                    char = sourceget()
                    dir_ = 1
                    if char == "<":
                        if source.peek not in LOOKBEHINDASSERTCHARS:
                            raise Exception("syntax error")
                        dir_ = -1  # lookbehind
                        char = sourceget()
                    p = _parse_sub(source, state)
                    if not sourcematch(")"):
                        raise Exception("unbalanced parenthesis")
                    if char == "=":
                        subpatternappend((ASSERT, (dir_, p)))
                    else:
                        subpatternappend((ASSERT_NOT, (dir_, p)))
                    continue
                elif sourcematch("("):
                    # conditional backreference group
                    condname = ""
                    while True:
                        char = sourceget()
                        if char is None:
                            raise Exception("unterminated name")
                        if char == ")":
                            break
                        condname = condname + char
                    group = 2
                    if isname(condname):
                        condgroup = state.groupdict.get(condname)
                        if condgroup is None:
                            raise Exception("unknown group name")
                    else:
                        try:
                            condgroup = int(condname)
                        except ValueError:
                            raise Exception("bad character in group name")
                else:
                    # flags
                    if not source.peek or source.peek not in FLAGS:
                        raise Exception("unexpected end of pattern")
                    while source.peek in FLAGS:
                        state.flags = state.flags | FLAGS[sourceget()]

            if group:
                # parse group contents
                if group == 2:
                    group = None  # anonymous / non-capturing
                else:
                    group = state.opengroup(name)
                if condgroup:
                    p = _parse_sub_cond(source, state, condgroup)
                else:
                    p = _parse_sub(source, state)
                if not sourcematch(")"):
                    raise Exception("unbalanced parenthesis")
                if group is not None:
                    state.closegroup(group)
                subpatternappend((SUBPATTERN, (group, p)))
            else:
                while True:
                    char = sourceget()
                    if char is None:
                        raise Exception("unexpected end of pattern")
                    if char == ")":
                        break
                    raise Exception("unknown extension")

        elif this == "^":
            subpatternappend((AT, AT_BEGINNING))

        elif this == "$":
            subpattern.append((AT, AT_END))

        elif this and this[0] == "\\":
            code = _escape(source, this, state)
            subpatternappend(code)

        else:
            raise Exception("parser error")

    return subpattern


def parse(s: str, flags: int = 0, pattern: Optional[Pattern] = None):
    # parse 're' pattern into list of (opcode, argument) tuples
    source = Tokenizer(s)

    if pattern is None:
        pattern = Pattern()
    pattern.flags = flags
    pattern.str = s

    p = _parse_sub(source, pattern, 0)

    tail = source.get()
    if tail == ")":
        raise Exception("unbalanced parenthesis")
    elif tail:
        raise Exception("bogus characters at end of regular expression")

    if flags & SRE_FLAG_DEBUG:
        p.dump()

    if not (flags & SRE_FLAG_VERBOSE) and p.pattern.flags & SRE_FLAG_VERBOSE:
        # The VERBOSE flag was switched on inside the pattern;
        # to be on the safe side, parse the whole thing again.
        return parse(s, p.pattern.flags)

    return p


def parse_template(source: str, pattern: Pattern):
    # parse 're' replacement string into list of literals and group references
    s = Tokenizer(source)
    sget = s.get
    p = []
    a = p.append

    def literal(lit, p=p, pappend=a):
        if p and p[-1][0] == LITERAL:
            # append to existing literal (note: previously used concatenation)
            p[-1] = (LITERAL, p[-1][1] + lit)
        else:
            pappend((LITERAL, lit))

    makechar = chr

    while True:
        this = sget()
        if this is None:
            break  # end of replacement string
        if this and this[0] == "\\":
            # group or escape
            c = this[1:2]
            if c == "g":
                name = ""
                if s.match("<"):
                    while True:
                        char = sget()
                        if char is None:
                            raise Exception("unterminated group name")
                        if char == ">":
                            break
                        name = name + char
                if not name:
                    raise Exception("bad group name")
                try:
                    index = int(name)
                    if index < 0:
                        raise Exception("negative group number")
                except ValueError:
                    if not isname(name):
                        raise Exception("bad character in group name")
                    try:
                        index = pattern.groupdict[name]
                    except KeyError:
                        raise IndexError("unknown group name")
                a((MARK, index))
            elif c == "0":
                if s.peek in OCTDIGITS:
                    this = this + sget()
                    if s.peek in OCTDIGITS:
                        this = this + sget()
                literal(makechar(int(this[1:], 8) & 0xFF))
            elif c in DIGITS:
                isoctal = False
                if s.peek in DIGITS:
                    this = this + sget()
                    if (
                        c in OCTDIGITS
                        and len(this) > 2
                        and this[2] in OCTDIGITS
                        and s.peek in OCTDIGITS
                    ):
                        this = this + sget()
                        isoctal = True
                        literal(makechar(int(this[1:], 8) & 0xFF))
                if not isoctal:
                    a((MARK, int(this[1:])))
            else:
                try:
                    this = makechar(ESCAPES[this][1])
                except KeyError:
                    pass
                literal(this)
        else:
            literal(this)

    # convert template to groups and literals lists
    i = 0
    groups = []
    literals = [None] * len(p)
    for c, s in p:
        if c == MARK:
            groups.append((i, s))
            # literals[i] is already None
        else:
            literals[i] = s
        i = i + 1
    return groups, literals


def expand_template(template, match):
    g = match.group
    sep = match.string[:0]
    groups, literals = template
    literals = literals[:]
    try:
        for index, group in groups:
            literals[index] = s = g(group)
            if s is None:
                raise Exception("unmatched group")
    except IndexError:
        raise Exception("invalid group reference")
    return sep.join(literals)

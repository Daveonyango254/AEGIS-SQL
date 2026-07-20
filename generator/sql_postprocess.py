"""Deterministic SQL post-processing for generated queries.

This module is intentionally dependency-free (standard library only) so it can
be unit-tested without torch / transformers and reused by both the SLM and LLM
generation paths.

The main fix here addresses a dominant BIRD failure mode: the SLM emits integer
division (``A / B``) where the BIRD ground-truth uses real division
(``CAST(A AS REAL) / B``). In SQLite, integer/integer division truncates toward
zero, so ratio queries silently return ``0`` and fail execution-accuracy even
though the SQL is otherwise correct. We rewrite the numerator of every division
operator with an explicit ``CAST(... AS REAL)`` when it is not already cast.
"""

import re

_FLOAT_LITERAL_RE = re.compile(r"^\d+\.\d+$")


def _find_division_ops(s: str) -> list[int]:
    """Return indices of top-level ``/`` division operators.

    Skips any ``/`` that appears inside a string literal ('...', "...") or a
    backtick-quoted identifier (`...`).
    """
    ops: list[int] = []
    in_quote = False
    quote_char = ""
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if in_quote:
            if c == quote_char:
                in_quote = False
        else:
            if c in ("'", '"', "`"):
                in_quote = True
                quote_char = c
            elif c == "/":
                ops.append(i)
        i += 1
    return ops


def _extract_left_operand(s: str, op_idx: int) -> tuple[int, int] | None:
    """Find the (start, end) span of the operand immediately left of ``op_idx``.

    Handles three operand shapes:
      * function call / parenthesised expr ending in ``)``  e.g. ``SUM(x)``, ``(a+b)``
      * backtick-quoted identifier, optionally alias-qualified  e.g. ``T1.`Free Meal``
      * plain identifier or numeric literal  e.g. ``T1.NumGE1500``, ``1500``
    """
    j = op_idx - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    if j < 0:
        return None
    end = j + 1

    if s[j] == ")":
        # Match the balanced opening parenthesis.
        depth = 0
        k = j
        while k >= 0:
            if s[k] == ")":
                depth += 1
            elif s[k] == "(":
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        if k < 0:
            return None
        start = k
        # Absorb a preceding function name / qualifier (e.g. the "SUM" in SUM(...)).
        m = start - 1
        while m >= 0 and (s[m].isalnum() or s[m] in "_.`"):
            m -= 1
        start = m + 1
        return (start, end)

    if s[j] == "`":
        k = j - 1
        while k >= 0 and s[k] != "`":
            k -= 1
        if k < 0:
            return None
        start = k
        # Absorb an alias qualifier such as ``T1.`` before the backtick.
        m = start - 1
        while m >= 0 and (s[m].isalnum() or s[m] in "_."):
            m -= 1
        start = m + 1
        return (start, end)

    # Plain identifier or numeric literal (allow dotted alias.column).
    k = j
    while k >= 0 and (s[k].isalnum() or s[k] in "_."):
        k -= 1
    start = k + 1
    if start >= end:
        return None
    return (start, end)


def apply_cast_fix(sql: str) -> str:
    """Wrap the numerator of each division in ``CAST(... AS REAL)``.

    Idempotent: operands already wrapped in CAST (or that already contain a
    CAST) are left untouched, so applying twice is a no-op.
    """
    if not sql or "/" not in sql:
        return sql

    for op_idx in reversed(_find_division_ops(sql)):
        span = _extract_left_operand(sql, op_idx)
        if span is None:
            continue
        start, end = span
        operand = sql[start:end]
        stripped = operand.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        # Already a float division or already cast -> nothing to do.
        if upper.startswith("CAST") or "CAST(" in upper:
            continue
        if _FLOAT_LITERAL_RE.match(stripped):
            continue
        sql = f"{sql[:start]}CAST({operand} AS REAL){sql[end:]}"

    return sql


# SQLite keywords that BIRD databases use as bare TABLE names (e.g. financial's
# `order`, european_football_2's `Match`). A reserved word in table position
# (right after FROM/JOIN/INTO/UPDATE) is a guaranteed syntax error unless quoted
# — models reliably emit it unquoted (`FROM order WHERE order_id = ...`), which
# crashed q158/q164 across runs. Quoting here is deterministic, idempotent, and
# model-agnostic (fixes the class for any generator).
_RESERVED_TABLE_WORDS = frozenset({
    "order", "match", "set", "to", "by", "group", "index", "where", "select",
    "from", "join", "on", "limit", "offset", "case", "when", "then", "else",
    "end", "values", "transaction", "left", "right", "inner", "outer", "cross",
    "union", "all", "and", "or", "not", "in", "is", "as", "exists", "between",
    "like", "having", "distinct", "table", "primary", "foreign", "key",
    "references", "default", "check", "using", "natural",
})

# The identifier token immediately after a table-introducing keyword. A leading
# backtick/quote or "(" (subquery) simply does not match, so those are skipped.
_TABLE_POSITION_RE = re.compile(
    r"\b(FROM|JOIN|INTO|UPDATE)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
)


def _quoted_spans(s: str) -> list[tuple[int, int]]:
    """Spans of string literals ('...', \"...\") and backtick identifiers (`...`)."""
    spans: list[tuple[int, int]] = []
    in_quote = False
    quote_char = ""
    start = 0
    for i, c in enumerate(s):
        if in_quote:
            if c == quote_char:
                in_quote = False
                spans.append((start, i + 1))
        elif c in ("'", '"', "`"):
            in_quote = True
            quote_char = c
            start = i
    if in_quote:  # unterminated literal: treat the tail as quoted
        spans.append((start, len(s)))
    return spans


def quote_reserved_tables(sql: str) -> str:
    """Backtick-quote reserved-word table names in FROM/JOIN/INTO/UPDATE position.

    The token after these keywords is unambiguously a table name, so quoting a
    reserved word there can never change semantics — it only turns a guaranteed
    SQLite syntax error into valid SQL. Matches inside string literals or
    already-quoted identifiers are left untouched; idempotent by construction
    (a quoted table no longer matches the identifier pattern).
    """
    if not sql:
        return sql
    spans = _quoted_spans(sql)

    def _in_quoted(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    def _repl(m: re.Match) -> str:
        keyword, ident = m.group(1), m.group(2)
        if ident.lower() in _RESERVED_TABLE_WORDS and not _in_quoted(m.start(2)):
            sep = m.group(0)[len(keyword):-len(ident)]  # preserve original spacing
            return f"{keyword}{sep}`{ident}`"
        return m.group(0)

    return _TABLE_POSITION_RE.sub(_repl, sql)


def finalize_sql(sql: str, enable_cast_fix: bool = True) -> str:
    """Light normalisation applied to extracted SQL before it leaves the generator."""
    if not sql:
        return sql
    sql = sql.strip()
    if enable_cast_fix:
        sql = apply_cast_fix(sql)
    sql = quote_reserved_tables(sql)
    return sql


_SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ANY_BLOCK_RE = re.compile(r"```\s*((?:SELECT|WITH)\b.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_START_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_EXPLANATION_STOPS = ("what is", "instructions:", "note:", "explanation:")


def extract_sql(output: str) -> str:
    """Extract one SQL statement from arbitrary model output.

    Shared by the SLM and LLM paths so chain-of-thought responses are handled
    identically everywhere. Precedence:

    1. The LAST fenced ```sql block anywhere in the text (CoT strategies reason
       first and emit the final query last; earlier blocks are partial snippets).
    2. Any fenced block that starts with SELECT/WITH.
    3. A line-scan for the last SELECT/WITH statement (handles truncated fences
       when the model ran out of tokens mid-```).
    4. The fence-stripped text itself if it already looks like SQL.

    Returns "" when no SQL can be found (callers drop empty candidates).
    """
    if not output:
        return ""
    text = output.strip()

    # 1-2. Fenced blocks, last one wins (CoT emits the final query last).
    blocks = _SQL_BLOCK_RE.findall(text) or _ANY_BLOCK_RE.findall(text)
    if blocks:
        candidate = blocks[-1].strip()
        if candidate:
            return candidate.rstrip(";").strip() + ";" if ";" in candidate else candidate

    # 3. Line-scan over statement starts. Direct outputs put the SQL first while
    # chain-of-thought puts it last, so instead of trusting position we collect a
    # candidate from every start and keep the most SQL-like one (a real query
    # references a table via FROM; prose lines like "SELECT the right table"
    # don't), breaking ties toward the LAST candidate (the CoT final answer).
    lines = text.split("\n")
    candidates = []
    for i, line in enumerate(lines):
        if not _SQL_START_RE.match(line):
            continue
        collected = []
        for raw in lines[i:]:
            stripped = raw.strip().strip("`")
            if collected and stripped.lower().startswith(_EXPLANATION_STOPS):
                break
            if stripped:
                collected.append(stripped)
            if ";" in stripped:
                break
        candidate = " ".join(collected)
        if ";" in candidate:
            candidate = candidate.split(";")[0].strip() + ";"
        if candidate:
            candidates.append(candidate)
    if candidates:
        scored = [
            (2 * (re.search(r"\bFROM\b", c, re.IGNORECASE) is not None)
             + (1 if c.endswith(";") else 0), idx, c)
            for idx, c in enumerate(candidates)
        ]
        scored.sort(key=lambda t: (t[0], t[1]))  # best score, then latest
        return scored[-1][2]

    # 4. Bare fence-stripped text that is already SQL.
    bare = text
    for fence in ("```sql", "```"):
        if bare.startswith(fence):
            bare = bare[len(fence):]
    if bare.endswith("```"):
        bare = bare[:-3]
    bare = bare.strip()
    return bare if _SQL_START_RE.match(bare) else ""

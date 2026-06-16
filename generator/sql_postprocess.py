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


def finalize_sql(sql: str, enable_cast_fix: bool = True) -> str:
    """Light normalisation applied to extracted SQL before it leaves the generator."""
    if not sql:
        return sql
    sql = sql.strip()
    if enable_cast_fix:
        sql = apply_cast_fix(sql)
    return sql

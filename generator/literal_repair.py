"""Deterministic post-hoc literal repair against the live database.

Value grounding at *prompt* time attaches the exact stored literals, but models
still copy the question's spelling into the SQL — the measured near-miss classes
on BIRD (each one turns a correct query into a 0-row / wrong answer):

* case / spelling one-off:  ``'Portuguese (Brazil)'`` vs stored ``'Portuguese (Brasil)'``
* diacritics:               ``'Volkan Baga'``        vs stored ``'Volkan Baǵa'``
* datetime suffix:          ``'2013-02-22 00:00:00'`` vs stored ``'2013-02-22'``

This module repairs the WINNING query after selection: for every
``[alias.]column = / LIKE 'literal'`` comparison it probes the database
(read-only, bounded); if the literal matches **zero** stored values but exactly
**one** near-match exists (case-insensitive → datetime-suffix → unicode-fold /
edit-distance ≤ 1), the literal is rewritten to the stored form. Ambiguity or
any doubt → no change. Deterministic, idempotent (an exact-matching literal is
never touched), and model-agnostic (fixes the class for any generator).

Standard library only (sqlite3 / re / unicodedata) — unit-testable offline.
"""

import re
import sqlite3
import time
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

try:  # pragma: no cover - loguru is always present in this repo
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("aegis.literal_repair")

# [alias.]column <op> 'literal'   — column plain or backticked (spaces/parens),
# operator = or LIKE. Literals containing % / _ under LIKE are wildcard patterns
# and are skipped (existence can't be checked meaningfully).
_COMPARISON_RE = re.compile(
    r"(?:(\w+)\s*\.\s*)?(?:`([^`]+)`|([A-Za-z_]\w*))\s*(=|LIKE)\s*'([^']*)'",
    re.IGNORECASE,
)

# FROM/JOIN table refs with an optional alias: `FROM trans T1`, `JOIN `order` AS o`.
_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?:`([^`]+)`|([A-Za-z_]\w*))(?:\s+(?:AS\s+)?([A-Za-z_]\w*))?",
    re.IGNORECASE,
)
# Words that follow a table name but are never aliases.
_NOT_ALIAS = {
    "on", "where", "inner", "left", "right", "outer", "cross", "join", "group",
    "order", "limit", "having", "union", "as", "set", "using", "natural",
}

_DATETIME_SUFFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00(?:\.0+)?$")

_MAX_CANDIDATE_VALUES = 300   # bounded fetch for fold / edit-distance matching
_PROBE_TIMEOUT_S = 5.0


def _fold(s: str) -> str:
    """Casefold + strip diacritics for accent-insensitive comparison."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).casefold()


def _edit_distance_le1(a: str, b: str) -> bool:
    """True iff levenshtein(a, b) <= 1 (early-exit, no DP table needed)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    # la <= lb, differ by 0 or 1
    i = j = 0
    edited = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if edited:
            return False
        edited = True
        if la == lb:
            i += 1
            j += 1          # substitution
        else:
            j += 1          # insertion into a
    return True


def _install_timeout(conn: sqlite3.Connection, seconds: float) -> None:
    deadline = time.time() + seconds
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 2000)


class _Prober:
    """Bounded read-only probes with per-connection column-existence cache."""

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.text_factory = lambda b: (
            b.decode(errors="replace") if isinstance(b, bytes) else b
        )
        _install_timeout(self.conn, _PROBE_TIMEOUT_S)
        self._columns: Dict[str, Set[str]] = {}

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def columns(self, table: str) -> Set[str]:
        key = table.lower()
        if key not in self._columns:
            try:
                rows = self.conn.execute(f"PRAGMA table_info(`{table}`)").fetchall()
                self._columns[key] = {r[1].lower() for r in rows}
            except sqlite3.Error:
                self._columns[key] = set()
        return self._columns[key]

    def exact_exists(self, table: str, column: str, literal: str) -> bool:
        try:
            row = self.conn.execute(
                f"SELECT 1 FROM `{table}` WHERE `{column}` = ? LIMIT 1", (literal,)
            ).fetchone()
            return row is not None
        except sqlite3.Error:
            return True  # cannot verify -> treat as fine, never rewrite blindly

    def nocase_matches(self, table: str, column: str, literal: str) -> List[str]:
        try:
            rows = self.conn.execute(
                f"SELECT DISTINCT `{column}` FROM `{table}` "
                f"WHERE `{column}` = ? COLLATE NOCASE LIMIT 3",
                (literal,),
            ).fetchall()
            return [r[0] for r in rows if r[0] is not None]
        except sqlite3.Error:
            return []

    def candidate_values(self, table: str, column: str, literal: str) -> List[str]:
        """Distinct stored values of comparable length (bounded) for fuzzy passes."""
        lo, hi = max(1, len(literal) - 2), len(literal) + 2
        try:
            rows = self.conn.execute(
                f"SELECT DISTINCT `{column}` FROM `{table}` "
                f"WHERE LENGTH(`{column}`) BETWEEN ? AND ? LIMIT ?",
                (lo, hi, _MAX_CANDIDATE_VALUES),
            ).fetchall()
            return [r[0] for r in rows if isinstance(r[0], str)]
        except sqlite3.Error:
            return []


def _alias_map(sql: str) -> Tuple[Dict[str, str], List[str]]:
    """(alias -> table, [tables in FROM/JOIN order])."""
    aliases: Dict[str, str] = {}
    tables: List[str] = []
    for m in _TABLE_REF_RE.finditer(sql):
        table = m.group(1) or m.group(2)
        alias = m.group(3)
        tables.append(table)
        aliases[table.lower()] = table
        if alias and alias.lower() not in _NOT_ALIAS:
            aliases[alias.lower()] = table
    return aliases, tables


def _find_stored_form(prober: _Prober, tables: List[str], column: str,
                      literal: str) -> Optional[str]:
    """The unique stored near-match for ``literal`` in ``column``, else None.

    Checks every candidate table that actually has the column; the literal is
    left alone if ANY of them contains it exactly (the query may be right).
    Fuzzy passes run in order of confidence and each requires a UNIQUE match.
    """
    holders = [t for t in tables if column.lower() in prober.columns(t)]
    if not holders:
        return None
    if any(prober.exact_exists(t, column, literal) for t in holders):
        return None  # literal is a real stored value -> never touch

    # Pass 1: case-insensitive exact.
    nocase = {v for t in holders for v in prober.nocase_matches(t, column, literal)}
    if len(nocase) == 1:
        return next(iter(nocase))
    if len(nocase) > 1:
        return None  # ambiguous

    # Pass 2: datetime suffix (pred '2013-02-22 00:00:00' vs stored '2013-02-22').
    m = _DATETIME_SUFFIX_RE.match(literal)
    if m and any(prober.exact_exists(t, column, m.group(1)) for t in holders):
        return m.group(1)

    # Pass 3: unicode-fold / edit-distance <= 1 over a bounded candidate set.
    folded = _fold(literal)
    matches: Set[str] = set()
    for t in holders:
        for value in prober.candidate_values(t, column, literal):
            fv = _fold(value)
            if fv == folded or _edit_distance_le1(fv, folded):
                matches.add(value)
                if len(matches) > 1:
                    return None  # ambiguous -> refuse to guess
    if len(matches) == 1:
        return next(iter(matches))
    return None


def repair_literals(sql: str, db_path: str) -> str:
    """Rewrite near-miss string literals in ``sql`` to their unique stored forms.

    Conservative by construction: a literal is only replaced when it matches
    nothing in the database AND exactly one near-match exists. Any probing
    failure leaves the SQL unchanged.
    """
    if not sql or not db_path or db_path == ":memory:" or "'" not in sql:
        return sql

    try:
        prober = _Prober(db_path)
    except sqlite3.Error as e:
        logger.debug(f"literal_repair: cannot open db ({e})")
        return sql

    try:
        aliases, tables = _alias_map(sql)
        if not tables:
            return sql

        # Right-to-left so replacements don't shift earlier match offsets.
        for m in reversed(list(_COMPARISON_RE.finditer(sql))):
            qualifier, col_bt, col_plain, op, literal = m.groups()
            column = col_bt or col_plain
            if not literal or literal.strip() == "":
                continue
            if op.upper() == "LIKE" and ("%" in literal or "_" in literal):
                continue  # wildcard pattern: existence check is meaningless
            if re.fullmatch(r"[\d.+-]+", literal):
                continue  # numeric-looking: exact numerics are not our classes

            if qualifier:
                table = aliases.get(qualifier.lower())
                candidate_tables = [table] if table else []
            else:
                candidate_tables = tables

            fixed = _find_stored_form(prober, candidate_tables, column, literal)
            if fixed is not None and fixed != literal:
                lit_start = m.end() - len(literal) - 1
                escaped = fixed.replace("'", "''")
                sql = f"{sql[:lit_start]}{escaped}{sql[lit_start + len(literal):]}"
                logger.info(f"LITERAL_REPAIR: '{literal}' -> '{fixed}' ({column})")
    except Exception as e:  # never let repair break generation
        logger.warning(f"literal_repair: skipped ({e})")
    finally:
        prober.close()
    return sql

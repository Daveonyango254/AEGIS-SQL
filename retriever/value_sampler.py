"""Database value grounding / value linking for schema elements.

A dominant BIRD failure mode is the model inventing literal values that do not
match what is stored in the database (e.g. writing ``'Continuation'`` when the
column actually stores ``'Continuation School'``, or filtering ``County Name``
when the literal ``'Fresno County Office of Education'`` lives in ``District
Name``). The schema prompt lists column *names* but never the *values*, so the
model has to guess.

This module samples real distinct values from the SQLite database and, for each
retrieved text column, surfaces:
  * values that lexically overlap the question tokens ("value linking"), and
  * all values for genuinely low-cardinality columns ("value grounding").

Design notes:
  * Standard library only (sqlite3) + a lazy/guarded loguru import, so the core
    is unit-testable without the project's heavier dependencies.
  * Distinct-value sampling is cached per ``(db_path, table, column)`` so the
    cost is paid once per column and amortised across the ~1.5k BIRD queries
    that hit only ~11 databases.
  * Every operation is wrapped defensively: value grounding must never break
    SQL generation.
"""

import re
import sqlite3
from typing import Dict, List, Optional

try:  # pragma: no cover - loguru is present in the real venv
    from loguru import logger
except Exception:  # pragma: no cover - fallback for lightweight test envs
    import logging

    logger = logging.getLogger("aegis.value_sampler")

# Cache: (db_path, table, column) -> list of distinct string values (sampled).
_DISTINCT_CACHE: Dict[tuple, List[str]] = {}

# Column data types that are NOT useful to ground with literal values.
_NUMERIC_TYPE_HINTS = (
    "int",
    "real",
    "float",
    "double",
    "decimal",
    "numeric",
    "date",
    "time",
    "year",
    "bool",
    "blob",
)

_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "what", "which",
    "who", "whom", "list", "show", "give", "name", "names", "number", "count",
    "all", "each", "have", "has", "many", "much", "their", "there", "please",
    "find", "between", "among", "into", "over", "under", "than", "then", "where",
    "when", "average", "total", "highest", "lowest", "most", "least", "more",
    "less", "evidence", "refers", "means", "value", "values", "column",
}


def _tokenize(text: str) -> List[str]:
    """Lower-cased alphanumeric tokens of length >= 3 (minus common stopwords)."""
    raw = re.findall(r"[A-Za-z0-9]+", text or "")
    tokens = []
    seen = set()
    for tok in raw:
        low = tok.lower()
        if len(low) < 3 or low in _STOPWORDS:
            continue
        if low not in seen:
            seen.add(low)
            tokens.append(low)
    return tokens


def _is_text_column(data_type: Optional[str]) -> bool:
    """Heuristic: ground only text-affinity (or unknown-type) columns."""
    if not data_type:
        return True  # Unknown type: SQLite often stores codes as TEXT.
    dt = data_type.lower()
    return not any(hint in dt for hint in _NUMERIC_TYPE_HINTS)


def get_distinct_values(
    conn: sqlite3.Connection,
    db_path: str,
    table: str,
    column: str,
    sample_limit: int = 200,
) -> List[str]:
    """Return up to ``sample_limit`` distinct non-null values for a column (cached)."""
    key = (db_path, table, column)
    cached = _DISTINCT_CACHE.get(key)
    if cached is not None:
        return cached

    values: List[str] = []
    try:
        cur = conn.cursor()
        cur.execute(
            f'SELECT DISTINCT "{table}"."{column}" FROM "{table}" '
            f'WHERE "{table}"."{column}" IS NOT NULL LIMIT {int(sample_limit)}'
        )
        for row in cur.fetchall():
            val = row[0]
            if val is None:
                continue
            sval = str(val).strip()
            if sval:
                values.append(sval)
    except sqlite3.Error as e:
        logger.debug(f"Value sampling failed for {table}.{column}: {e}")
        values = []

    _DISTINCT_CACHE[key] = values
    return values


def get_value_hints(
    db_path: str,
    schema_elements,
    query_text: str,
    max_display: int = 8,
    low_card_threshold: int = 12,
    sample_limit: int = 200,
) -> Dict[str, List[str]]:
    """Compute literal value hints for retrieved text columns.

    Args:
        db_path: Path to the SQLite database file.
        schema_elements: Iterable of SchemaElement (``name`` is ``"table.column"``).
        query_text: Natural-language question (already includes BIRD evidence).
        max_display: Max number of values to surface per column.
        low_card_threshold: Columns with <= this many distinct sampled values are
            shown in full (categorical/enum grounding).
        sample_limit: Max distinct values sampled per column.

    Returns:
        Dict mapping ``"table.column"`` -> list of literal value strings. Columns
        with no useful hints are omitted.
    """
    hints: Dict[str, List[str]] = {}
    tokens = _tokenize(query_text)

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        logger.debug(f"Could not open db for value grounding ({db_path}): {e}")
        return hints

    try:
        for elem in schema_elements:
            name = getattr(elem, "name", "")
            if "." not in name:
                continue
            if not _is_text_column(getattr(elem, "data_type", None)):
                continue
            table, column = name.split(".", 1)

            distinct = get_distinct_values(conn, db_path, table, column, sample_limit)
            if not distinct:
                continue

            chosen: List[str] = []
            seen = set()

            # 1) Value linking: values that lexically overlap the question.
            for val in distinct:
                low = val.lower()
                if any(tok in low for tok in tokens):
                    if val not in seen:
                        seen.add(val)
                        chosen.append(val)
                if len(chosen) >= max_display:
                    break

            # 2) Value grounding: enumerate genuinely low-cardinality columns.
            if len(distinct) <= low_card_threshold:
                for val in distinct:
                    if val not in seen:
                        seen.add(val)
                        chosen.append(val)
                    if len(chosen) >= max_display:
                        break

            if chosen:
                hints[name] = chosen[:max_display]
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return hints


def clear_cache() -> None:
    """Clear the distinct-value cache (useful for tests)."""
    _DISTINCT_CACHE.clear()

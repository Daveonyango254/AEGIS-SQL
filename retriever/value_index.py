"""Deterministic value retrieval: locate question literals inside the database.

Embedding similarity cannot tell which of several name-like columns actually
CONTAINS "Fresno County Office of Education" — only the data can. For each
literal extracted from the question we probe the database's text columns with
bounded LIKE queries and return the columns that hold a matching value, along
with the EXACT stored form of that value. Two wins:

1. Retrieval precision — a value hit is the strongest possible evidence that a
   column belongs in the schema slice (it anchors the WHERE clause).
2. Literal grounding — the exact stored value is injected as a prompt hint, so
   the model writes ``= 'Continuation School'`` instead of ``= 'Continuation'``
   (a dominant BIRD failure class).

Cost is tightly bounded: probes run only on TEXT-typed columns, each probe is a
single ``LIMIT 1`` LIKE lookup with a statement timeout, results are memoized per
(db, literal), and an overall per-query probe budget caps the worst case.

Stdlib only (sqlite3) — unit-testable without any model.
"""

import sqlite3
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:  # pragma: no cover - loguru is always present in this repo
    import logging

    logger = logging.getLogger(__name__)

# (db_path, literal_lower) -> list of (table.column, stored_value) hits.
_PROBE_CACHE: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

_TEXT_TYPES = ("TEXT", "VARCHAR", "CHAR", "STRING", "CLOB")


def _is_text_column(data_type: Optional[str]) -> bool:
    dt = (data_type or "").upper()
    return any(t in dt for t in _TEXT_TYPES) or dt == ""


def _quote_ident(name: str) -> str:
    """Quote an identifier for SQLite (handles spaces/parens in BIRD columns)."""
    return '`' + name.replace('`', '``') + '`'


def find_value_columns(
    db_path: str,
    schema_elements: List,
    literals: List[str],
    max_probes: int = 200,
    probe_ops: int = 200_000,
    relaxed: bool = False,
) -> Dict[str, List[Tuple[str, str]]]:
    """Locate each literal in the database's text columns.

    Args:
        db_path: SQLite database path.
        schema_elements: candidate columns (probing is restricted to this pool so
            the budget follows the retrieval slice, not the whole schema).
        literals: value-like strings extracted from the question/evidence.
        max_probes: overall cap on LIKE probes for this call (cost bound).
        probe_ops: per-probe SQLite VM-op budget (progress handler aborts scans
            beyond this, so one huge table cannot stall retrieval).
        relaxed: also try substring matching on the first word (recursive round 2
            uses this when the strict pass found nothing for an entity).

    Returns:
        {literal -> [(\"table.column\", exact_stored_value), ...]} — empty lists
        are omitted; at most 3 column hits per literal (more adds noise).
    """
    hits: Dict[str, List[Tuple[str, str]]] = {}
    if not db_path or db_path == ":memory:" or not literals:
        return hits

    text_cols = [e for e in schema_elements if "." in e.name and _is_text_column(e.data_type)]
    if not text_cols:
        return hits

    conn = None
    probes = 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode(errors="replace") if isinstance(b, bytes) else b

        for literal in literals:
            cache_key = (db_path, f"{literal.lower()}|relaxed={relaxed}")
            if cache_key in _PROBE_CACHE:
                if _PROBE_CACHE[cache_key]:
                    hits[literal] = _PROBE_CACHE[cache_key]
                continue

            found: List[Tuple[str, str]] = []
            patterns = [literal, f"%{literal}%"]
            if relaxed and " " in literal:
                patterns.append(f"%{literal.split()[0]}%")

            for elem in text_cols:
                if probes >= max_probes or len(found) >= 3:
                    break
                table, column = elem.name.split(".", 1)
                q = (
                    f"SELECT {_quote_ident(column)} FROM {_quote_ident(table)} "
                    f"WHERE {_quote_ident(column)} LIKE ? LIMIT 1"
                )
                for pat in patterns:
                    probes += 1
                    try:
                        # Bound each probe: SQLite progress handler aborts long scans.
                        conn.set_progress_handler(lambda: 1, probe_ops)
                        row = conn.execute(q, (pat,)).fetchone()
                        conn.set_progress_handler(None, 0)
                        if row and row[0] is not None:
                            found.append((elem.name, str(row[0])))
                            break  # exact form found; stop trying looser patterns
                    except sqlite3.Error:
                        conn.set_progress_handler(None, 0)
                        break  # this column is unprobeable; move on

            _PROBE_CACHE[cache_key] = found
            if found:
                hits[literal] = found
            if probes >= max_probes:
                logger.debug(f"value_index: probe budget ({max_probes}) exhausted")
                break
    except sqlite3.Error as e:
        logger.warning(f"value_index: probing skipped ({e})")
    finally:
        if conn is not None:
            conn.close()
    return hits


def clear_cache() -> None:
    """Reset the probe memo (tests / long-running processes)."""
    _PROBE_CACHE.clear()

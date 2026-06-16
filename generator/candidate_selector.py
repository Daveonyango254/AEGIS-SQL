"""Execution-guided candidate selection (self-consistency over results).

The single highest-ROI lever for execution accuracy on a fixed local SLM is to
stop trusting one greedy decode. Instead we:

  1. generate N candidate SQL strings (greedy + temperature samples),
  2. execute each against the *real* SQLite database,
  3. discard candidates that error out or return empty results, and
  4. pick the result that the most candidates agree on (majority vote over the
     *result set*, not the SQL text), tie-broken toward the greedy candidate.

This is robust to surface-form diversity: two different-looking queries that
compute the same answer reinforce each other, while one-off hallucinations are
outvoted.

Standard library only (sqlite3 + lazy loguru) so the core logic is unit-testable
without torch / transformers.
"""

import sqlite3
import time
from typing import Dict, List, Optional

try:  # pragma: no cover
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("aegis.candidate_selector")


def _install_timeout(conn: sqlite3.Connection, seconds: float) -> None:
    """Abort long-running queries via a time-based progress handler."""
    if seconds <= 0:
        return
    deadline = time.time() + seconds

    def _handler() -> int:
        return 1 if time.time() > deadline else 0

    # Invoke the handler roughly every N VM instructions.
    conn.set_progress_handler(_handler, 1000)


def _execute(
    conn: sqlite3.Connection, sql_text: str, timeout: float
) -> tuple[bool, Optional[frozenset], Optional[str]]:
    """Execute one candidate. Returns (ok, result_set, error_message).

    ``result_set`` is a frozenset of row tuples (order-independent, matching how
    BIRD execution accuracy compares results).
    """
    if not sql_text or not sql_text.strip():
        return False, None, "empty SQL"
    try:
        _install_timeout(conn, timeout)
        cur = conn.cursor()
        cur.execute(sql_text)
        rows = cur.fetchall()
        result = frozenset(tuple(r) for r in rows)
        return True, result, None
    except Exception as e:  # sqlite3.Error, TypeError on unhashable, etc.
        return False, None, str(e)
    finally:
        try:
            conn.set_progress_handler(None, 0)
        except Exception:
            pass


def select_best(
    candidate_sqls: List[str],
    db_path: str,
    timeout: float = 5.0,
) -> Dict:
    """Pick the best candidate by execution-guided majority vote.

    Args:
        candidate_sqls: Candidate SQL strings. The FIRST element is treated as
            the greedy/deterministic candidate and used for tie-breaking.
        db_path: Path to the SQLite database file.
        timeout: Per-candidate execution timeout (seconds).

    Returns:
        Dict with keys:
          best_sql:      chosen SQL string
          exec_ok:       whether the chosen SQL executed without error
          is_empty:      whether the chosen result set is empty
          error:         error message if the chosen SQL failed (else None)
          num_candidates / num_executed / num_nonempty / num_agree: diagnostics
    """
    candidates = [c for c in candidate_sqls if c and c.strip()]
    if not candidates:
        return {
            "best_sql": candidate_sqls[0] if candidate_sqls else "",
            "exec_ok": False,
            "is_empty": True,
            "error": "no candidates",
            "num_candidates": len(candidate_sqls),
            "num_executed": 0,
            "num_nonempty": 0,
            "num_agree": 0,
        }

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        logger.debug(f"Candidate selection could not open db {db_path}: {e}")
        return {
            "best_sql": candidates[0],
            "exec_ok": False,
            "is_empty": True,
            "error": f"db open failed: {e}",
            "num_candidates": len(candidates),
            "num_executed": 0,
            "num_nonempty": 0,
            "num_agree": 0,
        }

    num_executed = 0
    num_nonempty = 0
    first_ok_empty_idx: Optional[int] = None
    first_error: Optional[str] = None
    # result_set -> {"votes": int, "first_idx": int}
    groups: Dict[frozenset, Dict] = {}

    try:
        for idx, sql_text in enumerate(candidates):
            ok, result, err = _execute(conn, sql_text, timeout)
            if not ok:
                if first_error is None:
                    first_error = err
                continue
            num_executed += 1
            if result is None or len(result) == 0:
                if first_ok_empty_idx is None:
                    first_ok_empty_idx = idx
                continue
            num_nonempty += 1
            g = groups.get(result)
            if g is None:
                groups[result] = {"votes": 1, "first_idx": idx}
            else:
                g["votes"] += 1
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Priority 1: majority vote among non-empty results (tie-break -> earliest).
    if groups:
        best_group = min(
            groups.values(), key=lambda g: (-g["votes"], g["first_idx"])
        )
        idx = best_group["first_idx"]
        return {
            "best_sql": candidates[idx],
            "exec_ok": True,
            "is_empty": False,
            "error": None,
            "num_candidates": len(candidates),
            "num_executed": num_executed,
            "num_nonempty": num_nonempty,
            "num_agree": best_group["votes"],
        }

    # Priority 2: an executing-but-empty candidate (still valid SQL).
    if first_ok_empty_idx is not None:
        return {
            "best_sql": candidates[first_ok_empty_idx],
            "exec_ok": True,
            "is_empty": True,
            "error": None,
            "num_candidates": len(candidates),
            "num_executed": num_executed,
            "num_nonempty": 0,
            "num_agree": 0,
        }

    # Priority 3: everything errored -> fall back to the greedy candidate.
    return {
        "best_sql": candidates[0],
        "exec_ok": False,
        "is_empty": True,
        "error": first_error or "all candidates failed",
        "num_candidates": len(candidates),
        "num_executed": 0,
        "num_nonempty": 0,
        "num_agree": 0,
    }

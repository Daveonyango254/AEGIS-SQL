"""Corrective self-consistency (CSC-SQL) engine: group, pick top-2, merge-revise.

Implements the selection method the CscSQL merge checkpoint was trained for:

    candidates ──execute──> vote groups (key = frozenset of result rows)
                                │
                     top-2 groups by votes
                                │
             merge-revision prompt (drafts + results + reduced schema)
                                │
                merge model samples m candidates ──> re-vote ──> final SQL

Plain-vote self-consistency fails exactly when the majority group is wrong; the
merge model was RL-trained to adjudicate between the top two disagreeing result
groups using their execution evidence (CSC-SQL, arXiv:2505.13271). Grouping and
selection logic here mirrors the reference ``major_voting2`` implementation.

Stdlib only (sqlite3 via candidate_selector) — unit-testable without torch.
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from loguru import logger

from generator.candidate_selector import _execute, _install_timeout


@dataclass
class VoteGroup:
    """A set of candidates whose execution produced the same result set."""

    sql: str                      # representative = FIRST sql that produced the result
    result: Optional[frozenset]   # None => execution error group
    votes: int = 0
    members: List[str] = field(default_factory=list)


def group_by_execution(
    candidates: Sequence[str], db_path: str, timeout: float = 30.0
) -> List[VoteGroup]:
    """Execute every candidate and group them by result set (majority voting).

    Mirrors CSC-SQL's grouping: hashable ``frozenset(rows)`` keys accumulate
    votes; the representative SQL of a group is the first one seen. Errored
    candidates are collected in a trailing error group (never wins unless it
    is all there is). Returns groups sorted by votes, descending.
    """
    groups: dict = {}
    errors = VoteGroup(sql="", result=None, votes=0)
    if not db_path or db_path == ":memory:":
        # No database to vote on: one synthetic group per unique SQL, first wins.
        seen = []
        for sql in candidates:
            if sql not in seen:
                seen.append(sql)
        return [VoteGroup(sql=s, result=frozenset(), votes=1, members=[s]) for s in seen]

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        logger.warning(f"csc: cannot open db ({e}); falling back to first candidate")
        return [VoteGroup(sql=candidates[0], result=frozenset(), votes=1,
                          members=list(candidates))] if candidates else []

    try:
        conn.text_factory = lambda b: b.decode(errors="replace") if isinstance(b, bytes) else b
        _install_timeout(conn, timeout)
        for sql in candidates:
            ok, result, _err = _execute(conn, sql, timeout)
            if not ok:
                if not errors.sql:
                    errors.sql = sql
                errors.votes += 1
                errors.members.append(sql)
                continue
            key = result  # frozenset of rows (order-independent)
            if key not in groups:
                groups[key] = VoteGroup(sql=sql, result=key, votes=0)
            groups[key].votes += 1
            groups[key].members.append(sql)
    finally:
        conn.close()

    ranked = sorted(groups.values(), key=lambda g: -g.votes)
    if errors.votes:
        ranked.append(errors)  # error group ranks last regardless of votes
    return ranked


def top2(groups: List[VoteGroup]) -> List[VoteGroup]:
    """The two highest-vote NON-ERROR groups (one group => it stands alone)."""
    real = [g for g in groups if g.result is not None]
    return real[:2] if real else groups[:1]


def csc_select(
    candidates: Sequence[str],
    db_path: str,
    merge_fn: Optional[Callable[[List[VoteGroup]], List[str]]] = None,
    merge_when_unanimous: bool = False,
    timeout: float = 30.0,
) -> str:
    """Full corrective self-consistency selection over a candidate pool.

    Args:
        candidates: deduplicated candidate SQL strings (real tokens).
        db_path: SQLite database for execution voting.
        merge_fn: callable taking the top-2 ``VoteGroup``s and returning the
            merge model's fresh candidates. ``None`` disables the merge stage
            (plain self-consistency).
        merge_when_unanimous: also run the merge stage when all candidates
            agree (costs a decode for little gain; CSC merges only on
            disagreement, which is the default here).
        timeout: per-candidate execution timeout in seconds.

    Returns the final selected SQL ("" for an empty pool).
    """
    candidates = [c for c in candidates if c and c.strip()]
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    groups = group_by_execution(candidates, db_path, timeout=timeout)
    if not groups:
        return candidates[0]

    contenders = top2(groups)
    disagreement = len(contenders) >= 2 and contenders[0].result != contenders[1].result

    if merge_fn is not None and (disagreement or merge_when_unanimous):
        try:
            fresh = merge_fn(contenders) or []
        except Exception as e:  # merge is an enhancement, never fatal
            logger.warning(f"csc: merge stage failed ({e}); using vote winner")
            fresh = []
        fresh = [f for f in fresh if f and f.strip()]
        if fresh:
            # Re-vote over the merge outputs; the merge answer overrides the
            # original vote only when it executes into a real group.
            merged_groups = group_by_execution(fresh, db_path, timeout=timeout)
            for g in merged_groups:
                if g.result is not None:
                    logger.info(
                        f"csc: merge revised the answer "
                        f"({g.votes}/{len(fresh)} merge votes)"
                    )
                    return g.sql
    return contenders[0].sql

"""Rank fusion, metadata boosting, and the FK-aware adaptive schema budget.

Pure functions over (column-name, score) structures — stdlib only, fully
unit-testable. This is the precision engine of the multi-step retriever:

* ``rrf_fuse``           — combine per-sub-query rankings (Reciprocal Rank Fusion).
* ``apply_boosts``       — metadata-aware score adjustments (name match, value
                           hits, type expectations).
* ``adaptive_budget``    — choose the final table set + per-table columns. Keeps
                           evidence-backed tables, adds foreign-key BRIDGE tables
                           so the slice is always join-connected (the financial/
                           formula_1 failure class), and gives simple queries a
                           small clean slice instead of the fixed 60-column dump
                           (the measured 7.5-tables-for-1.76-needed noise ratio).

Complexity: fusion O(S·K log K) for S sub-queries of depth K; the bridge search
is a BFS per kept-table pair on the FK graph (both tiny for BIRD schemas).
"""

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Reciprocal Rank Fusion constant (Cormack et al.): dampens the head so one
# sub-query cannot dominate, while agreement across sub-queries accumulates.
RRF_K = 60

# Boost weights, expressed in RRF-score units (a rank-1 hit contributes
# 1/(60+1) ≈ 0.016). A value hit outweighs any single rank because data-level
# evidence is categorically stronger than embedding similarity.
BOOST_VALUE_HIT = 0.10
BOOST_NAME_MATCH = 0.02
BOOST_TYPE_MATCH = 0.008
FULL_QUESTION_WEIGHT = 2.0  # the full-question ranking counts double in RRF


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> Dict[str, float]:
    """Fuse ranked lists of column names into one score per column.

    ``rankings[0]`` is treated as the full-question ranking and weighted
    ``FULL_QUESTION_WEIGHT``; entity sub-queries weigh 1.0 each.
    """
    scores: Dict[str, float] = defaultdict(float)
    for qi, ranking in enumerate(rankings):
        weight = FULL_QUESTION_WEIGHT if qi == 0 else 1.0
        for rank, name in enumerate(ranking):
            scores[name] += weight / (k + rank + 1)
    return dict(scores)


def apply_boosts(
    scores: Dict[str, float],
    question_tokens: Set[str],
    value_hit_columns: Set[str],
    wants_aggregate: bool,
    has_numbers: bool,
    numeric_columns: Set[str],
) -> Dict[str, float]:
    """Metadata-aware boosting on fused scores (returns a new dict).

    * Name match: a column whose table/column tokens appear in the question is
      almost always relevant (cheap lexical prior the dense model can miss).
    * Value hit: the column literally CONTAINS a question literal.
    * Type expectation: aggregate/numeric questions need numeric columns.
    """
    out = dict(scores)
    for name, score in scores.items():
        boost = 0.0
        table, _, column = name.partition(".")
        tokens = set(
            t.lower()
            for part in (table, column)
            for t in part.replace("_", " ").replace("(", " ").replace(")", " ").split()
        )
        if tokens & question_tokens:
            boost += BOOST_NAME_MATCH
        if name in value_hit_columns:
            boost += BOOST_VALUE_HIT
        if (wants_aggregate or has_numbers) and name in numeric_columns:
            boost += BOOST_TYPE_MATCH
        out[name] = score + boost
    return out


def _fk_adjacency(foreign_keys: Iterable) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for fk in foreign_keys or []:
        adj[fk.from_table].add(fk.to_table)
        adj[fk.to_table].add(fk.from_table)
    return adj


def _bridge_tables(kept: Set[str], adj: Dict[str, Set[str]], max_hops: int = 3) -> Set[str]:
    """Tables needed to join-connect the kept set (BFS shortest paths).

    Gold queries frequently traverse semantically irrelevant intermediate tables
    (e.g. financial's ``disp`` linking ``client`` to ``account``); retrieval never
    surfaces them because nothing in the question mentions them. Walking the FK
    graph between every kept pair and adding the intermediate nodes guarantees
    the model can always express the join path.
    """
    bridges: Set[str] = set()
    kept_list = sorted(kept)
    for i, src in enumerate(kept_list):
        for dst in kept_list[i + 1:]:
            # BFS from src to dst, tracking parents to recover the path.
            parents = {src: None}
            frontier = deque([(src, 0)])
            while frontier:
                node, depth = frontier.popleft()
                if node == dst:
                    # walk back, collecting intermediates
                    cur = parents[dst]
                    while cur is not None and cur != src:
                        bridges.add(cur)
                        cur = parents[cur]
                    break
                if depth >= max_hops:
                    continue
                for nxt in adj.get(node, ()):
                    if nxt not in parents:
                        parents[nxt] = node
                        frontier.append((nxt, depth + 1))
    return bridges - kept


def adaptive_budget(
    scores: Dict[str, float],
    value_hit_columns: Set[str],
    schema,
    max_tables: Optional[int] = None,
    per_table_columns: int = 10,
) -> List[str]:
    """Select the final ordered column slice under an adaptive budget.

    RECALL-FIRST (measured lesson: capping tables at 4 dropped ground-truth
    tables on FK-maze databases and cost 14/100 queries outright): by default
    (``max_tables=None``) EVERY scored table is kept and noise is controlled
    purely through the per-table COLUMN cap. A table cap can still be set for
    ablations; value-hit tables are always exempt from it, and FK bridge tables
    are added so every kept pair is join-connected. Column selection per table:
    value hits + primary/foreign-key columns (join keys are never dropped) + the
    top-scored columns up to ``per_table_columns``.

    Returns column names ordered by (table evidence, column score) — the order
    the prompt renderer will use.
    """
    if not scores:
        return []

    table_score: Dict[str, float] = defaultdict(float)
    by_table: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for name, score in scores.items():
        table = name.split(".", 1)[0]
        table_score[table] += score
        by_table[table].append((name, score))

    value_tables = {c.split(".", 1)[0] for c in value_hit_columns}
    ranked_tables = sorted(table_score, key=lambda t: -table_score[t])
    kept: Set[str] = set()
    for t in ranked_tables:
        if max_tables is not None and len(kept) >= max_tables and t not in value_tables:
            continue
        kept.add(t)

    # Join connectivity: bridge tables enter with only their key columns.
    adj = _fk_adjacency(getattr(schema, "foreign_keys", None))
    bridges = _bridge_tables(kept, adj)

    # Key columns per table (PKs + FK endpoints) — never dropped.
    key_cols: Dict[str, Set[str]] = defaultdict(set)
    for t, pks in (getattr(schema, "primary_keys", None) or {}).items():
        for c in pks:
            key_cols[t].add(f"{t}.{c}")
    for fk in getattr(schema, "foreign_keys", None) or []:
        key_cols[fk.from_table].add(f"{fk.from_table}.{fk.from_column}")
        key_cols[fk.to_table].add(f"{fk.to_table}.{fk.to_column}")

    result: List[str] = []
    seen: Set[str] = set()
    for t in sorted(kept, key=lambda t: -table_score[t]) + sorted(bridges):
        cols = sorted(by_table.get(t, []), key=lambda cs: -cs[1])
        chosen: List[str] = []
        # 1) value hits and key columns are mandatory
        for name, _ in cols:
            if name in value_hit_columns or name in key_cols.get(t, ()):
                chosen.append(name)
        # bridge tables that never scored still contribute their key columns
        for name in sorted(key_cols.get(t, ())):
            if name not in chosen:
                chosen.append(name)
        # 2) fill with top-scored columns up to the per-table budget
        for name, _ in cols:
            if len(chosen) >= per_table_columns:
                break
            if name not in chosen:
                chosen.append(name)
        for name in chosen:
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result

"""Corrective schema selection — precision-oriented pruning of retrieved columns.

Diagnostic finding (BIRD-dev): hybrid retrieval + FK expansion already achieves
~99% table recall, but delivers ~7 tables/query when the gold query needs ~2.
That noise causes the model to pick the wrong tables / over-join. This module
prunes the retrieved set down to the *anchor* tables (the strongest semantic
hits) plus their FK-connected neighbours, while guaranteeing join keys and
primary keys are kept — preserving recall and slashing noise.

Standard-library only (duck-typed over ForeignKey attributes) so it is
unit-testable without torch / FlagEmbedding.
"""

from typing import Dict, List, Optional


def _table(col_name: str) -> str:
    return col_name.split(".", 1)[0] if "." in col_name else col_name


def corrective_select(
    ranked_columns: List[str],
    foreign_keys: Optional[List] = None,
    primary_keys: Optional[Dict[str, List[str]]] = None,
    anchor_top_n: int = 12,
    max_neighbor_tables: int = 3,
) -> List[str]:
    """Prune a ranked column list to anchor tables + FK neighbours.

    Args:
        ranked_columns: ``"table.column"`` names, best-first (semantic order).
        foreign_keys: objects with ``from_table/from_column/to_table/to_column``.
        primary_keys: ``{table: [pk_col, ...]}``.
        anchor_top_n: how many top-ranked columns define the anchor tables.
        max_neighbor_tables: cap on FK-connected tables added to the anchors.

    Returns:
        Pruned, order-preserving list of ``"table.column"`` names: every retrieved
        column whose table is an anchor or kept neighbour, plus the FK join keys
        and primary keys of the kept tables.
    """
    foreign_keys = foreign_keys or []
    primary_keys = primary_keys or {}
    if not ranked_columns:
        return []

    # Anchor tables = tables of the strongest semantic hits (preserve order).
    anchors: List[str] = []
    for c in ranked_columns[:anchor_top_n]:
        t = _table(c)
        if t not in anchors:
            anchors.append(t)
    anchor_set = set(anchors)

    # FK neighbours of the anchor tables (order-stable, capped).
    neighbors: List[str] = []
    for fk in foreign_keys:
        ft, tt = fk.from_table, fk.to_table
        if ft in anchor_set and tt not in anchor_set and tt not in neighbors:
            neighbors.append(tt)
        if tt in anchor_set and ft not in anchor_set and ft not in neighbors:
            neighbors.append(ft)
    neighbors = neighbors[:max_neighbor_tables]
    keep = anchor_set | set(neighbors)

    # Keep retrieved columns from kept tables (drops the long tail of noise tables).
    out: List[str] = [c for c in ranked_columns if _table(c) in keep]
    seen = set(out)

    def _add(name: str) -> None:
        if name not in seen:
            out.append(name)
            seen.add(name)

    # Guarantee join keys between kept tables and their primary keys are present.
    for fk in foreign_keys:
        if fk.from_table in keep and fk.to_table in keep:
            _add(f"{fk.from_table}.{fk.from_column}")
            _add(f"{fk.to_table}.{fk.to_column}")
    for t in keep:
        for pk in primary_keys.get(t, []):
            _add(f"{t}.{pk}")

    return out

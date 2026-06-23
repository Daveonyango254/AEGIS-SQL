"""Unit tests for corrective RAG schema selection (retriever/corrective.py).

Diagnostic showed retrieval has ~99% table recall but ~7 tables/query of noise
(gold needs ~2). corrective_select prunes to anchor tables + FK neighbours while
keeping join keys, cutting noise without losing the needed tables.

Run with: python tests/test_corrective_retrieval.py
"""

import importlib.util
import os
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "corrective", os.path.join(ROOT, "retriever/corrective.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fk(ft, fc, tt, tc):
    return SimpleNamespace(from_table=ft, from_column=fc, to_table=tt, to_column=tc)


# satscores is the anchor (top hits); schools is an FK neighbour; extra is noise.
FKS = [
    _fk("satscores", "cds", "schools", "CDSCode"),
    _fk("frpm", "CDSCode", "schools", "CDSCode"),
]
PKS = {"schools": ["CDSCode"], "satscores": ["cds"]}

RANKED = [
    "satscores.AvgScrMath", "satscores.cds", "satscores.NumTstTakr",  # anchor
    "frpm.Enrollment", "frpm.CDSCode",                                  # noise table
    "schools.School", "schools.County",                                # neighbour
    "extra.foo", "extra.bar",                                          # pure noise
]


def test_prunes_noise_tables_but_keeps_anchor_and_neighbour():
    corrective_select = _load().corrective_select
    out = corrective_select(RANKED, FKS, PKS, anchor_top_n=3, max_neighbor_tables=3)
    tables = {c.split(".", 1)[0] for c in out}
    assert "satscores" in tables          # anchor kept
    assert "schools" in tables            # FK neighbour kept
    assert "extra" not in tables          # pure-noise table dropped
    # join keys present for the kept-table FK
    assert "satscores.cds" in out and "schools.CDSCode" in out


def test_single_table_query_drops_unrelated_noise():
    corrective_select = _load().corrective_select
    ranked = ["schools.School", "schools.County", "schools.CDSCode", "extra.y", "extra.z"]
    out = corrective_select(ranked, FKS, PKS, anchor_top_n=3, max_neighbor_tables=3)
    tables = {c.split(".", 1)[0] for c in out}
    assert "extra" not in tables                          # unrelated noise gone
    assert {"schools.School", "schools.County"} <= set(out)


def test_empty_input():
    assert _load().corrective_select([], FKS, PKS) == []


if __name__ == "__main__":
    test_prunes_noise_tables_but_keeps_anchor_and_neighbour()
    test_single_table_query_drops_unrelated_noise()
    test_empty_input()
    print("All corrective-retrieval tests passed.")

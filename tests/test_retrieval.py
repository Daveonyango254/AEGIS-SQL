"""Tests for retrieval diagnostics (evaluation/analyze_retrieval.py).

Validates SQL table extraction and the recall/gen-match accounting that
separates retrieval failures from generation failures. Stdlib only.

Run with: python tests/test_retrieval.py
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_retrieval", os.path.join(ROOT, "evaluation/analyze_retrieval.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_tables():
    m = _load()
    assert m.extract_tables(
        "SELECT T1.x FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.c = T2.c"
    ) == {"frpm", "schools"}
    # backticks + single table
    assert m.extract_tables("SELECT * FROM `satscores`") == {"satscores"}
    # subquery FROM (SELECT ...) must not count 'select' as a table
    assert m.extract_tables(
        "SELECT a FROM (SELECT b FROM t1) JOIN t2 ON a=b"
    ) == {"t1", "t2"}
    assert m.extract_tables("") == set()


def test_analyze_separates_retrieval_from_generation():
    m = _load()
    rows = [
        # retrieval contains both GT tables, model picks them -> recall hit + gen match
        {"difficulty": "simple",
         "ground_truth_sql": "SELECT 1 FROM frpm JOIN schools ON a=b",
         "predicted_sql": "SELECT 1 FROM frpm JOIN schools ON a=b",
         "retrieved_tables": ["frpm", "schools", "satscores"],
         "num_retrieved_columns": 30},
        # retrieval contains both, but model used only frpm -> recall hit, NO gen match
        {"difficulty": "moderate",
         "ground_truth_sql": "SELECT 1 FROM frpm JOIN schools ON a=b",
         "predicted_sql": "SELECT 1 FROM frpm",
         "retrieved_tables": ["frpm", "schools"],
         "num_retrieved_columns": 20},
        # retrieval MISSED schools -> recall miss
        {"difficulty": "challenging",
         "ground_truth_sql": "SELECT 1 FROM frpm JOIN schools ON a=b",
         "predicted_sql": "SELECT 1 FROM frpm",
         "retrieved_tables": ["frpm"],
         "num_retrieved_columns": 10},
    ]
    b = m.analyze(rows)["OVERALL"]
    assert b["n"] == 3
    assert b["recall_hit"] == 2          # first two have GT ⊆ retrieved
    assert b["gen_match"] == 1           # only the first matches GT tables exactly
    assert len(b["retrieval_misses"]) == 1
    assert b["retrieval_misses"][0][1] == ["schools"]


if __name__ == "__main__":
    test_extract_tables()
    test_analyze_separates_retrieval_from_generation()
    print("All retrieval-diagnostic tests passed.")

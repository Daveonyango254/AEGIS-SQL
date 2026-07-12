"""Unit tests for the CSC engine + OmniSQL templates (AEGIS v1 core).

Covers: <answer>-tag extraction, the OmniSQL generation/merge prompt structure
(evidence prepended, drafts + execution results before Instructions, reduced
schema), execution-vote grouping (majority, first-seen representative, error
group last, top-2), and csc_select's merge-override semantics — all offline
(temp SQLite, no torch).

Run with: python tests/test_csc.py
"""

import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aegis_types import ForeignKey, Schema, SchemaElement  # noqa: E402
from generator.csc import csc_select, group_by_execution, top2  # noqa: E402
from generator.sql_postprocess import extract_sql  # noqa: E402
from prompts import omnisql  # noqa: E402


def _schema():
    cols = [
        SchemaElement(element_type="column", name="schools.CDSCode", data_type="TEXT",
                      description="school id", example_values=["011"]),
        SchemaElement(element_type="column", name="schools.County", data_type="TEXT"),
        SchemaElement(element_type="column", name="satscores.cds", data_type="TEXT"),
    ]
    return Schema(
        database_id="ca", tables=["schools", "satscores"], columns=cols,
        foreign_keys=[ForeignKey(from_table="satscores", from_column="cds",
                                 to_table="schools", to_column="CDSCode")],
        primary_keys={"schools": ["CDSCode"]},
    )


def _db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
    conn.commit(); conn.close()
    return path


# --- extraction ---------------------------------------------------------------

def test_answer_tag_extraction():
    out = "<think>join on cds and filter</think><answer>SELECT a FROM t WHERE x=1</answer>"
    assert extract_sql(out) == "SELECT a FROM t WHERE x=1"
    # fenced body inside the tag is unwrapped
    out2 = "<answer>```sql\nSELECT b FROM t\n```</answer>"
    assert extract_sql(out2) == "SELECT b FROM t"


# --- templates ------------------------------------------------------------------

def test_generation_prompt_structure():
    schema = _schema()
    dd = omnisql.build_db_details(schema)
    assert "CREATE TABLE schools" in dd
    assert "PRIMARY KEY (CDSCode)" in dd
    assert "FOREIGN KEY (cds) REFERENCES schools(CDSCode)" in dd
    assert "example: [011]" in dd
    p = omnisql.build_generation_prompt("How many?", "K-12 means grades", dd)
    assert p.index("K-12 means grades") < p.index("How many?")  # evidence PREPENDED
    assert "<answer>" in p                                       # think wrapper
    assert "Take a deep breath" in p                             # OmniSQL tail


def test_merge_prompt_structure():
    schema = _schema()
    p = omnisql.build_merge_prompt(
        "How many?", "", schema,
        [("SELECT 1 FROM schools", [(1,)]), ("SELECT 2 FROM satscores", None)],
    )
    assert "draft SQL and its corresponding execution result" in p  # task line swapped
    assert "1. SELECT 1 FROM schools" in p
    assert "【Execution result】" in p
    assert "Execution error" in p                                    # None => error text
    assert p.index("draft SQL and execute result") < p.index("Instructions:")


def test_reduced_schema_only_referenced_tables():
    schema = _schema()
    reduced = omnisql.build_reduced_db_details(schema, ["SELECT 1 FROM schools"])
    assert "CREATE TABLE schools" in reduced
    assert "satscores" not in reduced
    # unparseable drafts fall back to the full schema
    full = omnisql.build_reduced_db_details(schema, ["garbage"])
    assert "satscores" in full


def test_result_normalization_caps():
    long_rows = [(i,) for i in range(50)]
    text = omnisql.normalize_execution_result(long_rows)
    assert text.startswith("The execution results of the first twenty")
    assert len(text) <= 1000


# --- grouping + selection -------------------------------------------------------

def test_grouping_majority_and_error_group():
    db = _db()
    try:
        cands = ["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x = 1",
                 "SELECT x FROM t WHERE x=2", "SELECT broken"]
        gs = group_by_execution(cands, db)
        assert gs[0].votes == 2 and gs[0].sql == cands[0]   # first-seen representative
        assert gs[-1].result is None                          # error group ranks last
        pair = top2(gs)
        assert len(pair) == 2 and all(g.result is not None for g in pair)
    finally:
        os.unlink(db)


def test_csc_merge_overrides_vote():
    db = _db()
    try:
        cands = ["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x=1",
                 "SELECT x FROM t WHERE x=2"]
        got = csc_select(cands, db, merge_fn=lambda gs: ["SELECT x FROM t WHERE x=2"])
        assert got == "SELECT x FROM t WHERE x=2"             # merge answer wins
        assert csc_select(cands, db) == cands[0]              # no merge => vote winner
    finally:
        os.unlink(db)


def test_csc_merge_failure_falls_back_to_vote():
    db = _db()
    try:
        cands = ["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x=2"]

        def broken(_groups):
            raise RuntimeError("merge model unavailable")

        assert csc_select(cands, db, merge_fn=broken) == cands[0]
    finally:
        os.unlink(db)


def test_csc_no_db_returns_first():
    assert csc_select(["SELECT 1", "SELECT 2"], None) == "SELECT 1"
    assert csc_select([], None) == ""


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

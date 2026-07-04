"""Unit tests for the multi-step retrieval pipeline (RAG v2).

Covers the pure stages (decomposition, RRF fusion, boosts, adaptive budget with
FK bridging) plus value retrieval against a real temp SQLite database and the
end-to-end pipeline with a fake scored retriever. No torch / FlagEmbedding.

Run with: python tests/test_rag.py
"""

import contextlib
import os
import sqlite3
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# retriever/__init__ imports schema_retriever which imports torch at module
# level; stub it so the pure pipeline stages are testable offline.
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.no_grad = contextlib.nullcontext
    sys.modules["torch"] = _torch

from aegis_types import ForeignKey, Language, Query, Schema, SchemaElement  # noqa: E402
from retriever.query_decompose import decompose  # noqa: E402
from retriever.fusion import adaptive_budget, apply_boosts, rrf_fuse  # noqa: E402
from retriever.value_index import clear_cache, find_value_columns  # noqa: E402


# --- decomposition -------------------------------------------------------------

def test_decompose_extracts_quoted_and_proper_nouns():
    dq = decompose(
        'How many schools in "Fresno County Office" scored above 500?',
        evidence="SAT score refers to AvgScrMath",
    )
    assert "Fresno County Office" in dq.literals
    assert "500" in dq.numbers
    assert dq.wants_aggregate  # "How many"
    assert dq.sub_queries[0].startswith("How many schools")   # full question first
    assert any("AvgScrMath" in s for s in dq.sub_queries)     # evidence reference mined


def test_decompose_skips_question_words():
    dq = decompose("Which state has the most singers?")
    assert all(l.lower() not in ("which", "state") for l in dq.literals)


# --- fusion / boosts / budget ----------------------------------------------------

def _schema():
    cols = [
        SchemaElement(element_type="column", name=n, data_type=t)
        for n, t in [
            ("client.client_id", "INTEGER"), ("client.gender", "TEXT"),
            ("disp.disp_id", "INTEGER"), ("disp.client_id", "INTEGER"),
            ("disp.account_id", "INTEGER"),
            ("account.account_id", "INTEGER"), ("account.frequency", "TEXT"),
            ("noise.noise_id", "INTEGER"), ("noise.blob", "TEXT"),
        ]
    ]
    return Schema(
        database_id="financial", tables=["client", "disp", "account", "noise"],
        columns=cols,
        foreign_keys=[
            ForeignKey(from_table="disp", from_column="client_id",
                       to_table="client", to_column="client_id"),
            ForeignKey(from_table="disp", from_column="account_id",
                       to_table="account", to_column="account_id"),
        ],
        primary_keys={"client": ["client_id"], "disp": ["disp_id"], "account": ["account_id"]},
    )


def test_rrf_agreement_beats_single_high_rank():
    # 'a' is mid-ranked by ALL sub-queries; 'b' is top-ranked by one only.
    fused = rrf_fuse([["b", "a", "c"], ["a", "c"], ["a", "c"]])
    assert fused["a"] > fused["b"]


def test_boosts_value_hit_dominates():
    scores = {"client.gender": 0.02, "noise.blob": 0.02}
    out = apply_boosts(
        scores, question_tokens={"female"}, value_hit_columns={"client.gender"},
        wants_aggregate=False, has_numbers=False, numeric_columns=set(),
    )
    assert out["client.gender"] > out["noise.blob"]


def test_budget_inserts_fk_bridge_table():
    """client + account kept on evidence; disp (the join bridge) must be added."""
    schema = _schema()
    scores = {  # strong evidence for client + account only; disp never scored
        "client.client_id": 0.5, "client.gender": 0.4,
        "account.account_id": 0.5, "account.frequency": 0.4,
        "noise.noise_id": 0.01,
    }
    chosen = adaptive_budget(scores, set(), schema, max_tables=2, per_table_columns=5)
    tables = {c.split(".", 1)[0] for c in chosen}
    assert {"client", "account", "disp"} <= tables      # bridge inserted
    assert "disp.client_id" in chosen                    # with its join keys
    assert "disp.account_id" in chosen
    assert "noise" not in tables                         # noise stays out


def test_budget_keeps_value_hit_tables_and_caps_columns():
    schema = _schema()
    scores = {f"client.{c}": 0.5 for c in ("client_id", "gender")}
    scores.update({"noise.blob": 0.05, "noise.noise_id": 0.04})
    chosen = adaptive_budget(
        scores, value_hit_columns={"noise.blob"}, schema=schema,
        max_tables=1, per_table_columns=2,
    )
    tables = {c.split(".", 1)[0] for c in chosen}
    assert "noise" in tables            # value-hit table always kept
    assert "noise.blob" in chosen       # the hit column is mandatory
    per_table = {}
    for c in chosen:
        per_table.setdefault(c.split(".", 1)[0], []).append(c)
    # key columns may exceed the cap, but plain columns respect it
    assert len(per_table["client"]) <= 3


# --- value retrieval (real sqlite) ----------------------------------------------

def test_value_index_finds_exact_stored_literal():
    clear_cache()
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE schools (name TEXT, county TEXT)")
        conn.execute("INSERT INTO schools VALUES ('Continuation School', 'Fresno')")
        conn.commit(); conn.close()
        elems = [
            SchemaElement(element_type="column", name="schools.name", data_type="TEXT"),
            SchemaElement(element_type="column", name="schools.county", data_type="TEXT"),
        ]
        hits = find_value_columns(path, elems, ["Continuation"], relaxed=False)
        assert "Continuation" in hits
        col, stored = hits["Continuation"][0]
        assert col == "schools.name"
        assert stored == "Continuation School"   # EXACT stored form, not the fragment
    finally:
        os.unlink(path)


def test_value_index_no_hits_for_absent_literal():
    clear_cache()
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.commit(); conn.close()
        elems = [SchemaElement(element_type="column", name="t.x", data_type="TEXT")]
        assert find_value_columns(path, elems, ["Nonexistent"]) == {}
    finally:
        os.unlink(path)


# --- end-to-end pipeline with a fake scored retriever ---------------------------

class _FakeScoredRetriever:
    """Returns a fixed ranking; mimics SchemaRetriever's scored API."""

    def __init__(self, schema):
        self.schema = schema
        self.model = None  # pipeline never touches .model unless reranking

    def retrieve_scored(self, query_text, top_k=40):
        ranked = [c for c in self.schema.columns]
        return [(c, 1.0 / (i + 1)) for i, c in enumerate(ranked)][:top_k]

    def score_table_cards(self, query_text):
        return {}


def test_pipeline_end_to_end_budget_and_grounding():
    from config import AEGISConfig
    from retriever.pipeline import MultiStepRetriever

    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE client (client_id INTEGER, gender TEXT)")
        conn.execute("INSERT INTO client VALUES (1, 'Female')")
        conn.commit(); conn.close()

        schema = _schema()
        cfg = AEGISConfig()
        cfg.rag.rerank = False  # no model in this test
        pipe = MultiStepRetriever(_FakeScoredRetriever(schema), schema, cfg)
        q = Query(text='How many "Female" clients are there?',
                  language=Language.ENGLISH, database_id="financial")
        clear_cache()
        elements = pipe.retrieve(q, path)

        names = [e.name for e in elements]
        assert names, "pipeline returned an empty slice"
        assert any(n.startswith("client.") for n in names)
        gender = next(e for e in elements if e.name == "client.gender")
        assert gender.example_values == ["Female"]       # exact stored literal attached
        # cached schema elements must NOT be mutated (copies only)
        orig = next(c for c in schema.columns if c.name == "client.gender")
        assert not orig.example_values
    finally:
        os.unlink(path)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

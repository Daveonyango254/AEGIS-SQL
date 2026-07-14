"""Wiring + node tests for the AEGIS v2 LangGraph pipeline (offline, mocked).

Exercises the sequential fallback engine (identical node semantics to LangGraph,
minus arm parallelism) with a mocked model cache + temp SQLite: the mode router,
execution-vote node, the disagreement-triggered judge, the bounded refine loop,
and the prediction-contract dict.

Run with: python tests/test_graph_v2.py
"""

import contextlib
import os
import sqlite3
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_Err = type("_Err", (Exception,), {})


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)


_stub("torch", no_grad=contextlib.nullcontext)
_stub("transformers", AutoTokenizer=object, AutoModelForCausalLM=object)
_stub("openai", Client=object, RateLimitError=_Err, APITimeoutError=_Err, APIConnectionError=_Err)
_stub("anthropic", Anthropic=object, RateLimitError=_Err, APITimeoutError=_Err, APIConnectionError=_Err)
_stub("sqlglot", parse_one=lambda *a, **k: None, ParseError=_Err)

_mc = types.ModuleType("workflow.model_cache")
_mc._cache = None
_mc.get_cache = lambda: _mc._cache
_wf = types.ModuleType("workflow"); _wf.__path__ = []
sys.modules["workflow"] = _wf
sys.modules["workflow.model_cache"] = _mc

from aegis_types import Language, Query, RoutingDecision, Schema, SchemaElement  # noqa: E402
from config import AEGISConfig  # noqa: E402
from agents.graph import GraphOrchestrator  # noqa: E402


def _make_db():
    fd, path = tempfile.mkstemp(suffix=".sqlite"); os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    conn.commit(); conn.close()
    return path


def _schema():
    cols = [SchemaElement(element_type="column", name="t.x", data_type="INTEGER")]
    return Schema(database_id="d", tables=["t"], columns=cols,
                  foreign_keys=[], primary_keys={"t": ["x"]})


class _FakeSLM:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = 0

    def complete(self, prompt, n=1, temperature=None, system_prompt=None,
                 max_tokens=None, raw=False):
        self.calls += 1
        return self.scripted.pop(0) if self.scripted else []


class _FakeLLM:
    def __init__(self, outs):
        self.outs = outs
        self.total_tokens = 50

    def complete(self, prompt, n=1, temperature=None, system_prompt=None,
                 max_tokens=None, raw=False):
        return ["2"] if raw else list(self.outs)


class _FakeRetriever:
    def __init__(self, cols):
        self._cols = cols

    def retrieve_scored(self, q, top_k=40):
        return [(c, 1.0) for c in self._cols]

    def score_table_cards(self, q):
        return {}


class _FakeCache:
    def __init__(self, schema, slm, llm=None):
        self._slm = slm
        self._llm = llm
        self._retriever = _FakeRetriever(schema.columns)

    def get_schema_retriever(self, db_id, schema):
        return self._retriever

    def get_slm(self, model_id=None):
        return self._slm

    def get_llm_generator(self):
        return self._llm


def _config(mode):
    c = AEGISConfig()
    c.mode = mode
    c.retrieval.schema_mode = "full"
    c.retrieval.value_retrieval = False
    c.refine.rounds = 0
    return c


CONTRACT = ("sql", "routing_decision", "verification_result", "generation_source",
            "winner_arm", "candidates_local", "candidates_remote",
            "retrieved_tables", "num_retrieved_columns", "cost_usd", "privacy_loss")


def test_local_mode_vote_and_contract():
    db = _make_db()
    try:
        schema = _schema()
        cfg = _config("local"); cfg.selection.judge = "off"
        slm = _FakeSLM([["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x = 1",
                         "SELECT x FROM t WHERE x=2"]])
        _mc._cache = _FakeCache(schema, slm)
        q = Query(text="x is one", language=Language.ENGLISH, database_id="d")
        r = GraphOrchestrator(cfg).run(
            {"query": q, "schema": schema, "db_path": db, "database_id": "d"})
        for k in CONTRACT:
            assert k in r, f"missing {k}"
        assert r["sql"].text == "SELECT x FROM t WHERE x=1"     # majority group
        assert r["winner_arm"] == "local"
        assert r["routing_decision"] == RoutingDecision.LOCAL
        assert r["candidates_local"] == 3 and r["candidates_remote"] == 0
    finally:
        os.unlink(db)


def test_ensemble_pools_both_arms():
    db = _make_db()
    try:
        schema = _schema()
        cfg = _config("ensemble"); cfg.selection.judge = "off"
        cfg.generation.remote_candidates = 1
        cfg.generation.remote_strategies = ["direct"]
        slm = _FakeSLM([["SELECT x FROM t WHERE x=1"]])
        llm = _FakeLLM(["SELECT x FROM t WHERE x=3"])
        _mc._cache = _FakeCache(schema, slm, llm=llm)
        q = Query(text="pool", language=Language.ENGLISH, database_id="d")
        r = GraphOrchestrator(cfg).run(
            {"query": q, "schema": schema, "db_path": db, "database_id": "d"})
        assert r["generation_source"] == "ensemble"
        assert r["candidates_local"] == 1 and r["candidates_remote"] == 1
        assert r["winner_arm"] in ("local", "remote")
        assert r["sql"].text.startswith("SELECT x FROM t")
    finally:
        os.unlink(db)


def test_judge_breaks_disagreement():
    db = _make_db()
    try:
        schema = _schema()
        cfg = _config("ensemble"); cfg.selection.judge = "remote"
        cfg.generation.remote_candidates = 1
        cfg.generation.remote_strategies = ["direct"]
        # local -> {x=1}, remote -> {x=2}; two groups disagree -> judge fires.
        slm = _FakeSLM([["SELECT x FROM t WHERE x=1"]])
        llm = _FakeLLM(["SELECT x FROM t WHERE x=2"])  # raw judge reply = "2"
        _mc._cache = _FakeCache(schema, slm, llm=llm)
        q = Query(text="pick", language=Language.ENGLISH, database_id="d")
        r = GraphOrchestrator(cfg).run(
            {"query": q, "schema": schema, "db_path": db, "database_id": "d"})
        assert r["winner_arm"].startswith("judge")   # judge adjudicated
    finally:
        os.unlink(db)


def test_refine_repairs_empty_winner():
    db = _make_db()
    try:
        schema = _schema()
        cfg = _config("local"); cfg.selection.judge = "off"; cfg.refine.rounds = 1
        # sole candidate returns empty -> refine -> revision returns a good query
        slm = _FakeSLM([["SELECT x FROM t WHERE x=999"],       # generation
                        ["SELECT x FROM t WHERE x=1"]])         # revision (raw=False)
        _mc._cache = _FakeCache(schema, slm)
        q = Query(text="fix", language=Language.ENGLISH, database_id="d")
        r = GraphOrchestrator(cfg).run(
            {"query": q, "schema": schema, "db_path": db, "database_id": "d"})
        assert r["sql"].text == "SELECT x FROM t WHERE x=1"
        assert r["winner_arm"] == "refine"
    finally:
        os.unlink(db)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

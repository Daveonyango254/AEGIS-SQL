"""Logic + wiring tests for the multi-agent booster.

Runs offline by stubbing the heavy optional deps (torch/transformers/openai/
anthropic/sqlglot) and faking the two ``workflow`` helpers the orchestrator pulls,
so we can exercise the *real* selector/refiner (against a temp SQLite DB) and the
*real* orchestrator wiring with a mocked model cache.

Run with: python tests/test_agents.py
"""

import contextlib
import os
import sqlite3
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- stub heavy/unavailable modules before importing the agents --------------
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

# Fake the workflow helpers so importing the orchestrator does NOT pull the
# langgraph pipeline (workflow/__init__ eagerly imports it). compute_cost is the
# real one-liner; get_cache returns whatever the test installs.
_wf = types.ModuleType("workflow"); _wf.__path__ = []
sys.modules["workflow"] = _wf
_cost = types.ModuleType("workflow.costing")
_cost.compute_cost = lambda source, tok, rc, lc: (tok * rc if source == "llm" else lc)
sys.modules["workflow.costing"] = _cost
_mc = types.ModuleType("workflow.model_cache")
_mc._cache = None
_mc.get_cache = lambda: _mc._cache
sys.modules["workflow.model_cache"] = _mc

from aegis_types import Language, Query, RoutingDecision, Schema, SchemaElement  # noqa: E402
from config import AEGISConfig  # noqa: E402
from agents.context import RunContext  # noqa: E402
from agents.selector import SelectorAgent  # noqa: E402
from agents.refiner import RefinerAgent  # noqa: E402
from agents.orchestrator import MultiAgentOrchestrator  # noqa: E402


# --- fixtures ----------------------------------------------------------------

def _make_db():
    """A temp SQLite DB with table t(x) holding rows 1,2,3."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    conn.commit()
    conn.close()
    return path


def _ctx(db_path, generate_fn=None, config=None):
    cols = [SchemaElement(element_type="column", name="t.x", data_type="INTEGER")]
    schema = Schema(database_id="d", tables=["t"], columns=cols, foreign_keys=[], primary_keys={})
    q = Query(text="rows where x is 1", language=Language.ENGLISH, database_id="d")
    return RunContext(
        query=q, gen_query=q, schema=schema, schema_elements=cols, db_path=db_path,
        config=config or AEGISConfig(), source="slm",
        generate_fn=generate_fn or (lambda p, n, t, sp: []),
        reconstruct_fn=lambda s: s, recon_map=None, expose_keys=True,
    )


# --- selector ----------------------------------------------------------------

def test_selector_majority_vote():
    db = _make_db()
    try:
        cands = ["SELECT x FROM t WHERE x = 1", "SELECT x FROM t WHERE x = 1", "SELECT x FROM t WHERE x = 2"]
        best, info = SelectorAgent(AEGISConfig()).select(cands, _ctx(db))
        assert best in cands[:2]              # majority result {(1,)} wins
        assert info["num_agree"] >= 2
    finally:
        os.unlink(db)


def test_selector_judge_breaks_split_vote():
    db = _make_db()
    try:
        # All three results distinct -> no majority -> judge decides.
        cands = ["SELECT x FROM t WHERE x = 1", "SELECT x FROM t WHERE x = 2", "SELECT x FROM t WHERE x = 3"]
        judge = lambda p, n, t, sp: ["2"]    # judge picks candidate #2
        best, _ = SelectorAgent(AEGISConfig()).select(cands, _ctx(db), judge_fn=judge)
        assert best == cands[1]
    finally:
        os.unlink(db)


def test_selector_parse_choice():
    parse = SelectorAgent(AEGISConfig())._parse_choice
    assert parse("2", 3) == 1
    assert parse("The best is 3.", 3) == 2
    assert parse("99", 3) is None           # out of range
    assert parse("none", 3) is None


# --- refiner -----------------------------------------------------------------

def test_refiner_repairs_empty_result():
    db = _make_db()
    try:
        # Mock model: the repair attempt returns a query that yields rows.
        gen = lambda p, n, t, sp: ["SELECT x FROM t WHERE x = 1"]
        out = RefinerAgent(AEGISConfig()).refine("SELECT x FROM t WHERE x = 999", _ctx(db, gen))
        assert out == "SELECT x FROM t WHERE x = 1"
    finally:
        os.unlink(db)


def test_refiner_keeps_original_when_repair_not_better():
    db = _make_db()
    try:
        # Repair also returns empty -> keep the original (never accept a worse result).
        gen = lambda p, n, t, sp: ["SELECT x FROM t WHERE x = 888"]
        original = "SELECT x FROM t WHERE x = 999"
        out = RefinerAgent(AEGISConfig()).refine(original, _ctx(db, gen))
        assert out == original
    finally:
        os.unlink(db)


# --- orchestrator wiring (local path) ----------------------------------------

class _FakeRetriever:
    def __init__(self, cols):
        self._cols = cols

    def retrieve(self, query, **kwargs):
        return self._cols


class _FakeSLM:
    """Returns a fixed candidate; records that complete() was called."""
    def __init__(self, sql):
        self.sql = sql
        self.calls = 0

    def complete(self, prompt, n=1, temperature=None, system_prompt=None):
        self.calls += 1
        return [self.sql]

    def generate(self, query, schema_elements, schema=None):
        from aegis_types import SQL
        return SQL(text=self.sql, dialect="sqlite", source="slm", verified=False)


class _FakeRouter:
    def route(self, query, schema_elements):
        return RoutingDecision.LOCAL


class _FakeCache:
    def __init__(self, cols, sql):
        self._retriever = _FakeRetriever(cols)
        self._slm = _FakeSLM(sql)

    def get_schema_retriever(self, db_id, schema):
        return self._retriever

    def get_router(self):
        return _FakeRouter()

    def get_slm_generator(self):
        return self._slm


def test_orchestrator_local_contract():
    cols = [SchemaElement(element_type="column", name="t.x", data_type="INTEGER")]
    schema = Schema(database_id="d", tables=["t"], columns=cols, foreign_keys=[], primary_keys={})
    q = Query(text="all x", language=Language.ENGLISH, database_id="d")

    # Install the mocked cache for this run.
    _mc._cache = _FakeCache(cols, "SELECT x FROM t")

    config = AEGISConfig()
    config.agents.judge_enabled = False  # keep the wiring test deterministic
    result = MultiAgentOrchestrator(config).run(
        {"query": q, "schema": schema, "db_path": None, "database_id": "d"}
    )

    # The contract dict the evaluation harness reads back.
    for key in ("sql", "routing_decision", "abstracted_prompt", "verification_result",
                "generation_source", "retrieved_tables", "num_retrieved_columns",
                "cost_usd", "privacy_loss"):
        assert key in result, f"missing contract key: {key}"
    assert result["sql"].text == "SELECT x FROM t"
    assert result["routing_decision"] == RoutingDecision.LOCAL
    assert result["generation_source"] == "slm"
    assert result["abstracted_prompt"] is None       # no abstraction on local path
    assert result["retrieved_tables"] == ["t"]
    assert result["num_retrieved_columns"] == 1
    assert _mc._cache._slm.calls >= 1                 # the SLM was actually driven


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

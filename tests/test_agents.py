"""Wiring tests for the AEGIS v1 orchestrator (offline, mocked models).

Verifies: local-mode candidate flow through CSC selection, ensemble pooling
(local + remote candidates deduplicated), the merge stage being invoked on
disagreement, and the prediction-contract dict the evaluation harness reads.

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

# --- stub heavy deps + the model cache before importing the orchestrator ------
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
from agents.orchestrator import MultiAgentOrchestrator  # noqa: E402


# --- fixtures -------------------------------------------------------------------

def _make_db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
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
    """Scripted local model: returns queued responses per complete() call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete(self, prompt, n=1, temperature=None, system_prompt=None,
                 max_tokens=None, raw=False):
        self.calls.append({"prompt": prompt, "n": n, "raw": raw})
        return self.script.pop(0) if self.script else []


class _FakeLLM:
    def __init__(self, outs):
        self.outs = outs
        self.total_tokens = 100

    def complete(self, prompt, n=1, temperature=None, system_prompt=None,
                 max_tokens=None, raw=False):
        return list(self.outs)


class _FakeRetriever:
    def __init__(self, cols):
        self._cols = cols

    def retrieve_scored(self, q, top_k=40):
        return [(c, 1.0) for c in self._cols]

    def score_table_cards(self, q):
        return {}


class _FakeCache:
    """Model cache double: SLMs keyed by model id, plus retriever/LLM."""

    def __init__(self, schema, slms, llm=None):
        self._slms = slms
        self._llm = llm
        self._retriever = _FakeRetriever(schema.columns)

    def get_schema_retriever(self, db_id, schema):
        return self._retriever

    def get_slm(self, model_id=None):
        return self._slms[model_id]

    def get_llm_generator(self):
        return self._llm


def _config(mode):
    c = AEGISConfig()
    c.mode = mode
    c.retrieval.schema_mode = "full"
    c.retrieval.value_retrieval = False   # keep wiring test deterministic
    c.selection.judge = "off"
    c.refine.rounds = 0
    return c


CONTRACT_KEYS = ("sql", "routing_decision", "abstracted_prompt", "verification_result",
                 "generation_source", "retrieved_tables", "num_retrieved_columns",
                 "cost_usd", "privacy_loss")


# --- tests ----------------------------------------------------------------------

def test_local_mode_contract_and_vote():
    db = _make_db()
    try:
        schema = _schema()
        cfg = _config("local")
        gen = _FakeSLM([["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x = 1",
                         "SELECT x FROM t WHERE x=2"]])
        _mc._cache = _FakeCache(schema, {cfg.models.generator: gen})
        cfg.csc.enabled = False

        q = Query(text="x equals one", language=Language.ENGLISH, database_id="d")
        result = MultiAgentOrchestrator(cfg).run(
            {"query": q, "schema": schema, "db_path": db, "database_id": "d"})

        for key in CONTRACT_KEYS:
            assert key in result, f"missing contract key {key}"
        assert result["sql"].text == "SELECT x FROM t WHERE x=1"   # majority vote
        assert result["routing_decision"] == RoutingDecision.LOCAL
        assert result["generation_source"] == "slm"
        assert result["retrieved_tables"] == ["t"]
        assert result["abstracted_prompt"] is None
    finally:
        os.unlink(db)


def test_merge_stage_adjudicates_disagreement():
    db = _make_db()
    try:
        schema = _schema()
        cfg = _config("local")
        # generator: 1-1 split between two results -> disagreement -> merge runs
        gen = _FakeSLM([["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x=2"]])
        merger = _FakeSLM([["SELECT x FROM t WHERE x=2"]])
        _mc._cache = _FakeCache(schema, {cfg.models.generator: gen,
                                         cfg.models.merger: merger})

        q = Query(text="pick", language=Language.ENGLISH, database_id="d")
        result = MultiAgentOrchestrator(cfg).run(
            {"query": q, "schema": schema, "db_path": db, "database_id": "d"})

        assert result["sql"].text == "SELECT x FROM t WHERE x=2"   # merge overrode
        assert merger.calls, "merge model was never invoked"
        # the merge prompt must carry the draft + execution-result block
        assert "【Execution result】" in merger.calls[0]["prompt"]
    finally:
        os.unlink(db)


def test_chunked_sampling_oom_backoff():
    """_sample_chunked halves the chunk on CUDA OOM and still delivers n samples."""
    import types as _types

    # torch stub with a cuda namespace the backoff path touches.
    torch_stub = sys.modules["torch"]
    class _OOM(Exception):
        pass
    torch_stub.cuda = _types.SimpleNamespace(
        OutOfMemoryError=_OOM, empty_cache=lambda: None)

    from generator.slm_generator import SLMGenerator

    gen = SLMGenerator.__new__(SLMGenerator)          # no model load
    gen.config = _types.SimpleNamespace(chunk_size=4)
    calls = []

    def fake_run(inputs, max_tokens, do_sample, temperature, num_return_sequences=1):
        calls.append(num_return_sequences)
        if num_return_sequences > 2:                  # batches >2 "don't fit"
            raise _OOM()
        return [f"SELECT {len(calls)}"] * num_return_sequences

    gen._run_generation = fake_run
    out = SLMGenerator._sample_chunked(gen, None, n=6, max_tokens=64, temperature=0.8)
    assert len(out) == 6                              # all samples delivered
    assert calls[0] == 4 and max(calls[1:]) <= 2      # halved after the OOM


def test_ensemble_pools_and_dedupes():
    db = _make_db()
    try:
        schema = _schema()
        cfg = _config("ensemble")
        cfg.csc.enabled = False
        cfg.generation.remote_candidates = 2
        cfg.generation.remote_strategies = ["direct"]
        # local emits A; remote emits A (dup) + B  -> pool = [A, B]; A wins 2-way tie? no:
        # dedupe keeps one A; vote: A returns {x=1}, B returns {x=2}; equal 1-1 -> first group wins.
        gen = _FakeSLM([["SELECT x FROM t WHERE x=1"]])
        llm = _FakeLLM(["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x=2"])
        _mc._cache = _FakeCache(schema, {cfg.models.generator: gen}, llm=llm)

        q = Query(text="pool", language=Language.ENGLISH, database_id="d")
        result = MultiAgentOrchestrator(cfg).run(
            {"query": q, "schema": schema, "db_path": db, "database_id": "d"})

        assert result["generation_source"] == "ensemble"
        assert result["routing_decision"] == RoutingDecision.REMOTE
        assert result["cost_usd"] > 0                     # local const + remote tokens
        assert result["sql"].text.startswith("SELECT x FROM t")
    finally:
        os.unlink(db)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

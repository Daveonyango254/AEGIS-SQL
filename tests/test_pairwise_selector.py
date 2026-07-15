"""Offline tests for the trained pairwise selector (no torch/model needed).

Exercises the prompt format (must stay in sync with the notebook), the execution
preview against a temp SQLite, and the round-robin tournament with a scripted
comparison function. Run with: python tests/test_pairwise_selector.py
"""

import contextlib
import os
import sqlite3
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Importing the `agents` package pulls in the model stack; stub the heavy deps so
# this runs offline (the tournament is exercised via a scripted comparison).
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
_wf = types.ModuleType("workflow"); _wf.__path__ = []
sys.modules["workflow"] = _wf
_cost = types.ModuleType("workflow.costing")
_cost.compute_cost = lambda source, tok, rc, lc: (tok * rc if source == "llm" else lc)
sys.modules["workflow.costing"] = _cost
_mc = types.ModuleType("workflow.model_cache")
_mc._cache = None
_mc.get_cache = lambda: _mc._cache
sys.modules["workflow.model_cache"] = _mc

from agents.pairwise_selector import (  # noqa: E402
    SELECTOR_SYSTEM, PairwiseSelector, build_selector_prompt, execute_preview,
    judge_fn_compare, parse_ab, tournament_select,
)


def test_prompt_format_matches_notebook():
    p = build_selector_prompt(
        question="How many schools?", evidence="ss refers to state special",
        schema_block="CREATE TABLE schools (x INTEGER)",
        sql_a="SELECT 1", prev_a="[(1,)]", sql_b="SELECT 2", prev_b="[(2,)]",
    )
    assert "Database schema:" in p
    assert "CREATE TABLE schools" in p
    assert "ss refers to state special" in p and "How many schools?" in p
    assert "Candidate A:" in p and "Candidate B:" in p
    assert "Execution result A:" in p and "Execution result B:" in p
    assert p.rstrip().endswith("Reply with exactly A or B.")
    assert "expert SQL judge" in SELECTOR_SYSTEM


def test_execute_preview_rows_and_errors():
    fd, path = tempfile.mkstemp(suffix=".sqlite"); os.close(fd)
    try:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
        conn.commit(); conn.close()
        assert "1" in execute_preview(path, "SELECT x FROM t ORDER BY x")
        assert execute_preview(path, "SELECT x FROM nope").startswith("ERROR")
        assert execute_preview(path, "SELECT x FROM t WHERE x=999") == "(0 rows)"
        assert execute_preview(path, "") == "ERROR: empty query"
    finally:
        os.unlink(path)


class _ScriptedSelector(PairwiseSelector):
    """Overrides model loading + comparison so the tournament is testable offline.

    ``_compare`` returns 'A' whenever the winner SQL is on side A (and 'B' when
    the winner is on side B), so ``winner`` should sweep the round-robin.
    """

    def __init__(self, winner, **kw):
        super().__init__("fake/model", **kw)
        self.winner = winner

    def _ensure(self):
        self._model = object()  # non-None so select() proceeds
        return True

    def _compare(self, question, evidence, schema_block, sql_a, prev_a, sql_b, prev_b):
        if sql_a == self.winner:
            return "A"
        if sql_b == self.winner:
            return "B"
        return "A"  # arbitrary for pairs not involving the winner


def _db():
    fd, path = tempfile.mkstemp(suffix=".sqlite"); os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit(); conn.close()
    return path


def test_tournament_picks_the_winner():
    db = _db()
    try:
        cands = ["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x=2",
                 "SELECT x FROM t WHERE x=3"]
        sel = _ScriptedSelector(winner=cands[2], max_candidates=4)
        picked = sel.select(cands, question="q", evidence="",
                            schema_block="CREATE TABLE t (x INTEGER)", db_path=db)
        assert picked == cands[2]      # swept the round-robin despite being last
    finally:
        os.unlink(db)


def test_single_candidate_short_circuits():
    sel = _ScriptedSelector(winner="x")
    assert sel.select(["only"], question="q", evidence="", schema_block="s", db_path=":memory:") == "only"
    # in-memory / no db path -> cannot execute previews -> None (caller keeps vote)
    assert sel.select(["a", "b"], question="q", evidence="", schema_block="s", db_path=":memory:") is None


def test_parse_ab():
    assert parse_ab("A") == "A" and parse_ab("B") == "B"
    assert parse_ab("The answer is b.") == "B"
    assert parse_ab("") == "A" and parse_ab("neither") == "A"  # default


def test_reuse_tournament_via_text_judge():
    """The no-training path: a text judge_fn drives the same tournament."""
    db = _db()
    try:
        cands = ["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x=2",
                 "SELECT x FROM t WHERE x=3"]
        winner = cands[1]

        # Fake judge_fn(prompt, n, temperature, system_prompt) -> [reply]; replies
        # "A" when the winner is Candidate A in the prompt, else "B".
        def judge_fn(prompt, n, temperature, system_prompt):
            a_block = prompt.split("Candidate A:")[1].split("Execution result")[0]
            return ["A" if winner.strip() in a_block else "B"]

        compare = judge_fn_compare(judge_fn, "q", "", "CREATE TABLE t (x INTEGER)")
        picked = tournament_select(
            cands, question="q", evidence="", schema_block="CREATE TABLE t (x INTEGER)",
            db_path=db, compare=compare, max_candidates=4,
        )
        assert picked == winner
    finally:
        os.unlink(db)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

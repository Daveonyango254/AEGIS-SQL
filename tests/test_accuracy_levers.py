"""Tests for the non-architectural accuracy levers (offline, no torch).

1. Reserved-word table quoting in sql_postprocess (fixes the recurring
   ``FROM order`` syntax-error class, q158/q164).
2. OOM-safe chunked sampling in SLMGenerator._sample_chunked (n preserved
   across chunks, distinct per-chunk seeds, OOM backoff halves the chunk and
   never silently drops the pool).

Run with: python tests/test_accuracy_levers.py
"""

import contextlib
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- stub torch (with a real OutOfMemoryError type) before importing generator --
_OOM = type("OutOfMemoryError", (RuntimeError,), {})
_torch = types.ModuleType("torch")
_torch.no_grad = contextlib.nullcontext
_torch.cuda = types.SimpleNamespace(
    OutOfMemoryError=_OOM,
    empty_cache=lambda: None,
    is_available=lambda: False,
    manual_seed_all=lambda s: None,
)
_torch.manual_seed = lambda s: None
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault(
    "transformers",
    types.ModuleType("transformers"),
)
sys.modules["transformers"].AutoTokenizer = object
sys.modules["transformers"].AutoModelForCausalLM = object

# generator/__init__ also pulls the remote-LLM client; stub the SDKs.
_Err = type("_Err", (Exception,), {})
for _name, _attrs in (
    ("openai", {"OpenAI": object, "RateLimitError": _Err,
                "APITimeoutError": _Err, "APIConnectionError": _Err}),
    ("anthropic", {"Anthropic": object, "RateLimitError": _Err,
                   "APITimeoutError": _Err, "APIConnectionError": _Err}),
):
    _m = types.ModuleType(_name)
    for _k, _v in _attrs.items():
        setattr(_m, _k, _v)
    sys.modules.setdefault(_name, _m)

from generator.sql_postprocess import finalize_sql, quote_reserved_tables  # noqa: E402
from generator.slm_generator import SLMGenerator  # noqa: E402


# --- reserved-word table quoting -----------------------------------------------

def test_quotes_from_order():
    assert quote_reserved_tables(
        "SELECT district_id FROM account WHERE account_id IN "
        "(SELECT account_id FROM order WHERE order_id = 33333)"
    ) == (
        "SELECT district_id FROM account WHERE account_id IN "
        "(SELECT account_id FROM `order` WHERE order_id = 33333)"
    )


def test_quotes_join_with_alias():
    out = quote_reserved_tables("SELECT * FROM trans T1 JOIN order AS T2 ON T1.a = T2.a")
    assert "JOIN `order` AS T2" in out
    assert "FROM trans" in out                      # non-reserved untouched


def test_already_quoted_untouched_and_idempotent():
    sql = "SELECT * FROM `order` WHERE order_id = 1"
    assert quote_reserved_tables(sql) == sql
    once = quote_reserved_tables("SELECT * FROM order")
    assert quote_reserved_tables(once) == once      # idempotent


def test_order_by_not_mangled():
    sql = "SELECT * FROM trans ORDER BY amount DESC"
    assert quote_reserved_tables(sql) == sql


def test_string_literals_untouched():
    sql = "SELECT * FROM trans WHERE note = 'shipped FROM order dept'"
    assert quote_reserved_tables(sql) == sql


def test_subquery_and_case_insensitive():
    assert quote_reserved_tables("SELECT * FROM (SELECT 1)") == "SELECT * FROM (SELECT 1)"
    assert quote_reserved_tables("select * from Match") == "select * from `Match`"


def test_finalize_sql_applies_quoting():
    assert finalize_sql("SELECT * FROM order", enable_cast_fix=False) == "SELECT * FROM `order`"


# --- chunked sampling ----------------------------------------------------------

def _make_generator(chunk_size=2, seed=42):
    """SLMGenerator instance without model loading (__init__ bypassed)."""
    g = object.__new__(SLMGenerator)
    g.config = types.SimpleNamespace(
        local_chunk_size=chunk_size, generation_seed=seed, max_tokens=64
    )
    return g


def test_chunks_preserve_n_and_use_distinct_seeds():
    g = _make_generator(chunk_size=2, seed=42)
    calls = []

    def fake_run(inputs, max_tokens, do_sample, temperature, num_return_sequences=1, seed=None):
        calls.append((num_return_sequences, seed))
        return [f"SQL_{seed}_{k}" for k in range(num_return_sequences)]

    g._run_generation = fake_run
    texts = g._sample_chunked(None, max_tokens=64, temperature=0.8, n_samples=5)
    assert len(texts) == 5                                   # n preserved across chunks
    assert [c[0] for c in calls] == [2, 2, 1]                # chunked 2+2+1
    assert [c[1] for c in calls] == [42, 44, 46]             # distinct per-chunk seeds
    assert len(set(texts)) == 5                              # no duplicated chunks


def test_unseeded_chunks_pass_none():
    g = _make_generator(chunk_size=3, seed=None)
    seeds = []
    g._run_generation = lambda *a, **k: (seeds.append(k.get("seed")), ["x"] * k["num_return_sequences"])[1]
    out = g._sample_chunked(None, max_tokens=64, temperature=0.8, n_samples=4)
    assert len(out) == 4 and seeds == [None, None]           # 3+1, unseeded


def test_oom_halves_chunk_and_recovers():
    g = _make_generator(chunk_size=4, seed=10)
    sizes = []

    def fake_run(inputs, max_tokens, do_sample, temperature, num_return_sequences=1, seed=None):
        sizes.append(num_return_sequences)
        if num_return_sequences > 1:
            raise _OOM("CUDA out of memory")
        return [f"s{seed}"]

    g._run_generation = fake_run
    texts = g._sample_chunked(None, max_tokens=64, temperature=0.8, n_samples=3)
    assert len(texts) == 3                                   # ALL samples recovered
    assert sizes[0] == 4 or sizes[0] == 3                    # first try at full chunk
    assert sizes[-1] == 1                                    # backed off to singles


def test_single_sequence_oom_returns_partial_not_raise():
    g = _make_generator(chunk_size=1, seed=None)
    produced = {"n": 0}

    def fake_run(inputs, max_tokens, do_sample, temperature, num_return_sequences=1, seed=None):
        if produced["n"] >= 2:
            raise _OOM("CUDA out of memory")
        produced["n"] += 1
        return ["ok"]

    g._run_generation = fake_run
    texts = g._sample_chunked(None, max_tokens=64, temperature=0.8, n_samples=5)
    assert texts == ["ok", "ok"]                             # partial pool, no exception


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

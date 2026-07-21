"""Tests for deterministic post-hoc literal repair (real temp SQLite, offline).

Covers the three measured near-miss classes (case, diacritics/one-letter,
datetime suffix), the conservative guarantees (exact match untouched, ambiguity
untouched, numerics/wildcards skipped), and alias resolution.

Run with: python tests/test_literal_repair.py
"""

import contextlib
import os
import sqlite3
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# literal_repair is stdlib-only, but importing it through the `generator`
# package triggers generator/__init__ (torch/openai imports) — stub them.
_Err = type("_Err", (Exception,), {})
for _name, _attrs in (
    ("torch", {"no_grad": contextlib.nullcontext}),
    ("transformers", {"AutoTokenizer": object, "AutoModelForCausalLM": object}),
    ("openai", {"OpenAI": object, "RateLimitError": _Err,
                "APITimeoutError": _Err, "APIConnectionError": _Err}),
    ("anthropic", {"Anthropic": object, "RateLimitError": _Err,
                   "APITimeoutError": _Err, "APIConnectionError": _Err}),
):
    _m = types.ModuleType(_name)
    for _k, _v in _attrs.items():
        setattr(_m, _k, _v)
    sys.modules.setdefault(_name, _m)

from generator.literal_repair import repair_literals  # noqa: E402


def _db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE set_translations (id INTEGER, language TEXT, translation TEXT)"
    )
    conn.execute("INSERT INTO set_translations VALUES (1, 'Portuguese (Brasil)', 'x')")
    conn.execute("INSERT INTO set_translations VALUES (2, 'German', 'y')")
    conn.execute("CREATE TABLE cards (id INTEGER, artist TEXT)")
    conn.execute("INSERT INTO cards VALUES (1, 'Volkan Baǵa')")   # Volkan Baǵa
    conn.execute("CREATE TABLE Laboratory (ID INTEGER, Date TEXT)")
    conn.execute("INSERT INTO Laboratory VALUES (1, '2013-02-22')")
    conn.execute("CREATE TABLE dup (v TEXT)")
    conn.executemany("INSERT INTO dup VALUES (?)", [("Aa",), ("AA",)])  # ambiguous nocase
    conn.commit()
    conn.close()
    return path


def test_one_letter_spelling_fixed():
    db = _db()
    try:
        out = repair_literals(
            "SELECT id FROM set_translations WHERE language = 'Portuguese (Brazil)'", db
        )
        assert "'Portuguese (Brasil)'" in out
    finally:
        os.unlink(db)


def test_diacritics_fixed_with_alias():
    db = _db()
    try:
        out = repair_literals(
            "SELECT T1.id FROM cards AS T1 WHERE T1.artist = 'Volkan Baga'", db
        )
        assert "Volkan Baǵa" in out
    finally:
        os.unlink(db)


def test_datetime_suffix_stripped():
    db = _db()
    try:
        out = repair_literals(
            "SELECT ID FROM Laboratory WHERE Date = '2013-02-22 00:00:00'", db
        )
        assert "= '2013-02-22'" in out
    finally:
        os.unlink(db)


def test_exact_match_untouched_and_idempotent():
    db = _db()
    try:
        sql = "SELECT id FROM set_translations WHERE language = 'German'"
        assert repair_literals(sql, db) == sql
        once = repair_literals(
            "SELECT id FROM set_translations WHERE language = 'german'", db
        )
        assert repair_literals(once, db) == once            # idempotent
        assert "'German'" in once                            # case fixed once
    finally:
        os.unlink(db)


def test_ambiguous_left_alone():
    db = _db()
    try:
        sql = "SELECT v FROM dup WHERE v = 'aa'"             # matches Aa AND AA
        assert repair_literals(sql, db) == sql
    finally:
        os.unlink(db)


def test_wildcards_numerics_and_unknown_columns_skipped():
    db = _db()
    try:
        for sql in (
            "SELECT id FROM cards WHERE artist LIKE '%Baga%'",   # wildcard pattern
            "SELECT id FROM cards WHERE id = '123'",             # numeric-looking
            "SELECT id FROM cards WHERE nope = 'Volkan Baga'",   # column not in table
        ):
            assert repair_literals(sql, db) == sql
    finally:
        os.unlink(db)


def test_no_db_or_no_literals_noop():
    assert repair_literals("SELECT 1", ":memory:") == "SELECT 1"
    assert repair_literals("SELECT 1 FROM t", "/nonexistent/x.sqlite") == "SELECT 1 FROM t"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

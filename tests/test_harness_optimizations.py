"""Standalone smoke tests for the harness-optimization core logic.

These exercise the three dependency-free modules (no torch / transformers /
pydantic required) against a synthetic SQLite database that mimics the BIRD
failure modes seen in the evaluation predictions:

  * integer-division ratio bug          -> sql_postprocess.apply_cast_fix
  * literal value mismatch              -> value_sampler.get_value_hints
  * pick the correct answer among many  -> candidate_selector.select_best

Run with the plain interpreter (no project deps needed):
    python tests/test_harness_optimizations.py
"""

import importlib.util
import os
import sqlite3
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, rel_path):
    """Load a module file directly, bypassing package __init__ (which pulls torch)."""
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sql_postprocess = _load("sql_postprocess", "generator/sql_postprocess.py")
value_sampler = _load("value_sampler", "retriever/value_sampler.py")
candidate_selector = _load("candidate_selector", "generator/candidate_selector.py")


# --------------------------------------------------------------------------- #
# Minimal SchemaElement stand-in (avoids importing aegis_types -> pydantic).
# --------------------------------------------------------------------------- #
class Elem:
    def __init__(self, name, data_type=None):
        self.name = name
        self.data_type = data_type
        self.example_values = []


def build_db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE frpm ("
        "id INTEGER, EdOption TEXT, FreeCount INTEGER, Enrollment INTEGER, County TEXT)"
    )
    rows = [
        (1, "Continuation School", 50, 200, "Alameda"),
        (2, "Continuation School", 30, 300, "Alameda"),
        (3, "District Community Day School", 10, 100, "Fresno"),
        (4, "County Community School", 5, 100, "Fresno"),
        (5, "Continuation School", 90, 100, "Alameda"),
    ]
    cur.executemany("INSERT INTO frpm VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_cast_fix():
    print("\n[1] sql_postprocess.apply_cast_fix")
    cases = [
        # (input, must_contain, must_not_break)
        (
            "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm",
            "CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`",
        ),
        (
            "SELECT T1.NumGE1500 / T1.NumTstTakr FROM satscores AS T1",
            "CAST(T1.NumGE1500 AS REAL) / T1.NumTstTakr",
        ),
        (
            "SELECT SUM(a) / COUNT(*) FROM t",
            "CAST(SUM(a) AS REAL) / COUNT(*)",
        ),
    ]
    for src, expected in cases:
        out = sql_postprocess.apply_cast_fix(src)
        check(f"wraps numerator: {src[:40]}...", expected in out, f"got: {out}")

    # Idempotent: already-cast stays untouched, double-apply is a no-op.
    already = "SELECT CAST(a AS REAL) / b FROM t"
    once = sql_postprocess.apply_cast_fix(already)
    twice = sql_postprocess.apply_cast_fix(once)
    check("idempotent on already-cast", once == already and twice == already, f"{once!r}")

    # No false-positives on non-division SQL.
    plain = "SELECT a, b FROM t WHERE a > 5"
    check("leaves non-division SQL alone", sql_postprocess.apply_cast_fix(plain) == plain)

    # Float division actually computed (integer truncation avoided).
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(a INTEGER, b INTEGER)")
    conn.execute("INSERT INTO t VALUES (1, 2)")
    int_res = conn.execute("SELECT a / b FROM t").fetchone()[0]
    real_res = conn.execute(
        sql_postprocess.apply_cast_fix("SELECT a / b FROM t")
    ).fetchone()[0]
    check("integer division -> 0, cast -> 0.5", int_res == 0 and abs(real_res - 0.5) < 1e-9,
          f"int={int_res} real={real_res}")
    conn.close()


def test_value_grounding(db_path):
    print("\n[2] value_sampler.get_value_hints")
    value_sampler.clear_cache()
    elems = [
        Elem("frpm.EdOption", "TEXT"),
        Elem("frpm.County", "TEXT"),
        Elem("frpm.FreeCount", "INTEGER"),  # numeric -> should be skipped
    ]
    # Question mentions "continuation" (lowercase) -> must surface the full literal.
    q = "How many continuation schools are in Alameda county?"
    hints = value_sampler.get_value_hints(db_path, elems, q)

    check("links 'continuation' -> 'Continuation School'",
          "Continuation School" in hints.get("frpm.EdOption", []),
          f"got: {hints.get('frpm.EdOption')}")
    check("links 'alameda' -> 'Alameda'",
          "Alameda" in hints.get("frpm.County", []),
          f"got: {hints.get('frpm.County')}")
    check("skips numeric column", "frpm.FreeCount" not in hints)


def test_candidate_selection(db_path):
    print("\n[3] candidate_selector.select_best")

    # Correct answer: COUNT continuation schools in Alameda = 3.
    correct = "SELECT COUNT(*) FROM frpm WHERE EdOption = 'Continuation School' AND County = 'Alameda'"
    wrong_literal = "SELECT COUNT(*) FROM frpm WHERE EdOption = 'Continuation' AND County = 'Alameda'"  # 0 rows-of-interest -> returns 0
    broken = "SELECT COUNT(*) FROM frpm WHERE NoSuchCol = 'x'"  # execution error

    # Greedy (first) is the broken one; two sampled candidates agree on the correct answer.
    candidates = [broken, correct, correct, wrong_literal]
    info = candidate_selector.select_best(candidates, db_path, timeout=5)
    check("drops erroring candidate", info["exec_ok"] is True, str(info))
    check("majority vote picks correct SQL", info["best_sql"] == correct, info["best_sql"])
    check("reports agreement >= 2", info["num_agree"] >= 2, str(info["num_agree"]))

    # All-broken -> falls back to greedy (first) candidate, exec_ok False.
    allbad = ["SELECT * FROM nope", "SELECT bad syntax ("]
    info2 = candidate_selector.select_best(allbad, db_path, timeout=5)
    check("all-fail falls back to greedy", info2["best_sql"] == allbad[0] and not info2["exec_ok"], str(info2))

    # Combined: cast-fixed ratio executes and is selected over a non-executing variant.
    ratio_raw = "SELECT FreeCount / Enrollment FROM frpm WHERE id = 1"
    ratio_fixed = sql_postprocess.apply_cast_fix(ratio_raw)
    val = sqlite3.connect(db_path).execute(ratio_fixed).fetchone()[0]
    check("cast-fixed ratio computes 0.25 not 0", abs(val - 0.25) < 1e-9, f"got {val}")


if __name__ == "__main__":
    db = build_db()
    try:
        test_cast_fix()
        test_value_grounding(db)
        test_candidate_selection(db)
    finally:
        os.remove(db)

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    raise SystemExit(1 if FAIL else 0)

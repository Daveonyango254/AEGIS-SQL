"""Unit tests for the shared SQL extractor (generator/sql_postprocess.extract_sql).

Guards the CoT-extraction fix: the previous LLM extractor only stripped fences at
the start/end of the text, so a chain-of-thought response (reasoning first, then a
fenced ```sql block) returned the whole essay as "SQL" and every CoT candidate was
silently wasted.
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "sql_postprocess", os.path.join(ROOT, "generator/sql_postprocess.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def test_reasoning_then_fenced_block():
    out = (
        "1. We need schools and satscores joined on cds.\n"
        "2. Filter county = 'Alameda'.\n"
        "```sql\nSELECT AvgScrMath FROM satscores JOIN schools ON cds = CDSCode;\n```"
    )
    got = M.extract_sql(out)
    assert got.startswith("SELECT AvgScrMath")
    assert "We need" not in got


def test_last_block_wins_over_partial_snippets():
    out = (
        "First a snippet:\n```sql\nSELECT 1;\n```\nThen the final query:\n"
        "```sql\nSELECT name FROM t WHERE x = 2;\n```"
    )
    assert "x = 2" in M.extract_sql(out)


def test_truncated_fence_falls_back_to_line_scan():
    # Model ran out of tokens mid-answer: opening fence, no closing fence.
    out = "Reasoning about joins...\n```sql\nSELECT a FROM t WHERE b = 1"
    got = M.extract_sql(out)
    assert got.startswith("SELECT a FROM t")


def test_direct_sql_first_with_trailing_explanation():
    out = "SELECT COUNT(*) FROM schools;\nExplanation: counts all schools."
    got = M.extract_sql(out)
    assert got == "SELECT COUNT(*) FROM schools;"


def test_prose_select_line_loses_to_real_query():
    # A CoT step line starting with SELECT (no FROM) must not beat the real query.
    out = (
        "SELECT the relevant columns first\n"
        "Then join.\n"
        "SELECT name FROM client WHERE id = 3;"
    )
    got = M.extract_sql(out)
    assert "FROM client" in got


def test_judge_style_reply_yields_empty():
    # Non-SQL replies (the selection judge answers "2") must not be coerced.
    assert M.extract_sql("2") == ""
    assert M.extract_sql("The best candidate is number 3.") == ""


def test_bare_fenced_sql():
    assert M.extract_sql("```sql\nSELECT x FROM t\n```") == "SELECT x FROM t"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

"""Unit tests for the generation strategies and judge prompt (dependency-free)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis_types import ForeignKey, Language, Query, Schema, SchemaElement
from prompts import sql_strategies as S


def _fixture():
    cols = [
        SchemaElement(element_type="column", name="schools.CDSCode", data_type="TEXT"),
        SchemaElement(element_type="column", name="satscores.cds", data_type="TEXT"),
        SchemaElement(element_type="column", name="satscores.AvgScrMath", data_type="INTEGER"),
    ]
    schema = Schema(
        database_id="ca", tables=["schools", "satscores"], columns=cols,
        foreign_keys=[ForeignKey(from_table="satscores", from_column="cds",
                                 to_table="schools", to_column="CDSCode")],
        primary_keys={"schools": ["CDSCode"]},
    )
    query = Query(text="average math score in Alameda?", language=Language.ENGLISH,
                  database_id="ca", evidence="K-12 means kindergarten to 12th grade")
    return query, cols, schema


def test_direct_uses_default_system_prompt():
    query, cols, schema = _fixture()
    system, user = S.build_prompt(S.DIRECT, query, cols, schema=schema)
    assert system is None                      # direct => generator's default system prompt
    assert "CREATE TABLE schools" in user
    assert query.text in user


def test_query_plan_has_cot_scaffold_and_context():
    query, cols, schema = _fixture()
    system, user = S.build_prompt(S.QUERY_PLAN, query, cols, schema=schema)
    assert system and "```sql" in system       # reasoning-friendly system prompt
    assert "execution plan" in user.lower()     # plan-first instructions
    assert "FOREIGN KEY RELATIONSHIPS" in user  # join keys exposed
    assert query.text in user
    assert query.evidence in user               # evidence threaded in


def test_divide_and_conquer_scaffold():
    query, cols, schema = _fixture()
    _, user = S.build_prompt(S.DIVIDE_AND_CONQUER, query, cols, schema=schema)
    assert "divide and conquer" in user.lower()
    assert "```sql" in user


def test_unknown_strategy_raises():
    query, cols, schema = _fixture()
    try:
        S.build_prompt("nonsense", query, cols, schema=schema)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_judge_prompt_lists_candidates_and_asks_for_number():
    query, cols, schema = _fixture()
    system, user = S.build_judge_prompt(
        query, "CREATE TABLE schools (...)", ["SELECT 1", "SELECT 2", "SELECT 3"],
        result_previews=["(1,)", "(2,)", None],
    )
    assert "number" in system.lower()
    assert "[1]" in user and "[2]" in user and "[3]" in user
    assert "result: (1,)" in user               # execution preview shown to judge
    assert "1-3" in user


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

"""Unit tests for the M-Schema prompt builder (prompts/m_schema.py).

M-Schema is the native schema serialization SQL-specialist models are trained
on; feeding it (full schema, types, PK, FK, example values) is what lets the
specialist model reach its benchmark accuracy. Loaded directly so the test
needs no torch / pydantic.

Run with: python tests/test_m_schema.py
"""

import importlib.util
import os
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m_schema", os.path.join(ROOT, "prompts/m_schema.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _col(name, dtype, desc=None, examples=None):
    return SimpleNamespace(
        name=name, data_type=dtype, description=desc, example_values=examples or []
    )


def _fk(ft, fc, tt, tc):
    return SimpleNamespace(from_table=ft, from_column=fc, to_table=tt, to_column=tc)


ELEMENTS = [
    _col("schools.CDSCode", "TEXT", "school id", ["01100170109835"]),
    _col("schools.School", "TEXT", "school name", ["Alameda High"]),
    _col("schools.StatusType", "TEXT", None, ["Active", "Closed", "Merged"]),
    _col("frpm.CDSCode", "TEXT"),
    _col("frpm.Enrollment (K-12)", "REAL", None, [1087.0]),
]
PKS = {"schools": ["CDSCode"], "frpm": ["CDSCode"]}
FKS = [_fk("frpm", "CDSCode", "schools", "CDSCode")]


def test_m_schema_structure():
    m = _load()
    out = m.build_m_schema("california_schools", ELEMENTS, FKS, PKS)
    assert "【DB_ID】 california_schools" in out
    assert "# Table: schools" in out and "# Table: frpm" in out
    # column with type, PK marker, and examples
    assert "(CDSCode:TEXT, school id, Primary Key, Examples: ['01100170109835'])" in out
    # categorical examples surfaced (value grounding)
    assert "Examples: ['Active', 'Closed', 'Merged']" in out
    # foreign key block
    assert "【Foreign keys】" in out
    assert "frpm.CDSCode = schools.CDSCode" in out
    # numeric example is unquoted
    assert "Examples: [1087.0]" in out


def test_m_schema_prompt_matches_native_template():
    m = _load()
    q = "How many active schools are there?"
    p = m.build_m_schema_prompt(
        "california_schools", ELEMENTS,
        question=q,
        evidence="active means StatusType = 'Active'",
        foreign_keys=FKS, primary_keys=PKS,
    )
    # Native specialist structure: dialect-expert instruction, question FIRST and
    # REPEATED, schema + evidence sections, terminal ```sql fence.
    assert "SQLite expert" in p
    # Two 【User Question】 section headers (the instruction also names it once).
    assert p.count("【User Question】") == 3
    assert p.count(q) == 2  # the question text itself is repeated
    assert "【Database Schema】" in p and "# Table: schools" in p
    assert "【Evidence】" in p and "StatusType = 'Active'" in p
    assert p.rstrip().endswith("```sql")
    # question appears before the schema body (question-first ordering)
    assert p.index(q) < p.index("# Table: schools")
    # no few-shot scaffolding for the specialist
    assert "Example 1" not in p and "CREATE TABLE" not in p


if __name__ == "__main__":
    test_m_schema_structure()
    test_m_schema_prompt_matches_native_template()
    print("All M-Schema tests passed.")

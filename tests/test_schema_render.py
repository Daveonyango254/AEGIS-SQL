"""Unit tests for the shared DDL schema renderer (dependency-free)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis_types import ForeignKey, Schema, SchemaElement
from prompts.schema_render import render_schema_ddl


def _schema():
    cols = [
        SchemaElement(element_type="column", name="schools.CDSCode", data_type="TEXT",
                      description="school id", example_values=["01100170109835"]),
        SchemaElement(element_type="column", name="schools.County Name", data_type="TEXT",
                      description="county", example_values=["Alameda", "Kern"]),
        SchemaElement(element_type="column", name="satscores.cds", data_type="TEXT"),
        SchemaElement(element_type="column", name="satscores.AvgScrMath", data_type="INTEGER"),
    ]
    schema = Schema(
        database_id="ca", tables=["schools", "satscores"], columns=cols,
        foreign_keys=[ForeignKey(from_table="satscores", from_column="cds",
                                 to_table="schools", to_column="CDSCode")],
        primary_keys={"schools": ["CDSCode"]},
    )
    return cols, schema


def test_create_tables_and_keys():
    cols, schema = _schema()
    schema_str, fk_hint, tables = render_schema_ddl(cols, schema=schema, expose_keys=True)
    assert "CREATE TABLE schools" in schema_str
    assert "CREATE TABLE satscores" in schema_str
    assert "CDSCode TEXT PRIMARY KEY" in schema_str          # PK marker
    assert "`County Name`" in schema_str                      # backticked (space)
    assert "examples: 'Alameda', 'Kern'" in schema_str        # value grounding
    assert "satscores.cds = schools.CDSCode" in fk_hint        # FK join hint
    assert tables == ["schools", "satscores"]


def test_expose_keys_false_hides_keys():
    cols, schema = _schema()
    schema_str, fk_hint, _ = render_schema_ddl(cols, schema=schema, expose_keys=False)
    assert "PRIMARY KEY" not in schema_str
    assert fk_hint == ""


def test_fk_filtered_to_retrieved_tables():
    # Only retrieve satscores columns -> FK to schools must be dropped (schools absent).
    cols, schema = _schema()
    only_sat = [c for c in cols if c.name.startswith("satscores.")]
    _, fk_hint, tables = render_schema_ddl(only_sat, schema=schema, expose_keys=True)
    assert tables == ["satscores"]
    assert fk_hint == ""  # dangling FK to a non-retrieved table is not shown


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")

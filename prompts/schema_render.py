"""Dependency-free rendering of a retrieved schema slice into a DDL prompt block.

This is the single source of truth for how AEGIS turns a list of retrieved
``SchemaElement`` columns (plus the populated ``Schema`` for keys) into the
``CREATE TABLE`` text that every generation strategy shows the model. Factoring
it out keeps the local SLM prompt, the remote LLM prompt, and the multi-agent
reasoning strategies byte-for-byte consistent, so a candidate's quality reflects
the *strategy*, not an accidental difference in schema formatting.

Pure stdlib — safe to import and unit-test without torch/transformers.
"""

from typing import List, Tuple


def _needs_backticks(col_name: str) -> bool:
    """SQLite identifiers with spaces/parens/hyphens must be quoted."""
    return " " in col_name or "(" in col_name or "-" in col_name


def render_schema_ddl(
    schema_elements: List,
    schema=None,
    expose_keys: bool = True,
    max_example_values: int = 8,
) -> Tuple[str, str, List[str]]:
    """Render the retrieved columns as ``CREATE TABLE`` statements + FK hints.

    Args:
        schema_elements: retrieved columns; each has ``.name`` ("table.column"),
            ``.data_type``, ``.description`` and ``.example_values``.
        schema: the populated ``Schema`` (for real ``foreign_keys`` / ``primary_keys``).
            ``query.schema`` is never populated, so keys must come from here.
        expose_keys: emit ``PRIMARY KEY`` markers + explicit FK join hints. These
            target the multi-table JOIN reasoning the paper names as the bottleneck.
        max_example_values: cap on grounded value hints shown per column.

    Returns:
        ``(schema_str, fk_hint_str, table_names)`` where ``schema_str`` is the
        CREATE TABLE block, ``fk_hint_str`` is the explicit JOIN-key block (may be
        empty), and ``table_names`` is the ordered list of tables in the slice.
    """
    # Group retrieved columns by their owning table, preserving first-seen order.
    tables = {}
    for elem in schema_elements:
        if "." not in elem.name:
            continue
        table, _ = elem.name.split(".", 1)
        tables.setdefault(table, []).append(elem)

    # Real keys come from the populated Schema, filtered to the retrieved slice.
    primary_keys = (getattr(schema, "primary_keys", None) or {}) if expose_keys else {}
    fk_relationships = []
    if expose_keys and getattr(schema, "foreign_keys", None):
        table_set = set(tables.keys())
        # Only keep FKs whose BOTH endpoints are in the slice — a join hint that
        # references a table the model can't see is noise.
        for fk in schema.foreign_keys:
            if fk.from_table in table_set and fk.to_table in table_set:
                fk_relationships.append(fk)

    # CREATE TABLE statements, one column per line with inline comments.
    schema_str = ""
    for table, cols in tables.items():
        pk_cols = set(primary_keys.get(table, []))
        schema_str += f"CREATE TABLE {table} (\n"
        for col in cols:
            raw_col = col.name.split(".", 1)[1]
            col_name = f"`{raw_col}`" if _needs_backticks(raw_col) else raw_col
            col_type = col.data_type if col.data_type else "TEXT"
            schema_str += f"  {col_name} {col_type}"
            if raw_col in pk_cols:
                schema_str += " PRIMARY KEY"
            # Inline comment: human description + grounded example values. Value
            # hints anchor exact string literals (e.g. 'Continuation School').
            comment_parts = []
            if col.description:
                comment_parts.append(col.description)
            if getattr(col, "example_values", None):
                vals = ", ".join(f"'{v}'" for v in col.example_values[:max_example_values])
                comment_parts.append(f"examples: {vals}")
            if comment_parts:
                schema_str += " -- " + " | ".join(comment_parts)
            schema_str += ",\n"
        schema_str = schema_str.rstrip(",\n") + "\n);\n\n"

    # Explicit foreign-key relationships as JOIN hints.
    fk_hint_str = ""
    if fk_relationships:
        fk_hint_str = "-- FOREIGN KEY RELATIONSHIPS (Use these for JOINs):\n"
        for fk in fk_relationships:
            fk_hint_str += (
                f"--   {fk.from_table}.{fk.from_column} "
                f"= {fk.to_table}.{fk.to_column}\n"
            )
        fk_hint_str += "\n"

    return schema_str, fk_hint_str, list(tables.keys())

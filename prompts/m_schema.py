"""M-Schema prompt construction for SQL-specialist models (XiYanSQL / CscSQL).

CscSQL-Merge-Qwen2.5-Coder and the XiYanSQL-QwenCoder family it is built on were
trained/evaluated with the **M-Schema** schema serialization, not raw `CREATE
TABLE` DDL. Feeding such a specialist its native format (full schema, types,
primary keys, foreign keys, and example values) — without few-shot crutches —
is what unlocks its benchmark-level accuracy.

M-Schema layout (per XiYan-SQL, arXiv:2411.08599)::

    【DB_ID】 my_db
    【Schema】
    # Table: table_a
    [
    (col1:INTEGER, the id, Primary Key, Examples: [1, 2, 3]),
    (col2:TEXT, a description, Examples: ['Active', 'Closed']),
    ]
    # Table: table_b
    [
    (...),
    ]
    【Foreign keys】
    table_b.a_id = table_a.col1

This module is dependency-free (duck-typed over the SchemaElement / ForeignKey
attributes) so it can be unit-tested without torch / pydantic.
"""

from typing import Dict, Iterable, List, Optional


def _column_name(full_name: str) -> str:
    """Return the bare column name from a ``"table.column"`` element name."""
    return full_name.split(".", 1)[1] if "." in full_name else full_name


def _table_name(full_name: str) -> str:
    return full_name.split(".", 1)[0] if "." in full_name else ""


def _format_examples(values: Iterable, limit: int = 3) -> str:
    """Format up to ``limit`` example values, quoting non-numeric ones."""
    out = []
    for v in list(values)[:limit]:
        s = str(v).strip()
        if not s:
            continue
        # Numbers unquoted; everything else single-quoted (collapse inner quotes).
        if _looks_numeric(s):
            out.append(s)
        else:
            out.append("'" + s.replace("'", "") + "'")
    return ", ".join(out)


def _looks_numeric(s: str) -> bool:
    # Leading-zero codes (e.g. '01100170109835', '007') are identifiers, not
    # numbers — keep them quoted so the model treats them as string literals.
    if len(s) > 1 and s[0] == "0" and s[1].isdigit():
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_m_schema(
    db_id: str,
    schema_elements: List,
    foreign_keys: Optional[List] = None,
    primary_keys: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Render schema elements as an M-Schema string.

    Args:
        db_id: Database identifier.
        schema_elements: Iterable of objects with ``name`` ("table.column"),
            ``data_type``, ``description`` and ``example_values`` attributes.
        foreign_keys: Objects with ``from_table/from_column/to_table/to_column``.
        primary_keys: ``{table: [pk_col, ...]}``.

    Returns:
        The M-Schema representation as a single string.
    """
    primary_keys = primary_keys or {}
    foreign_keys = foreign_keys or []

    # Group columns by table, preserving first-seen order.
    tables: "Dict[str, list]" = {}
    for el in schema_elements:
        name = getattr(el, "name", "")
        if "." not in name:
            continue
        tables.setdefault(_table_name(name), []).append(el)

    lines: List[str] = [f"【DB_ID】 {db_id}", "【Schema】"]
    for table, cols in tables.items():
        pk_cols = set(primary_keys.get(table, []))
        lines.append(f"# Table: {table}")
        lines.append("[")
        for i, el in enumerate(cols):
            col = _column_name(getattr(el, "name", ""))
            dtype = (getattr(el, "data_type", None) or "TEXT")
            parts = [f"{col}:{dtype}"]
            desc = getattr(el, "description", None)
            if desc:
                parts.append(str(desc).strip().replace("\n", " "))
            if col in pk_cols:
                parts.append("Primary Key")
            examples = _format_examples(getattr(el, "example_values", []) or [])
            if examples:
                parts.append(f"Examples: [{examples}]")
            comma = "," if i < len(cols) - 1 else ""
            lines.append(f"({', '.join(parts)}){comma}")
        lines.append("]")

    if foreign_keys:
        fk_lines = []
        seen = set()
        for fk in foreign_keys:
            key = (
                f"{fk.from_table}.{fk.from_column} = {fk.to_table}.{fk.to_column}"
            )
            if key not in seen:
                seen.add(key)
                fk_lines.append(key)
        if fk_lines:
            lines.append("【Foreign keys】")
            lines.extend(fk_lines)

    return "\n".join(lines)


def build_m_schema_prompt(
    db_id: str,
    schema_elements: List,
    question: str,
    evidence: str = "",
    foreign_keys: Optional[List] = None,
    primary_keys: Optional[Dict[str, List[str]]] = None,
    dialect: str = "SQLite",
) -> str:
    """Build the full native prompt body in the XiYanSQL/CscSQL template.

    Mirrors the model's documented ``nl2sqlite`` template exactly: a dialect-expert
    instruction, the question FIRST, then the M-Schema body and evidence, then the
    question REPEATED, and a terminal ```sql fence that primes a clean code block.
    These structural details are load-bearing for the specialist model.

    Deliberately minimal — no few-shot examples or verbose instructions.
    """
    m_schema = build_m_schema(db_id, schema_elements, foreign_keys, primary_keys)
    q = question.strip()
    ev = (evidence or "").strip()

    sections = [
        f"You are a {dialect} expert. Read and understand the 【Database Schema】 "
        f"and the 【Evidence】, then use {dialect} knowledge to generate a single "
        f"executable SQL query that answers the 【User Question】.",
        "",
        "【User Question】",
        q,
        "【Database Schema】",
        m_schema,
        "【Evidence】",
        ev if ev else "None",
        "【User Question】",
        q,
        "```sql",
    ]
    return "\n".join(sections)

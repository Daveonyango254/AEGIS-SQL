"""OmniSQL / CSC-SQL native prompt templates (dependency-free).

The CscSQL Qwen2.5-Coder checkpoints were trained on the OmniSQL template — NOT
our DDL-few-shot prompt and NOT XiYanSQL M-Schema. Reproducing their published
accuracy (7B: 69.19% BIRD-dev) requires byte-level fidelity to three things,
all taken from the CSC-SQL reference implementation (CycloneBoy/csc_sql):

1. The OmniSQL generation prompt: full-schema DDL with per-column comments and
   example values, evidence PREPENDED to the question, instructions block, and a
   reasoning wrapper that asks for ``<think>…</think><answer>SQL</answer>``.
2. The merge-revision prompt: same skeleton, task line swapped, the two draft
   SQLs + their (normalized) execution results inserted before ``Instructions:``,
   and the schema REDUCED to the tables referenced by the two drafts.
3. Result normalization: execution results shown to the merge model are capped
   (>20 rows summarized, >1000 chars truncated).

Everything here is pure string assembly over our ``Schema``/``SchemaElement``
types — unit-testable without torch.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Generation prompt (OmniSQL) + think wrapper
# ---------------------------------------------------------------------------

_OMNISQL_TEMPLATE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
"""

# The GRPO checkpoints emit <think>…</think><answer>SELECT …</answer>; this
# instruction (appended by CSC-SQL's `make_prefix` in 'think' mode) elicits it.
_THINK_SUFFIX = (
    "\nShow your work in <think> </think> tags. And return the final SQLite SQL "
    "query that starts with keyword `SELECT` in <answer> </answer> tags, for "
    "example <answer>SELECT AVG(rating_score) FROM movies</answer>."
)

_MERGE_TASK_LINE = (
    "You are a data science expert. Below, you are provided with a database "
    "schema, a natural language question, some draft SQL and its corresponding "
    "execution result. Your task is to understand the schema and generate a "
    "valid SQL query to answer the question."
)

_GEN_TASK_LINE = (
    "You are a data science expert. Below, you are provided with a database "
    "schema and a natural language question. Your task is to understand the "
    "schema and generate a valid SQL query to answer the question."
)


def _quote_ident(name: str) -> str:
    """Backtick-quote identifiers that need it (spaces/parens/hyphens)."""
    if re.search(r"[^\w]", name):
        return f"`{name}`"
    return name


def build_db_details(
    schema,
    schema_elements: Optional[Sequence] = None,
    max_examples: int = 3,
) -> str:
    """Render the schema as OmniSQL-style DDL with comments and example values.

    Args:
        schema: the populated ``Schema`` (tables, columns, FKs, PKs).
        schema_elements: optional column subset (linked mode); ``None`` renders
            the FULL schema (the CSC-SQL 69.19% setting — table linking off).
        max_examples: example values shown per column comment.
    """
    elements = list(schema_elements) if schema_elements else [
        c for c in schema.columns if "." in c.name
    ]
    by_table: Dict[str, List] = {}
    for e in elements:
        if "." not in e.name:
            continue
        t, _ = e.name.split(".", 1)
        by_table.setdefault(t, []).append(e)

    pks = getattr(schema, "primary_keys", None) or {}
    fks = getattr(schema, "foreign_keys", None) or []
    parts: List[str] = []
    for table, cols in by_table.items():
        lines = []
        for col in cols:
            cname = col.name.split(".", 1)[1]
            ctype = col.data_type or "TEXT"
            comment_bits = [col.description] if col.description else [cname]
            if getattr(col, "example_values", None):
                vals = ", ".join(str(v) for v in col.example_values[:max_examples])
                comment_bits.append(f"example: [{vals}]")
            lines.append(f"    {_quote_ident(cname)} {ctype}, -- {' | '.join(comment_bits)}")
        # PRIMARY KEY clause
        table_pks = [c for c in pks.get(table, []) if any(
            col.name.split(".", 1)[1] == c for col in cols)]
        if table_pks:
            lines.append(f"    PRIMARY KEY ({', '.join(_quote_ident(c) for c in table_pks)}),")
        # FOREIGN KEY clauses (only when both endpoints are rendered)
        rendered = set(by_table.keys())
        for fk in fks:
            if fk.from_table == table and fk.to_table in rendered:
                lines.append(
                    f"    FOREIGN KEY ({_quote_ident(fk.from_column)}) REFERENCES "
                    f"{_quote_ident(fk.to_table)}({_quote_ident(fk.to_column)}),"
                )
        body = "\n".join(lines).rstrip(",")
        parts.append(f"CREATE TABLE {_quote_ident(table)} (\n{body}\n);")
    return "\n\n".join(parts)


def build_generation_prompt(
    question: str,
    evidence: str,
    db_details: str,
    think: bool = True,
) -> str:
    """The OmniSQL generation prompt; evidence is PREPENDED to the question."""
    q = f"{evidence.strip()}\n{question.strip()}" if (evidence or "").strip() else question.strip()
    prompt = _OMNISQL_TEMPLATE.format(db_details=db_details, question=q)
    return prompt + _THINK_SUFFIX if think else prompt


# ---------------------------------------------------------------------------
# Merge-revision prompt (the merge checkpoint's trained distribution)
# ---------------------------------------------------------------------------

def normalize_execution_result(rows, max_rows: int = 20, max_chars: int = 1000) -> str:
    """Normalize an execution result for display in the merge prompt.

    Mirrors CSC-SQL's ``normal_execute_result``: long result sets are summarized
    to their first rows, and the final string is hard-capped.
    """
    if rows is None:
        return "Execution error"
    rows = list(rows)
    if len(rows) > max_rows:
        text = (
            "The execution results of the first twenty columns are: "
            f"{rows[:max_rows]}"
        )
    else:
        text = str(rows)
    return text[:max_chars]


def referenced_tables(sql: str) -> List[str]:
    """Table names referenced by a SQL string (FROM/JOIN clauses)."""
    return list(dict.fromkeys(
        m.group(1) for m in re.finditer(
            r"(?:from|join)\s+`?([A-Za-z_]\w*)`?", sql or "", re.IGNORECASE)
    ))


def build_reduced_db_details(schema, candidate_sqls: Sequence[str], max_examples: int = 3) -> str:
    """DDL reduced to the tables referenced by the candidate SQLs.

    CSC-SQL passes the merge model only the schema slice the two drafts touch;
    falls back to the full schema if nothing can be parsed out of the drafts.
    """
    tables = set()
    for sql in candidate_sqls:
        tables.update(t.lower() for t in referenced_tables(sql))
    if not tables:
        return build_db_details(schema, None, max_examples)
    elements = [
        c for c in schema.columns
        if "." in c.name and c.name.split(".", 1)[0].lower() in tables
    ]
    if not elements:
        return build_db_details(schema, None, max_examples)
    return build_db_details(schema, elements, max_examples)


def build_merge_prompt(
    question: str,
    evidence: str,
    schema,
    candidates: Sequence[Tuple[str, object]],
    think: bool = True,
    max_sql_chars: int = 500,
) -> str:
    """The CSC merge-revision prompt: drafts + execution results before Instructions.

    Args:
        candidates: ``[(sql, execution_rows_or_None), ...]`` — normally the
            top-2 vote groups' representative SQLs with their result sets.
    """
    reduced = build_reduced_db_details(schema, [sql for sql, _ in candidates])
    base = build_generation_prompt(question, evidence, reduced, think=False)
    base = base.replace(_GEN_TASK_LINE, _MERGE_TASK_LINE, 1)

    blocks = []
    for i, (sql, rows) in enumerate(candidates):
        blocks.append(
            f"{i + 1}. {sql[:max_sql_chars]}\n【Execution result】\n"
            f"{normalize_execution_result(rows)}\n"
        )
    drafts = (
        "Here are some corresponding draft SQL and execute result: \n"
        + "\n".join(blocks)
        + "\nInstructions:"
    )
    merged = base.replace("Instructions:", drafts, 1)
    return merged + _THINK_SUFFIX if think else merged

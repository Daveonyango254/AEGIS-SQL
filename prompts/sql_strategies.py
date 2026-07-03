"""Reasoning strategies for diverse candidate generation (the booster's core).

Diversity is what makes execution-guided selection work: if every candidate is
generated the same way, a majority vote just re-confirms the same mistake. We
therefore prompt the model in several complementary styles and pool the results.
The styles are drawn from the SOTA the paper cites:

  * ``direct``             — the proven AEGIS DDL prompt (few-shot, terse). Fast,
                            strong on simple queries.
  * ``query_plan``         — CHASE-SQL's query-plan chain-of-thought: reason about
                            the JOIN/filter plan first. Targets the multi-table
                            JOIN reasoning the paper names as the bottleneck.
  * ``divide_and_conquer`` — DIN-SQL/CHASE decomposition into sub-questions, then
                            compose. Targets nested/complex queries.

Each builder returns ``(system_prompt, user_prompt)``. A ``system_prompt`` of
``None`` means "use the generator's own default system prompt" — used by
``direct`` so it reproduces the existing, tuned local prompt exactly.

Pure prompt assembly (only depends on the lightweight prompt manager + the
dependency-free schema renderer); safe to unit-test without torch.
"""

from typing import List, Optional, Tuple

from prompts.prompt_manager import get_prompt_manager
from prompts.schema_render import render_schema_ddl

# Strategy identifiers (also the values accepted in config `agents.strategies`).
DIRECT = "direct"
QUERY_PLAN = "query_plan"
DIVIDE_AND_CONQUER = "divide_and_conquer"
ALL_STRATEGIES = (DIRECT, QUERY_PLAN, DIVIDE_AND_CONQUER)

# Shared system prompt for the chain-of-thought strategies. Unlike the terse
# "output only SQL" local default, this leaves room to reason before the query.
_COT_SYSTEM_PROMPT = (
    "You are an expert data analyst who writes correct SQLite queries. "
    "Reason carefully about the schema, then give the final query inside a "
    "single ```sql code block."
)

_QUERY_PLAN_INSTRUCTIONS = (
    "Think step by step about the execution plan before writing SQL:\n"
    "1. Which tables are needed, and how do they join? Use the FOREIGN KEY "
    "RELATIONSHIPS above for the exact join columns.\n"
    "2. Which filters/conditions does the question (and evidence) require? Match "
    "literal values to the example values shown for each column.\n"
    "3. Any grouping, aggregation, ordering, or limit?\n"
    "Then write the final SQLite query. Put ONLY the final query in a ```sql block."
)

_DIVIDE_INSTRUCTIONS = (
    "Solve this by divide and conquer:\n"
    "1. Break the question into smaller sub-questions.\n"
    "2. Write a partial SQL snippet for each sub-question.\n"
    "3. Compose them into one final SQLite query (use subqueries/CTEs/JOINs as "
    "needed, with the foreign keys above for joins).\n"
    "Put ONLY the final composed query in a ```sql block."
)


def _question_block(query) -> str:
    """Question + optional BIRD evidence, in a consistent labelled format."""
    text = f"Question: {query.text}\n"
    evidence = getattr(query, "evidence", "") or ""
    if evidence.strip():
        text = f"Evidence: {evidence.strip()}\n" + text
    return text


def build_direct_prompt(
    query, schema_elements: List, schema=None, expose_keys: bool = True
) -> str:
    """Reproduce the tuned AEGIS DDL prompt (schema → FK hints → few-shot → question).

    This is the same assembly the local SLM has always used; it is shared here so
    ``SLMGenerator._format_prompt_ddl`` and the booster's ``direct`` strategy can
    never drift apart.
    """
    schema_str, fk_hint_str, table_names = render_schema_ddl(
        schema_elements, schema=schema, expose_keys=expose_keys
    )
    pm = get_prompt_manager()
    examples = pm.format_slm_examples()
    generic_fk_hints = pm.get_slm_fk_hint(table_names)
    instructions = pm.get_slm_instructions()
    question = pm.format_slm_question(query.text, getattr(query, "evidence", "") or "")
    return f"{schema_str}{fk_hint_str}{generic_fk_hints}{examples}{instructions}{question}"


def build_query_plan_prompt(
    query, schema_elements: List, schema=None, expose_keys: bool = True
) -> str:
    """CHASE-SQL query-plan chain-of-thought prompt."""
    schema_str, fk_hint_str, _ = render_schema_ddl(
        schema_elements, schema=schema, expose_keys=expose_keys
    )
    return (
        f"{schema_str}{fk_hint_str}"
        f"{_question_block(query)}\n{_QUERY_PLAN_INSTRUCTIONS}\n"
    )


def build_decompose_prompt(
    query, schema_elements: List, schema=None, expose_keys: bool = True
) -> str:
    """DIN-SQL/CHASE divide-and-conquer prompt."""
    schema_str, fk_hint_str, _ = render_schema_ddl(
        schema_elements, schema=schema, expose_keys=expose_keys
    )
    return (
        f"{schema_str}{fk_hint_str}"
        f"{_question_block(query)}\n{_DIVIDE_INSTRUCTIONS}\n"
    )


def build_prompt(
    strategy: str, query, schema_elements: List, schema=None, expose_keys: bool = True
) -> Tuple[Optional[str], str]:
    """Dispatch to a strategy builder.

    Returns ``(system_prompt, user_prompt)``. ``system_prompt`` is ``None`` for
    ``direct`` (use the generator's default) and the shared CoT system prompt for
    the reasoning strategies.
    """
    if strategy == DIRECT:
        return None, build_direct_prompt(query, schema_elements, schema, expose_keys)
    if strategy == QUERY_PLAN:
        return _COT_SYSTEM_PROMPT, build_query_plan_prompt(
            query, schema_elements, schema, expose_keys
        )
    if strategy == DIVIDE_AND_CONQUER:
        return _COT_SYSTEM_PROMPT, build_decompose_prompt(
            query, schema_elements, schema, expose_keys
        )
    raise ValueError(f"Unknown generation strategy: {strategy!r}")


# --- Pairwise / listwise selection judge (CHASE-SQL selection agent) ----------

_JUDGE_SYSTEM_PROMPT = (
    "You are a meticulous SQL reviewer. Given a question and several candidate "
    "SQLite queries, pick the single candidate that most correctly answers the "
    "question against the schema. Answer with ONLY the candidate number."
)


def build_judge_prompt(
    query,
    schema_block: str,
    candidates: List[str],
    result_previews: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Build the selection-judge prompt comparing candidate queries.

    Args:
        query: the natural-language query (with optional evidence).
        schema_block: the rendered CREATE TABLE block (so the judge sees the schema).
        candidates: candidate SQL strings (already deduplicated).
        result_previews: optional short string previews of each candidate's executed
            result set, which give the judge execution evidence to compare against.

    Returns ``(system_prompt, user_prompt)``. The user prompt asks for a 1-based
    index; the caller parses the integer and is robust to a malformed answer.
    """
    lines = [schema_block, _question_block(query), "Candidates:"]
    for i, sql in enumerate(candidates, start=1):
        block = f"[{i}] {sql.strip()}"
        if result_previews and i - 1 < len(result_previews) and result_previews[i - 1]:
            block += f"\n    -> result: {result_previews[i - 1]}"
        lines.append(block)
    lines.append(
        f"\nWhich candidate (1-{len(candidates)}) best answers the question? "
        "Reply with only the number."
    )
    return _JUDGE_SYSTEM_PROMPT, "\n".join(lines)

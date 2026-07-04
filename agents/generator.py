"""Candidate-Generator agent (the core booster).

Generates a diverse pool of candidate SQLs by prompting the model with several
complementary reasoning strategies. Diversity across strategies is what gives the
downstream execution-guided selector something to vote between — the single most
reliable test-time-compute lever in the text-to-SQL literature (CHASE-SQL,
CSC-SQL, XiYan-SQL).
"""

from typing import List

from loguru import logger

from generator.sql_postprocess import finalize_sql
from prompts.sql_strategies import build_prompt


class CandidateGeneratorAgent:
    """Produce a deduplicated pool of candidate SQL strings (real tokens)."""

    def __init__(self, config) -> None:
        agents = config.agents
        # Strategies are chosen per path in generate(): the specialist SLM gets
        # direct self-consistency, the general LLM gets the CoT strategies too.
        self.local_strategies = list(agents.local_strategies)
        self.remote_strategies = list(agents.remote_strategies)
        self.per_strategy = agents.candidates_per_strategy
        self.temperature = agents.generation_temperature
        # CoT strategies reason before emitting SQL; the default 512-token budget
        # truncated the final ```sql block mid-fence on complex queries.
        self.cot_max_tokens = getattr(agents, "cot_max_tokens", 1024)
        self.enable_cast_fix = getattr(config.slm, "enable_cast_fix", True)

    def generate(self, ctx) -> List[str]:
        """Generate candidates across the strategies for the active path.

        Remote (general LLM) uses the CoT-inclusive ``remote_strategies``; local
        (specialist SLM) uses ``local_strategies`` (direct self-consistency). For
        each strategy we build the strategy-specific prompt from ``ctx.gen_query``
        (abstracted on the remote path), sample ``per_strategy`` candidates, then map
        each back to real tokens via ``ctx.reconstruct_fn`` and normalize it. Order
        is preserved and exact duplicates are dropped so identical candidates don't
        inflate the majority vote.
        """
        strategies = self.remote_strategies if ctx.source == "llm" else self.local_strategies
        pool: List[str] = []
        seen = set()
        for strategy in strategies:
            system_prompt, user_prompt = build_prompt(
                strategy,
                ctx.gen_query,
                ctx.schema_elements,
                schema=ctx.schema,
                expose_keys=ctx.expose_keys,
            )
            try:
                # Reasoning strategies get a larger decode budget than direct ones.
                max_tokens = self.cot_max_tokens if strategy != "direct" else None
                raw = ctx.generate_fn(
                    user_prompt, self.per_strategy, self.temperature, system_prompt,
                    max_tokens,
                )
            except Exception as e:
                logger.warning(f"Generator: strategy '{strategy}' failed ({e})")
                continue

            for text in raw:
                # Reconstruct (remote: placeholder->real; local: identity) then apply
                # deterministic fixes (idempotent CAST-as-REAL for ratio queries).
                sql = finalize_sql(ctx.reconstruct_fn(text), enable_cast_fix=self.enable_cast_fix)
                key = sql.strip().lower()
                if sql and key not in seen:
                    seen.add(key)
                    pool.append(sql)

        logger.info(
            f"Generator[{ctx.source}]: {len(pool)} unique candidates from "
            f"{len(strategies)} strategies x {self.per_strategy}"
        )
        return pool

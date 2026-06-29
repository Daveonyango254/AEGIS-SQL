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
        self.strategies = list(agents.strategies)
        self.per_strategy = agents.candidates_per_strategy
        self.temperature = agents.generation_temperature
        self.enable_cast_fix = getattr(config.slm, "enable_cast_fix", True)

    def generate(self, ctx) -> List[str]:
        """Generate candidates across all enabled strategies.

        For each strategy we build the strategy-specific prompt from ``ctx.gen_query``
        (abstracted on the remote path), sample ``per_strategy`` candidates, then map
        each back to real tokens via ``ctx.reconstruct_fn`` and normalize it. Order
        is preserved and exact duplicates are dropped so identical candidates don't
        inflate the majority vote.
        """
        pool: List[str] = []
        seen = set()
        for strategy in self.strategies:
            system_prompt, user_prompt = build_prompt(
                strategy,
                ctx.gen_query,
                ctx.schema_elements,
                schema=ctx.schema,
                expose_keys=ctx.expose_keys,
            )
            try:
                raw = ctx.generate_fn(
                    user_prompt, self.per_strategy, self.temperature, system_prompt
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
            f"Generator: {len(pool)} unique candidates from "
            f"{len(self.strategies)} strategies x {self.per_strategy}"
        )
        return pool

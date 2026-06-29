"""Refiner agent — execution-feedback self-correction (MAC-SQL / RetrySQL).

Runs the selected query against the database; if it errors or returns an empty
result set (the signature of a literal/JOIN mistake), it shows the model the
execution feedback and asks for one corrected query, keeping the repair only if it
strictly improves. Bounded by ``agents.refine_rounds`` so cost stays predictable
and an accepted result is never replaced by a worse one.
"""

from loguru import logger

from generator.candidate_selector import select_best
from generator.sql_postprocess import finalize_sql
from prompts.sql_strategies import build_direct_prompt

_EMPTY_FEEDBACK = (
    "The query executed but returned an empty result set (0 rows). Re-check the "
    "filter values against the example values shown for each column, and the JOIN keys."
)


class RefinerAgent:
    """Repair a query using real execution feedback, bounded and improvement-only."""

    def __init__(self, config) -> None:
        self.rounds = config.agents.refine_rounds
        self.timeout = config.agents.selection_timeout
        self.enable_cast_fix = getattr(config.slm, "enable_cast_fix", True)

    def refine(self, sql_text: str, ctx) -> str:
        """Return a possibly-repaired query (never worse than the input)."""
        if not sql_text or not ctx.db_path or ctx.db_path == ":memory:" or self.rounds <= 0:
            return sql_text

        current = sql_text
        for round_idx in range(self.rounds):
            info = select_best([current], ctx.db_path, timeout=self.timeout)
            if info.get("exec_ok") and not info.get("is_empty"):
                return current  # healthy result — nothing to fix

            feedback = info.get("error") or _EMPTY_FEEDBACK
            feedback = self._sanitize(feedback, ctx)

            prompt = build_direct_prompt(
                ctx.gen_query, ctx.schema_elements, ctx.schema, ctx.expose_keys
            )
            prompt += (
                f"\n\n-- The previous attempt was rejected:\n-- {feedback}\n"
                f"-- Generate a corrected SQL query.\n-- SQL:"
            )
            try:
                raw = ctx.generate_fn(prompt, 1, 0.0, None)
            except Exception as e:
                logger.warning(f"Refiner: repair generation failed ({e})")
                break
            if not raw:
                break

            repaired = finalize_sql(ctx.reconstruct_fn(raw[0]), enable_cast_fix=self.enable_cast_fix)
            if not repaired:
                break

            r_info = select_best([repaired], ctx.db_path, timeout=self.timeout)
            # Accept a repair only when it is strictly better:
            #  - a clean non-empty result always wins, or
            #  - if the current query errors, an executing repair is an improvement.
            if r_info.get("exec_ok") and not r_info.get("is_empty"):
                logger.info(f"Refiner: round {round_idx + 1} produced a clean result")
                return repaired
            if not info.get("exec_ok") and r_info.get("exec_ok"):
                current = repaired

        return current

    @staticmethod
    def _sanitize(feedback: str, ctx) -> str:
        """On the remote path, strip real tokens from feedback before it is sent out."""
        if ctx.source == "llm" and ctx.recon_map is not None:
            try:
                from verifier.feedback_generator import FeedbackGenerator

                return FeedbackGenerator().sanitize_feedback_for_remote_retry(
                    feedback, ctx.recon_map.real_to_placeholder
                )
            except Exception:
                return feedback
        return feedback

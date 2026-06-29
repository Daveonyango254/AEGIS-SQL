"""Selector agent — pick the best candidate from the pool.

Primary signal is **execution-guided self-consistency**: run every candidate on
the real database and take the majority result set (reusing the proven
``candidate_selector.select_best``). When the vote is split (no majority), an
optional **pairwise/listwise judge** breaks the tie (CHASE-SQL's selection agent).

The judge always runs on the *local, trusted* model, so it adds zero leakage even
on the remote path — preserving the paper's content-independent-routing guarantee.
"""

import re
from typing import List, Optional, Tuple

from loguru import logger

from generator.candidate_selector import select_best
from prompts.schema_render import render_schema_ddl
from prompts.sql_strategies import build_judge_prompt


class SelectorAgent:
    """Choose the winning candidate via execution consistency + an optional judge."""

    def __init__(self, config) -> None:
        agents = config.agents
        self.judge_enabled = agents.judge_enabled
        self.timeout = agents.selection_timeout
        self.max_judge_candidates = agents.max_judge_candidates

    def select(self, candidates: List[str], ctx, judge_fn=None) -> Tuple[str, dict]:
        """Return ``(best_sql, info)`` where ``info`` is the execution diagnostics.

        ``judge_fn`` is the trusted local generator's ``complete``-style callable;
        if ``None`` or judging is disabled, selection is execution-consistency only.
        """
        if not candidates:
            return "", {"num_candidates": 0}
        if len(candidates) == 1:
            return candidates[0], {"num_candidates": 1, "num_agree": 1}

        # Without an executable DB we cannot vote on result sets; take the first
        # (the greedy/direct candidate) as the safest default.
        if not ctx.db_path or ctx.db_path == ":memory:":
            return candidates[0], {"num_candidates": len(candidates), "num_executed": 0}

        info = select_best(candidates, ctx.db_path, timeout=self.timeout)
        best = info.get("best_sql", candidates[0])

        # Tie-break with the judge only when execution gave no clear majority but
        # at least two candidates returned non-empty results worth comparing.
        split_vote = info.get("num_agree", 0) <= 1 and info.get("num_nonempty", 0) >= 2
        if self.judge_enabled and judge_fn and split_vote:
            judged = self._judge(candidates, ctx, judge_fn)
            if judged:
                logger.info("Selector: judge broke a split vote")
                best = judged

        return best, info

    def _judge(self, candidates: List[str], ctx, judge_fn) -> Optional[str]:
        """Ask the local model to pick the best of the top candidates."""
        top = candidates[: self.max_judge_candidates]
        schema_block, _, _ = render_schema_ddl(
            ctx.schema_elements, schema=ctx.schema, expose_keys=ctx.expose_keys
        )
        system_prompt, user_prompt = build_judge_prompt(ctx.query, schema_block, top)
        try:
            out = judge_fn(user_prompt, 1, 0.0, system_prompt)
        except Exception as e:
            logger.warning(f"Selector: judge failed ({e})")
            return None
        if not out:
            return None
        idx = self._parse_choice(out[0], len(top))
        return top[idx] if idx is not None else None

    @staticmethod
    def _parse_choice(text: str, n: int) -> Optional[int]:
        """Parse a 1-based candidate index from the judge's reply; None if invalid."""
        m = re.search(r"\d+", text)
        if not m:
            return None
        choice = int(m.group()) - 1
        return choice if 0 <= choice < n else None

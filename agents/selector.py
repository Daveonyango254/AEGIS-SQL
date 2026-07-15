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

        # Optional trained pairwise selector (CHASE-SQL lever). Constructed here
        # but the model loads lazily on first use, so this stays import-safe and
        # zero-cost when unconfigured. Empty selector_model => heuristic judge only.
        self.pairwise = None
        model_id = (getattr(agents, "selector_model", "") or "").strip()
        if model_id:
            try:
                from agents.pairwise_selector import PairwiseSelector

                mcfg = config.slm
                self.pairwise = PairwiseSelector(
                    model_id,
                    cache_dir=getattr(mcfg, "cache_dir", None),
                    hf_token=self._resolve_token(getattr(mcfg, "hf_token", None)),
                    device=getattr(mcfg, "device", "auto"),
                    torch_dtype=getattr(mcfg, "torch_dtype", "float16"),
                    max_candidates=self.max_judge_candidates,
                    exec_timeout=self.timeout,
                )
                logger.info(f"SelectorAgent: trained pairwise selector enabled ({model_id})")
            except Exception as e:  # never let selector wiring break selection
                logger.warning(f"SelectorAgent: pairwise selector unavailable ({e})")
                self.pairwise = None

    @staticmethod
    def _resolve_token(token):
        """Resolve a ${ENV_VAR} HF token to its value (config subst may leave it)."""
        import os

        if isinstance(token, str) and token.startswith("${") and token.endswith("}"):
            return os.getenv(token[2:-1])
        return token

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

        # Trained pairwise selector (if configured) arbitrates among the top
        # candidates, replacing the heuristic judge. The execution-vote winner is
        # placed first so it wins ties. This is the CHASE-SQL selection lever.
        if self.pairwise is not None:
            try:
                ordered = [best] + [c for c in candidates if c != best]
                schema_block, _, _ = render_schema_ddl(
                    ctx.schema_elements, schema=ctx.schema, expose_keys=ctx.expose_keys
                )
                picked = self.pairwise.select(
                    ordered,
                    question=ctx.query.text,
                    evidence=getattr(ctx.query, "evidence", "") or "",
                    schema_block=schema_block,
                    db_path=ctx.db_path,
                )
                if picked:
                    if picked != best:
                        logger.info("Selector: trained pairwise selector overrode execution vote")
                    return picked, {**info, "selector": "trained_pairwise"}
            except Exception as e:  # fall through to heuristic on any failure
                logger.warning(f"Selector: pairwise selection failed ({e}); using execution vote")

        # Tie-break with the heuristic judge only when execution gave no clear
        # majority but at least two candidates returned non-empty results.
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

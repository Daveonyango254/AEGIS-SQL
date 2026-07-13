"""AEGIS v1 orchestrator — accuracy-first corrective self-consistency pipeline.

    Schema stage (recall-first)            [agents/schema_linker.py]
      └─> Candidate pool (mode-dependent, parallel)
            local:  CSC GRPO generator, native OmniSQL prompt, n samples
            remote: LLM, direct + query-plan strategies, n samples each
      └─> Execution-vote grouping over the POOLED candidates    [generator/csc.py]
      └─> CSC merge-revision on the top-2 disagreeing groups (the merge
          checkpoint's trained skill) → re-vote
      └─> Judge tie-break (equal-vote groups, judge model configurable)
      └─> Bounded revision-based refine (error/empty results only)
      └─> Reviewer: 3-stage verification                        [agents/reviewer.py]

Returns the exact prediction-contract dict the evaluation harness has always
consumed, so results/logs stay comparable across branches. Cost and privacy are
ISOLATED here: no router, no DP abstraction (both parked in the repo untouched);
per-query cost is still reported (fixed local cost + token-billed remote).
"""

from typing import List, Optional

from loguru import logger

from aegis_types import SQL, RoutingDecision
from generator.csc import VoteGroup, csc_select, group_by_execution
from generator.sql_postprocess import finalize_sql
from prompts import omnisql
from prompts.schema_render import render_schema_ddl
from prompts.sql_strategies import build_judge_prompt, build_prompt
from workflow.model_cache import get_cache

# Reporting constants (cost is isolated, not optimized — still recorded).
REMOTE_TOKEN_COST_USD = 1.5e-05
LOCAL_COMPUTE_COST_USD = 1e-04


class MultiAgentOrchestrator:
    """Run one query through the v1 pipeline; ``run()`` is the eval entry point."""

    def __init__(self, config) -> None:
        from agents.reviewer import ReviewerAgent
        from agents.schema_linker import SchemaLinkerAgent

        self.config = config
        self.mode = config.mode
        self.linker = SchemaLinkerAgent(config)
        self.reviewer = ReviewerAgent(config)
        if config.models.generator == config.models.merger:
            logger.info(
                "Single-model mode: one checkpoint serves generation and merge "
                "(the default; fits any 20GB+ GPU in fp16)"
            )

    # ------------------------------------------------------------------ main --

    def run(self, initial_state: dict) -> dict:
        query = initial_state["query"]
        schema = initial_state["schema"]
        db_path = initial_state.get("db_path")
        db_id = initial_state.get("database_id") or getattr(query, "database_id", "")
        cache = get_cache()

        # --- Schema stage ------------------------------------------------------
        retriever = cache.get_schema_retriever(db_id, schema)
        elements, tables, num_columns = self.linker.link(retriever, query, schema, db_path)
        db_details = omnisql.build_db_details(schema, elements)

        # --- Candidate pool ----------------------------------------------------
        # The remote arm is pure I/O, so in ensemble mode it runs CONCURRENTLY
        # with local GPU generation — per-query wall time is max(local, remote)
        # instead of their sum. Every candidate is tagged with the arm that
        # produced it so the prediction record can report which arm WON.
        pool: List[str] = []
        origin: dict = {}  # normalized sql -> "local" | "remote" | "merge" | "refine"
        n_local = n_remote = 0
        llm = None
        remote_future = None
        use_remote = (
            self.mode in ("remote", "ensemble")
            and self.config.generation.remote_candidates > 0
        )
        if use_remote:
            from concurrent.futures import ThreadPoolExecutor

            llm = cache.get_llm_generator()
            executor = ThreadPoolExecutor(max_workers=1)
            remote_future = executor.submit(
                self._remote_candidates, llm, query, elements, schema
            )
            executor.shutdown(wait=False)
        if self.mode in ("local", "ensemble"):
            local = self._local_candidates(cache, query, db_details)
            n_local = len(local)
            for sql in local:
                origin.setdefault(sql.strip().lower(), "local")
            pool += local
        if remote_future is not None:
            try:
                remote = remote_future.result()
                n_remote = len(remote)
                for sql in remote:
                    origin.setdefault(sql.strip().lower(), "remote")
                pool += remote
            except Exception as e:
                logger.warning(f"Orchestrator: remote arm failed ({e})")

        pool = self._dedupe(pool)
        if not pool:
            logger.error("Orchestrator: empty candidate pool; emitting trivial query")
            pool = [self._trivial(elements)]

        # --- CSC selection: vote -> merge-revise -> re-vote ---------------------
        timeout = self.config.selection.timeout_seconds
        merge_fn = None
        if self.config.csc.enabled:
            inner_merge = self._merge_fn(cache, query, schema)

            def merge_fn(groups):
                out = inner_merge(groups)
                for sql in out:  # merge outputs are their own arm in the report
                    origin.setdefault(sql.strip().lower(), "merge")
                return out

        best = csc_select(pool, db_path, merge_fn=merge_fn, timeout=timeout)

        # --- Judge tie-break (equal top votes and CSC didn't adjudicate) --------
        best = self._judge_tiebreak(cache, llm, query, elements, schema, pool, best, db_path)

        # --- Bounded refine (revision prompt on error/empty results) ------------
        refined = self._refine(cache, query, schema, best, db_path)
        if refined.strip().lower() != best.strip().lower():
            origin.setdefault(refined.strip().lower(), "refine")
        best = refined

        sql = SQL(text=best, dialect="sqlite", source=self._source(), verified=False)
        verification_result = self.reviewer.review(sql, schema, db_path)

        cost = LOCAL_COMPUTE_COST_USD if self.mode != "remote" else 0.0
        if llm is not None:
            cost += llm.total_tokens * REMOTE_TOKEN_COST_USD

        return {
            "sql": sql,
            "routing_decision": self._route(),
            "abstracted_prompt": None,   # privacy isolated in v1
            "verification_result": verification_result,
            "generation_source": self._source(),
            # Which arm produced the FINAL answer ("local"/"remote"/"merge"/
            # "refine") + pool sizes — the per-query routing record for
            # analyzing local-vs-remote wins in ensemble mode.
            "winner_arm": origin.get(best.strip().lower(), self._source()),
            "candidates_local": n_local,
            "candidates_remote": n_remote,
            "retrieved_tables": tables,
            "num_retrieved_columns": num_columns,
            "cost_usd": cost,
            "privacy_loss": 0.0,
        }

    # ------------------------------------------------------------- generation --

    def _local_candidates(self, cache, query, db_details: str) -> List[str]:
        """Sample n candidates from the GRPO generator with its native prompt."""
        gen = cache.get_slm(self.config.models.generator)
        prompt = omnisql.build_generation_prompt(
            query.text, getattr(query, "evidence", "") or "", db_details
        )
        raw = gen.complete(
            prompt,
            n=self.config.generation.local_candidates,
            temperature=self.config.generation.temperature,
            max_tokens=self.config.generation.max_tokens,
        )
        out = [finalize_sql(r) for r in raw]
        logger.info(f"Orchestrator: {len(out)} local candidates")
        return [o for o in out if o]

    def _remote_candidates(self, llm, query, elements, schema) -> List[str]:
        """Sample candidates from the remote LLM across its reasoning strategies."""
        gcfg = self.config.generation
        out: List[str] = []
        for strategy in gcfg.remote_strategies:
            system_prompt, user_prompt = build_prompt(
                strategy, query, elements, schema=schema, expose_keys=True
            )
            try:
                raw = llm.complete(
                    user_prompt,
                    n=gcfg.remote_candidates,
                    temperature=gcfg.temperature,
                    system_prompt=system_prompt,
                    max_tokens=gcfg.max_tokens,
                )
            except Exception as e:
                logger.warning(f"Orchestrator: remote strategy '{strategy}' failed ({e})")
                continue
            out += [finalize_sql(r) for r in raw if r]
        logger.info(f"Orchestrator: {len(out)} remote candidates")
        return [o for o in out if o]

    # -------------------------------------------------------------- csc stages --

    def _merge_fn(self, cache, query, schema):
        """Build the merge-revision callable for csc_select (loads merger lazily)."""

        def merge(groups: List[VoteGroup]) -> List[str]:
            merger = cache.get_slm(self.config.models.merger)
            candidates = [
                (g.sql, list(g.result) if g.result is not None else None) for g in groups
            ]
            prompt = omnisql.build_merge_prompt(
                query.text, getattr(query, "evidence", "") or "", schema, candidates
            )
            raw = merger.complete(
                prompt,
                n=self.config.csc.merge_candidates,
                temperature=self.config.generation.temperature,
                max_tokens=self.config.generation.max_tokens,
            )
            return [finalize_sql(r) for r in raw if r]

        return merge

    def _judge_tiebreak(
        self, cache, llm, query, elements, schema, pool, best, db_path
    ) -> str:
        """Break an exact vote tie between the top two result groups.

        Runs only when the top-2 groups have EQUAL votes (csc's merge already
        adjudicated genuine disagreements; the judge is the residual tie-break).
        """
        jcfg = self.config.selection
        if jcfg.judge == "off" or not db_path or db_path == ":memory:":
            return best
        groups = [g for g in group_by_execution(pool, db_path, timeout=jcfg.timeout_seconds)
                  if g.result is not None]
        if len(groups) < 2 or groups[0].votes != groups[1].votes:
            return best

        top = [g.sql for g in groups[: jcfg.max_judge_candidates]]
        schema_block, _, _ = render_schema_ddl(elements, schema=schema, expose_keys=True)
        system_prompt, user_prompt = build_judge_prompt(query, schema_block, top)
        judge = self._judge_model(cache, llm, jcfg.judge)
        if judge is None:
            return best
        try:
            reply = judge(user_prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Orchestrator: judge failed ({e})")
            return best
        import re

        m = re.search(r"\d+", reply or "")
        if m and 1 <= int(m.group()) <= len(top):
            logger.info("Orchestrator: judge broke an exact vote tie")
            return top[int(m.group()) - 1]
        return best

    def _judge_model(self, cache, llm, mode: str):
        """Judge callable by config: remote LLM when available, else local merger."""
        if mode in ("remote", "auto") and llm is not None:
            return lambda p, sp: (llm.complete(p, n=1, temperature=0.0,
                                               system_prompt=sp, raw=True) or [""])[0]
        if mode in ("local", "auto"):
            slm = cache.get_slm(self.config.models.merger)
            return lambda p, sp: (slm.complete(p, n=1, temperature=0.0,
                                               system_prompt=sp, raw=True) or [""])[0]
        return None

    def _refine(self, cache, query, schema, best: str, db_path) -> str:
        """Revision-based repair when the winner errors or returns empty.

        Uses the merge model's single-draft revision distribution: show the draft
        + its execution outcome, ask for a corrected query, keep the revision
        only if it strictly improves (never replace a working result).
        """
        rounds = self.config.refine.rounds
        if rounds <= 0 or not best or not db_path or db_path == ":memory:":
            return best
        timeout = self.config.selection.timeout_seconds
        current = best
        for _ in range(rounds):
            groups = group_by_execution([current], db_path, timeout=timeout)
            g = groups[0] if groups else None
            healthy = g is not None and g.result is not None and len(g.result) > 0
            if healthy:
                return current
            rows = list(g.result) if (g and g.result is not None) else None
            merger = cache.get_slm(self.config.models.merger)
            prompt = omnisql.build_merge_prompt(
                query.text, getattr(query, "evidence", "") or "", schema,
                [(current, rows)],
            )
            raw = merger.complete(prompt, n=1, temperature=0.0,
                                  max_tokens=self.config.generation.max_tokens)
            fixed = finalize_sql(raw[0]) if raw else ""
            if not fixed or fixed == current:
                break
            f_groups = group_by_execution([fixed], db_path, timeout=timeout)
            fg = f_groups[0] if f_groups else None
            if fg is not None and fg.result is not None and len(fg.result) > 0:
                logger.info("Orchestrator: refine produced a clean result")
                return fixed
            if fg is not None and fg.result is not None and (g is None or g.result is None):
                current = fixed  # executing beats erroring; loop may improve further
        return current

    # ---------------------------------------------------------------- helpers --

    def _source(self) -> str:
        return {"local": "slm", "remote": "llm"}.get(self.mode, "ensemble")

    def _route(self) -> RoutingDecision:
        # Reporting compatibility: 'local' maps to LOCAL; remote/ensemble involve
        # the remote model, so they report REMOTE (privacy is isolated in v1).
        return RoutingDecision.LOCAL if self.mode == "local" else RoutingDecision.REMOTE

    @staticmethod
    def _dedupe(pool: List[str]) -> List[str]:
        seen, out = set(), []
        for sql in pool:
            key = sql.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(sql)
        return out

    @staticmethod
    def _trivial(elements) -> str:
        for e in elements:
            if "." in e.name:
                return f"SELECT * FROM {e.name.split('.', 1)[0]} LIMIT 10"
        return "SELECT 1"

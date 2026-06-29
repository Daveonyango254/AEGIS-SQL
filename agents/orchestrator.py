"""MultiAgentOrchestrator — the booster pipeline and drop-in for ``graph.invoke``.

Wires the agents into the paper's skeleton and returns the exact result dict the
evaluation harness expects, so results/logs are collected identically:

    SchemaLinker -> Router -> [ LOCAL: SLM | REMOTE: abstract -> LLM -> reconstruct ]
                 -> Generate (multi-strategy) -> Select -> Refine -> Review

Reuses the cached models (no reloads), the content-independent router, the DP
abstraction layer, and the 3-stage verifier. Privacy is preserved: on the remote
path the model only ever sees abstracted text, candidates are reconstructed inside
the trust boundary, and the optional selection judge runs on the local model.
"""

from typing import Optional

from loguru import logger

from aegis_types import SQL, Query, RoutingDecision
from workflow.costing import compute_cost
from workflow.model_cache import get_cache

from agents.context import RunContext
from agents.schema_linker import SchemaLinkerAgent
from agents.generator import CandidateGeneratorAgent
from agents.selector import SelectorAgent
from agents.refiner import RefinerAgent
from agents.reviewer import ReviewerAgent


class MultiAgentOrchestrator:
    """Run a single query through the multi-agent booster harness."""

    def __init__(self, config) -> None:
        self.config = config
        self.linker = SchemaLinkerAgent(config)
        self.generator = CandidateGeneratorAgent(config)
        self.selector = SelectorAgent(config)
        self.refiner = RefinerAgent(config)
        self.reviewer = ReviewerAgent(config)
        self.expose_keys = getattr(config.slm, "expose_keys", True)

    def run(self, initial_state: dict) -> dict:
        """Process one query; return the prediction-contract dict."""
        query = initial_state["query"]
        schema = initial_state["schema"]
        db_path = initial_state.get("db_path")
        db_id = initial_state.get("database_id") or getattr(query, "database_id", "")

        cache = get_cache()

        # --- Query Planner: schema linking -------------------------------------
        retriever = cache.get_schema_retriever(db_id, schema)
        schema_elements, retrieved_tables, num_columns = self.linker.link(
            retriever, query, schema, db_path
        )

        # --- Content-Independent Router ---------------------------------------
        route = cache.get_router().route(query, schema_elements)

        # --- Build the path-specific generation context ------------------------
        if route == RoutingDecision.REMOTE:
            ctx, llm, abstracted_prompt = self._remote_context(
                query, schema, schema_elements, db_path
            )
        else:
            ctx, abstracted_prompt = self._local_context(
                query, schema, schema_elements, db_path
            ), None
            llm = None

        # --- Generate -> Select -> Refine --------------------------------------
        candidates = self.generator.generate(ctx)
        if not candidates:
            candidates = [self._fallback_candidate(ctx, query, schema_elements, schema, llm, abstracted_prompt)]

        judge_fn = self._judge_fn(cache) if self.selector.judge_enabled else None
        best_text, _ = self.selector.select(candidates, ctx, judge_fn=judge_fn)
        best_text = self.refiner.refine(best_text, ctx)

        sql = SQL(text=best_text, dialect="sqlite", source=ctx.source, verified=False)

        # --- Reviewer: 3-stage verification ------------------------------------
        verification_result = self.reviewer.review(sql, schema, db_path)

        # --- Cost (token-billed remote, fixed local) ---------------------------
        ccfg = self.config.cost
        if ctx.source == "llm" and llm is not None:
            cost = compute_cost("llm", llm.total_tokens, ccfg.remote_token_cost, ccfg.local_compute_cost)
        else:
            cost = compute_cost("slm", 0, ccfg.remote_token_cost, ccfg.local_compute_cost)

        return {
            "sql": sql,
            "routing_decision": route,
            "abstracted_prompt": abstracted_prompt,
            "verification_result": verification_result,
            "generation_source": ctx.source,
            "retrieved_tables": retrieved_tables,
            "num_retrieved_columns": num_columns,
            "cost_usd": cost,
            "privacy_loss": 0.0,  # aggregate ℒ_priv is computed offline by metrics.py
        }

    # ------------------------------------------------------------------ paths --

    def _local_context(self, query, schema, schema_elements, db_path) -> RunContext:
        """Local path: the SLM sees raw inputs; no abstraction/reconstruction."""
        slm = get_cache().get_slm_generator()

        def generate_fn(prompt, n, temperature, system_prompt):
            return slm.complete(prompt, n=n, temperature=temperature, system_prompt=system_prompt)

        return RunContext(
            query=query, gen_query=query, schema=schema, schema_elements=schema_elements,
            db_path=db_path, config=self.config, route=RoutingDecision.LOCAL, source="slm",
            generate_fn=generate_fn, reconstruct_fn=lambda s: s, recon_map=None,
            expose_keys=self.expose_keys,
        )

    def _remote_context(self, query, schema, schema_elements, db_path):
        """Remote path: abstract the query, generate on the LLM, reconstruct candidates."""
        from abstraction.dp_abstractor import DPAbstractor
        from abstraction.placeholder_vocab import PlaceholderVocabulary
        from abstraction.sensitivity_policy import SensitivityPolicy
        from abstraction.reconstruction import ReconstructionModule

        pcfg = self.config.privacy
        vocab = PlaceholderVocabulary(vocab_size=pcfg.placeholder_vocab_size)
        policy = SensitivityPolicy(pcfg.sensitivity_policy)
        abstractor = DPAbstractor(pcfg, vocab, policy, embedding_model=None)
        abstracted_prompt, recon_map = abstractor.abstract(query, schema_elements)

        # Prompts are built from the abstracted text so no sensitive value leaves
        # the trust boundary; reconstruction restores real tokens for execution.
        gen_query = Query(
            text=abstracted_prompt.text, language=query.language,
            database_id=query.database_id,
            evidence=abstracted_prompt.evidence or getattr(query, "evidence", ""),
        )

        recon = ReconstructionModule()
        recon.register_map("q", recon_map)

        def reconstruct_fn(text: str) -> str:
            return recon.reconstruct(
                SQL(text=text, dialect="sqlite", source="llm", verified=False), "q"
            ).text

        llm = get_cache().get_llm_generator()

        def generate_fn(prompt, n, temperature, system_prompt):
            return llm.complete(prompt, n=n, temperature=temperature, system_prompt=system_prompt)

        ctx = RunContext(
            query=query, gen_query=gen_query, schema=schema, schema_elements=schema_elements,
            db_path=db_path, config=self.config, route=RoutingDecision.REMOTE, source="llm",
            generate_fn=generate_fn, reconstruct_fn=reconstruct_fn, recon_map=recon_map,
            expose_keys=self.expose_keys,
        )
        return ctx, llm, abstracted_prompt

    # -------------------------------------------------------------- fallbacks --

    def _judge_fn(self, cache):
        """The selection judge always runs on the local (trusted) model."""
        try:
            slm = cache.get_slm_generator()
        except Exception:
            return None

        def judge(prompt, n, temperature, system_prompt):
            return slm.complete(prompt, n=n, temperature=temperature, system_prompt=system_prompt)

        return judge

    def _fallback_candidate(self, ctx, query, schema_elements, schema, llm, abstracted_prompt) -> str:
        """If every strategy returned nothing, fall back to the proven single-shot path."""
        from generator.sql_postprocess import finalize_sql

        try:
            if ctx.source == "llm" and llm is not None:
                sql = llm.generate(abstracted_prompt, schema_elements, schema=schema)
                text = ctx.reconstruct_fn(sql.text)
            else:
                slm = get_cache().get_slm_generator()
                text = slm.generate(query, schema_elements, schema=schema).text
            return finalize_sql(text, enable_cast_fix=getattr(self.config.slm, "enable_cast_fix", True))
        except Exception as e:
            logger.error(f"Orchestrator fallback failed ({e}); emitting trivial query")
            first_table = retrieved_first_table(schema_elements)
            return f"SELECT * FROM {first_table} LIMIT 10" if first_table else "SELECT 1"


def retrieved_first_table(schema_elements) -> Optional[str]:
    """First table name in the retrieved slice (for the trivial fallback)."""
    for elem in schema_elements:
        if "." in elem.name:
            return elem.name.split(".", 1)[0]
    return None

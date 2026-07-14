"""AEGIS v2 — model-agnostic LangGraph pipeline (CHASE-style, no CSC coupling).

v1's arm analysis showed the CSC merge-revision — the stage welded to one 7B
checkpoint — decided only 2/100 ensemble queries. v2 removes that coupling and
rebuilds the pipeline around the components that carry the accuracy in the
literature and in our own runs, expressed as a LangGraph:

                              ┌──────────────┐
                              │ schema_link  │   tools: retriever, value index
                              └──────┬───────┘
                          MODE ROUTER (conditional edges)
                     local ▼        both ▼         ▼ remote
              ┌──────────────┐  ┌──────────────────────────┐
              │ generate_local│  │ generate_remote          │   candidates merge via
              │ (any HF model)│  │ (any OpenAI/Anthropic id)│   an additive reducer
              └──────┬───────┘  └───────────┬──────────────┘
                     └───────────┬──────────┘
                              ┌──▼───────────┐
                              │ vote          │  tool: SQLite execution
                              │ (result-set   │  (frozenset grouping, BIRD
                              │  grouping)    │   EX semantics)
                              └──┬───────────┘
              top groups disagree?│                 agree/empty
              ┌───────────────────▼──┐          ┌──────────────┐
              │ judge (LLM tournament │──────────▶ refine_gate  │
              │ over group reps +     │          └──────┬───────┘
              │ execution previews)   │    error/empty & rounds left?
              └───────────────────────┘          ┌──────▼───────┐
                                                 │ refine       │──▶ vote (loop)
                                                 └──────┬───────┘
                                                 ┌──────▼───────┐
                                                 │ verify        │──▶ END
                                                 └──────────────┘

Every node is a PURE function over the state dict, so the pipeline runs
identically under LangGraph (installed on the eval box; parallel arm fan-out)
or under the built-in sequential fallback (no dependency; used by offline
tests and as a safety net).
"""

import operator
from typing import Annotated, List, Optional, Tuple, TypedDict

from loguru import logger

from aegis_types import SQL, RoutingDecision
from generator.csc import group_by_execution
from generator.sql_postprocess import finalize_sql
from prompts import omnisql
from prompts.schema_render import render_schema_ddl
from prompts.sql_strategies import (
    build_judge_prompt,
    build_prompt,
    build_revision_prompt,
)
from workflow.model_cache import get_cache

# Reporting constants (cost is isolated, not optimized — still recorded).
REMOTE_TOKEN_COST_USD = 1.5e-05
LOCAL_COMPUTE_COST_USD = 1e-04


class GraphState(TypedDict, total=False):
    """The LangGraph state. ``candidates`` uses an additive reducer so the two
    generator arms can run in PARALLEL branches and their outputs merge."""

    # inputs
    query: object
    schema: object
    db_path: Optional[str]
    db_id: str
    # schema stage
    elements: list
    tables: List[str]
    num_columns: int
    db_details: str
    # generation (each item: (sql, arm)) — additive across parallel branches
    candidates: Annotated[list, operator.add]
    # selection
    best: str
    winner_arm: str
    # refine loop
    refine_count: int
    # output
    verification_result: object


# ---------------------------------------------------------------------------
# Nodes (pure functions over GraphState; models come from the shared cache)
# ---------------------------------------------------------------------------


def make_nodes(config):
    """Build the node functions bound to a config (closures keep them pure)."""
    from agents.schema_linker import SchemaLinkerAgent

    linker = SchemaLinkerAgent(config)
    gcfg = config.generation

    def schema_link(state: GraphState) -> dict:
        """Recall-first schema slice + value grounding (v1's proven stage)."""
        cache = get_cache()
        retriever = cache.get_schema_retriever(state["db_id"], state["schema"])
        elements, tables, n_cols = linker.link(
            retriever, state["query"], state["schema"], state.get("db_path")
        )
        return {
            "elements": elements,
            "tables": tables,
            "num_columns": n_cols,
            "db_details": omnisql.build_db_details(state["schema"], elements),
            "candidates": [],
            "refine_count": 0,
        }

    def generate_local(state: GraphState) -> dict:
        """Local arm: ANY HF causal model, its native OmniSQL prompt, n samples."""
        slm = get_cache().get_slm(config.models.generator)
        prompt = omnisql.build_generation_prompt(
            state["query"].text,
            getattr(state["query"], "evidence", "") or "",
            state["db_details"],
        )
        raw = slm.complete(
            prompt, n=gcfg.local_candidates,
            temperature=gcfg.temperature, max_tokens=gcfg.max_tokens,
        )
        out = [(finalize_sql(r), "local") for r in raw if r]
        logger.info(f"graph_v2: {len(out)} local candidates")
        return {"candidates": [c for c in out if c[0]]}

    def generate_remote(state: GraphState) -> dict:
        """Remote arm: ANY OpenAI/Anthropic model across reasoning strategies."""
        llm = get_cache().get_llm_generator()
        out: List[Tuple[str, str]] = []
        for strategy in gcfg.remote_strategies:
            system_prompt, user_prompt = build_prompt(
                strategy, state["query"], state["elements"],
                schema=state["schema"], expose_keys=True,
            )
            try:
                raw = llm.complete(
                    user_prompt, n=gcfg.remote_candidates,
                    temperature=gcfg.temperature,
                    system_prompt=system_prompt, max_tokens=gcfg.max_tokens,
                )
            except Exception as e:
                logger.warning(f"graph_v2: remote strategy '{strategy}' failed ({e})")
                continue
            out += [(finalize_sql(r), "remote") for r in raw if r]
        logger.info(f"graph_v2: {len(out)} remote candidates")
        return {"candidates": [c for c in out if c[0]]}

    def vote(state: GraphState) -> dict:
        """Tool node: execute every candidate; majority result-set group wins."""
        pool = _dedupe(state.get("candidates", []))
        if not pool:
            return {"best": _trivial(state.get("elements", [])), "winner_arm": "error"}
        texts = [sql for sql, _ in pool]
        groups = group_by_execution(
            texts, state.get("db_path"), timeout=config.selection.timeout_seconds
        )
        best = groups[0].sql if groups else texts[0]
        arm = dict((s.strip().lower(), a) for s, a in pool).get(
            best.strip().lower(), "unknown"
        )
        return {"best": best, "winner_arm": arm}

    def judge(state: GraphState) -> dict:
        """CHASE-style selection: an LLM tournament over the disagreeing
        result groups, each shown with an execution preview. Runs on the
        strongest configured model (remote when available, else local)."""
        pool = _dedupe(state.get("candidates", []))
        texts = [sql for sql, _ in pool]
        groups = [
            g for g in group_by_execution(
                texts, state.get("db_path"), timeout=config.selection.timeout_seconds)
            if g.result is not None
        ]
        if len(groups) < 2:
            return {}
        top = groups[: config.selection.max_judge_candidates]
        schema_block, _, _ = render_schema_ddl(
            state["elements"], schema=state["schema"], expose_keys=True
        )
        previews = [omnisql.normalize_execution_result(list(g.result))[:200] for g in top]
        system_prompt, user_prompt = build_judge_prompt(
            state["query"], schema_block, [g.sql for g in top], result_previews=previews
        )
        reply = _judge_model(config)(user_prompt, system_prompt)
        import re

        m = re.search(r"\d+", reply or "")
        if m and 1 <= int(m.group()) <= len(top):
            pick = top[int(m.group()) - 1]
            arm = dict((s.strip().lower(), a) for s, a in pool).get(
                pick.sql.strip().lower(), "judge"
            )
            logger.info("graph_v2: judge adjudicated a disagreement")
            return {"best": pick.sql, "winner_arm": f"judge:{arm}"}
        return {}

    def refine(state: GraphState) -> dict:
        """Execution-feedback revision of the winner (any model, generic prompt)."""
        best = state.get("best", "")
        groups = group_by_execution(
            [best], state.get("db_path"), timeout=config.selection.timeout_seconds
        )
        g = groups[0] if groups else None
        feedback = (
            "the query raised an execution error" if g is None or g.result is None
            else "the query executed but returned an empty result set (0 rows)"
        )
        schema_block, _, _ = render_schema_ddl(
            state["elements"], schema=state["schema"], expose_keys=True
        )
        system_prompt, user_prompt = build_revision_prompt(
            state["query"], schema_block, best, feedback
        )
        raw = _refiner_model(config)(user_prompt, system_prompt)
        fixed = finalize_sql(raw) if raw else ""
        out = {"refine_count": state.get("refine_count", 0) + 1}
        if fixed and fixed.strip().lower() != best.strip().lower():
            f_groups = group_by_execution(
                [fixed], state.get("db_path"), timeout=config.selection.timeout_seconds
            )
            fg = f_groups[0] if f_groups else None
            # Improvement-only: adopt the revision when it produces a clean result.
            if fg is not None and fg.result is not None and len(fg.result) > 0:
                logger.info("graph_v2: refine produced a clean result")
                out.update({"best": fixed, "winner_arm": "refine"})
        return out

    def verify(state: GraphState) -> dict:
        """Reviewer: the shared 3-stage grammar → schema → execution verifier."""
        from verifier.review import run_verification

        sql = SQL(text=state.get("best", ""), dialect="sqlite",
                  source="graph_v2", verified=False)
        vr = run_verification(
            sql, state["schema"], state.get("db_path"),
            vcfg=config.verifier, generation_count=10_000,
        )
        return {"verification_result": vr, "best": sql.text}

    return {
        "schema_link": schema_link,
        "generate_local": generate_local,
        "generate_remote": generate_remote,
        "vote": vote,
        "judge": judge,
        "refine": refine,
        "verify": verify,
    }


# ---------------------------------------------------------------------------
# Router + loop predicates (conditional edges)
# ---------------------------------------------------------------------------


def route_mode(config) -> List[str]:
    """MODE ROUTER: which generator arms run after schema_link."""
    return {
        "local": ["generate_local"],
        "remote": ["generate_remote"],
        "ensemble": ["generate_local", "generate_remote"],
    }[config.mode]


def needs_judge(state: GraphState, config) -> bool:
    """Judge fires when the top two result groups DISAGREE (CHASE selection),
    not only on exact ties — selection is the documented accuracy bottleneck."""
    if config.selection.judge == "off":
        return False
    pool = _dedupe(state.get("candidates", []))
    if len(pool) < 2 or not state.get("db_path"):
        return False
    groups = [
        g for g in group_by_execution(
            [s for s, _ in pool], state["db_path"],
            timeout=config.selection.timeout_seconds)
        if g.result is not None
    ]
    return len(groups) >= 2


def needs_refine(state: GraphState, config) -> bool:
    """Refine while the winner errors/returns-empty and rounds remain."""
    if state.get("refine_count", 0) >= config.refine.rounds:
        return False
    best = state.get("best", "")
    if not best or not state.get("db_path"):
        return False
    groups = group_by_execution(
        [best], state["db_path"], timeout=config.selection.timeout_seconds
    )
    g = groups[0] if groups else None
    return g is None or g.result is None or len(g.result) == 0


# ---------------------------------------------------------------------------
# Graph assembly (LangGraph when installed; sequential fallback otherwise)
# ---------------------------------------------------------------------------


def build_langgraph(config):
    """Compile the pipeline as a LangGraph StateGraph (parallel arm fan-out)."""
    from langgraph.graph import END, StateGraph

    nodes = make_nodes(config)
    g = StateGraph(GraphState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.set_entry_point("schema_link")
    arms = route_mode(config)
    for arm in arms:                      # fan out: arms run in parallel
        g.add_edge("schema_link", arm)
    for arm in arms:                      # fan in: candidates merge additively
        g.add_edge(arm, "vote")

    g.add_conditional_edges(
        "vote", lambda s: "judge" if needs_judge(s, config) else "refine_gate",
        {"judge": "judge", "refine_gate": "verify_or_refine"},
    )
    # judge falls through to the same refine gate
    g.add_node("verify_or_refine", lambda s: {})   # pass-through junction
    g.add_edge("judge", "verify_or_refine")
    g.add_conditional_edges(
        "verify_or_refine",
        lambda s: "refine" if needs_refine(s, config) else "verify",
        {"refine": "refine", "verify": "verify"},
    )
    g.add_edge("refine", "verify_or_refine")       # bounded loop via refine_count
    g.add_edge("verify", END)
    return g.compile()


class GraphOrchestrator:
    """v2 entry point — same ``run(initial_state) -> contract dict`` as v1.

    Uses LangGraph when importable; otherwise executes the SAME node functions
    sequentially (identical semantics minus arm parallelism), so the pipeline
    has no hard dependency and offline tests exercise real logic.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.nodes = make_nodes(config)
        try:
            self.graph = build_langgraph(config)
            logger.info("graph_v2: LangGraph engine")
        except ImportError:
            self.graph = None
            logger.info("graph_v2: sequential fallback engine (langgraph not installed)")

    def run(self, initial_state: dict) -> dict:
        state: dict = {
            "query": initial_state["query"],
            "schema": initial_state["schema"],
            "db_path": initial_state.get("db_path"),
            "db_id": initial_state.get("database_id")
            or getattr(initial_state["query"], "database_id", ""),
        }
        llm_tokens_before = 0

        if self.graph is not None:
            out = self.graph.invoke(state)
        else:
            out = self._run_sequential(state)

        counts = {"local": 0, "remote": 0}
        for _, arm in _dedupe(out.get("candidates", [])):
            counts[arm] = counts.get(arm, 0) + 1

        cost = LOCAL_COMPUTE_COST_USD if self.config.mode != "remote" else 0.0
        try:  # remote tokens accumulate on the per-query LLM instance
            llm = get_cache().get_llm_generator() if self.config.mode != "local" else None
        except Exception:
            llm = None
        # NOTE: get_llm_generator returns a fresh instance; token-based cost is
        # therefore approximated by candidate count in v2 (cost is isolated).
        cost += counts.get("remote", 0) * 1500 * REMOTE_TOKEN_COST_USD

        return {
            "sql": SQL(text=out.get("best", ""), dialect="sqlite",
                       source=self._source(), verified=False),
            "routing_decision": (
                RoutingDecision.LOCAL if self.config.mode == "local"
                else RoutingDecision.REMOTE
            ),
            "abstracted_prompt": None,
            "verification_result": out.get("verification_result"),
            "generation_source": self._source(),
            "winner_arm": out.get("winner_arm", "unknown"),
            "candidates_local": counts.get("local", 0),
            "candidates_remote": counts.get("remote", 0),
            "retrieved_tables": out.get("tables", []),
            "num_retrieved_columns": out.get("num_columns", 0),
            "cost_usd": cost,
            "privacy_loss": 0.0,
        }

    def _run_sequential(self, state: dict) -> dict:
        """The fallback engine: same nodes, same routing, in order."""
        state.update(self.nodes["schema_link"](state))
        for arm in route_mode(self.config):
            add = self.nodes[arm](state)
            state["candidates"] = state.get("candidates", []) + add["candidates"]
        state.update(self.nodes["vote"](state))
        if needs_judge(state, self.config):
            state.update(self.nodes["judge"](state))
        while needs_refine(state, self.config):
            state.update(self.nodes["refine"](state))
        state.update(self.nodes["verify"](state))
        return state

    def _source(self) -> str:
        return {"local": "slm", "remote": "llm"}.get(self.config.mode, "ensemble")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _dedupe(pool: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen, out = set(), []
    for sql, arm in pool:
        key = (sql or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append((sql, arm))
    return out


def _trivial(elements) -> str:
    for e in elements:
        if "." in e.name:
            return f"SELECT * FROM {e.name.split('.', 1)[0]} LIMIT 10"
    return "SELECT 1"


def _judge_model(config):
    """Judge callable: remote when configured/available, else the local model."""
    cache = get_cache()
    mode = config.selection.judge
    if mode in ("remote", "auto") and config.mode != "local":
        llm = cache.get_llm_generator()
        return lambda p, sp: (llm.complete(p, n=1, temperature=0.0,
                                           system_prompt=sp, raw=True) or [""])[0]
    slm = cache.get_slm(config.models.generator)
    return lambda p, sp: (slm.complete(p, n=1, temperature=0.0,
                                       system_prompt=sp, raw=True) or [""])[0]


def _refiner_model(config):
    """Refiner callable: mirrors the judge's model choice (strongest available)."""
    cache = get_cache()
    if config.mode != "local":
        llm = cache.get_llm_generator()
        return lambda p, sp: (llm.complete(p, n=1, temperature=0.0,
                                           system_prompt=sp) or [""])[0]
    slm = cache.get_slm(config.models.generator)
    return lambda p, sp: (slm.complete(p, n=1, temperature=0.0,
                                       system_prompt=sp) or [""])[0]

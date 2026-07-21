# AEGIS-SQL — Architecture (`aegis_feat_2`)

> ## ✅ Architecture decision (final, locked)
>
> **The `graph` pipeline (LangGraph: schema-link → route → [local SLM | remote abstract→LLM→
> reconstruct] → verify → bounded repair) is the system architecture.** The `multi_agent`
> "booster" harness has been **removed** after a full A/B: it never beat the simpler graph on
> any tested configuration (local SLM 46–49% vs graph 49%; remote gpt-4o 53% vs graph 55%),
> even after adding a pairwise-tournament selector, while costing ~40% more wall-time. The
> ambiguity resolver (`query_planner/`) was also removed (unused). The DP-abstraction / router
> **privacy** code is **kept** — it is the paper's three-axis thesis and the graph's remote path
> uses it.
>
> **Removed:** `agents/` (booster), `query_planner/` (ambiguity), the `orchestrator`/`agents:`/
> `ambiguity:` config, and the eval-driver toggle (now graph-only).
> **Final numbers (100q, seed 42, RAG v2 recall-fix):** graph local **49%**, graph remote
> **55%**. Retrieval recall is 98%; the ceiling is the generator, not the harness (§10–11).
>
> Sections 5 and the booster parts of §2 below are retained as the **experimental record** that
> led to this decision — they describe code no longer in the tree.

---

> **Historical scope note (pre-decision).** The sections below were written while the
> multi-agent booster was still a candidate. They remain accurate about the RAG v2 retriever,
> the privacy layer, verification, config, and the results/regression analysis — all still in
> the shipped graph system — but references to the "booster / `agents/` / `multi_agent`
> orchestrator / `selector_model`" describe the removed experiment, not the current code.

---

## 1. System overview

AEGIS-SQL turns a natural-language question (plus BIRD "evidence" hints) into an executable
SQLite query. The design premise, established empirically earlier in the project, is that
**the harness around the model — not the model — is the bottleneck**: a single-shot 7B SLM
and single-shot gpt-4o both plateaued ~11 points below gpt-4o's public BIRD number. So the
architecture is a *booster*: diverse candidate generation + execution-guided selection +
bounded self-correction, wrapped around a recall-first retriever.

Every query flows through one linear, explicitly-coded pipeline (no graph engine on the
active path):

```
                         ┌───────────────── privacy boundary (remote path only) ─────────────────┐
 Question + Evidence     │                                                                         │
        │                │   abstract → LLM (gpt-4o) generate → reconstruct                        │
        ▼                │                                                                         │
  SchemaLinker ── Router ─┤                                                                         │
  (RAG v2 slice)  (route) └───────────────────────────────────────────────────────────────────────┘
        │                    │  (local path)
        │                    ▼
        │            SLM (CscSQL-7B) generate  ──►  Candidate pool (multi-strategy, dedup)
        │                                                    │
        ▼                                                    ▼
  retrieved_tables                             Selector  (execution-vote majority
  num_columns                                             + optional trusted judge)
                                                             │
                                                             ▼
                                                  Refiner (bounded exec-feedback repair)
                                                             │
                                                             ▼
                                                  Reviewer (3-stage verification)
                                                             │
                                                             ▼
                                          predictions.jsonl contract dict
```

Two pipelines exist in the repo; the **active** one is selected by `orchestrator: multi_agent`
in `config.yaml`:

| `orchestrator` | Entry point | Status |
|---|---|---|
| `multi_agent` (**active**) | `agents/orchestrator.py :: MultiAgentOrchestrator` | The booster; everything below describes this path. |
| `graph` | `workflow/graph.py :: build_aegis_graph` | The original LangGraph pipeline, kept only for A/B comparison. |

Both consume the same `initial_state` dict and return the same contract dict, so the
evaluation tooling downstream is identical.

---

## 2. Data flow (one query, end to end)

Source: `agents/orchestrator.py :: MultiAgentOrchestrator.run()`.

1. **Schema linking** (`agents/schema_linker.py`). `cache.get_schema_retriever(db_id, schema)`
   returns a `SchemaRetriever` with precomputed BGE-M3 embeddings. `SchemaLinkerAgent.link()`
   produces `(schema_elements, retrieved_tables, num_columns)`:
   - if `slm.full_schema` and the DB is small (`≤ full_schema_max_columns`), inline every column;
   - else if `rag.multi_step` (default), run the RAG v2 `MultiStepRetriever` (§4);
   - else the legacy single-shot top-k + FK-closure retriever (also the pass-through fallback).
   Then best-effort **value grounding** attaches exact stored DB literals to text columns
   (via `dataclasses.replace` on *copies* — never mutating the shared cached schema objects).

2. **Routing** (`router/content_independent_router.py`, via `cache.get_router()`). Returns a
   `RoutingDecision`. The default config sets `router.force_local: true`, so every query routes
   **LOCAL** and the abstraction / remote-LLM code never executes.

3. **Path context** (`RunContext`, `agents/context.py`). The orchestrator builds a
   path-specific context carrying a model-agnostic `generate_fn(prompt, n, temperature,
   system_prompt, max_tokens)`:
   - **Local:** `generate_fn = slm.complete`, `reconstruct_fn = identity`, `source = "slm"`.
   - **Remote:** `_remote_context()` deep-imports `abstraction.*`, rewrites the question into
     placeholders, `generate_fn = llm.complete`, `reconstruct_fn = placeholder→real`,
     `source = "llm"` (§7).

4. **Candidate generation** (`agents/generator.py`). For each strategy in the path's list
   (`local_strategies` vs `remote_strategies`), build the strategy prompt
   (`prompts/sql_strategies.py :: build_prompt`), sample `candidates_per_strategy`, reconstruct +
   `finalize_sql` each, and dedup by lowercased text into a pool. Empty pool → single-shot
   fallback, then a trivial `SELECT * FROM <first_table> LIMIT 10`.

5. **Selection** (`agents/selector.py`). `generator.candidate_selector.select_best` executes
   every candidate and takes the **majority result-set** (execution-guided self-consistency).
   On a split vote with ≥2 non-empty results, an optional **judge** (trusted local SLM by
   default) breaks the tie.

6. **Refinement** (`agents/refiner.py`). Executes the winner; if it errors or returns 0 rows,
   it shows the model the (sanitized) execution feedback and asks for one corrected query,
   bounded by `agents.refine_rounds` and **improvement-only** (a repair is kept only if strictly
   better).

7. **Review** (`agents/reviewer.py` → `verifier/review.py`). 3-stage verification: grammar
   (sqlglot) → schema (table/column existence) → execution. Produces the `VerificationResult`.

8. **Cost + contract** (`workflow/costing.py`). Token-billed for remote, fixed for local.
   `run()` returns:
   ```
   { sql, routing_decision, abstracted_prompt, verification_result, generation_source,
     retrieved_tables, num_retrieved_columns, cost_usd, privacy_loss }
   ```
   `run_bird_evaluation.py` flattens this into `predictions.jsonl` and auto-runs EX + VES.

---

## 3. Model components

| Component | Default model | Where | Role |
|---|---|---|---|
| **Local SLM (FSLM)** | `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-7B-Instruct` | `generator/slm_generator.py` | Local SQL generation (greedy candidate + temperature samples). |
| **Remote LLM (FLLM)** | `gpt-4o` (OpenAI) | `generator/llm_fallback.py` | Remote generation on the abstracted query (privacy path). |
| **Embeddings** | `BAAI/bge-m3` | `retriever/embedding_models.py` | Dense + sparse + ColBERT vectors for hybrid schema retrieval and re-rank. |
| **Model cache** | — | `workflow/model_cache.py` | Process-singleton loaders (SLM, LLM, retriever per DB, router) so nothing reloads mid-run; `warmup()` pre-encodes every unique DB schema. |

The SLM generation contract (`generator/slm_generator.py`):
- `generate_candidates()` / `complete()` return **candidate 1 = greedy** (`do_sample=False`,
  deterministic) plus **candidates 2..n = temperature samples** (`do_sample=True`, batched).
  On the active multi_agent path the sampling temperature is `agents.generation_temperature`
  (**0.7**); `slm.selection_temperature` (0.8) is only the fallback used when a caller passes no
  temperature.
- ⚠️ **No torch/CUDA seed is set before sampling** — see §11 (this makes the sampled
  candidates, and therefore the final EX, vary run-to-run). A new `slm.generation_seed` knob
  fixes this when set.

---

## 4. The RAG v2 retrieval pipeline

Source: `retriever/pipeline.py` (orchestration) + `retriever/fusion.py` (pure scoring/budget)
+ `retriever/query_decompose.py` + `retriever/value_index.py`. Active when `rag.multi_step: true`.

**Why it exists.** Measured on the 1,534-query run, the legacy single-shot retriever gave
simple questions ~7.5 tables when 1.76 were needed (noise that inverted simple-vs-moderate
accuracy), and 247/1,534 predictions used the wrong table *set*, concentrated in FK-maze DBs
(financial 43%). RAG v2 trades that blunt top-k for a precise, agentic loop:

```
decompose ─► per-sub-query hybrid search ─► RRF fusion (+ table-card boost)
     │                                             │
   literals ─► value retrieval (DB LIKE probes) ───┤
                                                    ▼
             metadata boosts ─► adaptive budget (evidence tables + FK bridges + column caps)
                                                    │
                       coverage check ── uncovered entity? ─► relaxed round 2
                                                    │
                            optional ColBERT re-rank of the final slice
```

Stage detail:

1. **Decompose** (`query_decompose.py`). Extract literals (quoted strings, proper-noun spans,
   numbers) and build focused sub-queries — the *full question first* (so decomposition can only
   add recall), then evidence column references, then one per entity.
2. **Per-sub-query hybrid search.** Each sub-query is scored against the precomputed BGE-M3
   dense+sparse column embeddings (`retrieve_scored`).
3. **RRF fusion** (`fusion.py :: rrf_fuse`). Reciprocal-Rank-Fusion across sub-queries; the full
   question counts double so one entity can't dominate.
4. **Table-card boost.** A whole-table semantic match lifts every member column (catches
   table-level evidence individual column embeddings miss).
5. **Value retrieval** (`value_index.py`). For each literal, bounded read-only `LIKE` probes on
   TEXT columns locate the exact stored value — the strongest possible evidence a column belongs
   in the slice, and the literal is grounded into the prompt (`= 'Continuation School'`, not
   `'Continuation'`).
6. **Metadata boosts + adaptive budget** (`fusion.py :: apply_boosts`, `adaptive_budget`).
   Boost name-matches, value-hits, and numeric columns for aggregate questions; then choose the
   final table set + per-table columns (see §11 — this is where the recall regression lives).
7. **Coverage round.** If a literal is still uncovered, a relaxed round 2 re-probes with looser
   matching.
8. **ColBERT re-rank** (`rag.rerank`). Optional multi-vector re-ordering of the final slice with
   the same BGE-M3 model.

---

## 5. The multi-agent booster

Six single-responsibility agents, wired in `agents/orchestrator.py`, sharing one `RunContext`.
Diversity across strategies is the lever the execution-vote selector exploits.

| Agent | File | Responsibility |
|---|---|---|
| **SchemaLinker** | `agents/schema_linker.py` | Retrieve the focused schema slice (RAG v2 / legacy) + value grounding. |
| **Router** | `router/content_independent_router.py` | Content-independent LOCAL/REMOTE decision (forced LOCAL by default). |
| **CandidateGenerator** | `agents/generator.py` | Multi-strategy sampling → reconstruct → finalize → dedup pool. |
| **Selector** | `agents/selector.py` | Execution-vote majority + optional trusted-model tie-break judge. |
| **Refiner** | `agents/refiner.py` | Bounded, improvement-only execution-feedback repair. |
| **Reviewer** | `agents/reviewer.py` | 3-stage verification verdict. |

**Model-aware strategies.** The local SLM is a *specialist* (CscSQL/CSC-SQL, trained for direct
generation), so `local_strategies = [direct]` (execution-guided self-consistency); CoT tends to
hurt it and adds latency. The remote LLM is a *general reasoner*, so
`remote_strategies = [direct, query_plan]` (adds chain-of-thought). Strategies live in
`prompts/sql_strategies.py` (`direct`, `query_plan`, `divide_and_conquer`).

**Selection.** `select_best` groups candidates by `frozenset(result_rows)` and votes. The judge
(CHASE-SQL-style) fires only on a genuine split and always runs on the **trusted local model**
on the remote path, so it adds zero leakage (preserves the paper's Theorem 1).

---

## 6. Verification & repair

`verifier/review.py :: run_verification` — one implementation shared by both pipelines:

1. **Grammar** — `GrammarVerifier` (sqlglot parse). Fail → `GRAMMAR_FAIL`.
2. **Schema** — `SchemaVerifier` checks every table/column exists. Fail → `SCHEMA_FAIL`.
3. **Execution** — `ExecutionVerifier` runs the query (timeout `verifier.timeout_seconds`, 30s in
   the eval config). An executed-but-empty result is a *soft* fail that can trigger one
   value-aware repair (`repair_on_empty`), bounded by `max_repair_attempts`.

A `VerificationStatus.PASS` means the SQL **parses, references real columns, and executes without
error** — it does **not** mean the answer is correct. This is why a run can show 90/100 "pass"
but 47/100 EX (§10): the gap is wrong-but-executable SQL.

---

## 7. The privacy layer (parked, but part of the thesis)

The paper's contribution is *content-independent routing with DP abstraction*: sensitive queries
go remote only after value-like tokens are replaced by semantic placeholders, and candidates are
reconstructed inside the trust boundary. This lives in `abstraction/` (DP abstractor,
placeholder vocab, reconstruction, sensitivity policy), `router/`, and `query_planner/`.

On the **default local config it does not execute** — `_remote_context()` deep-imports
`abstraction.*` only on the REMOTE route, and `router.force_local: true` forces LOCAL. It is kept
because it is the research thesis and drives the hybrid/remote experiments; it is *isolated*, not
wired into the local accuracy path. **Do not delete it during cleanup.**

---

## 8. Configuration reference (`config.yaml` → `config.py`)

| Block | Key knobs | Notes |
|---|---|---|
| `orchestrator` | `multi_agent` \| `graph` | Selects the active pipeline. |
| `embedding` | `model`, `device` | BGE-M3 for retrieval. |
| `slm` | `model`, `num_candidates`, `selection_temperature`, `expose_keys`, `enable_value_grounding`, `full_schema`, `retrieval_top_k`, `max_expanded_tables` | Local generation + legacy-retrieval knobs. |
| `llm` | `provider`, `model`, `temperature` | Remote arm (gpt-4o). |
| `privacy` | `epsilon`, `abstraction_enabled`, `value_aware_abstraction` | DP abstraction (remote path only). |
| `router` | `force_local`, `force_remote`, `threshold_complexity` | Routing; `force_local: true` for the local eval. |
| `rag` | `multi_step`, `max_tables`, `per_table_columns`, `per_query_top_k`, `max_rounds`, `value_retrieval`, `table_cards`, `rerank` | The RAG v2 pipeline. |
| `agents` | `local_strategies`, `remote_strategies`, `candidates_per_strategy`, `generation_temperature`, `refine_rounds`, `judge_enabled`, `judge_model`, `selection_timeout` | The booster. |
| `verifier` | `timeout_seconds` (30), `max_repair_attempts`, `repair_on_empty` | 3-stage verification. |
| `evaluation` | `seed`, `metrics` | Eval reproducibility + metric list. |

---

## 9. Evaluation harness

- **Driver:** `run_bird_evaluation.py` — loads BIRD (stratified sample by difficulty via
  `--num_queries`/`--seed`), warms the cache, runs the orchestrator per query, writes
  `predictions.jsonl` + a `config_snapshot.yaml`, then auto-runs EX and VES.
- **Metrics:** `evaluation/evaluator_ex.py` (Execution Accuracy — result-set match vs gold) and
  `evaluation/evaluator_ves.py` (Valid Efficiency Score). `evaluation/analyze_retrieval.py`
  computes table recall from `predictions.jsonl`.
- **Reproducibility caveat:** `--seed` fixes *which* queries are sampled, **not** the SLM's
  candidate decoding (§11).

Command (local 100-query sample used for the runs below):
```
python run_bird_evaluation.py --config config.yaml --seed 42 --num_queries 100 --stratify \
       --output_name test0.1_100_local
```

---

## 10. Results

**Latest local run** (`test0.1_100_local`, 100q, seed 42, `force_local`, `orchestrator=multi_agent`):

| Difficulty | EX | n |
|---|---|---|
| Simple | 48.33% | 60 |
| Moderate | 43.33% | 30 |
| Challenging | 50.00% | 10 |
| **Overall** | **47.00%** | 100 |

Prediction breakdown (from `predictions.jsonl`): 100/100 routed local; 90/100 pass verification,
10 execution-fail; **0 empty/stub predictions**; grammar & schema valid on all 100; generation
took 25.3 min (~15 s/query; 13 queries > 15 s). **Table recall: 85/100** (§11).

**Project history for context:**

| Configuration | Scope | EX | Table recall |
|---|---|---|---|
| Remote gpt-4o + RAG v2 (ensemble) | full 1,534 | **61.93%** | — |
| Local SLM + RAG v2 (best earlier) | 100q | ~50% | ~86% |
| Local SLM + RAG v2 (baseline this investigation) | 100q | 47.00% | 85% |
| **Local SLM + RAG v2 + recall fix** | 100q | **46.00%** | **98%** |
| Local SLM single-shot (pre-booster) | 100q | ~42–45% | — |

**The full 2×2: orchestrator × model (same 100q, seed 42, RAG v2 recall-fix, healthy GPU).**

| Model arm | `graph` (single-shot) | `multi_agent` (booster) |
|---|---|---|
| **Local SLM** (CscSQL-7B) | **48–49%** | 46% |
| **Remote gpt-4o** (no abstraction) | **55%** | 53% |

Two clean findings:

1. **The model is the lever, not the harness.** Swapping SLM→gpt-4o adds **+6–7 EX** on both
   orchestrators; swapping graph→booster moves nothing (or slightly down). Retrieval recall (98%),
   verification, and pipeline choice are all second-order next to the generator.

2. **The booster does not beat the simple graph on *either* path** — it is marginally behind on
   both (46 vs 48–49 local; 53 vs 55 remote), and ~2.5× slower on remote (mean 11.6 s vs 4.5 s;
   31.6 vs 20 min wall). The expectation that diversity + judge would pay off with the *general*
   reasoner did not hold. Diffing the two remote runs: they differ on 89/100 predictions, but most
   are **cosmetic** (alias style); on the handful of *semantic* divergences the booster sometimes
   selects a **plausible-but-wrong** candidate over gpt-4o's clean greedy answer — e.g. q303, where
   graph emits `SUM(CASE WHEN bond_type='=' ...)` (correct) and the booster picks `COUNT(*)`
   (wrong). This is the CHASE-SQL selection-gap in miniature: **without a *trained* selector, extra
   candidates add noise the execution-vote + heuristic judge cannot reliably resolve, so selection
   is net-neutral-to-negative.** (Minor: inline `MetricsCalculator` vs standalone `evaluator_ex`
   disagree by ~1 pt on the graph runs — evaluator quirk, not the pipeline.)

**Implication.** The booster's promise (candidates + selection > single-shot) is currently
*unrealized* because the selector is heuristic. The decisive experiment is to swap in the trained
pairwise selector (`aegis-selector`, CHASE-SQL: +4.17 EX) and re-run the 2×2.

**Booster-fate experiment (wired, ready to run).** `agents/pairwise_selector.py` implements the
trained selector: it loads the fine-tuned model, executes each top candidate for a result preview,
and runs a round-robin **both-orderings** tournament (position-bias-cancelled) using the notebook's
exact `A`/`B` prompt; the winner replaces the heuristic judge. It is gated by one config key and
lives entirely inside `agents/`, so it is removed with the booster if the verdict is negative.

Two ways to run it — pick by how much time you have:

- **Fast, no training** — `agents.selector_model: pairwise`. Runs the same round-robin tournament
  but powered by the **model already loaded** (the 7B SLM on the local arm, gpt-4o on the remote
  arm) instead of a trained model. This isolates the *selection mechanism* (pairwise A/B over the
  top candidates, always, both orderings) from the *current* heuristic (listwise, split-vote-only) —
  no download, no training, one config flag. Best first probe of the booster's fate.
- **Full lever** — train + publish the 3B selector (`aegis-selector/selector_finetune.ipynb` →
  `Daveonyango254/aegis-sql-selector-3b`), then `agents.selector_model: Daveonyango254/aegis-sql-selector-3b`.
  Only worth the training time if the fast probe is promising or inconclusive.

Then re-run the 2×2 (both orchestrators × local/remote, 100q seed 42) and compare `multi_agent` EX
against the `graph` baselines above. On the remote arm the pairwise judge is gpt-4o itself — a
strong reasoner adjudicating its own candidates, which is where a selection lift is most likely.

**Decision rule.** If `multi_agent` + trained selector clears the `graph` baseline by more than
noise (>~3 pts) on the remote arm, the booster is justified — keep it. If not, the booster is
retired: delete `agents/` + the `orchestrator` toggle and run the `graph` pipeline only (the
simpler, equal-or-better baseline), per the streamlining plan in `CLEANUP.md`.

**The decisive measurement.** The recall fix (§11) raised table recall **85% → 98%** — it works
exactly as designed — yet EX stayed **flat (47 → 46%, a 1-query swing = noise)**. Spot-checking the
recovered queries confirms why: they now retrieve *all* their tables (often the full DB schema),
but the 7B SLM still produces semantically wrong SQL — skipped joins (`account⋈disp`), wrong table
for a column (`races.url` vs `seasons.url`), misread questions, wrong aggregation units.
**Conclusion: schema linking is not the accuracy bottleneck on this branch — SQL generation /
selection is.** Retrieval is now effectively solved (98% recall); the ceiling is the model's
reasoning, which is exactly where the remote arm's 61.93% comes from.

---

## 11. Regression analysis — why the last local run was 47% (not 50%+)

**Question posed:** did something in the config change, or is this a real regression? Two checks:

### 11a. Config check — no meaningful drift (red herring)

Diffing the run's `config_snapshot.yaml` against the branch `config.yaml` yields **exactly one
difference**: `router.force_local` is `true` in the snapshot and `false` in the branch default.
That is expected — the local eval forces local routing. Every accuracy-relevant knob
(`rag.*`, `agents.*`, `slm.*`, `verifier.*`) is identical. **The 47% is not caused by a config
change.** The snapshot *is* the 47% run's config.

### 11b. Root cause — retrieval **table recall dropped to 85%** (dominant)

Parsing the ground-truth SQL for each query and comparing its tables against `retrieved_tables`:

- **Table recall = 85/100.** 15 queries are missing a required table from retrieval and are
  therefore **unwinnable regardless of the model.**
- The misses cluster in FK-maze DBs — the `financial` database dominates: **`client` dropped 5×,
  `disp` dropped 3×** (q100, q128, q159, q164, q172, q187, q189, …), plus `rulings`, `votes`,
  `postlinks`, `seasons`, `laptimes`, `fastest_lap_times`, `set_translations`, `country`.

**Mechanism** (`retriever/fusion.py :: adaptive_budget`, default `max_tables=4`):
1. Tables are ranked by summed member-column score; only the top `max_tables` are kept (plus any
   table holding a value-hit).
2. `_bridge_tables()` then adds FK **intermediates** *between kept tables* to keep the slice
   join-connected.
3. **The gap:** an *endpoint* evidence table that scores below the top 4 and has no value-hit is
   dropped — and because it is an endpoint (not an intermediate between two kept tables), the
   bridge logic cannot recover it. In `financial`, a question about clients whose SQL joins
   `account → disp → client` loses `client` (low score, no literal probed) and then `disp` too
   (its only reason to exist was to bridge to the now-absent `client`). That is exactly the
   `client`/`disp` miss pattern in the data.

The legacy single-shot path (`rag.multi_step: false`) used `retrieval_top_k=80` + FK closure and
had ~99% table recall — which is why the `SchemaLinker` docstring still (now inaccurately) claims
"~99% table recall". RAG v2 improved *precision* on simple queries but regressed *recall* on
FK-maze queries.

### 11c. Contributing cause — non-deterministic sampling (the 47 vs 50 gap is partly noise)

`--seed 42` only fixes which queries are sampled. The SLM generates candidates 2..n with
`do_sample=True` at `agents.generation_temperature = 0.7` and **no torch/CUDA seed**
(`generator/slm_generator.py :: _run_generation`), and the execution-vote winner can be one of
those sampled candidates. So the final EX **varies run-to-run**. On a 100-query sample the 95%
confidence half-width is ≈ ±9 points; a 3-point swing (47 vs 50) is well inside run-to-run noise.
In other words: part of the "regression" is not a regression at all — it is the same system
landing on a different sample of its own stochastic output. (The only seeds in the repo,
`evaluation/sampling.py`, seed *query selection*, not decoding.)

### 11d. Recommended adjustments (to durably clear 50%+)

Ranked by expected impact. Fixes **①** and **④** are **implemented on this branch** (independently
confirmed by an adversarial verification pass); **②** and **③** are documented levers the user can
enable on the GPU box if ① is not enough.

**① Recall-first adaptive budget — IMPLEMENTED** (`retriever/fusion.py`, `retriever/pipeline.py`,
`config.rag`). Two composable parts:
   - `rag.max_tables` raised **4 → 6** (mean retrieved was 4.44, so this mainly helps the FK-maze
     queries needing 5–7 tables).
   - `adaptive_budget` now takes `protected_tables` — tables the question **names** (with
     singular/plural folding, computed by `MultiStepRetriever._name_matched_tables`) bypass the
     cap alongside value-hit tables. Protecting `client` keeps it in `kept`, which lets the
     existing bridge pass pull `disp` back. *Expected:* recall 85% → ~90–95%; each recovered
     query is a potential +1 EX. **Limitation (verified):** name-match only recovers tables named
     *literally* — an endpoint referred to indirectly ("account owners" → `client`) still drops;
     that residual needs ② or ③.

**② One-hop FK closure of anchors — documented lever.** In `adaptive_budget`, also pull the FK
neighbours (key columns only) of value-/name-matched *anchor* tables into `kept`, mirroring the
legacy `_expand_with_foreign_keys`. Recovers indirectly-referenced endpoints. *Risk:* medium —
must be bounded (anchors only, cap added tables) or it re-inflates the prompt on FK-dense schemas.
Enable this if ① leaves recall below ~97%.

**③ Instant safe restore — `rag.multi_step: false`.** Falls back to the legacy single-shot path
(`top_k=80` + one-hop FK closure, **no table cap**, documented ~99% recall). One flag, instantly
reversible; trades RAG v2's precision/noise-reduction for guaranteed recall. Use as the A/B
baseline / stabiliser.

**④ Reproducible sampling — IMPLEMENTED** (`slm.generation_seed`, default `null`). When set (e.g.
42), `_run_generation` seeds torch/CUDA before `do_sample=True`, so candidate sampling — and thus
EX — is stable run-to-run. No accuracy gain, but it removes the "is 47 vs 50 real or noise?"
ambiguity so the recall fix can be measured cleanly.

> **Measured outcome (100q, seed 42, GPU healthy):** the fix raised table recall **85% → 98%**
> (better than the projected 90–95%), but EX was **flat at 46%** — retrieval was not the
> bottleneck. Recovered queries get their tables and still generate wrong SQL. So ① is the *correct
> retrieval result to keep* (clean 98% recall for the paper), but the accuracy lever is elsewhere —
> see §11e.

### 11e. Where the accuracy actually lives (the real bottleneck)

With recall at 98% and EX flat, the ceiling is the **local 7B SLM's SQL reasoning + candidate
selection**, not schema linking. The evidence: given the full correct schema, the model still
skips required joins, picks the wrong table for a projected column, and misreads question intent.
The levers that move this number, in ROI order:

1. **A trained pairwise selection model** — CHASE-SQL's ablation shows +4.17 EX over execution
   voting, and the oracle-vs-achieved gap (82.8 vs ~73) is the largest single lever in the
   literature. The `aegis-selector` notebook (already built, separate repo) trains exactly this;
   wiring its Hub model into `SelectorAgent`'s judge is the highest-value next step.
2. **A stronger / better-prompted generator** — the remote gpt-4o arm already reaches **61.93%** on
   the same retrieval. For a local-first system, a larger or SQL-specialist-fine-tuned SLM, or more
   diverse candidates (raise `agents.candidates_per_strategy` / add `query_plan` on the local path
   and measure), is the lever.
3. **Adaptive schema breadth** — because the weak model sometimes does *worse* with the full schema
   (more tables = more ways to pick wrong), give simple/single-table queries a tight slice and
   expand only for join-maze queries. This is refinement, not a fix — it's EX-neutral here — but it
   removes the small simple-query noise the `max_tables: 6` bump introduced.

**Bottom line:** the retrieval regression is fixed and retrieval is no longer where points are
lost. Do not spend further effort tuning RAG for accuracy; invest in selection (①) and the
generator (②).

---

## 12. Design rationale & trade-offs

- **Booster over bigger model.** Diverse candidates + execution voting is the cheapest reliable
  test-time-compute lever in the text-to-SQL literature (CHASE-SQL, CSC-SQL, XiYan-SQL). It lifts
  a fixed model without retraining.
- **Recall-first retrieval — mostly.** Strong models are hurt more by a *missing* table than by a
  few extra columns. RAG v2 tried to also win precision on simple queries; the lesson from §11 is
  that the recall floor must be protected first (hence the recall-first budget fix).
- **Model-aware strategies.** Specialist SLM ≠ general LLM: forcing CoT on the specialist hurt
  accuracy and added a latency tail, so the strategy list is chosen per path.
- **Privacy isolated, not entangled.** The DP abstraction is confined to the remote path so it
  never taxes the local accuracy number, while remaining available for the hybrid/privacy
  experiments the paper reports.
- **One verification implementation.** `verifier/review.py` is shared by both pipelines so the
  A/B comparison is apples-to-apples.
- **Explicit code over a graph.** The booster's control flow (generate → select → refine) is
  linear and far more debuggable as plain Python than as a graph; the LangGraph pipeline is kept
  only for comparison.

---

## 13. Repository layout

```
agents/            ← ACTIVE booster: orchestrator + 6 agents + RunContext
retriever/         ← RAG v2: pipeline, fusion (budget), decompose, value_index, value_sampler,
                     schema_retriever, embedding_models
generator/         ← slm_generator, llm_fallback, candidate_selector, sql_postprocess
prompts/           ← schema_render, sql_strategies, prompt_manager, templates.yaml
verifier/          ← grammar / schema / execution verifiers, feedback_generator, review
workflow/          ← model_cache, embedding_cache, costing  (+ graph, state — legacy pipeline)
abstraction/       ← PARKED privacy layer (DP abstraction, reconstruction, vocab, policy)
router/            ← content-independent router (privacy thesis; used on the remote path)
evaluation/        ← BIRD loader, EX/VES evaluators, sampling, retrieval analyzer, metrics
config.py / config.yaml   ← Pydantic config + the active config
run_bird_evaluation.py    ← the canonical eval driver
tests/             ← offline unit tests (mock models / in-memory sqlite)
```

See `CLEANUP.md` (generated alongside this doc) for the streamlining pass: which redundant
entry-points / duplicate tests were removed and what was deliberately kept (all parked privacy
code, the legacy graph unless A/B is dropped).

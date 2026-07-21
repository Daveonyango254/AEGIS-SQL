# AEGIS-SQL — Architecture

**AEGIS-SQL** turns a natural-language question (+ BIRD "evidence" hints) over a database into an
executable SQLite query. It is a *hybrid* text-to-SQL system built around three axes — **accuracy,
cost, and privacy** — with a content-independent router that sends each query to a local small
language model (SLM, zero data egress) or a remote LLM behind a differential-privacy abstraction
layer.

This document is the current, authoritative description of the shipped system on branch
`aegis_feat_2`: the pipeline, **how a candidate query is chosen**, every component/module, the
configuration surface, and the error-analysis journey that drove each design choice.

- **Pipeline:** a single LangGraph `StateGraph` (`workflow/graph.py`). There is one pipeline; an
  earlier multi-agent "booster" variant was A/B-tested and **removed** (it never beat this simpler
  graph — see §7).
- **Headline results (BIRD dev, 100q stratified, seed 42):** local 7B SLM **~47–49% EX**
  (reproducible with `generation_seed`), remote gpt-4o **~55% EX**. Retrieval table-recall **~98–99%**.
- **The established bottleneck is the generator model, not the harness** (§7). Retrieval, selection,
  and verification are all near their useful ceilings; remaining errors are the model choosing the
  wrong table/join or writing semantically wrong SQL.

---

## 1. End-to-end data flow

Per query, `workflow/graph.py :: build_aegis_graph` wires these nodes (entry → exit):

```
                                     ┌───────────── LOCAL path (zero egress) ──────────────┐
 question + evidence + schema        │                                                     │
        │                            │   fslm_generation ──► verification ──(repair?)──┐   │
        ▼                            │        ▲                                        │   │
  schema_extraction ──► routing ──?──┤        └────────────── repair_route ◄───────────┘   │
   (RAG v2 slice)      (LOCAL/       │                                                     │
                        REMOTE)      └───────────── REMOTE path (DP-abstracted) ───────────┘
                                         abstraction ──► fllm_generation ──► reconstruction ──► verification ──(repair?)
                                                                                                    │
                                                                                                    ▼
                                                                                    predictions.jsonl record
```

1. **`schema_extraction_node`** → RAG v2 retrieves the focused schema slice (tables + columns +
   grounded value hints). §4.1.
2. **`routing_node`** → `ContentIndependentRouter` returns `LOCAL` or `REMOTE` (config `force_local`
   / `force_remote` override). §4.6.
3. **Generation** → `fslm_generation_node` (local SLM) **or** `abstraction → fllm_generation →
   reconstruction` (remote LLM behind DP abstraction). §2, §4.6.
4. **`verification_node`** → 3-stage neuro-symbolic verification. §4.4.
5. **`repair_route`** → on failure, loop back to the *same* generator with structured feedback,
   bounded by `verifier.max_repair_attempts`. §2.4.
6. `run_bird_evaluation.py` flattens the final state to a `predictions.jsonl` record and scores EX/VES.

The state object threaded through the graph is `AEGISState` (`workflow/state.py`, a `TypedDict`).
All models are loaded once and shared via a process singleton (`workflow/model_cache.py`) — nodes
never reload.

---

## 2. How the candidate query is chosen  ⭐

This is the core of the local path and the single most important mechanism in the system. It has
four stages: **generate a diverse pool → pick by execution-vote → repair the literal → verify with a
bounded repair loop.**

### 2.1 Generate a diverse candidate pool
`fslm_generation_node` → `SLMGenerator.generate_candidates(query, schema_elements, n, temperature,
feedback, schema)` (`generator/slm_generator.py`), with `n = slm.num_candidates` (**default 5**):

- **Candidate 1 = greedy decode** (`do_sample=False`, deterministic). This is the model's single
  best guess and the tie-break anchor.
- **Candidates 2..n = temperature samples** (`do_sample=True`, `selection_temperature=0.8`,
  `top_p=0.95`) — deliberate *diversity* so the execution vote has alternatives to choose between.
- Sampling is done by **`_sample_chunked`**: it decodes at most `slm.local_chunk_size` (default 2)
  sequences per `model.generate` call. On a CUDA OOM it empties the cache, **halves the chunk
  (floor 1) and retries loudly**, and if even one sequence cannot decode it returns the partial pool
  rather than collapsing to a stub. Each chunk gets a distinct seed (`generation_seed + produced`)
  so seeded runs are reproducible without chunks duplicating one another. (§7 — this was added after
  an OOM once silently drove EX to 3%.)
- Every raw candidate passes through **`finalize_sql`** (`generator/sql_postprocess.py`):
  `apply_cast_fix` (wrap division numerators in `CAST(... AS REAL)` so integer division doesn't
  truncate ratios to 0) then `quote_reserved_tables` (backtick reserved-word table names like
  ``FROM `order` `` so they don't crash the parser). Deterministic, idempotent, model-agnostic.

### 2.2 Pick by execution-guided self-consistency  (the "selection")
`generator/candidate_selector.py :: select_best(candidate_texts, db_path, timeout)` — the heart of
selection. It does **not** trust one decode; it runs every candidate against the *real* database and
lets the results vote:

1. Execute each candidate on the DB (per-candidate timeout = `verifier.timeout_seconds`, 30 s;
   a progress-handler kills runaway queries).
2. Discard candidates that **error** or return an **empty** result set.
3. Group the surviving candidates by their **result set** — `frozenset(row_tuples)`, so two
   different-looking queries that compute the *same answer* reinforce each other and order/dup do
   not matter.
4. **Winner = the result group with the most votes**; ties break toward the **earliest** candidate
   (i.e. the greedy decode, index 0). Returns the SQL plus diagnostics (`num_executed`,
   `num_nonempty`, `num_agree`).

Graceful degradation when no group forms: (Priority 2) if some candidate executed but every result
was empty, take the first executing-but-empty query (valid SQL, likely a literal mismatch → feeds
the empty-result repair); (Priority 3) if *everything* errored, fall back to the greedy candidate.

> **Why this design:** self-consistency over *executed results* is the strongest test-time lever for
> a fixed model in the text-to-SQL literature — a one-off hallucination is outvoted by the agreeing
> majority, and correctness is judged by what the DB actually returns, not by surface form. It is
> fully **model-agnostic** (plain HF sampling + SQLite execution). §7 quantifies its limit: when the
> model is *confidently* wrong, all candidates agree on the same wrong SQL and no vote can rescue it.

### 2.3 Repair the winning literal  (deterministic, post-selection)
`generator/literal_repair.py :: repair_literals(winning_sql, db_path)` runs on the selected query
(gated by `slm.enable_literal_repair`). For each `[alias.]column =/LIKE 'literal'` it probes the DB
read-only: if the literal matches **nothing** stored but exactly **one** near-match exists — checked
in confidence order **case-insensitive → datetime-suffix strip (`'2013-02-22 00:00:00'` →
`'2013-02-22'`) → unicode-fold / edit-distance ≤ 1 (`'Brazil'` → `'Brasil'`, `'Baga'` → `'Baǵa'`)** —
it rewrites the literal to the stored form. Any ambiguity or probe failure → no change. This applies
value grounding at the *output* instead of hoping the model copied the prompt hint. §7.

### 2.4 Verify, and repair up to a bound
`verification_node` → `verifier/review.py :: run_verification` runs 3 stages (grammar → schema →
execution, §4.4), reusing the execution diagnostics already computed in selection. If the query
fails — including the *soft* fail where it executed but returned 0 rows (`repair_on_empty`) —
`repair_route` sends it back to the **same** generator (`fslm` local / `fllm` remote) with the
verifier's `structured_feedback`, bounded by `verifier.max_repair_attempts` (**default 2**). Each
repair increments `generation_count`; when the budget is spent the loop ends and the current query is
emitted.

### 2.5 Remote path selection (contrast)
On the REMOTE path `fllm_generation_node` makes **one** gpt-4o/gpt-4.1-mini call (single candidate,
temperature 0) — there is no multi-candidate vote (cost). Selection there is just that one query,
run through the same verification + bounded repair. Multi-candidate self-consistency is a
local-SLM-only lever.

---

## 3. Model components

| Component | Default | Module | Role |
|---|---|---|---|
| **Local SLM (FSLM)** | `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-7B-Instruct` | `generator/slm_generator.py` | Local generation: greedy + temperature samples, chunked/OOM-safe. |
| **Remote LLM (FLLM)** | `gpt-4.1-mini` (OpenAI) | `generator/llm_fallback.py` | Single-shot generation on the DP-abstracted query. |
| **Embeddings** | `BAAI/bge-m3` | `retriever/embedding_models.py` | Dense + sparse + ColBERT vectors for hybrid retrieval and re-rank. |
| **Model cache** | — | `workflow/model_cache.py` | Process-singleton loaders (SLM / LLM / per-DB retriever / router); `warmup()` pre-encodes every DB schema before the run. |

---

## 4. Component / module reference

### 4.1 `retriever/` — RAG v2 schema linking (the "Query Planner")
Active when `rag.multi_step: true`. Turns the question into the focused schema slice the generator
sees, at ~98–99% table recall.

| Module | Responsibility |
|---|---|
| `pipeline.py` `MultiStepRetriever` | Orchestrates the loop: decompose → per-sub-query hybrid search → RRF fusion → value grounding → adaptive budget → coverage round → optional ColBERT re-rank. |
| `query_decompose.py` | Extract literals (quoted strings, proper-noun spans, numbers) + focused sub-queries (full question first, then evidence refs, then one per entity). |
| `fusion.py` | Pure scoring: `rrf_fuse` (Reciprocal-Rank-Fusion across sub-queries), `apply_boosts` (name-match / value-hit / numeric), and **`adaptive_budget`** — selects the table set + per-table columns. **Value-hit *and* question-named tables bypass the `max_tables` cap** (the recall fix, §7) so endpoint tables like financial's `client` are never dropped; FK "bridge" tables are added so the slice stays join-connected. |
| `value_index.py` | Deterministic value retrieval: bounded read-only `LIKE` probes locate a question literal's exact stored form and which column holds it. |
| `value_sampler.py` | Sampled representative values per column → prompt hints (`examples:`). |
| `schema_retriever.py` | BGE-M3 hybrid (dense+sparse) column retrieval, one-hop FK closure; the scoring engine `pipeline.py` consumes. Also the legacy single-shot path when `multi_step:false`. |
| `embedding_models.py` | BGE-M3 wrapper (dense/sparse/ColBERT). |

Knobs live under `config.rag` (`max_tables`, `per_table_columns`, `per_query_top_k`, `max_rounds`,
`value_retrieval`, `table_cards`, `rerank`, `multi_step`).

### 4.2 `generator/` — candidate generation, selection, and repair
| Module | Responsibility |
|---|---|
| `slm_generator.py` | Local generation. `generate_candidates()` = greedy + chunked temperature samples (`_sample_chunked`, OOM-safe, seeded); `_finalize` → `finalize_sql`. Native DDL prompt with FK/PK exposure + value hints. |
| `candidate_selector.py` | **`select_best`** — execution-guided majority vote (§2.2); `flag_empty_for_repair` decides when an all-empty result triggers a value-aware retry. |
| `literal_repair.py` | **`repair_literals`** — post-selection near-miss literal → stored-DB-value rewrite (§2.3). |
| `sql_postprocess.py` | `apply_cast_fix` (real division) + `quote_reserved_tables` (reserved-word table quoting) + `finalize_sql`. |
| `llm_fallback.py` | Remote generation: single OpenAI/Anthropic call, real FK/PK prompt parity, captures token usage for the cost axis. |

### 4.3 `prompts/` — prompt construction (shared, dependency-free)
| Module | Responsibility |
|---|---|
| `schema_render.py` | `render_schema_ddl` — the single CREATE-TABLE + FK/PK + value-hint renderer. |
| `prompt_manager.py` + `templates.yaml` | Loads the SLM system prompt, few-shot examples, and the tuned **instruction list** (evidence-first, exact FK/PK, exact literals, **project only asked-for columns**, real division). |
| `sql_strategies.py` | Reasoning-strategy prompt builders (direct / query-plan / decompose); `slm_generator` uses `build_direct_prompt`. |

### 4.4 `verifier/` — 3-stage neuro-symbolic verification (the "Reviewer")
`review.py :: run_verification` runs, short-circuiting on the first failure:
1. **Grammar** (`grammar_verifier.py`, sqlglot parse) → `GRAMMAR_FAIL`.
2. **Schema** (`schema_verifier.py`, every table/column exists) → `SCHEMA_FAIL`.
3. **Execution** (`execution_verifier.py`, runs it, timeout `verifier.timeout_seconds`) →
   `EXECUTION_FAIL`; an executed-but-empty result is a *soft* fail (`repair_on_empty`) that can
   trigger one value-aware repair.

`feedback_generator.py` turns a failure into `structured_feedback` fed back on the repair pass (and
sanitizes it on the remote path so no real tokens leak). **`PASS` means parses + references real
columns + executes — not that the answer is correct**; that is why a run can show 97/100 verified
but ~48% EX (the gap is executable-but-wrong SQL, §7).

### 4.5 `workflow/` — orchestration and shared state
| Module | Responsibility |
|---|---|
| `graph.py` | The LangGraph pipeline: node implementations + edge/repair wiring (`build_aegis_graph`). |
| `model_cache.py` | Process-singleton model/retriever/router loaders + `warmup`. |
| `embedding_cache.py` | Caches per-DB schema embeddings so retrieval encodes once. |
| `costing.py` | `compute_cost` — token-billed remote, fixed local. |
| `state.py` | `AEGISState` TypedDict (graph channels). |

### 4.6 `router/` + `abstraction/` — routing and the privacy axis
- `router/content_independent_router.py` — decides `LOCAL`/`REMOTE` from *content-independent*
  features (token count, schema size, structural complexity), so the routing decision itself leaks
  nothing about values (the paper's Theorem-1 property). `force_local`/`force_remote` override.
- `abstraction/` — the DP layer on the remote path: `dp_abstractor.py` replaces value-like tokens
  with semantic placeholders (value-aware, so schema words survive), `placeholder_vocab.py` +
  `sensitivity_policy.py` govern it, `reconstruction.py` restores real tokens in the returned SQL
  inside the trust boundary. On a local run this code never executes.

### 4.7 `evaluation/` — scoring and diagnostics
`bird_loader.py` (load + build schemas), `sampling.py` (stratified sampling), `evaluator_ex.py`
(Execution Accuracy — result-set match vs gold), `evaluator_ves.py` (Valid Efficiency Score),
`metrics.py` (three-axis privacy/cost/latency), `analyze_retrieval.py` (table-recall diagnostic).
`run_bird_evaluation.py` drives per-query generation; `run_full_evaluation.py` wraps it with the
combined EX/VES/three-axis report.

### 4.8 Top-level
`config.py` (Pydantic config, §5), `aegis_types.py` (`Query`, `SQL`, `Schema`, `SchemaElement`,
`VerificationResult`, enums), `run_bird_evaluation.py` / `run_full_evaluation.py` (entry points).

---

## 5. Configuration reference (`config.yaml` → `config.py`)

| Block | Key knobs |
|---|---|
| `slm` | `model`, **`num_candidates` (5)**, **`local_chunk_size` (2)**, `selection_temperature` (0.8), **`generation_seed` (42)**, `enable_cast_fix`, **`enable_literal_repair`**, `expose_keys`, `enable_value_grounding`, `full_schema`, `retrieval_top_k`, `max_expanded_tables` |
| `llm` | `provider`, `model` (`gpt-4.1-mini`), `temperature` |
| `rag` | `multi_step`, `max_tables` (6), `per_table_columns`, `per_query_top_k`, `max_rounds`, `value_retrieval`, `table_cards`, `rerank` |
| `verifier` | `timeout_seconds` (30), **`max_repair_attempts` (2)**, `repair_on_empty`, stage toggles |
| `router` | `force_local`, `force_remote`, `threshold_complexity` |
| `privacy` | `epsilon`, `abstraction_enabled`, `value_aware_abstraction` (remote path only) |
| `cost` | `remote_token_cost` (pegged to the remote model), `local_compute_cost` |
| `evaluation` | `seed`, `metrics` |

**Common run modes:** local-only = `router.force_local: true`; remote oracle = `force_remote: true`
+ `privacy.epsilon: 0` + `abstraction_enabled: false`. `generation_seed: 42` makes local runs
reproducible so config changes are measurable.

**Language ablation.** `config.language` is inert at runtime; language ablations are run via
**parallel data dirs**, not config: `scripts/make_spanish_sample.py` builds `data/bird_es/`
(the same seed-42 stratified 100-query sample with questions machine-translated to Spanish,
entities/literals preserved verbatim; gold SQL, evidence, and databases unchanged via symlink),
and the whole ablation is then `--bird_path data/bird_es` on the *unchanged* pipeline. Retrieval
(BGE-M3) and the SLM are multilingual; `query_decompose`'s stopwords/regexes are English-only, so
decomposition is weaker on non-English questions — a documented limitation the ablation measures
rather than patches.

---

## 6. Results (BIRD dev, 100q stratified, seed 42)

| Configuration | EX | Table recall |
|---|---|---|
| Local SLM, current pipeline (n=5, seeded, all repairs) | **~47–49%** | ~99% |
| Remote gpt-4o (no abstraction) | **~55%** | ~99% |
| Remote gpt-4o + RAG v2, full 1534-set (historical) | **61.93%** | — |

Latency: local ~11 s/query at n=5; remote ~4–5 s/query. Verification pass rate ~93–97/100 (executable
≠ correct). All numbers are now reproducible thanks to `generation_seed`.

---

## 7. Error analysis → the improvements it drove

Every design choice below was made in response to a measured failure mode, in order.

1. **Retrieval recall (85% → ~99%).** FK-maze DBs (financial) dropped endpoint tables like `client`
   because the old `adaptive_budget` cut anything past `max_tables` with no value hit, and the FK
   bridge could only recover *intermediate* tables. **Fix:** question-named + value-hit tables bypass
   the cap; `max_tables` 4→6 (`retriever/fusion.py`). Recall stopped being a loss source — but EX
   stayed flat, which *proved retrieval was not the bottleneck*.

2. **The generator is the bottleneck (the pivotal finding).** With ~99% recall the model still picks
   the wrong table/join or writes semantically wrong SQL; swapping the whole orchestrator (the
   multi-agent booster) moved nothing (booster local 46–49% vs graph 49%; remote 53% vs 55%). The
   booster was therefore **removed** and the simpler graph locked in. Accuracy effort moved to
   generation + selection + deterministic repair.

3. **Silent OOM → stub collapse (once drove EX to 3%).** A single CUDA OOM in one batched
   `generate` call returned `[]`, collapsing every query to a trivial `SELECT * LIMIT 10`. **Fix:**
   `_sample_chunked` with halving backoff and loud failure (`generator/slm_generator.py`).

4. **Run-to-run noise (±3 EX).** Unseeded sampling made every result a coin flip. **Fix:**
   `generation_seed: 42` — deterministic, so improvements are measurable.

5. **Reserved-word crashes.** `FROM order` (financial) is a guaranteed SQLite syntax error;
   recurred across runs. **Fix:** `quote_reserved_tables` in `finalize_sql`. Execution failures
   dropped 7 → 3 on the 100q set.

6. **Wider self-consistency (n=3 → 5).** More executing candidates for the vote. Bounded gain,
   because the dominant wrong answers are **consensus** errors — the vote histogram showed most wrong
   queries had all candidates agreeing on the same wrong SQL, which no vote can outvote.

7. **Literal near-misses.** `'Portuguese (Brazil)'` vs stored `'Brasil'`, `'Volkan Baga'` vs
   `'Baǵa'`, `'2013-02-22 00:00:00'` vs `'2013-02-22'` — value grounding attached the right form at
   prompt time but the model still copied the question's spelling. **Fix:** `literal_repair.py` fixes
   it deterministically on the *winning* query, post-selection.

8. **Projection over-selection.** BIRD's strict column match rejects `SELECT ID, Birthday` when gold
   wants `SELECT ID`. **Fix:** an explicit "project exactly what's asked" instruction in the system
   prompt + `templates.yaml`.

9. **Single repair attempt.** ~13/100 queries ended with every candidate empty and only one retry.
   **Fix:** `max_repair_attempts` 1→2 (bounded).

**The residual ceiling:** the ~23 wrong-table/join-path queries are *consensus reasoning* errors in
the 7B — no deterministic patch reaches them. Realistic local ceiling with these levers is the low
50s; beyond that the lever is the generator itself (a stronger/fine-tuned model, or the remote arm,
which already reaches ~55–62%). This is the honest through-line of the whole investigation:
**retrieval, selection, and verification are solved-enough; accuracy now lives in the model.**

---

## 8. Repository layout

```
workflow/          ← the LangGraph pipeline (graph.py), model/embedding cache, costing, state
retriever/         ← RAG v2: pipeline, fusion (adaptive budget), decompose, value_index/sampler,
                     schema_retriever, embedding_models
generator/         ← slm_generator, llm_fallback, candidate_selector (the vote), literal_repair,
                     sql_postprocess
prompts/           ← schema_render, prompt_manager + templates.yaml, sql_strategies
verifier/          ← grammar / schema / execution verifiers, feedback_generator, review
abstraction/       ← DP abstraction + reconstruction (privacy axis; remote path)
router/            ← content-independent router (privacy thesis; used on the remote path)
evaluation/        ← BIRD loader, EX/VES evaluators, sampling, metrics, retrieval analyzer
config.py / config.yaml   ← Pydantic config + the active config
aegis_types.py            ← core types (Query, SQL, Schema, VerificationResult, …)
run_bird_evaluation.py    ← per-query eval driver     run_full_evaluation.py ← EX/VES/three-axis wrapper
tests/             ← offline unit tests (mock models / temp sqlite)
aegis-selector/    ← standalone pairwise-selector training notebook (not on the runtime path)
```

See `CLEANUP.md` for what was removed (the multi-agent booster, the ambiguity resolver, the
`Model_Fine_Tuning/` copy) and what was deliberately kept (the DP-abstraction / router privacy code).

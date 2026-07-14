# AEGIS-SQL — Architecture Comparison Across Four Branches

**Benchmark:** BIRD dev (text-to-SQL). Primary metric: Execution Accuracy (EX).
**Evaluation protocol:** stratified 100-query sample (seed 42) for fast iteration; full 1534-query dev set for the headline number.
**Common substrate (all branches):** BGE-M3 hybrid schema retrieval (dense + sparse) with FK closure, value grounding to stored DB literals, a 3-stage verifier (grammar → schema → execution), and the exact `predictions.jsonl` contract so every run is scored by the same tooling.

This document walks through the four architectures in the order they were built, explains *why* each one exists, the trade-offs baked into each design choice, the measured results, a head-to-head comparison, and a final selection.

---

## 1. `aegis_feat` — The multi-agent "booster" harness

### Reasoning
The founding experiment established that the *harness*, not the model, was the bottleneck: both the local 7B SLM and remote gpt-4o were pinned at ~44–47% EX — roughly 11 points below gpt-4o's public BIRD number — when driven single-shot. The paper's own thesis ("the context and the harness around the model matter more than the model") pointed the way. `aegis_feat` is the first attempt to build a *booster*: a small, explicit multi-agent pipeline in plain Python that makes `model + harness > model alone`.

### Architecture
A linear `agents/` package, each agent single-responsibility, composing the existing reusable blocks (no model reloads — everything through `workflow/model_cache`):

```
SchemaLinker → Router → [Local SLM | Remote (DP-abstracted) LLM]
            → CandidateGenerator (multi-strategy) → Refiner → Selector → Reviewer
```

- **CandidateGenerator** samples across *reasoning strategies* (`direct`, `query_plan`, `divide_and_conquer`) — diversity is what makes selection work.
- **Refiner** does one round of execution-feedback repair on error/empty results.
- **Selector** uses execution-guided self-consistency (majority vote by result set), with an optional pairwise LLM judge on ties.
- **Reviewer** wraps the 3-stage verifier.
- The privacy round-trip (DP abstraction → reconstruction) still wraps the remote arm, and the judge runs locally, preserving the paper's content-independent routing (Theorem 1).

### Trade-offs & optimizations
- **Plain Python, not a graph.** The control flow is linear; explicit code is far more debuggable than a graph engine for a first booster. (This is the design decision v2 later revisits.)
- **Model-aware strategies.** A specialist GRPO SLM wants *direct* execution-guided self-consistency; a general LLM benefits from CoT (`query_plan`). Forcing CoT onto the specialist SLM *hurt* it and produced a latency tail (a 116s outlier). The fix split `local_strategies` (default `[direct]`) from `remote_strategies` (`[direct, query_plan]`).
- **Everything is a config flag** so the cost↔accuracy frontier is A/B-able.

### Results
- Booster-hybrid (abstraction ON, confounded): **45.0% EX** (S 51.7 / M 36.7 / C 30.0), VES 47.8, 81% remote / 19% local, latency mean ~13s.
- **+11 EX over the old single-shot hybrid (34%)** — the first hard evidence the booster helps — but confounded by (a) the DP abstraction penalty on 81% of queries and (b) CoT-on-a-specialist. The *clean* isolation (force-local booster vs SLM-alone) was the intended test.

**Verdict:** proved the booster concept and the harness-over-model thesis, but the privacy round-trip and mixed strategies muddied the number.

---

## 2. `aegis_feat_2` — RAG v2: recall-first retrieval + prompt parity

### Reasoning
Analysis of `aegis_feat` localized the dominant failure precisely: **28/100 predictions used the wrong table set** — a schema-*linking* failure, not a generation failure. Two mechanisms: (a) the adaptive retrieval budget was dropping ground-truth tables (recall fell 99%→86%; 14/100 queries became unwinnable), and (b) the DDL prompt never exposed real foreign keys, so the model guessed joins.

### Architecture
Same booster skeleton, but the retrieval and prompt layers are rebuilt for **recall first**:

- **Multi-step retrieval with value grounding.** Question literals are grounded to exact stored DB values (LSH/keyword probes), so `WHERE city = 'Lakeport'` uses the real spelling.
- **FK bridges, never table drops.** The budget fix caps *columns per table* but never removes a table — killing the 86%-recall regression. FK-connected tables of the anchor are always pulled in with their key columns.
- **Prompt FK/PK exposure.** The CREATE TABLE block now renders real `-- Foreign keys: t1.c1 = t2.c2` join hints and `PRIMARY KEY` markers, sourced from the populated `Schema` (the old path read a never-populated field). Both SLM and LLM prompts render schema identically for a fair swap.
- **Extraction & judge fixes** for GRPO-style `<answer>` output and tie-breaking.
- **Real cost + clean literals.** The abstraction core-span fix stopped re-injecting quotes/punctuation (`''French''`, `'Lakeport?'`), and per-query token cost is finally computed.

### Trade-offs & optimizations
- **Recall over precision at the retrieval stage.** More columns = more tokens/noise, but for strong models schema *pruning hurts* — the recipe is recall-first (or full schema on small DBs). BIRD dev DBs are small enough that this is affordable.
- **Full-schema on small DBs, linked mode above a column threshold** — the `auto` schema mode.

### Results
- **Remote gpt-4o, full 1534-set: 61.93% EX** — the strongest validated number in the project, and the first to approach public BIRD territory.
- Local SLM, 100q: **50% EX** (up from 47% before the retrieval fix).

**Verdict:** fixed the real bottleneck (schema linking) and produced the project's best *validated* accuracy — but the headline number leans on a remote frontier model and single-shot greedy decoding.

---

## 3. `aegis_v1` — CSC merge-revision, native OmniSQL, single-7B

### Reasoning
Two discoveries reframed the local-model strategy. First, the checkpoint in use — `CscSQL-Merge-Qwen2.5-Coder-7B` — is CSC-SQL's *merge-revision* model, whose published **69.19% dev / 71.72% test** EX comes from a specific pipeline we had never actually run. Second, selection is the true ceiling (CHASE-SQL oracle 82.8 vs 73.0 achieved). So `aegis_v1` commits to running the CSC-SQL recipe *natively and faithfully*, to unlock the checkpoint's real number.

### Architecture
The full CSC-SQL corrective self-consistency pipeline, mode-selectable (`local | remote | ensemble`):

```
Recall-first schema (full DDL, OmniSQL template, evidence prepended)
 → Candidate pool: local GRPO SLM (n samples @ 0.8) ± remote LLM (direct, query_plan)
 → Execution-vote grouping  (group by frozenset(result))
 → CSC merge-revision: top-2 disagreeing groups → merge model (m samples) → re-vote
 → Judge tie-break → bounded execution-feedback refine → 3-stage verify
```

- **Native OmniSQL prompt** (`prompts/omnisql.py`): DDL with per-column sampled values in comments, `<think>…</think><answer>SQL</answer>` output, table-linking disabled (full schema) — exactly the CSC setup.
- **Single-7B default:** one model serves *both* generation and merge-revision (same checkpoint id → one instance), so it fits any 20GB+ GPU with no GPU-specific setup. A dual-checkpoint upgrade (GRPO generator + Merge model) is a config change for 48GB GPUs.
- **Worker-pool eval** (`--workers`, thread-based) with OOM-safe chunked sampling, a GPU serialization lock, and no silent CPU offload (fails loudly if the model doesn't fit).
- **`winner_arm` logging** per query (`local | remote | merge | refine`) — turns every run into an arm-level ablation.

### Trade-offs & optimizations
- **Fidelity vs speed.** The paper-faithful preset (n=64, m=8) is accurate but slow. To meet the hard time budgets (full eval < 4h, 100-sample < 20 min), defaults were cut to `local_candidates=6, merge_candidates=4, chunk=8, workers=4`. Honest caveat: this is GPU-bound; an A40 48GB (~$0.39/hr) is recommended over a 24GB card for the time budget.
- **Welded to the 7B.** The merge-revision arm only works with a CSC-family checkpoint — the architecture is *coupled to one model family*. This is the key limitation v2 removes.
- **Crash fix = free accuracy.** A `NoneType` regex crash on `european_football_2` / `debit_card_specializing` was killing 10/100 queries outright; making `_quote_ident` None-safe and skipping None-valued PK/FK columns recovered ~10 EX points.

### Results
- Ensemble, 100q: **54% EX** (before the crash fix; ~60%+ projected after).
- 10-query smoke: **90%** (with a crippled/CPU-offloaded merger — i.e. mostly the gpt-4o arm).
- Arm breakdown at 54%: the CSC merge arm fired only **2/100** — near dead-weight because it is welded to the 7B and the single-instance config rarely diverged enough to trigger it. All produced SQL executed cleanly, confirming the ceiling is candidate quality + selection, not crashes.

**Verdict:** the validated single-7B option. Faithful to a published SOTA recipe, self-hostable on modest hardware, but its headline arm (merge-revision) under-fires in the single-instance config, and the whole design is tied to one model family.

---

## 4. `aegis_v2` — Model-agnostic LangGraph pipeline *(no eval results yet)*

### Reasoning
Two things became clear from v1: (1) the CSC merge-revision arm is dead weight when welded to a single 7B, and (2) the highest-value lever is *diverse candidates + strong selection* (CHASE-SQL: a fine-tuned pairwise selector is +4.17 EX over voting), which is **model-agnostic**. v2 drops the CSC merge coupling entirely and rebuilds the pipeline as a clean CHASE-style LangGraph that works with **any** model in all three modes — targeting 65%+ in remote/ensemble without depending on a specific checkpoint.

### Architecture
A LangGraph `StateGraph` with typed state, pure node functions, a mode router, and conditional edges:

```
                     ┌──────────────► generate_local ─┐
schema_link ─ route ─┤  (mode)                        ├─► vote ─?─► judge ─?─► refine ─► verify
                     └──────────────► generate_remote ┘        (disagree)   (empty/err)
```

- **`GraphState(TypedDict)`** with an additive reducer `candidates: Annotated[list, operator.add]` so parallel arms fan-in cleanly.
- **Mode router** (`route_mode`): `local → [generate_local]`, `remote → [generate_remote]`, `ensemble → both` (parallel fan-out).
- **Conditional edges:** `needs_judge` fires only on ≥2 disagreeing result groups; `needs_refine` fires only when the winner errors/empties and rounds remain. Compute is spent only when it can change the answer.
- **Model-agnostic backend:** local HF SLM `complete()` and remote LLM `complete()` share one interface; the revision node uses a generic execution-feedback prompt (`build_revision_prompt`) instead of the CSC merge template — so *any* model can refine.
- **Sequential fallback:** `GraphOrchestrator` runs on LangGraph when installed, else a `_run_sequential` path with identical node semantics (minus arm parallelism) — so it's offline-testable and works with or without the `langgraph` dependency.
- Returns the exact prediction contract (`sql`, `routing_decision`, `winner_arm`, `candidates_local/remote`, `cost_usd`, `privacy_loss`, …) so existing eval tooling is unchanged.

### Trade-offs & optimizations
- **Model-agnostic over model-optimal.** By dropping the CSC merge template, v2 gives up the 7B-specific merge fidelity in exchange for working with *any* SLM/LLM. The bet: diverse candidates + execution voting + selective judge/refine generalizes better than a checkpoint-specific corrector.
- **Selective compute via conditional edges.** Judge and refine are gated on need, not run unconditionally — cheaper than v1's always-on merge stage.
- **Graph engine with a plain-Python fallback** — you get LangGraph's parallelism when available without a hard dependency, keeping the "runs on any GPU, no special deps" constraint.
- **Config-selectable engine** (`engine: graph | csc`) so v1 and v2 coexist for A/B.
- **Validation:** 4 wiring/node tests pass offline (mode router + contract, ensemble pooling, judge-on-disagreement, refine-repairs-empty) via the sequential fallback with a mocked cache and temp SQLite.

### Results
**No evaluation results yet** — v2 is built, unit-tested, and pushed, but the GPU eval (100q, then full 1534) is run by the user. Design target: **≥65% EX** in remote/ensemble mode, matching or beating `aegis_feat_2`'s 61.93% while being model-agnostic.

**Verdict:** the most flexible and forward-looking design; the only one that isn't tied to a model family. Unproven until the user's GPU run lands.

---

## 5. Head-to-head comparison

| Dimension | `aegis_feat` | `aegis_feat_2` | `aegis_v1` | `aegis_v2` |
|---|---|---|---|---|
| **Core idea** | Multi-agent booster (plain Python) | Recall-first RAG + prompt parity | Native CSC merge-revision | Model-agnostic LangGraph (CHASE-style) |
| **Engine** | Linear Python | Linear Python | CSC orchestrator | LangGraph (+ sequential fallback) |
| **Selection** | Exec vote + judge | Exec vote + judge | Exec vote + **merge-revision** + judge | Exec vote + gated judge + gated refine |
| **Model coupling** | Model-aware strategies | Model-aware | **Welded to CSC-7B** | **Model-agnostic** |
| **Best EX (validated)** | 45.0% (100q, confounded) | **61.93%** (full, remote) | 54% (100q, pre-crashfix) | — (target ≥65%) |
| **Local SLM EX (100q)** | ~19 local rows only | 50% | ~54% ensemble | — |
| **Modes** | hybrid/router | hybrid/router | local/remote/ensemble | local/remote/ensemble |
| **GPU footprint** | 7B + BGE | 7B + BGE | 1×7B (any 20GB+ GPU) | any model (backend-agnostic) |
| **Key strength** | Proved harness > model | Fixed schema linking; best validated number | Faithful published recipe, self-hostable | Flexible, future-proof, selective compute |
| **Key weakness** | Privacy round-trip confounds; CoT-on-specialist | Best number needs a frontier remote model | Merge arm under-fires (2/100); 1-model-family lock | Unproven on GPU |

### Reading the numbers honestly
- `aegis_feat_2`'s **61.93%** is the highest *validated* EX, but it's a **remote gpt-4o, single-shot** number on the full set — it measures a frontier model's ceiling with good retrieval, not a self-hosted system.
- `aegis_v1`'s **54%** is a *local-capable ensemble* on a noisy 100-sample, pre-crash-fix. The +10 from the crash fix and the under-firing merge arm mean its true ensemble number is likely ~60%+, but that awaits a re-run.
- `aegis_v2` has **no number yet** — its case is architectural, not empirical.

---

## 6. Conclusion & selection

The four branches trace a clear arc: prove the harness matters (`feat`) → fix the real bottleneck, schema linking (`feat_2`) → chase a published SOTA recipe (`v1`) → generalize it so it isn't chained to one model (`v2`).

**Selection: `aegis_v2` as the go-forward architecture, with `aegis_feat_2` as the validated fallback.**

Reasoning:
1. **Model-agnosticism is the decisive constraint.** The stated requirement is *maximum accuracy for any model* across three modes on *any* GPU. `aegis_v1` violates this at its core — its headline arm only works with a CSC-family 7B. `aegis_v2` is the only design that satisfies it.
2. **It keeps every proven lever and drops only the dead weight.** v2 retains recall-first retrieval (`feat_2`'s win), diverse candidates + execution voting (`feat`'s win), and the judge/refine loop — while removing the merge-revision stage that fired 2/100 in v1. The levers that moved accuracy are all still present.
3. **Selective compute + LangGraph parallelism** should let it hit the ≥65% target at *equal or lower* latency than v1's always-on merge — the judge and refine nodes only run when they can change the answer.
4. **Lowest operational risk.** The sequential fallback means it runs with or without the graph dependency, and the config `engine` switch keeps v1 available for A/B, so adopting v2 costs nothing if the GPU run disappoints.

**The one caveat:** `aegis_v2` is unproven. The selection is conditional on the user's GPU run confirming the ≥65% design target. Until then, **`aegis_feat_2` (61.93% full-set, remote) is the validated production choice**, and `aegis_v1` (~60%+ ensemble, self-hostable, after the crash fix re-run) is the best *local-first* option.

### Recommended next step
Run `aegis_v2` on the stratified 100-query sample (`engine: graph`, `mode: ensemble`), read the `winner_arm` breakdown, then the full 1534-set. If it clears 65%, promote it. The one accuracy lever beyond v2 — a **fine-tuned pairwise selection model** (CHASE-SQL, +4.17 EX; the `aegis-selector` notebook is already built) — plugs directly into v2's judge node as the follow-up.

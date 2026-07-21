# Repository cleanup — `aegis_feat_2`

Streamlining pass around the active multi-agent pipeline. The guiding rule: **remove only
files that are dead (unreferenced), redundant, or one-off scaffolding — and never anything on
the active eval path, the parked research thesis, or the diagnostics that measure the system.**
Every removal was grep-verified to have no live import references, and the full offline test
suite + `py_compile` stay green afterward.

## Removed (6 files)

| File | Why it's safe to remove |
|---|---|
| `run_parallel_predictions.py` | Legacy graph-only parallel driver — calls `build_aegis_graph` + `graph.invoke` with no `multi_agent` branch, so it can't exercise the active booster. Superseded by `run_bird_evaluation.py`. Unreferenced. |
| `run_e2e_test.py` | One-off in-memory smoke test hardcoding a `company_db` schema over the legacy graph. Referenced only by a print string in `test_config.py` (also removed). |
| `add_ex_to_predictions.py` | Standalone post-hoc utility that re-executes predicted vs gold SQL to append an EX flag — duplicates `evaluation/evaluator_ex.py`, which `run_bird_evaluation.py` already invokes. Unreferenced. |
| `test_config.py` | Top-level manual smoke script (not pytest-collected in `tests/`) exercising the legacy graph + `LLMFallback` A/B path. Unreferenced. |
| `test_fk_expansion.py` | Top-level manual integration script needing `data/bird` + a model download; the offline FK-bridging coverage now lives in `tests/test_rag.py`. Unreferenced. |
| `test_multilingual_retrieval.py` | Top-level manual script needing GPU/BGE-M3/data; overlaps `tests/test_rag.py` / `tests/test_retrieval.py` as the heavyweight live variant. Unreferenced. |

Net: the top-level tree drops from 8 loose scripts to the two real entry points
(`run_bird_evaluation.py`, `run_full_evaluation.py`) plus `config.py` / `aegis_types.py`.

## Deliberately KEPT (and why)

These were candidates in an aggressive audit but are retained on purpose:

- **`run_full_evaluation.py`** — this is the user's *actual* driver. The uploaded
  `full_evaluation.log` header ("AEGIS-SQL BIRD-dev Full Evaluation Pipeline") is emitted by
  this file; it orchestrates `run_bird_evaluation.py` + EX + VES + the three-axis report.
  Removing it would break the documented run command.
- **`evaluation/analyze_retrieval.py`** (+ `tests/test_retrieval.py`) — the table-recall
  diagnostic. It is exactly the tool that measures the retrieval fix from the regression
  analysis; deleting it would remove the ability to verify the recall improvement.
- **`tests/test_costing.py`** — guards `workflow/costing.py`, which *is* on the active path
  (`compute_cost` fills `cost_usd`). Keeping the test keeps that coverage.
- **`tests/test_repair_loop_termination.py`** — guards the legacy-graph repair loop; kept
  because the legacy graph itself is kept (below).
- **`abstraction/`, `router/`, `query_planner/`** — the paper's DP-abstraction / content-
  independent-routing thesis. `router/` is live on every query (it makes the routing decision);
  `abstraction/` fires on the remote path; `query_planner/` is parked. **Kept as thesis code.**
- **`workflow/graph.py`, `workflow/state.py`, `workflow/__init__.py`** — the legacy LangGraph
  A/B pipeline. These are **import-coupled**: `workflow/__init__.py` eagerly imports
  `build_aegis_graph`, so `workflow.model_cache` / `workflow.costing` (both active) transitively
  load `graph.py`. Deleting them requires editing `__init__.py` + `run_bird_evaluation.py`'s
  top-level import in the same change. Left intact to avoid breaking the active path; remove only
  as a deliberate "drop the graph A/B" change.
- **`scripts/install_embedding_model.py`** — one-off BGE-M3 setup helper; harmless, useful on a
  fresh pod.

**`Model_Fine_Tuning/` — REMOVED.** The self-contained offline SLM fine-tuning pipeline (imported
by nothing at runtime; dependency was one-directional) was deleted wholesale: the fine-tuning is
maintained in a separate repo, so the copy here was dead weight. Nothing on the graph/eval path
referenced it.

## Verification

```
git ls-files '*.py' | xargs python -m py_compile      # all compile
# offline suite (each file is a runnable script):
python tests/test_rag.py            # 11 passed
python tests/test_retrieval.py      # passed
python tests/test_agents.py         # 7 passed
python tests/test_schema_render.py  # 3 passed
python tests/test_sql_strategies.py # 5 passed
python tests/test_extract_sql.py    # 7 passed
python tests/test_costing.py        # passed
python tests/test_harness_optimizations.py  # 20 passed
python tests/test_repair_loop_termination.py # passed
PYTHONPATH=. python tests/test_abstraction_value_aware.py  # passed
```

## Optional next steps (not done — they need a decision)

- **Drop the legacy graph A/B** to delete `workflow/graph.py` + `workflow/state.py`: edit
  `workflow/__init__.py` to stop eagerly importing `build_aegis_graph`, make
  `run_bird_evaluation.py`'s graph import conditional, then remove the files and
  `tests/test_repair_loop_termination.py`. Only worth it if you no longer need the graph as a
  comparison baseline.
- **Remove `Model_Fine_Tuning/` wholesale** if the paper uses only the published checkpoint.

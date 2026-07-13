# AEGIS-SQL

A **corrective self-consistency** text-to-SQL system for the BIRD benchmark:
one simple pipeline that runs a local 7B SQL specialist, a remote LLM, or an
**ensemble of both**, and converts a diverse candidate pool into a single
high-accuracy answer through execution voting and merge-revision.

Design principles: **highest accuracy, least complexity, any GPU.** One
orchestrator, one ~80-line config, no GPU-specific dependencies. (The research
repo also parks the paper's differential-privacy abstraction and
content-independent router — isolated from this pipeline, reactivatable for the
privacy study.)

---

## Final Architecture

```
                          Question + Evidence
                                  │
             ┌────────────────────▼─────────────────────┐
             │ 1. SCHEMA STAGE (recall-first)           │
             │  full schema DDL (≤120 cols, the default │
             │  for BIRD) or multi-step linked retrieval│
             │  + value grounding: question literals    │
             │  located IN the DB, exact stored forms   │
             └────────────────────┬─────────────────────┘
                                  │
        ┌─────────────────────────▼──────────────────────────┐
        │ 2. CANDIDATE POOL     (mode: local|remote|ensemble) │
        │  local:  CSC 7B, native OmniSQL prompt,             │
        │          n sampled candidates (OOM-safe chunks)     │
        │  remote: LLM, direct + query-plan strategies        │
        │  (arms run CONCURRENTLY in ensemble mode)           │
        └─────────────────────────┬───────────────────────────┘
                                  │
             ┌────────────────────▼─────────────────────┐
             │ 3. EXECUTION-VOTE GROUPING               │
             │  run every candidate; group by result    │
             │  set; majority vote (BIRD EX semantics)  │
             └────────────────────┬─────────────────────┘
                                  │ top-2 groups disagree?
             ┌────────────────────▼─────────────────────┐
             │ 4. CSC MERGE-REVISION                    │
             │  merge model sees both drafts + their    │
             │  execution results (reduced schema) and  │
             │  writes the corrected SQL → re-vote      │
             └────────────────────┬─────────────────────┘
                                  │
             ┌────────────────────▼─────────────────────┐
             │ 5. JUDGE tie-break → bounded REFINE      │
             │    (error/empty results only)            │
             └────────────────────┬─────────────────────┘
                                  │
             ┌────────────────────▼─────────────────────┐
             │ 6. REVIEWER: grammar → schema → execution│
             └────────────────────┬─────────────────────┘
                                  ▼
                    predictions.jsonl (winner_arm logged)
```

**Why this shape** (each stage earned its place empirically):
- *Recall-first schema*: dropping one needed table caps a query's EX at 0 —
  measured on BIRD (`financial`), and confirmed by the schema-linking literature.
- *Diverse pool + execution voting*: the most reliable test-time-compute lever
  (CHASE-SQL, CSC-SQL); ensemble diversity across model families raises the
  pool's oracle ceiling.
- *Merge-revision*: plain voting fails exactly when the majority is wrong; the
  CSC merge checkpoint was RL-trained to adjudicate the top-2 disagreeing
  groups from their execution evidence (CSC-SQL 7B: 69.19% BIRD-dev).

## Repo layout

```
agents/          orchestrator, schema linker, reviewer   (the pipeline)
generator/       SLM + LLM wrappers, CSC engine, SQL postprocessing
prompts/         OmniSQL + strategy + judge prompt builders (pure, tested)
retriever/       hybrid retrieval, value index, fusion, multi-step pipeline
verifier/        grammar / schema / execution verification
evaluation/      BIRD loader, EX/VES evaluators, retrieval analyzer
workflow/        model cache (thread-safe), embedding cache, costing
abstraction/, router/   parked privacy components (paper; not in this pipeline)
config.yaml      the single config (~80 lines)
```

## Setup

```bash
pip install -r requirements.txt          # torch, transformers, sqlglot, ...
echo "OPENAI_API_KEY=sk-..." >> .env     # remote/ensemble modes
echo "HF_HUB_TOKEN=hf_..."   >> .env     # faster model downloads
# BIRD dev set under data/bird/ (dev.json + dev_databases/)
```

**Hardware**: any GPU with ≥20GB VRAM runs the default (a single 7B fp16
serves generation and merge-revision). No vLLM, no quantization, no
GPU-specific setup. A model that doesn't fit fails loudly at load — it never
silently offloads to CPU. Dual-checkpoint upgrade for 48GB GPUs: point
`models.generator` at the GRPO checkpoint in `config.yaml`.

## Running evaluations

```bash
# 100-query stratified sample (seed-fixed, reproducible)
python run_full_evaluation.py --num_queries 100 --seed 42 --stratify --output_name exp_100

# Full BIRD-dev (1,534 queries)
python run_full_evaluation.py --output_name full_dev

# Retrieval diagnostics on any run
python evaluation/analyze_retrieval.py evaluation/output/exp_100/predictions.jsonl
```

Switch what you're testing with one key in `config.yaml`:

| `mode:` | Pool | Use |
|---|---|---|
| `local` | 7B SLM only | zero API cost, the local-model story |
| `remote` | remote LLM only | model-alone baseline |
| `ensemble` | both, pooled | maximum accuracy |

**Throughput**: queries run through a worker pool (`--workers`, default 3);
the GPU serializes internally while remote API calls, SQLite execution voting,
and verification overlap with it. Rough guide on an A40-class GPU with the
defaults (`local_candidates: 16`): 100 queries ≲ 20 min, full 1,534 ≲ 4 h;
`remote` mode is network-bound and much faster. On slower GPUs, lower
`generation.local_candidates` (8 keeps most of the accuracy — pass@k saturates
near 8) or raise `--workers`.

## Reading results

Every run writes to `evaluation/output/<name>/`: `predictions.jsonl`,
`ex_results.txt` (EX by difficulty), `ves_results.txt`, `evaluation_report.json`,
`config_snapshot.yaml`, `evaluation.log`.

Each prediction records **which arm produced the final answer** — the routing
record for analyzing local-vs-remote wins in ensemble mode:

```json
{"winner_arm": "merge",        // local | remote | merge | refine
 "candidates_local": 16, "candidates_remote": 8, ...}
```

```bash
# Arm-win breakdown of a run
python - <<'EOF'
import json, collections
arms = collections.Counter(json.loads(l)["winner_arm"]
                           for l in open("evaluation/output/exp_100/predictions.jsonl"))
print(dict(arms))
EOF
```

## Key config knobs (`config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `mode` | `ensemble` | candidate pool: local / remote / ensemble |
| `generation.local_candidates` | 16 | SLM samples per query (64 = CSC paper preset) |
| `generation.remote_candidates` | 4 | remote samples per strategy (0 = off) |
| `csc.enabled` / `csc.merge_candidates` | true / 8 | merge-revision stage |
| `retrieval.schema_mode` | `auto` | full schema ≤120 cols, linked above |
| `selection.judge` | `auto` | tie-break on exact vote ties |
| `refine.rounds` | 1 | execution-feedback repair on error/empty |

## Tests

```bash
for t in tests/test_*.py; do python "$t"; done   # offline; no GPU/API needed
```

## References

CSC-SQL (arXiv:2505.13271) · CHASE-SQL (arXiv:2410.01943) · OmniSQL template
(RUCKBReasoning/OmniSQL) · BIRD benchmark (bird-bench.github.io)

## Citation

```bibtex
@article{aegis_sql_2025,
  title={Three-Axis Constrained Optimization for Hybrid NL2SQL},
  author={David Onyango and MINDS Lab},
  year={2025}
}
```

## License

MIT License

# aegis-selector

Fine-tune a **pairwise SQL selection model** — the highest-leverage selection
component for text-to-SQL pipelines (CHASE-SQL's ablation: a fine-tuned pairwise
selector beats self-consistency voting by **+4.17 EX** on BIRD).

Given a question, a schema, and two candidate SQLite queries with their execution
results, the model replies `A` or `B` for the one that answers correctly. It is
the drop-in upgrade for the tie-break judge in
[AEGIS-SQL](https://github.com/Daveonyango254/AEGIS-SQL) (`aegis_v1` branch).

## What's here

One notebook, end to end:

| | Stage | Notes |
|---|---|---|
| 1 | Download BIRD-train | questions + SQLite databases, rerun-safe |
| 2 | Candidate generation | `cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-7B-Instruct`, n=8 @ temp 0.8, **resumable jsonl checkpoints** |
| 3 | Execution labeling | result-set match vs gold (BIRD EX definition), timeouts handled |
| 4 | Pair construction | order-randomized A/B, capped per question, **split by database** (no leakage) |
| 5 | LoRA fine-tune | `Qwen/Qwen2.5-Coder-3B-Instruct`, r=16, single-token supervision |
| 6 | Evaluate | pairwise accuracy on held-out databases |
| 7 | Deploy | pushes **adapter + merged model** to the Hugging Face Hub with a model card |
| 8 | Inference | loads the merged model back from the Hub and runs a pick |

## Requirements

- 1× GPU with ≥ 24 GB VRAM (generation runs a 7B fp16; training a 3B with LoRA).
- ~6 GB disk for BIRD-train, ~7 GB for model caches.
- A Hugging Face account + **write** token (`HF_TOKEN` env var or the login widget).

**Timings** (A100-class): generation ~3–5 h @ 2,000 questions, labeling ~15 min,
training ~1–2 h. Set `N_QUESTIONS = 100` in the config cell for a ~1-hour pilot
first — every expensive stage checkpoints, so pilot work carries over.

## Run

```bash
pip install -r requirements.txt
jupyter lab selector_finetune.ipynb   # run top to bottom
```

## Output

- `your-hf-user/aegis-sql-selector-3b-lora` — LoRA adapter (small, iterable)
- `your-hf-user/aegis-sql-selector-3b` — merged fp16 model (plain-`transformers` inference)

The final notebook section shows the AEGIS v1 integration snippet.

## References

- CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection
  in Text-to-SQL — [arXiv:2410.01943](https://arxiv.org/abs/2410.01943)
- CSC-SQL (generator checkpoints) — [arXiv:2505.13271](https://arxiv.org/abs/2505.13271)
- BIRD benchmark — [bird-bench.github.io](https://bird-bench.github.io/)

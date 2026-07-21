# AEGIS-SQL: Three-Axis Constrained Optimization for Hybrid NL2SQL

A production-grade implementation of **AEGIS-SQL**, a hybrid natural language to SQL system that balances **utility (accuracy)**, **privacy**, and **cost** through intelligent routing, differential privacy abstraction, and neuro-symbolic verification.

---

## Architecture Overview

**Agent Names:** In the system architecture, components are referred to by their agent names:
- **Query Planner Agent** → Encompasses `SchemaRetriever` (schema extraction) + `ContentIndependentRouter` (routing logic)
- **Reviewer Agent** → Encompasses `GrammarVerifier`, `SchemaVerifier`, `ExecutionVerifier`, `FeedbackGenerator` classes

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AEGIS-SQL WORKFLOW                           │
└─────────────────────────────────────────────────────────────────────┘

Input: Natural Language Query + Database Schema
   │
   ▼
┌───────────────────────────────────────────────────────────────────┐
│                    QUERY PLANNER AGENT                            │
│                                                                   │
│  Step 1: Schema Extraction                                        │
│  └─> Retrieve relevant schema elements via multilingual          │
│      embeddings (BGE-M3 RAG)                                      │
│                                                                   │
│  Step 2: Content-Independent Routing                              │
│  └─> Route based on query complexity & cost constraints           │
│      Decision: LOCAL (FSLM) or REMOTE (FLLM)                      │
└───────────────────────────────────────────────────────────────────┘
           │
           ├─────────────────────┬──────────────────────┐
           │                     │                      │
           ▼ LOCAL               ▼ REMOTE              │
    ┌─────────────┐      ┌──────────────┐             │
    │   FSLM      │      │ Abstraction  │             │
    │ (CscSQL-7B) │      │ (DP ε-mech)  │             │
    │             │      └──────┬───────┘             │
    └──────┬──────┘             │                     │
           │                    ▼                     │
           │             ┌──────────────┐             │
           │             │    FLLM      │             │
           │             │ (GPT-4/Claude)│            │
           │             └──────┬───────┘             │
           │                    │                     │
           │                    ▼                     │
           │             ┌──────────────┐             │
           │             │Reconstruction│             │
           │             │(Placeholders)│             │
           │             └──────┬───────┘             │
           │                    │                     │
           └────────────────────┴─────────────────────┘
                                │
                                ▼
                  ┌──────────────────────┐
                  │  REVIEWER AGENT      │
                  │  (3-stage)           │
                  │  • Grammar           │
                  │  • Schema            │
                  │  • Execution         │
                  └──────────┬───────────┘
                             │
                             ▼
                    Output: Verified SQL
```

---

## Key Features

### 🔒 **Privacy-Preserving**
- **Differential Privacy (DP) Abstraction**: Token-level ε-DP via exponential mechanism
- **Zero Leakage Local Path**: Queries routed to local FSLM never expose sensitive data
- **Formal Guarantees**: Privacy loss bounded by ℒ_priv = ε × E[|prompt|] × Pr(r=remote)

### 💰 **Cost-Optimized**
- **Content-Independent Routing**: Automatic LOCAL/REMOTE decision based on complexity
- **Budget Controls**: Per-query cost limits with configurable thresholds
- **Hybrid Architecture**: Use expensive remote LLMs only when necessary

### 🎯 **High Accuracy**
- **Execution-Guided Self-Consistency**: 1 greedy + 4 sampled candidates, executed against the
  real DB and selected by result-set majority vote (`slm.num_candidates`, chunked/OOM-safe,
  seeded for reproducible runs)
- **Deterministic Repairs**: post-hoc literal repair (near-miss values → exact stored DB form),
  reserved-word table quoting, CAST-as-REAL division fix
- **3-Stage Neuro-Symbolic Verification**: Grammar → Schema → Execution, with a bounded
  path-aware self-correction loop
- **RAG v2 Retrieval**: BGE-M3 hybrid dense+sparse, value grounding, FK-aware recall-first
  budget (~99% table recall)
- **Models**: `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-7B-Instruct` (local SQL specialist),
  `gpt-4.1-mini` / Claude (remote)
- **Multilingual**: BGE-M3 retrieval + multilingual SLM; language ablation kit included (see
  Evaluation)

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/aegis_sql.git
cd aegis_sql

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 2. Configuration

Edit `config.yaml` to customize behavior:

```yaml
# Force local-only mode (no API costs, zero privacy leakage)
router:
  force_local: true

# OR force remote-only mode (maximum accuracy)
router:
  force_remote: true

# OR hybrid mode (automatic routing)
router:
  threshold_complexity: 0.7  # 0-1 scale
```

### 3. Run Workflow

```python
from aegis_sql import AEGISConfig
from aegis_sql.workflow import build_aegis_graph
from aegis_sql.types import Query, Language

# Load configuration
config = AEGISConfig.from_yaml("config.yaml")

# Build workflow graph
graph = build_aegis_graph(config)

# Execute query
query = Query(
    text="Find all employees hired after 2020",
    language=Language.ENGLISH,
    database_id="company_db"
)

result = graph.invoke({
    "query": query,
    "database_id": "company_db"
})

print(result["sql"].text)
```

---

## Configuration Modes

### 🏠 **Local-Only Mode** (Zero Privacy Leakage, No API Costs)
```yaml
router:
  force_local: true
```
OR
```yaml
cost:
  budget_per_query: 0.0
```

### ☁️ **Remote-Only Mode** (Maximum Accuracy)
```yaml
router:
  force_remote: true
```
OR
```yaml
router:
  threshold_complexity: 0.0
```

### 🔄 **Hybrid Mode** (Automatic Routing - DEFAULT)
```yaml
router:
  threshold_complexity: 0.7  # Queries >= 0.7 complexity → REMOTE
cost:
  budget_per_query: 0.01  # $0.01 per query
```

### 🚫 **Disable Abstraction** (WARNING: Sends sensitive data to remote LLM!)
```yaml
privacy:
  abstraction_enabled: false
  reconstruction_enabled: false
```
OR
```yaml
privacy:
  epsilon: 0.0
```

---

## Evaluation

### Full Pipeline Evaluation

Run the complete BIRD-dev evaluation pipeline with all metrics (EX, VES, privacy, cost, latency):

```bash
# Quick test with 2 queries (recommended for testing setup)
python run_full_evaluation.py --bird_path data/bird --num_queries 2 --seed 42

# Test with 10 queries
python run_full_evaluation.py --bird_path data/bird --num_queries 10 --seed 42

# Sample 100 queries
python run_full_evaluation.py --bird_path data/bird --num_queries 100 --seed 42 --output_name exp_100

# Full BIRD-dev (1534 queries)
python run_full_evaluation.py --bird_path data/bird --output_name full_bird_dev

# Use existing predictions (skip generation, only compute metrics)
python run_full_evaluation.py --bird_path data/bird --predictions_file evaluation/output/exp/predictions.jsonl
```

**What it does:**
1. **Step 1**: Generates SQL predictions using AEGIS-SQL workflow
2. **Step 2**: Computes **EX (Execution Accuracy)** - executes predicted and ground truth SQL, compares results
3. **Step 3**: Computes **VES (Valid Efficiency Score)** - measures query efficiency with timing
4. **Step 4**: Computes three-axis metrics (privacy loss, cost, latency)
5. **Step 5**: Generates comprehensive evaluation report

### Prediction Generation

Generate predictions without running full evaluation metrics:

```bash
# Generate SQL for 10 test queries
python run_bird_evaluation.py --bird_path data/bird --num_queries 10 --seed 42

# Generate for 100 queries with custom output name
python run_bird_evaluation.py --bird_path data/bird --num_queries 100 --seed 42 --output_name exp_100

# Full BIRD-dev dataset (all 1534 queries)
python run_bird_evaluation.py --bird_path data/bird --output_name full_bird_dev
```

### Language Ablation (Spanish)

Test how EX changes when only the **question language** changes — same seed-42 stratified
100-query sample, same gold SQL, same databases, same pipeline. The kit builds a parallel
data dir `data/bird_es/` where the 100 sampled questions are machine-translated to Spanish
(named entities, quoted strings, and numbers preserved verbatim so DB value grounding is not
confounded; the BIRD `evidence` hint stays in English to isolate the language variable):

```bash
# One-time: build the Spanish mirror (needs OPENAI_API_KEY and data/bird)
python scripts/make_spanish_sample.py

# Spanish run — identical pipeline/config/seed, only the data dir differs
python run_full_evaluation.py --config config.yaml --seed 42 --num_queries 100 --stratify \
       --bird_path data/bird_es --output_name local_100_es

# Compare ex_results.txt against the English baseline run (same 100 question_ids)
```

Translations are checkpointed to `data/bird_es/translations.jsonl` (`question_id`, `en`,
`es`) for manual review. Note: retrieval (BGE-M3) and the SLM are multilingual, but query
decomposition's stopwords are English-only — that degradation is part of what the ablation
measures.

**Checkpoint/Resume:**
- If generation is interrupted, rerun the same command
- Automatically skips queries already in predictions.jsonl
- Protects against transient 429 rate limit errors with exponential backoff

### Metrics Computed

**BIRD Benchmark Metrics:**
- **EX (Execution Accuracy)**: Percentage of queries returning correct results by executing both predicted and ground truth SQL on actual databases
- **VES (Valid Efficiency Score)**: `sqrt(gt_time / pred_time) × 100` - measures both correctness and query efficiency

**Three-Axis Metrics:**
- **Privacy Loss**: ℒ_priv = ε × E[|prompt|] × Pr(r=remote)
- **Cost per Query**: Average USD cost (API usage)
- **Latency**: End-to-end processing time (SQL generation)

**All metrics broken down by difficulty**: Simple, Moderate, Challenging, Overall

### Output Files

Results saved to `evaluation/output/{experiment_name}/`:
```
evaluation/output/{experiment_name}/
├── predictions.jsonl           # Per-query predictions and metrics
├── evaluation_report.json      # Aggregated metrics summary
├── ex_results.txt             # EX evaluation details
├── ex_results_latency.json    # EX execution timing stats
├── ves_results.txt            # VES evaluation details
├── ves_results_latency.json   # VES timing stats
├── config_snapshot.yaml       # Configuration used
├── evaluation.log             # Detailed logs
└── full_evaluation.log        # Full pipeline logs
```

---

## Project Structure

```
aegis_sql/
├── __init__.py
├── config.py                    # Pydantic configuration models
├── config.yaml                  # Main configuration file
├── aegis_types.py               # Core type definitions
├── retriever/                   # RAG v2 schema linking
│   ├── pipeline.py             # Multi-step retrieve → fuse → value-ground → budget
│   ├── fusion.py               # RRF + recall-first adaptive table budget
│   ├── value_index.py          # DB value grounding (LIKE probes)
│   ├── schema_retriever.py
│   └── embedding_models.py
├── router/                      # Content-independent routing
│   └── content_independent_router.py
├── abstraction/                 # DP abstraction & reconstruction
│   ├── dp_abstractor.py
│   ├── placeholder_vocab.py
│   ├── reconstruction.py
│   └── sensitivity_policy.py
├── generator/                   # SQL generation, selection, repair
│   ├── slm_generator.py        # Local FSLM (CscSQL-Merge-Qwen2.5-Coder-7B); execution-vote candidates
│   ├── candidate_selector.py   # Execution-guided majority-vote selection
│   ├── literal_repair.py       # Post-hoc near-miss literal → stored DB value
│   ├── sql_postprocess.py      # CAST fix + reserved-word table quoting
│   └── llm_fallback.py         # Remote FLLM (gpt-4.1-mini, Claude)
├── verifier/                    # Neuro-symbolic verification
│   ├── grammar_verifier.py
│   ├── schema_verifier.py
│   ├── execution_verifier.py
│   └── feedback_generator.py
├── workflow/                    # LangGraph orchestration
│   ├── graph.py                # Workflow graph definition
│   └── state.py                # State management
└── evaluation/                  # Evaluation framework
    ├── evaluator_ex.py         # EX metric
    ├── evaluator_ves.py        # VES metric
    ├── metrics.py              # Three-axis metrics
    └── README.md               # Evaluation instructions
```

---

## Research Paper

Based on: **"Three-Axis Constrained Optimization for Hybrid NL2SQL"**

**Key Contributions:**
1. **Router-Before-Abstraction Architecture**: Local path has zero privacy leakage
2. **Content-Independent Routing**: Theorem 1 - Privacy amplification via hybrid routing
3. **Multilingual DP Abstraction**: Language-agnostic placeholder vocabulary
4. **Three-Axis Optimization**: Utility, Privacy, Cost trade-offs

---

## Implementation Status

### ✅ Fully Implemented Components

**Core System:**
- ✅ Configuration system with YAML and environment variables
- ✅ Type definitions (Query, SQL, Schema, etc.)
- ✅ LangGraph workflow orchestration
- ✅ State management and routing logic

**Query Planner (Schema Retrieval):**
- ✅ Schema retriever with pass-through mode (returns all schema elements)
- ✅ Support for multilingual embeddings (BGE-M3 ready)
- ⚙️ Future: Fine-tuned embeddings and FAISS indexing

**Content-Independent Router:**
- ✅ Complexity-based routing (local/remote/hybrid modes)
- ✅ Cost budget enforcement
- ✅ Force local/remote override options

**SQL Generation:**
- ✅ Local SLM generator with HuggingFace models (default: cycloneboy/CscSQL-Merge-Qwen2.5-Coder-7B-Instruct)
- ✅ Execution-vote self-consistency (n=5, chunked/OOM-safe, seeded)
- ✅ Deterministic post-processing: literal repair, reserved-word quoting, CAST fix
- ✅ LoRA adapter support
- ✅ Remote LLM fallback (OpenAI gpt-4.1-mini, Anthropic Claude)
- ✅ SQL extraction and formatting

**Privacy (DP Abstraction & Reconstruction):**
- ✅ Token-level DP abstraction with exponential mechanism
- ✅ Sensitivity detection (PII, proprietary data)
- ✅ Placeholder vocabulary with semantic categories
- ✅ Reconstruction with substitution tracking
- ⚙️ Future: Advanced NER for better entity detection

**Reviewer (Verification):**
- ✅ Grammar verification (sqlglot parser)
- ✅ Schema verification (element validation)
- ✅ Execution verification with timeout and sampling
- ✅ Feedback generation for refinement

**Evaluation Pipeline:**
- ✅ BIRD-dev dataset loader with schema extraction
- ✅ Stratified sampling by difficulty
- ✅ Full evaluation orchestration (predictions + metrics)
- ✅ EX and VES metric computation
- ✅ Three-axis metrics (privacy, cost, latency)
- ✅ Comprehensive logging and reporting

### 🎯 Testing Status
- ✅ E2E workflow tested with real SLM inference
- ✅ 10-query evaluation completed successfully
- ✅ All verifiers passing on generated SQL
- ✅ Zero privacy loss and cost on local path verified

### ⚙️ Future Enhancements
- Advanced schema retrieval with fine-tuned embeddings
- FAISS/vector database integration for large schemas
- Enhanced NER for abstraction
- Multi-turn refinement with feedback loop

---

## Citation

```bibtex
@article{aegis_sql_2025,
  title={Three-Axis Constrained Optimization for Hybrid NL2SQL},
  author={David Onyango and MINDS Lab},
  year={2025}
}
```

---

## License

MIT License

---

## Contact

For questions or issues, please open an issue on GitHub or contact the MINDS Lab.

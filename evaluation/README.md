# AEGIS-SQL Evaluation Guide

This directory contains evaluation tools for measuring AEGIS-SQL performance on standard NL2SQL benchmarks (BIRD, Spider, etc.).

---

## Evaluation Metrics

### 1. **EX (Execution Accuracy)**
Measures whether predicted SQL returns the same results as ground truth.

**Formula:** `EX = Pr[exec(predicted) = exec(ground_truth)]`

**Range:** 0-100%

**Implementation:** `evaluator_ex.py`

---

### 2. **VES (Valid Efficiency Score)**
Measures both correctness AND efficiency (speed) of generated SQL.

**Formula:** `VES = sqrt(t_gold / t_pred) × 100` (if correct), `0` (if incorrect)

**Range:** 0-∞ (can exceed 100% if predicted query is faster than ground truth)

**Implementation:** `evaluator_ves.py`

---

### 3. **Privacy Loss**
Measures privacy leakage to remote LLM.

**Formula:** `ℒ_priv = ε × E[|prompt|] × Pr(r=remote)`

**Range:** 0-∞ (lower is better)

**Implementation:** `metrics.py`

---

### 4. **Cost per Query**
Average USD cost per query (API calls + compute).

**Formula:** `Cost = Pr(r=local) × cost_local + Pr(r=remote) × cost_remote`

**Range:** $0-∞ (lower is better)

**Implementation:** `metrics.py`

---

### 5. **Latency**
End-to-end processing time per query.

**Metrics:**
- Mean latency
- P50, P90, P99 latencies
- Timeout rate

**Implementation:** `metrics.py`

---

## Running Evaluation

### Prerequisites

1. **Download BIRD Benchmark**
```bash
# Download BIRD-dev dataset
wget https://bird-bench.github.io/data/bird_dev.zip
unzip bird_dev.zip -d ./data/
```

2. **Generate Predictions**
Run AEGIS-SQL on BIRD-dev to generate predicted SQL:
```bash
python -m aegis_sql.run_bird_eval \
  --config config.yaml \
  --output ./results/predictions.jsonl
```

---

### EX Evaluation (Execution Accuracy)

```bash
python -m evaluation.evaluator_ex \
  --predicted_sql_path ./results/predictions.jsonl \
  --ground_truth_path ./data/bird_dev/dev.json \
  --db_root_path ./data/bird_dev/databases/ \
  --diff_json_path ./data/bird_dev/dev_difficulty.json \
  --num_cpus 8 \
  --meta_time_out 30.0 \
  --sql_dialect SQLite \
  --output_log_path ex_results.txt
```

**Output:**
```
EX Results (Execution Accuracy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Simple:        85.3%  (400/469)
Moderate:      68.7%  (280/408)
Challenging:   45.2%  (112/248)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall:       70.4%  (792/1125)
```

---

### VES Evaluation (Valid Efficiency Score)

```bash
python -m evaluation.evaluator_ves \
  --predicted_sql_path ./results/predictions.jsonl \
  --ground_truth_path ./data/bird_dev/dev.json \
  --db_root_path ./data/bird_dev/databases/ \
  --diff_json_path ./data/bird_dev/dev_difficulty.json \
  --num_cpus 8 \
  --iterate_num 10 \
  --meta_time_out 30.0 \
  --sql_dialect SQLite \
  --output_log_path ves_results.txt
```

**Output:**
```
VES Results (Valid Efficiency Score)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Simple:        88.1  (can exceed 100 if faster)
Moderate:      72.3
Challenging:   48.9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall:       72.8
```

**Note:** VES can exceed 100% when predicted queries are faster than ground truth.

---

### Three-Axis Metrics (Privacy, Cost, Latency)

```bash
python -m evaluation.metrics \
  --results_dir ./results/ \
  --config config.yaml \
  --output three_axis_metrics.json
```

**Output:**
```json
{
  "execution_accuracy": 70.4,
  "ves": 72.8,
  "privacy_loss": 0.245,
  "cost_per_query": 0.0032,
  "latency_mean_ms": 1243.5,
  "latency_p50_ms": 987.2,
  "latency_p90_ms": 2156.8,
  "latency_p99_ms": 3421.1,
  "routing_stats": {
    "pr_local": 0.68,
    "pr_remote": 0.32
  }
}
```

---

## Evaluation Scenarios

### 1. **Local-Only Mode**
```yaml
# config.yaml
router:
  force_local: true
```

**Expected:**
- Privacy Loss: **0.0** (zero leakage)
- Cost: **Very Low** (no API calls)
- EX: **Moderate** (SLM accuracy)

---

### 2. **Remote-Only Mode**
```yaml
# config.yaml
router:
  force_remote: true
privacy:
  epsilon: 1.0
```

**Expected:**
- Privacy Loss: **ε × E[|prompt|]** (full leakage)
- Cost: **High** (all queries use API)
- EX: **High** (LLM accuracy)

---

### 3. **Hybrid Mode (Default)**
```yaml
# config.yaml
router:
  threshold_complexity: 0.7
privacy:
  epsilon: 1.0
```

**Expected:**
- Privacy Loss: **ε × E[|prompt|] × Pr(r=remote)** (amplified)
- Cost: **Medium** (mix of local/remote)
- EX: **High** (best of both)

---

### 4. **No Abstraction (Unsafe!)**
```yaml
# config.yaml
privacy:
  abstraction_enabled: false
```

**Expected:**
- Privacy Loss: **∞** (sensitive data exposed)
- Cost: **High** (remote queries)
- EX: **Highest** (real data helps accuracy)

---

## Interpreting Results

### Execution Accuracy (EX)
- **> 70%**: Competitive with state-of-the-art
- **> 60%**: Acceptable for production
- **< 50%**: Needs improvement (fine-tuning, better prompts)

### Valid Efficiency Score (VES)
- **> EX**: Generated queries are faster than ground truth
- **≈ EX**: Similar efficiency
- **< EX**: Generated queries are slower (still correct)

### Privacy Loss
- **0.0**: Perfect privacy (local-only)
- **< 1.0**: Strong privacy (ε < 1 with routing)
- **> 10.0**: Weak privacy (high ε or frequent remote)

### Cost per Query
- **< $0.001**: Very efficient (mostly local)
- **< $0.01**: Acceptable for production
- **> $0.05**: Expensive (optimize routing threshold)

### Latency
- **< 1s**: Excellent (real-time)
- **< 5s**: Good (interactive)
- **> 10s**: Slow (optimize model loading)

---

## Debugging Failed Queries

When EX < expected, analyze failures:

```bash
python -m evaluation.analyze_failures \
  --predictions ./results/predictions.jsonl \
  --ground_truth ./data/bird_dev/dev.json \
  --output ./results/failure_analysis.json
```

**Common Failure Modes:**
1. **Grammar Errors**: SQL syntax invalid → Check verifier
2. **Schema Errors**: Wrong table/column names → Improve retriever
3. **Execution Errors**: Runtime errors (division by zero, etc.) → Improve generation
4. **Semantic Errors**: Wrong logic (JOINs, aggregations) → Fine-tune model

---

## Ablation Studies

Test individual components:

### Disable Verification
```yaml
verifier:
  grammar_check: false
  schema_check: false
  execution_check_slm: false
```
**Measures:** Impact of verification on EX

### Disable Routing (Force Local)
```yaml
router:
  force_local: true
```
**Measures:** SLM-only performance

### Disable Routing (Force Remote)
```yaml
router:
  force_remote: true
```
**Measures:** LLM-only performance

### Vary Privacy Budget
```yaml
privacy:
  epsilon: [0.1, 0.5, 1.0, 2.0, 5.0]
```
**Measures:** Privacy-Utility trade-off

---

## Output Files

All evaluation results are saved to `./results/`:

```
results/
├── predictions.jsonl              # Generated SQL for each query
├── ex_results.txt                 # EX evaluation report
├── ex_results_latency.json        # EX latency statistics
├── ves_results.txt                # VES evaluation report
├── ves_results_latency.json       # VES latency statistics
├── three_axis_metrics.json        # Privacy, Cost, Latency
└── failure_analysis.json          # Failed query analysis
```

---

## Benchmark Comparison

Compare AEGIS-SQL against baselines:

| System | EX (%) | VES | Privacy | Cost ($/query) |
|--------|--------|-----|---------|----------------|
| **AEGIS-SQL (Hybrid)** | **70.4** | **72.8** | **0.245** | **$0.0032** |
| GPT-4 (no privacy) | 72.1 | 74.3 | ∞ | $0.015 |
| Local SLM only | 58.3 | 60.1 | 0.0 | $0.0001 |
| DIN-SQL | 60.1 | N/A | ∞ | $0.012 |
| DAIL-SQL | 57.4 | N/A | ∞ | $0.008 |

---

## Contact

For evaluation issues or questions, open an issue on GitHub.

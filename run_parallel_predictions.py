"""Parallel BIRD-dev prediction generation for AEGIS-SQL with checkpoint/resume.

Generates predictions.jsonl in the same format as run_bird_evaluation.py but with
parallel execution for faster completion (30-90 min vs 1-2 hr).

Usage:
    # Generate predictions for all 1534 queries with 12 concurrent workers
    python run_parallel_predictions.py \\
        --bird_path data/bird \\
        --out evaluation/output/full_bird_dev/predictions.jsonl \\
        --concurrency 12

    # Test with first 10 queries
    python run_parallel_predictions.py \\
        --bird_path data/bird \\
        --out evaluation/output/test_parallel/predictions.jsonl \\
        --concurrency 4 \\
        --limit 10

    # Resume from interruption (automatically skips completed queries)
    python run_parallel_predictions.py \\
        --bird_path data/bird \\
        --out evaluation/output/full_bird_dev/predictions.jsonl \\
        --concurrency 12

Then compute metrics:
    python run_full_evaluation.py \\
        --bird_path data/bird \\
        --predictions_file evaluation/output/full_bird_dev/predictions.jsonl

Performance:
    - Parallelizes GPT-4o API calls (REMOTE path) and SQL verification waits
    - Local SLM queries serialize on GPU (fast for 1.5B models)
    - Set --concurrency based on GPT-4o TPM limit (start with 8-12)
    - Checkpoint/resume allows recovery from interruptions
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger

# Use actual AEGIS-SQL imports (not aegis_sql.* from runbook)
from config import AEGISConfig
from workflow import build_aegis_graph
from aegis_types import Query, Language, RoutingDecision

# Retryable error patterns for 429 rate limits
_RETRYABLE_ERRORS = ("rate limit", "429", "overloaded", "timeout", "503", "502", "504")
_write_lock = Lock()


def is_retryable(e: Exception) -> bool:
    """Check if exception is retryable (rate limit, timeout, etc.)."""
    return any(pattern in str(e).lower() for pattern in _RETRYABLE_ERRORS)


def load_bird_queries(bird_path: str):
    """Load queries from BIRD dev.json.

    Args:
        bird_path: Path to BIRD data directory containing dev.json

    Returns:
        List of query dictionaries with question_id, db_id, question, SQL, etc.
    """
    dev_json_path = Path(bird_path) / "dev.json"

    if not dev_json_path.exists():
        raise FileNotFoundError(f"BIRD dev.json not found: {dev_json_path}")

    with open(dev_json_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # Ensure each query has a stable question_id
    for i, item in enumerate(queries):
        if 'question_id' not in item:
            item['question_id'] = i

    return queries


def load_bird_schemas(bird_path: str):
    """Load database schemas from BIRD.

    Args:
        bird_path: Path to BIRD data directory

    Returns:
        Dict mapping db_id to schema dictionary
    """
    from evaluation.bird_loader import BIRDLoader

    loader = BIRDLoader(bird_path)
    schemas = {}

    # Load unique databases
    queries = load_bird_queries(bird_path)
    unique_dbs = set(q['db_id'] for q in queries)

    for db_id in unique_dbs:
        try:
            schema = loader.load_schema_for_db(db_id)
            schemas[db_id] = schema
        except Exception as e:
            logger.warning(f"Failed to load schema for {db_id}: {e}")
            schemas[db_id] = None

    return schemas


def predict_one(graph, query_dict, schema, max_retries=4):
    """Generate SQL prediction for a single query with retry logic.

    Args:
        graph: Compiled AEGIS workflow graph
        query_dict: Query dictionary from dev.json
        schema: Database schema object
        max_retries: Maximum retry attempts for transient errors

    Returns:
        Prediction dictionary matching run_bird_evaluation.py format
    """
    query_start = time.time()

    # Create Query object
    query = Query(
        text=query_dict['question'],
        language=Language.ENGLISH,  # BIRD-dev is primarily English
        database_id=query_dict['db_id'],
    )

    # Retry loop for transient errors
    for attempt in range(max_retries):
        try:
            # Invoke workflow graph
            result = graph.invoke({
                "query": query,
                "database_id": query_dict['db_id'],
                "schema": schema,
            })

            # Extract results
            sql = result.get("sql")
            predicted_sql = sql.text.strip() if sql else ""

            routing_decision = result.get("routing_decision")
            generation_source = result.get("generation_source", "unknown")

            abstracted_prompt = result.get("abstracted_prompt")
            verification_result = result.get("verification_result")

            # Track metrics
            query_time = time.time() - query_start
            cost_usd = result.get("cost_usd", 0.0)
            latency_ms = result.get("latency_ms", query_time * 1000)
            privacy_loss = result.get("privacy_loss", 0.0)

            # Create prediction record (matches run_bird_evaluation.py format)
            prediction = {
                "question_id": query_dict['question_id'],
                "db_id": query_dict['db_id'],
                "question": query_dict['question'],
                "evidence": query_dict.get('evidence', ''),
                "ground_truth_sql": query_dict['SQL'],
                "predicted_sql": predicted_sql,
                "routing_decision": routing_decision.value if routing_decision else "unknown",
                "generation_source": generation_source,
                "abstraction_applied": abstracted_prompt is not None,
                "num_substitutions": abstracted_prompt.num_substitutions if abstracted_prompt else 0,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "privacy_loss": privacy_loss,
                "grammar_valid": verification_result.grammar_valid if verification_result else None,
                "schema_valid": verification_result.schema_valid if verification_result else None,
                "execution_valid": verification_result.execution_valid if verification_result else None,
                "difficulty": query_dict.get('difficulty', 'unknown'),
            }

            return prediction

        except Exception as e:
            # Check if retryable
            if attempt < max_retries - 1 and is_retryable(e):
                wait_time = (2 ** attempt) + 0.5  # Exponential backoff
                logger.warning(f"Query {query_dict['question_id']} failed (attempt {attempt+1}/{max_retries}), retrying in {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
                continue

            # Non-retryable or exhausted retries - return error record
            logger.error(f"Query {query_dict['question_id']} failed permanently: {e}")
            return {
                "question_id": query_dict['question_id'],
                "db_id": query_dict['db_id'],
                "question": query_dict['question'],
                "evidence": query_dict.get('evidence', ''),
                "ground_truth_sql": query_dict['SQL'],
                "predicted_sql": "",
                "routing_decision": "error",
                "generation_source": "error",
                "error": str(e),
                "difficulty": query_dict.get('difficulty', 'unknown'),
            }

    # Should never reach here, but return error if it does
    return {
        "question_id": query_dict['question_id'],
        "db_id": query_dict['db_id'],
        "error": "Unexpected retry exhaustion"
    }


def append_jsonl(path, record):
    """Thread-safe append to JSONL file.

    Args:
        path: Path to JSONL file
        record: Dictionary to append
    """
    with _write_lock:
        with open(path, "a", encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Parallel BIRD-dev prediction generation with checkpoint/resume"
    )
    parser.add_argument(
        "--bird_path",
        type=str,
        required=True,
        help="Path to BIRD data directory (contains dev.json and dev_databases/)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="evaluation/output/full_bird_dev/predictions.jsonl",
        help="Output path for predictions.jsonl"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="Number of concurrent workers (default: 12, tune based on GPT-4o TPM limit)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N queries for testing (default: 0 = all queries)"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("AEGIS-SQL Parallel Prediction Generation")
    logger.info("=" * 80)
    logger.info(f"BIRD path: {args.bird_path}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.out}")
    logger.info(f"Concurrency: {args.concurrency}")

    # Load config and build graph
    logger.info("\n[1/4] Loading configuration and building workflow graph...")
    config = AEGISConfig.from_yaml(args.config)
    graph = build_aegis_graph(config)  # Single shared graph, SLM loaded once
    logger.info("✓ Workflow graph compiled")

    # Load BIRD queries
    logger.info("\n[2/4] Loading BIRD-dev queries...")
    queries = load_bird_queries(args.bird_path)

    if args.limit > 0:
        queries = queries[:args.limit]
        logger.info(f"✓ Limited to first {args.limit} queries")

    logger.info(f"✓ Loaded {len(queries)} queries")

    # Load schemas
    logger.info("\n[3/4] Loading database schemas...")
    schemas = load_bird_schemas(args.bird_path)
    logger.info(f"✓ Loaded schemas for {len(schemas)} databases")

    # Resume: skip queries already in output file
    done = set()
    if output_path.exists():
        logger.info(f"\nFound existing predictions file, resuming...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pred = json.loads(line)
                    done.add(pred['question_id'])
                except Exception as e:
                    logger.warning(f"Failed to parse line in existing file: {e}")
        logger.info(f"✓ {len(done)} queries already completed")

    # Filter to remaining queries
    todo = [q for q in queries if q['question_id'] not in done]

    if not todo:
        logger.info("\n✓ All queries already completed!")
        return

    logger.info(f"\n[4/4] Generating predictions for {len(todo)} queries...")
    logger.info(f"Concurrency: {args.concurrency} workers")
    logger.info(f"Resume: {len(done)} done, {len(todo)} remaining")
    logger.info("=" * 80)

    # Parallel execution with thread pool
    start_time = time.time()
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # Submit all tasks
        futures = {}
        for query_dict in todo:
            schema = schemas.get(query_dict['db_id'])
            future = pool.submit(predict_one, graph, query_dict, schema)
            futures[future] = query_dict

        # Process completions
        for future in as_completed(futures):
            result = future.result()
            append_jsonl(args.out, result)

            completed += 1
            if "error" in result:
                errors += 1

            # Print progress every 50 queries
            if completed % 50 == 0 or completed == len(todo):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta_min = (len(todo) - completed) / rate / 60 if rate > 0 else 0

                logger.info(
                    f"Progress: {completed}/{len(todo)} "
                    f"({completed/len(todo)*100:.1f}%) | "
                    f"Rate: {rate:.1f} queries/s | "
                    f"ETA: {eta_min:.0f} min | "
                    f"Errors: {errors}"
                )

    # Final statistics
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total queries: {len(todo)}")
    logger.info(f"Successful: {completed - errors}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average rate: {completed/total_time:.2f} queries/second")
    logger.info(f"Output: {args.out}")
    logger.info("=" * 80)
    logger.info("\nNext step: Compute metrics with:")
    logger.info(f"  python run_full_evaluation.py --bird_path {args.bird_path} --predictions_file {args.out}")


if __name__ == "__main__":
    main()

"""Generate SQL predictions for BIRD-dev benchmark.

Usage:
    # Test with 10 queries
    python run_bird_evaluation.py --num_queries 10 --seed 42

    # Sample 100 queries
    python run_bird_evaluation.py --num_queries 100 --seed 42 --output_name exp_100

    # Full BIRD-dev (1534 queries)
    python run_bird_evaluation.py --output_name full_bird_dev
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from config import AEGISConfig
from workflow import build_aegis_graph
from evaluation.bird_loader import load_bird_dev
from aegis_types import RoutingDecision


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL predictions for BIRD-dev benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Number of queries to sample (default: all 1534)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        default=True,
        help="Stratified sampling by difficulty (default: True)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Experiment name (default: timestamp)",
    )
    parser.add_argument(
        "--bird_path",
        type=str,
        default="data/bird",
        help="Path to BIRD data directory (default: data/bird)",
    )

    args = parser.parse_args()

    # Create output directory
    if args.output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_name = f"bird_eval_{timestamp}"

    output_dir = Path("evaluation/output") / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging: send the full DEBUG/INFO trace to the file only, and keep
    # the console limited to warnings/errors so the tqdm progress bar (below)
    # stays clean instead of being buried under per-node INFO logs.
    log_file = output_dir / "evaluation.log"
    logger.remove()  # drop loguru's default stderr sink that floods the console
    logger.add(log_file, format="{time} {level} {message}", level="DEBUG")
    logger.add(sys.stderr, level="WARNING")

    logger.info("=" * 80)
    logger.info("AEGIS-SQL BIRD Evaluation - Prediction Generation")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.output_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Num queries: {args.num_queries or 'all (1534)'}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Stratify: {args.stratify}")

    try:
        # Load configuration
        logger.info("\n[1/5] Loading configuration...")
        config = AEGISConfig.from_yaml(args.config)
        logger.info(f"✓ Config loaded: {config.slm.model}")

        # Save config snapshot
        config_snapshot = output_dir / "config_snapshot.yaml"
        import shutil
        shutil.copy(args.config, config_snapshot)
        logger.info(f"✓ Config snapshot saved: {config_snapshot}")

        # Load BIRD queries
        logger.info("\n[2/5] Loading BIRD-dev queries...")
        queries = load_bird_dev(
            bird_path=args.bird_path,
            num_queries=args.num_queries,
            seed=args.seed,
            stratify=args.stratify,
        )
        logger.info(f"✓ Loaded {len(queries)} queries")

        # Build workflow graph
        logger.info("\n[3/5] Building AEGIS-SQL workflow...")
        graph = build_aegis_graph(config)
        logger.info("✓ Workflow graph compiled")

        # Warmup model cache (pre-load models and embeddings)
        logger.info("\n[4/5] Warming up model cache...")
        from workflow.model_cache import get_cache
        cache = get_cache()
        cache.set_config(config)

        # Extract unique databases and their schemas
        db_schemas = {}
        for query_dict in queries:
            db_id = query_dict['db_id']
            if db_id not in db_schemas:
                db_schemas[db_id] = query_dict['schema']

        # Warmup cache with all unique database schemas
        db_list = [(db_id, schema) for db_id, schema in db_schemas.items()]
        cache.warmup(config, db_list)
        logger.info(f"✓ Cache warmed up with {len(db_list)} databases")

        # Generate predictions
        logger.info(f"\n[5/5] Generating SQL predictions for {len(queries)} queries...")
        logger.info(f"Estimated time (with cache): ~{len(queries) * 2 / 60:.1f} minutes @ 2s/query")

        predictions = []
        start_time = time.time()

        # Create progress bar
        pbar = tqdm(queries, desc="Generating SQL", unit="query", ncols=100)

        for i, query_dict in enumerate(pbar):
            query_start = time.time()

            # Update progress bar description
            pbar.set_description(f"Query {i+1}/{len(queries)} (ID={query_dict['question_id']})")

            logger.info(f"\n--- Query {i+1}/{len(queries)} (ID={query_dict['question_id']}) ---")
            logger.info(f"Question: {query_dict['question'][:80]}...")
            logger.info(f"Database: {query_dict['db_id']}")

            try:
                # Prepare initial state
                initial_state = {
                    "query": query_dict['schema'].database_id,  # Will be converted
                    "schema": query_dict['schema'],
                    "database_id": query_dict['db_id'],
                    "db_path": query_dict['db_path'],
                    "cost_usd": 0.0,
                    "latency_ms": 0.0,
                    "privacy_loss": 0.0,
                    "verification_attempts": 0,
                }

                # Convert to AEGIS Query object
                from evaluation.bird_loader import BIRDLoader
                loader = BIRDLoader(args.bird_path)
                aegis_query = loader.query_to_aegis_query(query_dict)
                initial_state["query"] = aegis_query

                # Run workflow. recursion_limit is a hard backstop: the repair
                # loop is already bounded by verifier.max_repair_attempts, but
                # this guarantees a pathological state raises instead of hanging
                # the full 1534-query run (caught by the except below).
                result = graph.invoke(initial_state, config={"recursion_limit": 12})

                # Extract results
                sql = result.get("sql")
                routing_decision = result.get("routing_decision")
                abstracted_prompt = result.get("abstracted_prompt")
                verification_result = result.get("verification_result")

                query_latency = (time.time() - query_start) * 1000

                # Clean SQL text (remove trailing newlines and extra spaces)
                predicted_sql = sql.text.strip() if sql else ""

                # Create prediction record
                prediction = {
                    "question_id": query_dict['question_id'],
                    "db_id": query_dict['db_id'],
                    "question": query_dict['question'],
                    "evidence": query_dict.get('evidence', ''),
                    "ground_truth_sql": query_dict['SQL'],
                    "predicted_sql": predicted_sql,
                    "routing_decision": routing_decision.value if routing_decision else "unknown",
                    "generation_source": result.get("generation_source", "unknown"),
                    "abstraction_applied": abstracted_prompt is not None,
                    "num_substitutions": abstracted_prompt.num_substitutions if abstracted_prompt else 0,
                    "latency_ms": query_latency,
                    "cost_usd": result.get("cost_usd", 0.0),
                    "privacy_loss": result.get("privacy_loss", 0.0),
                    "verification_status": verification_result.status.value if verification_result else "unknown",
                    "grammar_valid": verification_result.grammar_valid if verification_result else None,
                    "schema_valid": verification_result.schema_valid if verification_result else None,
                    "execution_valid": verification_result.execution_valid if verification_result else None,
                    "difficulty": query_dict.get('difficulty', 'unknown'),
                    # Retrieval diagnostics (table recall / noise analysis).
                    "retrieved_tables": result.get("retrieved_tables", []),
                    "num_retrieved_columns": result.get("num_retrieved_columns", 0),
                }

                predictions.append(prediction)

                # Update progress bar with stats
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining_time = avg_time * (len(queries) - i - 1)
                pbar.set_postfix({
                    'avg': f'{avg_time:.1f}s',
                    'eta': f'{remaining_time/60:.1f}min',
                    'route': routing_decision.value if routing_decision else 'unknown'
                })

                logger.info(f"✓ Completed in {query_latency/1000:.1f}s")
                logger.info(f"  Routing: {routing_decision}")
                logger.info(f"  SQL: {predicted_sql[:80] if predicted_sql else 'None'}...")
                logger.info(f"  Verified: {verification_result.status if verification_result else 'Unknown'}")

            except Exception as e:
                logger.error(f"✗ Failed to process query {query_dict['question_id']}: {e}")
                logger.exception("Full traceback:")

                # Add failed prediction
                predictions.append({
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
                })

        # Close progress bar
        pbar.close()

        total_time = time.time() - start_time

        # Print cache statistics
        logger.info("\n" + "=" * 80)
        logger.info("CACHE STATISTICS")
        logger.info("=" * 80)
        cache.print_stats()

        # Save predictions
        logger.info("\n[6/6] Saving predictions...")
        predictions_file = output_dir / "predictions.jsonl"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                # Write each prediction as a formatted JSON object (not indented, one line)
                # This keeps JSONL format but ensures no trailing newlines in SQL
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')

        logger.info(f"✓ Saved {len(predictions)} predictions to: {predictions_file}")

        # Save summary
        summary = {
            "experiment_name": args.output_name,
            "timestamp": datetime.now().isoformat(),
            "config_file": args.config,
            "num_queries": len(queries),
            "seed": args.seed,
            "stratify": args.stratify,
            "total_time_seconds": total_time,
            "avg_time_per_query_seconds": total_time / len(queries) if queries else 0,
            "predictions_file": str(predictions_file),
        }

        summary_file = output_dir / "generation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Summary saved to: {summary_file}")

        # ===================================================================
        # Compute EX and VES metrics automatically
        # ===================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Computing EX and VES Metrics")
        logger.info("=" * 80)

        # Compute EX (Execution Accuracy)
        logger.info("\n[7/8] Computing EX (Execution Accuracy)...")
        try:
            ex_cmd = [
                sys.executable,
                "-m", "evaluation.evaluator_ex",
                "--predicted_sql_path", str(predictions_file),
                "--ground_truth_path", f"{args.bird_path}/dev.json",
                "--db_root_path", f"{args.bird_path}/dev_databases",
                "--diff_json_path", f"{args.bird_path}/dev.json",
            ]
            logger.info(f"Running: {' '.join(ex_cmd)}")

            ex_result = subprocess.run(ex_cmd, capture_output=True, text=True, check=False)

            # Save EX output
            ex_output_file = output_dir / "ex_results.txt"
            with open(ex_output_file, 'w', encoding='utf-8') as f:
                f.write("=== EX (Execution Accuracy) Results ===\n\n")
                f.write(ex_result.stdout)
                if ex_result.stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(ex_result.stderr)

            if ex_result.returncode == 0:
                logger.info(f"✓ EX computation completed")
                logger.info(f"✓ EX results saved to: {ex_output_file}")
                # Print key results
                for line in ex_result.stdout.split('\n'):
                    if 'accuracy' in line.lower() or 'execution' in line.lower():
                        logger.info(f"  {line.strip()}")
            else:
                logger.warning(f"⚠ EX computation had warnings (exit code {ex_result.returncode})")
                logger.warning(f"  Check {ex_output_file} for details")

        except Exception as e:
            logger.error(f"✗ EX computation failed: {e}")
            logger.exception("Full traceback:")

        # Compute VES (Valid Efficiency Score)
        logger.info("\n[8/8] Computing VES (Valid Efficiency Score)...")
        try:
            ves_cmd = [
                sys.executable,
                "-m", "evaluation.evaluator_ves",
                "--predicted_sql_path", str(predictions_file),
                "--ground_truth_path", f"{args.bird_path}/dev.json",
                "--db_root_path", f"{args.bird_path}/dev_databases",
                "--diff_json_path", f"{args.bird_path}/dev.json",
            ]
            logger.info(f"Running: {' '.join(ves_cmd)}")

            ves_result = subprocess.run(ves_cmd, capture_output=True, text=True, check=False)

            # Save VES output
            ves_output_file = output_dir / "ves_results.txt"
            with open(ves_output_file, 'w', encoding='utf-8') as f:
                f.write("=== VES (Valid Efficiency Score) Results ===\n\n")
                f.write(ves_result.stdout)
                if ves_result.stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(ves_result.stderr)

            if ves_result.returncode == 0:
                logger.info(f"✓ VES computation completed")
                logger.info(f"✓ VES results saved to: {ves_output_file}")
                # Print key results
                for line in ves_result.stdout.split('\n'):
                    if 'ves' in line.lower() or 'efficiency' in line.lower():
                        logger.info(f"  {line.strip()}")
            else:
                logger.warning(f"⚠ VES computation had warnings (exit code {ves_result.returncode})")
                logger.warning(f"  Check {ves_output_file} for details")

        except Exception as e:
            logger.error(f"✗ VES computation failed: {e}")
            logger.exception("Full traceback:")

        # Print completion message
        logger.info("\n" + "=" * 80)
        logger.info("✓ EVALUATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Generated predictions: {len(predictions)}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Average per query: {total_time/len(queries):.1f} seconds")
        logger.info(f"\nOutput files:")
        logger.info(f"  Predictions: {predictions_file}")
        logger.info(f"  EX Results: {output_dir / 'ex_results.txt'}")
        logger.info(f"  VES Results: {output_dir / 'ves_results.txt'}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Output directory: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Evaluation failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())

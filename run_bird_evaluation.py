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
from agents import MultiAgentOrchestrator
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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent query workers. GPU decode is serialized internally, so "
        "workers overlap remote API calls + SQLite voting + verification with "
        "GPU work (default: 4; 1 = serial)",
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
        logger.info(f"✓ Config loaded: mode={config.mode}, generator={config.models.generator}")

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

        # Build the per-query pipeline. 'graph' = the v2 LangGraph; 'csc' = the v1
        # CSC merge-revision orchestrator (kept for A/B). Both return the same
        # prediction-contract dict, so everything downstream is identical.
        logger.info("\n[3/5] Building AEGIS-SQL pipeline...")
        if getattr(config, "engine", "graph") == "graph":
            from agents.graph import GraphOrchestrator
            orchestrator = GraphOrchestrator(config)
        else:
            orchestrator = MultiAgentOrchestrator(config)
        logger.info(f"✓ Pipeline ready (engine={config.engine}, mode={config.mode})")

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

        # Generate predictions with a WORKER POOL. The GPU serializes on a lock
        # inside the generator, so extra workers don't fight over VRAM — they
        # overlap everything else (remote API calls, SQLite execution voting,
        # verification, retrieval encoding) with the current query's GPU decode.
        # Works identically on any GPU; workers=1 restores the serial loop.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from evaluation.bird_loader import BIRDLoader

        loader = BIRDLoader(args.bird_path)
        logger.info(
            f"\n[5/5] Generating SQL predictions for {len(queries)} queries "
            f"({args.workers} workers)..."
        )

        def process_one(query_dict: dict) -> dict:
            """Run one query end-to-end and build its prediction record."""
            query_start = time.time()
            try:
                initial_state = {
                    "query": loader.query_to_aegis_query(query_dict),
                    "schema": query_dict['schema'],
                    "database_id": query_dict['db_id'],
                    "db_path": query_dict['db_path'],
                }
                result = orchestrator.run(initial_state)

                sql = result.get("sql")
                routing_decision = result.get("routing_decision")
                verification_result = result.get("verification_result")
                predicted_sql = sql.text.strip() if sql else ""

                return {
                    "question_id": query_dict['question_id'],
                    "db_id": query_dict['db_id'],
                    "question": query_dict['question'],
                    "evidence": query_dict.get('evidence', ''),
                    "ground_truth_sql": query_dict['SQL'],
                    "predicted_sql": predicted_sql,
                    "routing_decision": routing_decision.value if routing_decision else "unknown",
                    "generation_source": result.get("generation_source", "unknown"),
                    # Per-query arm report: which arm produced the FINAL answer
                    # ("local"/"remote"/"merge"/"refine") + candidate pool sizes.
                    "winner_arm": result.get("winner_arm", "unknown"),
                    "candidates_local": result.get("candidates_local", 0),
                    "candidates_remote": result.get("candidates_remote", 0),
                    "abstraction_applied": False,  # privacy isolated in v1
                    "num_substitutions": 0,
                    "latency_ms": (time.time() - query_start) * 1000,
                    "cost_usd": result.get("cost_usd", 0.0),
                    "privacy_loss": result.get("privacy_loss", 0.0),
                    "verification_status": verification_result.status.value if verification_result else "unknown",
                    "grammar_valid": verification_result.grammar_valid if verification_result else None,
                    "schema_valid": verification_result.schema_valid if verification_result else None,
                    "execution_valid": verification_result.execution_valid if verification_result else None,
                    "difficulty": query_dict.get('difficulty', 'unknown'),
                    "retrieved_tables": result.get("retrieved_tables", []),
                    "num_retrieved_columns": result.get("num_retrieved_columns", 0),
                }
            except Exception as e:
                logger.error(f"✗ Failed to process query {query_dict['question_id']}: {e}")
                logger.exception("Full traceback:")
                return {
                    "question_id": query_dict['question_id'],
                    "db_id": query_dict['db_id'],
                    "question": query_dict['question'],
                    "evidence": query_dict.get('evidence', ''),
                    "ground_truth_sql": query_dict['SQL'],
                    "predicted_sql": "",
                    "routing_decision": "error",
                    "generation_source": "error",
                    "winner_arm": "error",
                    "error": str(e),
                    "difficulty": query_dict.get('difficulty', 'unknown'),
                }

        start_time = time.time()
        order = {q['question_id']: i for i, q in enumerate(queries)}
        predictions = []
        pbar = tqdm(total=len(queries), desc="Generating SQL", unit="query", ncols=100)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(process_one, q) for q in queries]
            for future in as_completed(futures):
                pred = future.result()
                predictions.append(pred)
                elapsed = time.time() - start_time
                avg_time = elapsed / len(predictions)
                pbar.set_postfix({
                    'avg': f'{avg_time:.1f}s',
                    'eta': f'{avg_time * (len(queries) - len(predictions)) / 60:.1f}min',
                    'arm': pred.get('winner_arm', '?'),
                })
                pbar.update(1)
        pbar.close()

        # Restore the sampled order (workers complete out of order).
        predictions.sort(key=lambda p: order.get(p['question_id'], 0))

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

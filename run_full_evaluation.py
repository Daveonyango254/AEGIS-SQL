"""Full BIRD-dev evaluation pipeline orchestrator.

Runs the complete evaluation pipeline:
1. Generate SQL predictions (run_bird_evaluation.py)
2. Compute EX metrics (evaluator_ex.py)
3. Compute VES metrics (evaluator_ves.py)
4. Compute three-axis metrics (privacy, cost, latency)
5. Generate comprehensive report

Usage:
    # Run full pipeline with 10 queries
    python run_full_evaluation.py --num_queries 10 --seed 42

    # Use existing predictions file
    python run_full_evaluation.py --predictions_file evaluation/output/bird_eval_20240115_120000/predictions.jsonl

    # Full BIRD-dev evaluation
    python run_full_evaluation.py --output_name full_bird_dev
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from config import AEGISConfig
from evaluation.metrics import MetricsCalculator
from aegis_types import SQL, AbstractedPrompt, RoutingDecision


def run_prediction_generation(
    num_queries: int | None,
    seed: int,
    stratify: bool,
    output_name: str,
    config_path: str,
    bird_path: str
) -> Path:
    """Run prediction generation step.

    Args:
        num_queries: Number of queries to sample (None = all)
        seed: Random seed
        stratify: Stratified sampling flag
        output_name: Experiment name
        config_path: Path to config file
        bird_path: Path to BIRD data

    Returns:
        Path to predictions.jsonl file
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Generate SQL Predictions")
    logger.info("=" * 80)

    # Build command
    cmd = [
        sys.executable,
        "run_bird_evaluation.py",
        "--config", config_path,
        "--seed", str(seed),
        "--output_name", output_name,
        "--bird_path", bird_path,
    ]

    if num_queries is not None:
        cmd.extend(["--num_queries", str(num_queries)])

    if stratify:
        cmd.append("--stratify")

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info("\n" + "=" * 80)
    logger.info("STREAMING OUTPUT FROM run_bird_evaluation.py:")
    logger.info("=" * 80 + "\n")

    # Run prediction generation with real-time output streaming
    start_time = time.time()

    # Use subprocess.Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )

    # Stream output line by line
    for line in process.stdout:
        print(line, end='')  # Print to console in real-time
        sys.stdout.flush()

    # Wait for completion
    return_code = process.wait()
    elapsed = time.time() - start_time

    if return_code != 0:
        logger.error("\n\nPrediction generation failed!")
        raise RuntimeError("Prediction generation failed")

    logger.info(f"\n\n✓ Prediction generation completed in {elapsed/60:.1f} minutes")

    # Find predictions file
    predictions_file = Path("evaluation/output") / output_name / "predictions.jsonl"
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    logger.info(f"✓ Predictions saved: {predictions_file}")
    return predictions_file


def compute_ex_from_predictions(predictions_file: Path) -> dict:
    """Compute EX metrics from predictions file (simplified version).

    Since execution_valid is already computed during prediction generation,
    we can just aggregate those results.

    Args:
        predictions_file: Path to predictions.jsonl

    Returns:
        EX metrics dict
    """
    import json

    total = 0
    execution_valid_count = 0
    verification_pass_count = 0

    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pred = json.loads(line)
                total += 1
                if pred.get('execution_valid', False):
                    execution_valid_count += 1
                if pred.get('verification_status') == 'pass':
                    verification_pass_count += 1

    execution_accuracy = execution_valid_count / total if total > 0 else 0.0
    verification_pass_rate = verification_pass_count / total if total > 0 else 0.0

    return {
        'execution_accuracy': execution_accuracy,
        'verification_pass_rate': verification_pass_rate,
        'total_queries': total,
    }


def run_ex_evaluation(predictions_file: Path, bird_path: str) -> dict:
    """Run EX (Execution Accuracy) evaluation.

    Args:
        predictions_file: Path to predictions.jsonl
        bird_path: Path to BIRD data

    Returns:
        EX evaluation results dict
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Compute EX Metrics (Execution Accuracy)")
    logger.info("=" * 80)

    # Build command
    cmd = [
        sys.executable,
        "-m", "evaluation.evaluator_ex",
        "--predicted_sql_path", str(predictions_file),
        "--ground_truth_path", f"{bird_path}/dev.json",
        "--db_root_path", f"{bird_path}/dev_databases",
        "--diff_json_path", f"{bird_path}/dev.json",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    # Run EX evaluation
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error("EX evaluation failed!")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError("EX evaluation failed")

    logger.info(f"✓ EX evaluation completed in {elapsed:.1f} seconds")

    # Parse EX results from output. The evaluator prints (via print_data):
    #   Overall:     47.00% (100 queries)   (+ Simple/Moderate/Challenging lines)
    import re
    ex_results = {}
    # Use the LAST printed result block (the final aggregate the evaluator also
    # saves to ex_results.txt); an earlier intermediate block can differ by a query.
    overalls = re.findall(r'Overall:\s*([\d.]+)\s*%', result.stdout)
    if overalls:
        ex_results['execution_accuracy'] = float(overalls[-1]) / 100.0
    for diff in ('Simple', 'Moderate', 'Challenging'):
        dms = re.findall(rf'{diff}:\s*([\d.]+)\s*%', result.stdout)
        if dms:
            ex_results[f'{diff.lower()}_accuracy'] = float(dms[-1]) / 100.0

    if 'execution_accuracy' not in ex_results:
        logger.warning("Could not parse execution accuracy from output")
        ex_results['execution_accuracy'] = None

    logger.info(f"✓ EX Results: {ex_results}")
    return ex_results


def run_ves_evaluation(predictions_file: Path, bird_path: str) -> dict:
    """Run VES (Valid Efficiency Score) evaluation.

    Args:
        predictions_file: Path to predictions.jsonl
        bird_path: Path to BIRD data

    Returns:
        VES evaluation results dict
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Compute VES Metrics (Valid Efficiency Score)")
    logger.info("=" * 80)

    # Build command
    cmd = [
        sys.executable,
        "-m", "evaluation.evaluator_ves",
        "--predicted_sql_path", str(predictions_file),
        "--ground_truth_path", f"{bird_path}/dev.json",
        "--db_root_path", f"{bird_path}/dev_databases",
        "--diff_json_path", f"{bird_path}/dev.json",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    # Run VES evaluation
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error("VES evaluation failed!")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError("VES evaluation failed")

    logger.info(f"✓ VES evaluation completed in {elapsed:.1f} seconds")

    # Parse VES results from output. The evaluator prints (via print_data):
    #   Overall:     85.30% (100 queries)   where the value is the VES score.
    import re
    ves_results = {}
    overalls = re.findall(r'Overall:\s*([\d.]+)\s*%', result.stdout)
    if overalls:
        ves_results['ves_score'] = float(overalls[-1])

    if 'ves_score' not in ves_results:
        logger.warning("Could not parse VES score from output")
        ves_results['ves_score'] = None

    logger.info(f"✓ VES Results: {ves_results}")
    return ves_results


def compute_three_axis_metrics(
    predictions_file: Path,
    config: AEGISConfig
) -> dict:
    """Compute three-axis metrics (privacy, cost, latency).

    Args:
        predictions_file: Path to predictions.jsonl
        config: AEGIS configuration

    Returns:
        Three-axis metrics dict
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Compute Three-Axis Metrics (Privacy, Cost, Latency)")
    logger.info("=" * 80)

    # Load predictions
    predictions = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))

    logger.info(f"Loaded {len(predictions)} predictions")

    # Extract data for metrics computation
    generated_sqls = []
    abstracted_prompts = []
    routing_decisions = []

    for pred in predictions:
        # SQL
        sql_text = pred.get('predicted_sql', '')
        if sql_text:
            generated_sqls.append(SQL(text=sql_text))
        else:
            generated_sqls.append(None)

        # Routing decision
        routing_str = pred.get('routing_decision', 'unknown')
        if routing_str == 'local':
            routing_decisions.append(RoutingDecision.LOCAL)
        elif routing_str == 'remote':
            routing_decisions.append(RoutingDecision.REMOTE)
        else:
            routing_decisions.append(RoutingDecision.LOCAL)  # Default

        # Abstracted prompt (approximate from question)
        if pred.get('abstraction_applied', False):
            prompt_text = pred.get('question', '')
            num_subs = pred.get('num_substitutions', 0)
            abstracted_prompts.append(
                AbstractedPrompt(
                    text=prompt_text,
                    original_tokens=[],  # Not stored in predictions
                    placeholder_map={},  # Not stored in predictions
                    num_substitutions=num_subs,
                    epsilon=config.privacy.epsilon
                )
            )
        else:
            # No abstraction, use original question
            abstracted_prompts.append(
                AbstractedPrompt(
                    text=pred.get('question', ''),
                    original_tokens=[],
                    placeholder_map={},
                    num_substitutions=0,
                    epsilon=0.0
                )
            )

    # Initialize metrics calculator
    calculator = MetricsCalculator(
        epsilon=config.privacy.epsilon,
        remote_token_cost=config.cost.remote_token_cost,
        local_cost=config.cost.local_compute_cost
    )

    # Compute privacy loss
    privacy_loss = calculator.compute_privacy_loss(
        abstracted_prompts,
        routing_decisions
    )

    # Compute cost
    avg_cost = calculator.compute_cost_per_query(
        abstracted_prompts,
        routing_decisions,
        generated_sqls
    )

    # Compute latency statistics
    latencies = [p.get('latency_ms', 0.0) for p in predictions]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    # Compute routing statistics
    num_local = sum(1 for r in routing_decisions if r == RoutingDecision.LOCAL)
    num_remote = len(routing_decisions) - num_local
    pr_local = num_local / len(routing_decisions) if routing_decisions else 0.0
    pr_remote = num_remote / len(routing_decisions) if routing_decisions else 0.0

    # Verification statistics
    verification_stats = {
        'pass': sum(1 for p in predictions if p.get('verification_status') == 'pass'),
        'fail': sum(1 for p in predictions if p.get('verification_status') == 'fail'),
        'error': sum(1 for p in predictions if p.get('verification_status') == 'error'),
    }

    metrics = {
        'privacy_loss': privacy_loss,
        'cost_per_query_usd': avg_cost,
        'avg_latency_ms': avg_latency_ms,
        'pr_local': pr_local,
        'pr_remote': pr_remote,
        'num_local': num_local,
        'num_remote': num_remote,
        'verification_stats': verification_stats,
    }

    logger.info("✓ Three-axis metrics computed:")
    logger.info(f"  Privacy Loss: {privacy_loss:.4f}")
    logger.info(f"  Cost per Query: ${avg_cost:.6f}")
    logger.info(f"  Avg Latency: {avg_latency_ms:.1f}ms")
    logger.info(f"  Routing: Local={pr_local:.1%}, Remote={pr_remote:.1%}")

    return metrics


def generate_comprehensive_report(
    output_dir: Path,
    ex_results: dict,
    ves_results: dict,
    three_axis_metrics: dict,
    num_queries: int,
    experiment_name: str
) -> None:
    """Generate comprehensive evaluation report.

    Args:
        output_dir: Output directory
        ex_results: EX evaluation results
        ves_results: VES evaluation results
        three_axis_metrics: Three-axis metrics
        num_queries: Number of queries evaluated
        experiment_name: Experiment name
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Generate Comprehensive Report")
    logger.info("=" * 80)

    # Compile full report
    report = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'num_queries': num_queries,
        'evaluation_metrics': {
            'execution_accuracy': ex_results.get('execution_accuracy'),
            'ves_score': ves_results.get('ves_score'),
            'privacy_loss': three_axis_metrics['privacy_loss'],
            'cost_per_query_usd': three_axis_metrics['cost_per_query_usd'],
            'avg_latency_ms': three_axis_metrics['avg_latency_ms'],
        },
        'routing_statistics': {
            'pr_local': three_axis_metrics['pr_local'],
            'pr_remote': three_axis_metrics['pr_remote'],
            'num_local': three_axis_metrics['num_local'],
            'num_remote': three_axis_metrics['num_remote'],
        },
        'verification_statistics': three_axis_metrics['verification_stats'],
    }

    # Save report
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    logger.info(f"✓ Report saved: {report_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Queries: {num_queries}")
    logger.info("\n--- Accuracy Metrics ---")
    logger.info(f"Execution Accuracy (EX): {ex_results.get('execution_accuracy', 'N/A')}")
    logger.info(f"Valid Efficiency Score (VES): {ves_results.get('ves_score', 'N/A')}")
    logger.info("\n--- Three-Axis Metrics ---")
    logger.info(f"Privacy Loss: {three_axis_metrics['privacy_loss']:.4f}")
    logger.info(f"Cost per Query: ${three_axis_metrics['cost_per_query_usd']:.6f}")
    logger.info(f"Avg Latency: {three_axis_metrics['avg_latency_ms']:.1f}ms")
    logger.info("\n--- Routing Statistics ---")
    logger.info(f"Local: {three_axis_metrics['num_local']} ({three_axis_metrics['pr_local']:.1%})")
    logger.info(f"Remote: {three_axis_metrics['num_remote']} ({three_axis_metrics['pr_remote']:.1%})")
    logger.info("\n--- Verification Statistics ---")
    for status, count in three_axis_metrics['verification_stats'].items():
        logger.info(f"{status.capitalize()}: {count}")
    logger.info("\n" + "=" * 80)
    logger.info(f"Full report: {report_file}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run full BIRD-dev evaluation pipeline"
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
        help="Number of queries to evaluate (default: all 1534)",
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
        "--predictions_file",
        type=str,
        default=None,
        help="Use existing predictions file (skips step 1)",
    )

    args = parser.parse_args()

    # Create output name
    if args.output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_name = f"bird_eval_{timestamp}"

    output_dir = Path("evaluation/output") / args.output_name

    # Setup logging
    log_file = output_dir / "full_evaluation.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, format="{time} {level} {message}", level="DEBUG")

    logger.info("=" * 80)
    logger.info("AEGIS-SQL BIRD-dev Full Evaluation Pipeline")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.output_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Num queries: {args.num_queries or 'all (1534)'}")
    logger.info(f"Seed: {args.seed}")

    try:
        # Load config
        config = AEGISConfig.from_yaml(args.config)

        # Step 1: Generate predictions (or use existing)
        if args.predictions_file:
            predictions_file = Path(args.predictions_file)
            logger.info(f"\nUsing existing predictions: {predictions_file}")
            if not predictions_file.exists():
                raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        else:
            predictions_file = run_prediction_generation(
                num_queries=args.num_queries,
                seed=args.seed,
                stratify=args.stratify,
                output_name=args.output_name,
                config_path=args.config,
                bird_path=args.bird_path
            )

        # Determine number of queries from predictions file
        with open(predictions_file, 'r') as f:
            num_queries = sum(1 for _ in f)

        # Step 2: EX evaluation (execute SQL and compare results)
        ex_results = run_ex_evaluation(predictions_file, args.bird_path)

        # Step 3: VES evaluation (measure query efficiency)
        ves_results = run_ves_evaluation(predictions_file, args.bird_path)

        # Step 4: Three-axis metrics
        three_axis_metrics = compute_three_axis_metrics(predictions_file, config)

        # Step 5: Generate report
        generate_comprehensive_report(
            output_dir=output_dir,
            ex_results=ex_results,
            ves_results=ves_results,
            three_axis_metrics=three_axis_metrics,
            num_queries=num_queries,
            experiment_name=args.output_name
        )

        logger.info("\n✓ Full evaluation pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"\n✗ Evaluation pipeline failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Evaluate fine-tuned model on BIRD dev set.

Tests the fine-tuned model's performance on the BIRD development set
using the AEGIS-SQL evaluation pipeline.

Usage:
    python evaluate_model.py --adapter_path ../checkpoints/phi4 --output_name phi4_eval
    python evaluate_model.py --adapter_path Daveonyango254/aegis-sql-phi4-lora --output_name phi4_eval_from_hub
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from loguru import logger

# Add parent directory to path to import AEGIS modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.bird_loader import BIRDLoader
from evaluation.evaluator_ex import ExecutionAccuracyEvaluator
from generator.slm_generator import SLMGenerator
from config import SLMConfig, AEGISConfig


def evaluate_finetuned_model(
    adapter_path: str,
    base_model: str,
    output_dir: str,
    num_queries: int = None,
    seed: int = 42
) -> Dict[str, float]:
    """Evaluate a fine-tuned model on BIRD dev set.

    Args:
        adapter_path: Path to LoRA adapter (local or HuggingFace Hub)
        base_model: Base model name (e.g., "microsoft/Phi-4-mini-instruct")
        output_dir: Directory to save evaluation results
        num_queries: Number of queries to evaluate (None = all)
        seed: Random seed for sampling

    Returns:
        Dictionary of evaluation metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AEGIS-SQL Fine-tuned Model Evaluation")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Adapter: {adapter_path}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    # Load AEGIS config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        aegis_config = AEGISConfig.from_yaml(config_path)
    else:
        logger.warning("config.yaml not found, using defaults")
        aegis_config = AEGISConfig()

    # Override SLM config with fine-tuned model
    aegis_config.slm.model = base_model
    aegis_config.slm.adapter_path = adapter_path
    aegis_config.slm.device = "auto"

    # Load BIRD dev set
    logger.info("Loading BIRD dev dataset...")
    bird_loader = BIRDLoader(
        bird_dev_path=str(Path(__file__).parent.parent.parent / "data" / "bird"),
        seed=seed
    )

    if num_queries:
        queries = bird_loader.sample(num_queries)
        logger.info(f"Sampled {num_queries} queries for evaluation")
    else:
        queries = bird_loader.load_all()
        logger.info(f"Evaluating on full dev set ({len(queries)} queries)")

    # Load fine-tuned SLM
    logger.info("Loading fine-tuned model...")
    slm = SLMGenerator(aegis_config.slm)

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []

    for i, query_data in enumerate(queries):
        if (i + 1) % 50 == 0:
            logger.info(f"Generated {i + 1}/{len(queries)} predictions...")

        try:
            # Create query object
            from aegis_types import Query, Language

            query = Query(
                text=query_data['question'],
                language=Language.ENGLISH,
                database_id=query_data['db_id'],
                evidence=query_data.get('evidence', '')
            )

            # For evaluation, we need schema retrieval
            # This is simplified - in practice you'd use the full workflow
            # For now, we'll just pass empty schema (model should handle it)
            schema_elements = []

            # Generate SQL
            sql = slm.generate(query, schema_elements)

            predictions.append({
                'question_id': query_data.get('question_id', i),
                'db_id': query_data['db_id'],
                'question': query_data['question'],
                'evidence': query_data.get('evidence', ''),
                'gold_sql': query_data['SQL'],
                'predicted_sql': sql.text,
                'difficulty': query_data.get('difficulty', 'unknown')
            })

        except Exception as e:
            logger.error(f"Failed to generate SQL for query {i}: {e}")
            predictions.append({
                'question_id': query_data.get('question_id', i),
                'db_id': query_data['db_id'],
                'question': query_data['question'],
                'gold_sql': query_data['SQL'],
                'predicted_sql': 'SELECT 1;',  # Fallback
                'error': str(e)
            })

    # Save predictions
    predictions_file = output_path / "predictions.json"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.success(f"Saved predictions to: {predictions_file}")

    # Evaluate execution accuracy
    logger.info("Evaluating execution accuracy...")
    evaluator = ExecutionAccuracyEvaluator(
        bird_dev_path=str(Path(__file__).parent.parent.parent / "data" / "bird")
    )

    results = evaluator.evaluate_predictions(predictions)

    # Save metrics
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.success(f"Saved metrics to: {metrics_file}")

    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    logger.info(f"  Execution Accuracy (EX): {results['ex_accuracy']*100:.2f}%")
    logger.info(f"  Correct: {results['correct']}/{results['total']}")

    if 'by_difficulty' in results:
        logger.info("  By Difficulty:")
        for diff, stats in results['by_difficulty'].items():
            logger.info(f"    {diff}: {stats['correct']}/{stats['total']} ({stats['accuracy']*100:.1f}%)")

    logger.info("=" * 60)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on BIRD dev set"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter (local path or HuggingFace Hub)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Base model name (default: microsoft/Phi-4-mini-instruct)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output",
        help="Output directory for results (default: ../output)"
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Number of queries to evaluate (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Resolve output directory
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()

    # Create unique output directory based on model name
    adapter_name = Path(args.adapter_path).name if '/' not in args.adapter_path else args.adapter_path.split('/')[-1]
    output_dir = output_dir / adapter_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    results = evaluate_finetuned_model(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        output_dir=str(output_dir),
        num_queries=args.num_queries,
        seed=args.seed
    )

    logger.success("Evaluation complete!")


if __name__ == "__main__":
    main()

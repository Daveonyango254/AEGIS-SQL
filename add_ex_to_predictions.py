"""Add EX boolean field to predictions.jsonl.

This script reads predictions from predictions.jsonl, computes the EX (Execution Accuracy)
boolean field for each query by executing both predicted and ground truth SQL, and writes
results to final_results.jsonl.

Usage:
    python add_ex_to_predictions.py \
        --predictions evaluation/output/eval_output_1/full_bird_dev_1/predictions.jsonl \
        --bird_path data/bird \
        --output evaluation/output/eval_output_1/full_bird_dev_1/final_results.jsonl
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, List, Tuple

from loguru import logger


def normalize_sql(sql: str) -> str:
    """Normalize SQL by removing newlines and extra whitespace.

    Args:
        sql: SQL string that may contain newlines

    Returns:
        Normalized SQL with newlines replaced by spaces
    """
    sql = sql.replace('\n', ' ')
    sql = ' '.join(sql.split())
    return sql.strip()


def execute_sql(sql: str, db_path: str) -> Tuple[bool, List[Any]]:
    """Execute SQL query and return results.

    Args:
        sql: SQL query to execute
        db_path: Path to SQLite database

    Returns:
        Tuple of (success, results) where results is list of tuples
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        logger.debug(f"SQL execution failed: {e}")
        return False, []


def calculate_ex(predicted_sql: str, ground_truth_sql: str, db_path: str) -> bool:
    """Calculate EX (Execution Accuracy) by comparing query results.

    Args:
        predicted_sql: Predicted SQL query
        ground_truth_sql: Ground truth SQL query
        db_path: Path to database file

    Returns:
        True if results match, False otherwise
    """
    # Normalize SQL to handle newlines
    predicted_sql_clean = normalize_sql(predicted_sql)
    ground_truth_sql_clean = normalize_sql(ground_truth_sql)

    # Execute both queries
    pred_success, pred_results = execute_sql(predicted_sql_clean, db_path)
    gt_success, gt_results = execute_sql(ground_truth_sql_clean, db_path)

    # Both must execute successfully
    if not pred_success or not gt_success:
        return False

    # Compare results using set equality (same as official evaluator)
    return set(pred_results) == set(gt_results)


def main():
    parser = argparse.ArgumentParser(
        description="Add EX boolean field to predictions.jsonl"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.jsonl file"
    )
    parser.add_argument(
        "--bird_path",
        type=str,
        required=True,
        help="Path to BIRD data directory (contains dev_databases/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for final_results.jsonl (default: same dir as predictions)"
    )

    args = parser.parse_args()

    # Setup paths
    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = predictions_path.parent / "final_results.jsonl"

    bird_path = Path(args.bird_path)
    databases_path = bird_path / "dev_databases"

    if not databases_path.exists():
        raise FileNotFoundError(f"BIRD databases not found: {databases_path}")

    logger.info("=" * 80)
    logger.info("Adding EX Boolean Field to Predictions")
    logger.info("=" * 80)
    logger.info(f"Predictions: {predictions_path}")
    logger.info(f"BIRD databases: {databases_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 80)

    # Read predictions
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))

    logger.info(f"\nLoaded {len(predictions)} predictions")
    logger.info("Computing EX for each prediction...\n")

    # Process each prediction
    results = []
    ex_true_count = 0

    for i, pred in enumerate(predictions):
        db_id = pred['db_id']
        predicted_sql = pred.get('predicted_sql', '')
        ground_truth_sql = pred.get('ground_truth_sql', '')

        # Find database path
        db_path = databases_path / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            logger.warning(f"Database not found for {db_id}, marking EX=False")
            ex_value = False
        else:
            # Calculate EX
            ex_value = calculate_ex(predicted_sql, ground_truth_sql, str(db_path))

        # Add EX field to prediction
        pred_with_ex = pred.copy()
        pred_with_ex['EX'] = ex_value
        results.append(pred_with_ex)

        if ex_value:
            ex_true_count += 1

        # Progress logging
        if (i + 1) % 100 == 0 or (i + 1) == len(predictions):
            logger.info(
                f"Progress: {i + 1}/{len(predictions)} "
                f"({(i + 1) / len(predictions) * 100:.1f}%) | "
                f"EX=True: {ex_true_count}/{i + 1} "
                f"({ex_true_count / (i + 1) * 100:.2f}%)"
            )

    # Write results
    logger.info(f"\nWriting results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Final statistics
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total predictions: {len(results)}")
    logger.info(f"EX=True: {ex_true_count} ({ex_true_count / len(results) * 100:.2f}%)")
    logger.info(f"EX=False: {len(results) - ex_true_count} ({(len(results) - ex_true_count) / len(results) * 100:.2f}%)")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

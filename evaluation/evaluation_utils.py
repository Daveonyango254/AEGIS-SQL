"""Evaluation utility functions for BIRD benchmark.

Common utilities for EX and VES evaluators.
"""

import json
import sqlite3
from typing import List, Tuple, Any, Dict
from pathlib import Path


def load_jsonl(file_path: str) -> List[Dict]:
    """Load predictions from JSONL file or JSON array file.

    Args:
        file_path: Path to JSONL file or JSON array file

    Returns:
        List of dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to load as JSON array first
        try:
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Fall back to JSONL format (newline-delimited)
        f.seek(0)
        data = []
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data


def connect_db(sql_dialect: str, db_path: str):
    """Connect to database based on SQL dialect.

    Args:
        sql_dialect: SQL dialect (currently only "SQLite" is supported)
        db_path: Path to database file

    Returns:
        Database connection object
    """
    if sql_dialect.lower() == "sqlite":
        return sqlite3.connect(db_path)
    else:
        raise ValueError(f"Unsupported SQL dialect: {sql_dialect}")


def execute_sql(predicted_sql: str, ground_truth: str, db_path: str,
                sql_dialect: str = "SQLite", eval_fn=None):
    """Execute SQL queries and compare results (for EX evaluator).

    Args:
        predicted_sql: Predicted SQL query
        ground_truth: Ground truth SQL query
        db_path: Path to database
        sql_dialect: SQL dialect
        eval_fn: Evaluation function to compare results (e.g., calculate_ex)

    Returns:
        Result of eval_fn comparing predicted and ground truth results
    """
    conn = connect_db(sql_dialect, db_path)
    cursor = conn.cursor()

    # Execute predicted SQL
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()

    # Execute ground truth SQL
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()

    conn.close()

    # Apply evaluation function if provided
    if eval_fn:
        return eval_fn(predicted_res, ground_truth_res)

    return predicted_res, ground_truth_res


def package_sqls(
    sql_path: str,
    db_root_path: str,
    mode: str = 'pred',
    ground_truth_path: str = None,
    question_ids: List[int] = None
) -> Tuple[List[str], List[str]]:
    """Package SQL queries with their database paths.

    Args:
        sql_path: Path to SQL file or JSONL predictions
        db_root_path: Root directory containing database subdirectories
        mode: 'pred' for predictions, 'gt' for ground truth
        ground_truth_path: Path to ground truth file (for pred mode)
        question_ids: List of question IDs to load in specific order (for gt mode)

    Returns:
        Tuple of (sql_queries, db_paths)
        - sql_queries: List of SQL query strings
        - db_paths: List of corresponding database file paths
    """
    sql_path = Path(sql_path)
    sqls = []
    db_paths = []

    if mode == 'pred':
        # Load predictions from JSONL
        predictions = load_jsonl(str(sql_path))
        for pred in predictions:
            sql_query = pred.get('predicted_sql', '')
            db_id = pred.get('db_id', '')
            db_path = str(Path(db_root_path) / db_id / f"{db_id}.sqlite")
            sqls.append(sql_query)
            db_paths.append(db_path)

    elif mode == 'gt':
        # Load ground truth from dev.json
        gt_data = load_jsonl(str(sql_path))

        # Build question_id -> data mapping
        gt_map = {item.get('question_id'): item for item in gt_data}

        # If question_ids specified, load in that order
        if question_ids:
            for qid in question_ids:
                if qid in gt_map:
                    item = gt_map[qid]
                    sql_query = item.get('SQL', '')
                    db_id = item.get('db_id', '')
                    db_path = str(Path(db_root_path) / db_id / f"{db_id}.sqlite")
                    sqls.append(sql_query)
                    db_paths.append(db_path)
        else:
            # Load all ground truth in file order
            for item in gt_data:
                sql_query = item.get('SQL', '')
                db_id = item.get('db_id', '')
                db_path = str(Path(db_root_path) / db_id / f"{db_id}.sqlite")
                sqls.append(sql_query)
                db_paths.append(db_path)

    return sqls, db_paths


def sort_results(results: List[Tuple]) -> List[Tuple]:
    """Sort query results for comparison.

    Args:
        results: List of result tuples

    Returns:
        Sorted list of tuples
    """
    if not results:
        return results

    try:
        # Try to sort naturally
        return sorted(results)
    except TypeError:
        # If sorting fails (e.g., mixed types), convert to strings
        return sorted(results, key=lambda x: str(x))


def print_data(score_lists: List[float] = None, count_lists: List[int] = None,
               metric: str = "EX", result_log_file: str = None, data: List[Any] = None, top_k: int = None) -> None:
    """Print evaluation results and optionally save to file.

    Args:
        score_lists: List of accuracy scores [simple, moderate, challenging, overall]
        count_lists: List of query counts [simple, moderate, challenging, total]
        metric: Metric name (EX or VES)
        result_log_file: Path to save results
        data: Generic data to print (for backward compatibility)
        top_k: Number of items to print (None for all)
    """
    # Handle generic data printing (backward compatibility)
    if data is not None:
        if top_k:
            data = data[:top_k]
        for item in data:
            print(item)
        return

    # Handle evaluation results printing
    if score_lists and count_lists:
        simple_acc, moderate_acc, challenging_acc, overall_acc = score_lists
        simple_count, moderate_count, challenging_count, total_count = count_lists

        output = []
        output.append(f"\n{'=' * 80}")
        output.append(f"{metric} Evaluation Results")
        output.append(f"{'=' * 80}")
        output.append(f"Simple:      {simple_acc:.2f}% ({simple_count} queries)")
        output.append(f"Moderate:    {moderate_acc:.2f}% ({moderate_count} queries)")
        output.append(f"Challenging: {challenging_acc:.2f}% ({challenging_count} queries)")
        output.append(f"Overall:     {overall_acc:.2f}% ({total_count} queries)")
        output.append(f"{'=' * 80}")

        # Print to console
        for line in output:
            print(line)

        # Save to file if specified
        if result_log_file:
            Path(result_log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(result_log_file, 'w') as f:
                f.write('\n'.join(output))
            print(f"\nResults saved to: {result_log_file}")


def normalize_sql(sql: str) -> str:
    """Normalize SQL query for comparison.

    Args:
        sql: SQL query string

    Returns:
        Normalized SQL string
    """
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    # Convert to lowercase
    sql = sql.lower()
    # Remove trailing semicolon
    sql = sql.rstrip(';')
    return sql


def compare_results(pred_results: List[Tuple], gold_results: List[Tuple]) -> bool:
    """Compare predicted and gold results.

    Args:
        pred_results: Predicted query results
        gold_results: Ground truth query results

    Returns:
        True if results match, False otherwise
    """
    # Sort both result sets
    pred_sorted = sort_results(pred_results)
    gold_sorted = sort_results(gold_results)

    # Compare as sets (order-independent)
    return set(pred_sorted) == set(gold_sorted)


def load_predictions(predictions_file: str) -> Dict[int, Dict]:
    """Load predictions from JSONL file into dictionary keyed by question_id.

    Args:
        predictions_file: Path to predictions JSONL file

    Returns:
        Dictionary mapping question_id to prediction dictionary
    """
    predictions = {}
    data = load_jsonl(predictions_file)
    for pred in data:
        qid = pred.get('question_id')
        if qid is not None:
            predictions[qid] = pred
    return predictions

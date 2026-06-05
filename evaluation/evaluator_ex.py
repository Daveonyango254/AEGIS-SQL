"""
EX (Execution Accuracy) Evaluator
Measures whether predicted SQL returns the same results as ground truth
Consolidated implementation + wrapper in one file
"""

import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from func_timeout import func_timeout, FunctionTimedOut
import time

from evaluation.evaluation_utils import (
    load_jsonl,
    execute_sql,
    package_sqls,
    sort_results,
    print_data,
)
from evaluation.config import OUTPUT_DIR, DEFAULT_QUERY_TIMEOUT
from evaluation.latency_tracker import LatencyTracker

# Global result collector
exec_result = []


def result_callback(result):
    """Callback for multiprocessing results"""
    exec_result.append(result)


def calculate_ex(predicted_res, ground_truth_res):
    """
    Calculate EX metric: 1 if results match, 0 otherwise

    Args:
        predicted_res: Results from predicted SQL
        ground_truth_res: Results from ground truth SQL

    Returns:
        1 if results match, 0 otherwise
    """
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(
    predicted_sql, ground_truth, db_place, idx, meta_time_out, sql_dialect
):
    """
    Execute a single SQL query pair with timeout and latency tracking

    Args:
        predicted_sql: Predicted SQL query
        ground_truth: Ground truth SQL query
        db_place: Database path
        idx: Query index
        meta_time_out: Timeout in seconds
        sql_dialect: SQL dialect

    Returns:
        Dictionary with sql_idx, result, latency_ms, is_timeout, and success
    """
    start_time = time.time()
    is_timeout = False
    success = True
    error_msg = None

    try:
        res = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth, db_place, sql_dialect, calculate_ex),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f"timeout",)]
        res = 0
        is_timeout = True
        success = False
        error_msg = f"Query timeout after {meta_time_out}s"
    except Exception as e:
        result = [(f"error",)]
        res = 0
        success = False
        error_msg = str(e)

    latency_ms = (time.time() - start_time) * 1000

    result = {
        "sql_idx": idx,
        "res": res,
        "latency_ms": latency_ms,
        "is_timeout": is_timeout,
        "success": success,
        "error_msg": error_msg
    }
    return result


def run_sqls_parallel(
    sqls, db_places, num_cpus=1, meta_time_out=30.0, sql_dialect="SQLite"
):
    """
    Execute SQL query pairs in parallel

    Args:
        sqls: List of (predicted_sql, ground_truth) tuples
        db_places: List of database paths
        num_cpus: Number of CPU cores
        meta_time_out: Timeout per query
        sql_dialect: SQL dialect
    """
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_places[i],
                i,
                meta_time_out,
                sql_dialect,
            ),
            callback=result_callback,
        )
    pool.close()
    pool.join()


def compute_acc_by_diff(exec_results, diff_json_path, question_ids=None):
    """
    Compute accuracy broken down by difficulty level

    Args:
        exec_results: List of execution results
        diff_json_path: Path to difficulty labels file
        question_ids: Optional question_id list aligned with exec_results order

    Returns:
        Tuple of (simple_acc, moderate_acc, challenging_acc, overall_acc, count_lists)
    """
    num_queries = len(exec_results)
    results = [res["res"] for res in exec_results]
    contents = load_jsonl(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    qid_to_diff = {item.get("question_id"): item.get("difficulty") for item in contents}

    # Iterate in execution order and use question_id mapping when available.
    for i in range(len(exec_results)):
        content = None
        if question_ids and i < len(question_ids):
            difficulty = qid_to_diff.get(question_ids[i])
            if difficulty is not None:
                content = {"difficulty": difficulty}
        if content is None and i < len(contents):
            content = contents[i]
        if content is None:
            continue

        if content["difficulty"] == "simple":
            simple_results.append(exec_results[i])

        if content["difficulty"] == "moderate":
            moderate_results.append(exec_results[i])

        if content["difficulty"] == "challenging":
            challenging_results.append(exec_results[i])

    # Compute accuracy with protection against division by zero
    simple_acc = sum([res["res"] for res in simple_results]) / len(simple_results) if simple_results else 0
    moderate_acc = sum([res["res"] for res in moderate_results]) / len(moderate_results) if moderate_results else 0
    challenging_acc = sum([res["res"] for res in challenging_results]) / len(challenging_results) if challenging_results else 0
    all_acc = sum(results) / num_queries if num_queries > 0 else 0
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    return (
        simple_acc * 100,
        moderate_acc * 100,
        challenging_acc * 100,
        all_acc * 100,
        count_lists,
    )


def run_ex_evaluation(
    predicted_sql_path: str,
    ground_truth_path: str,
    db_root_path: str,
    diff_json_path: str,
    num_cpus: int = 1,
    meta_time_out: float = 30.0,
    sql_dialect: str = "SQLite",
    output_log_path: str = None
) -> dict:
    """
    Run EX (Execution Accuracy) evaluation - HIGH-LEVEL INTERFACE

    Args:
        predicted_sql_path: Path to predicted SQL file
        ground_truth_path: Path to ground truth file
        db_root_path: Root directory for databases
        diff_json_path: Path to difficulty labels
        num_cpus: Number of CPU cores
        meta_time_out: Timeout per query
        sql_dialect: SQL dialect
        output_log_path: Path to save results in output directory

    Returns:
        Dictionary with evaluation results including latency statistics
    """
    # Ensure output directory exists
    if output_log_path is None:
        output_log_path = OUTPUT_DIR / "ex_results.txt"
    else:
        output_log_path = OUTPUT_DIR / Path(output_log_path).name

    # Clear previous results
    exec_result.clear()

    # Initialize latency tracker
    latency_tracker = LatencyTracker()

    # Load predicted and ground truth SQLs
    # First load predictions to get the question_ids in sorted order
    from evaluation.evaluation_utils import load_predictions
    predictions = load_predictions(predicted_sql_path)
    question_ids_in_order = sorted(predictions.keys())

    # Load prediction queries
    pred_queries, db_paths = package_sqls(
        predicted_sql_path,
        db_root_path,
        mode='pred',
        ground_truth_path=ground_truth_path
    )

    # Load ground truth queries IN THE SAME ORDER as predictions
    # This ensures correct pairing: pred_queries[i] matches gt_queries[i]
    gt_queries, db_paths_gt = package_sqls(
        ground_truth_path,
        db_root_path,
        mode="gt",
        question_ids=question_ids_in_order
    )

    # Pair queries - now they are properly aligned by question_id
    query_pairs = list(zip(pred_queries, gt_queries))

    # Run parallel execution
    run_sqls_parallel(
        query_pairs,
        db_places=db_paths_gt,
        num_cpus=num_cpus,
        meta_time_out=meta_time_out,
        sql_dialect=sql_dialect,
    )

    # Sort and compute metrics
    sorted_results = sort_results(exec_result)

    # Record latencies from results
    for result in sorted_results:
        latency_tracker.record(
            query_id=result.get("sql_idx", 0),
            latency_ms=result.get("latency_ms", 0.0),
            is_timeout=result.get("is_timeout", False),
            success=result.get("success", True),
            error_msg=result.get("error_msg", None)
        )

    simple_acc, moderate_acc, challenging_acc, overall_acc, count_lists = compute_acc_by_diff(
        sorted_results, diff_json_path, question_ids_in_order
    )

    score_lists = [simple_acc, moderate_acc, challenging_acc, overall_acc]

    # Print and save results to output directory
    print_data(
        score_lists,
        count_lists,
        metric="EX",
        result_log_file=str(output_log_path)
    )

    # Save and display latency statistics
    latency_file = output_log_path.parent / f"{output_log_path.stem}_latency.json"
    latency_tracker.save_to_file(latency_file)
    print(f"\n[EX] Latency stats saved to: {latency_file}")

    # Print latency summary
    latency_tracker.print_summary()

    # Return structured results
    return {
        "overall_acc": overall_acc,
        "simple_acc": simple_acc,
        "moderate_acc": moderate_acc,
        "challenging_acc": challenging_acc,
        "counts": {
            "simple": count_lists[0],
            "moderate": count_lists[1],
            "challenging": count_lists[2],
            "total": count_lists[3]
        },
        "output_file": str(output_log_path),
        "latency_stats": latency_tracker.get_summary(),
        "per_question_results": sorted_results  # Required for database metrics grouping
    }


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--predicted_sql_path", type=str, required=True, default=""
    )
    args_parser.add_argument("--ground_truth_path", type=str, required=True, default="")
    args_parser.add_argument("--db_root_path", type=str, required=True, default="")
    args_parser.add_argument("--num_cpus", type=int, default=1)
    args_parser.add_argument("--meta_time_out", type=float, default=DEFAULT_QUERY_TIMEOUT)
    args_parser.add_argument("--diff_json_path", type=str, default="")
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args_parser.add_argument("--output_log_path", type=str, default="SQLite")
    args = args_parser.parse_args()

    # Run evaluation
    results = run_ex_evaluation(
        predicted_sql_path=args.predicted_sql_path,
        ground_truth_path=args.ground_truth_path,
        db_root_path=args.db_root_path,
        diff_json_path=args.diff_json_path,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
        sql_dialect=args.sql_dialect,
        output_log_path=args.output_log_path
    )

    print("=" * 80)
    print(f"Finished EX evaluation for {args.sql_dialect}")
    print("=" * 80)

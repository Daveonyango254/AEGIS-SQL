"""Query sampling utilities for reproducible evaluation.

Supports random sampling and stratified sampling by difficulty level.
"""

import random
from typing import List, Dict, Any, Optional

from loguru import logger


def sample_queries(
    queries: List[Dict[str, Any]],
    num_queries: int,
    seed: int = 42,
    stratify: bool = True
) -> List[Dict[str, Any]]:
    """Sample queries with reproducibility.

    Args:
        queries: List of query dictionaries
        num_queries: Number of queries to sample
        seed: Random seed for reproducibility
        stratify: If True, preserve difficulty distribution

    Returns:
        List of sampled queries, sorted by question_id

    Example:
        >>> queries = load_bird_queries()  # 1534 queries
        >>> sampled = sample_queries(queries, num_queries=100, seed=42, stratify=True)
        >>> len(sampled)
        100
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Validate num_queries
    if num_queries > len(queries):
        logger.warning(
            f"Requested {num_queries} queries but only {len(queries)} available. "
            f"Using all queries."
        )
        return sorted(queries, key=lambda x: x['question_id'])

    if stratify:
        logger.info(f"Stratified sampling {num_queries} queries with seed={seed}")
        sampled = _sample_stratified(queries, num_queries)
    else:
        logger.info(f"Random sampling {num_queries} queries with seed={seed}")
        sampled = random.sample(queries, num_queries)

    # Sort by question_id for consistent ordering
    sampled_sorted = sorted(sampled, key=lambda x: x['question_id'])

    # Log sampling statistics
    _log_sampling_stats(queries, sampled_sorted)

    return sampled_sorted


def _sample_stratified(
    queries: List[Dict[str, Any]],
    num_queries: int
) -> List[Dict[str, Any]]:
    """Stratified sampling by difficulty level.

    Preserves the difficulty distribution of the original dataset.

    Args:
        queries: List of query dictionaries
        num_queries: Number of queries to sample

    Returns:
        List of sampled queries (not sorted)
    """
    # Split by difficulty
    simple = [q for q in queries if q.get('difficulty') == 'simple']
    moderate = [q for q in queries if q.get('difficulty') == 'moderate']
    challenging = [q for q in queries if q.get('difficulty') == 'challenging']

    total = len(queries)

    # Proportional sampling
    n_simple = int(num_queries * len(simple) / total)
    n_moderate = int(num_queries * len(moderate) / total)
    n_challenging = num_queries - n_simple - n_moderate  # Remaining goes to challenging

    # Ensure we don't sample more than available
    n_simple = min(n_simple, len(simple))
    n_moderate = min(n_moderate, len(moderate))
    n_challenging = min(n_challenging, len(challenging))

    # Adjust if we sampled less than requested
    actual_sampled = n_simple + n_moderate + n_challenging
    if actual_sampled < num_queries:
        shortage = num_queries - actual_sampled
        # Add shortage to the largest category
        if len(simple) - n_simple >= shortage:
            n_simple += shortage
        elif len(moderate) - n_moderate >= shortage:
            n_moderate += shortage
        elif len(challenging) - n_challenging >= shortage:
            n_challenging += shortage

    # Sample from each difficulty level
    sampled = []
    if n_simple > 0:
        sampled.extend(random.sample(simple, n_simple))
    if n_moderate > 0:
        sampled.extend(random.sample(moderate, n_moderate))
    if n_challenging > 0:
        sampled.extend(random.sample(challenging, n_challenging))

    logger.debug(
        f"Stratified sampling: simple={n_simple}, moderate={n_moderate}, "
        f"challenging={n_challenging}"
    )

    return sampled


def _log_sampling_stats(
    original: List[Dict[str, Any]],
    sampled: List[Dict[str, Any]]
) -> None:
    """Log sampling statistics for verification.

    Args:
        original: Original query list
        sampled: Sampled query list
    """
    # Count by difficulty
    def count_difficulty(queries):
        counts = {'simple': 0, 'moderate': 0, 'challenging': 0}
        for q in queries:
            diff = q.get('difficulty', 'unknown')
            if diff in counts:
                counts[diff] += 1
        return counts

    orig_counts = count_difficulty(original)
    samp_counts = count_difficulty(sampled)

    logger.info(
        f"Original distribution: simple={orig_counts['simple']}, "
        f"moderate={orig_counts['moderate']}, challenging={orig_counts['challenging']}"
    )
    logger.info(
        f"Sampled distribution: simple={samp_counts['simple']}, "
        f"moderate={samp_counts['moderate']}, challenging={samp_counts['challenging']}"
    )

    # Calculate percentages
    total_orig = sum(orig_counts.values())
    total_samp = sum(samp_counts.values())

    if total_orig > 0:
        orig_pct = {k: 100 * v / total_orig for k, v in orig_counts.items()}
        samp_pct = {k: 100 * v / total_samp for k, v in samp_counts.items()}

        logger.info(
            f"Original percentages: simple={orig_pct['simple']:.1f}%, "
            f"moderate={orig_pct['moderate']:.1f}%, "
            f"challenging={orig_pct['challenging']:.1f}%"
        )
        logger.info(
            f"Sampled percentages: simple={samp_pct['simple']:.1f}%, "
            f"moderate={samp_pct['moderate']:.1f}%, "
            f"challenging={samp_pct['challenging']:.1f}%"
        )


def get_specific_queries(
    queries: List[Dict[str, Any]],
    question_ids: List[int]
) -> List[Dict[str, Any]]:
    """Get specific queries by question_id.

    Args:
        queries: List of query dictionaries
        question_ids: List of question IDs to retrieve

    Returns:
        List of queries with matching question_ids, sorted by question_id

    Example:
        >>> queries = load_bird_queries()
        >>> specific = get_specific_queries(queries, [0, 1, 2, 5, 10])
        >>> len(specific)
        5
    """
    id_set = set(question_ids)
    selected = [q for q in queries if q['question_id'] in id_set]

    if len(selected) < len(question_ids):
        missing = id_set - {q['question_id'] for q in selected}
        logger.warning(f"Missing question IDs: {sorted(missing)}")

    return sorted(selected, key=lambda x: x['question_id'])

"""Latency tracking for evaluation.

Tracks query execution times and computes statistics.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LatencyTracker:
    """Track query execution latencies.

    Attributes:
        latencies: Dictionary mapping query_id to latency in milliseconds
        start_times: Dictionary mapping query_id to start timestamp
    """

    latencies: Dict[int, float] = field(default_factory=dict)
    start_times: Dict[int, float] = field(default_factory=dict)

    def start(self, query_id: int) -> None:
        """Start timing for a query.

        Args:
            query_id: Query identifier
        """
        self.start_times[query_id] = time.time()

    def stop(self, query_id: int) -> float:
        """Stop timing for a query and record latency.

        Args:
            query_id: Query identifier

        Returns:
            Latency in milliseconds
        """
        if query_id not in self.start_times:
            return 0.0

        elapsed = (time.time() - self.start_times[query_id]) * 1000  # Convert to ms
        self.latencies[query_id] = elapsed
        del self.start_times[query_id]
        return elapsed

    def get_latency(self, query_id: int) -> Optional[float]:
        """Get recorded latency for a query.

        Args:
            query_id: Query identifier

        Returns:
            Latency in milliseconds, or None if not found
        """
        return self.latencies.get(query_id)

    def get_all_latencies(self) -> List[float]:
        """Get all recorded latencies.

        Returns:
            List of latencies in milliseconds
        """
        return list(self.latencies.values())

    def get_statistics(self) -> Dict[str, float]:
        """Compute latency statistics.

        Returns:
            Dictionary with min, max, mean, median, p95, p99 latencies
        """
        if not self.latencies:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        latency_list = sorted(self.get_all_latencies())
        n = len(latency_list)

        return {
            "min": latency_list[0],
            "max": latency_list[-1],
            "mean": sum(latency_list) / n,
            "median": latency_list[n // 2],
            "p95": latency_list[int(n * 0.95)] if n > 1 else latency_list[0],
            "p99": latency_list[int(n * 0.99)] if n > 1 else latency_list[0],
        }

    def reset(self) -> None:
        """Reset all tracked latencies."""
        self.latencies.clear()
        self.start_times.clear()

    def __len__(self) -> int:
        """Return number of tracked queries."""
        return len(self.latencies)

    def record(self, query_id: int, latency_ms: float, is_timeout: bool = False,
               success: bool = True, error_msg: Optional[str] = None) -> None:
        """Record a completed query with its latency and status.

        Args:
            query_id: Query identifier
            latency_ms: Latency in milliseconds
            is_timeout: Whether query timed out
            success: Whether query succeeded
            error_msg: Error message if failed
        """
        self.latencies[query_id] = latency_ms

    def save_to_file(self, filepath: str) -> None:
        """Save latency data to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        import json
        from pathlib import Path

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "latencies": self.latencies,
            "statistics": self.get_statistics(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        """Print latency statistics summary."""
        stats = self.get_statistics()
        print("\nLatency Statistics:")
        print(f"  Min:    {stats['min']:.2f} ms")
        print(f"  Max:    {stats['max']:.2f} ms")
        print(f"  Mean:   {stats['mean']:.2f} ms")
        print(f"  Median: {stats['median']:.2f} ms")
        print(f"  P95:    {stats['p95']:.2f} ms")
        print(f"  P99:    {stats['p99']:.2f} ms")

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics (alias for get_statistics).

        Returns:
            Dictionary with latency statistics
        """
        return self.get_statistics()

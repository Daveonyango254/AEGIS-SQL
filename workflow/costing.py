"""Per-query cost accounting for the hybrid workflow.

Standard-library only so it can be unit-tested without the heavy workflow deps.
The remote (LLM) path is billed by token usage; the local (SLM) path is a fixed
per-inference compute cost.
"""


def compute_cost(
    source: str,
    token_usage: int,
    remote_token_cost: float,
    local_compute_cost: float,
) -> float:
    """Compute the USD cost of a single generated query.

    Args:
        source: Generation source — "llm"/"fllm" for the remote path, anything
            else (e.g. "slm") for the local path.
        token_usage: Total remote tokens consumed (prompt + completion). Ignored
            for the local path.
        remote_token_cost: USD per remote token.
        local_compute_cost: Fixed USD per local SLM inference.

    Returns:
        Cost in USD for this query.
    """
    if source in ("llm", "fllm"):
        return max(0, int(token_usage)) * remote_token_cost
    return local_compute_cost

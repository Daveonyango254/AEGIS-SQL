"""Unit tests for per-query cost accounting (workflow/costing.py).

Guards the fix for the hybrid run reporting $0.00 for remote GPT-4o calls:
state["cost_usd"] was never populated. compute_cost bills the remote path by
token usage and the local path by a fixed compute cost.

Run with: python tests/test_costing.py
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "costing", os.path.join(ROOT, "workflow/costing.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REMOTE_TOKEN_COST = 0.000015
LOCAL_COMPUTE_COST = 0.0001


def test_remote_cost_scales_with_tokens():
    compute_cost = _load().compute_cost
    assert compute_cost("llm", 1000, REMOTE_TOKEN_COST, LOCAL_COMPUTE_COST) == 1000 * REMOTE_TOKEN_COST
    # "fllm" alias is billed the same way.
    assert compute_cost("fllm", 500, REMOTE_TOKEN_COST, LOCAL_COMPUTE_COST) == 500 * REMOTE_TOKEN_COST


def test_local_cost_is_fixed():
    compute_cost = _load().compute_cost
    # token_usage is ignored for the local path.
    assert compute_cost("slm", 9999, REMOTE_TOKEN_COST, LOCAL_COMPUTE_COST) == LOCAL_COMPUTE_COST
    assert compute_cost("slm", 0, REMOTE_TOKEN_COST, LOCAL_COMPUTE_COST) == LOCAL_COMPUTE_COST


def test_remote_zero_or_negative_tokens_safe():
    compute_cost = _load().compute_cost
    assert compute_cost("llm", 0, REMOTE_TOKEN_COST, LOCAL_COMPUTE_COST) == 0.0
    assert compute_cost("llm", -5, REMOTE_TOKEN_COST, LOCAL_COMPUTE_COST) == 0.0


if __name__ == "__main__":
    test_remote_cost_scales_with_tokens()
    test_local_cost_is_fixed()
    test_remote_zero_or_negative_tokens_safe()
    print("All costing tests passed.")

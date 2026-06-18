"""Regression test: the self-correction repair loop must terminate.

Root cause of a production GraphRecursionError: `generation_count` (the repair
counter incremented in fslm_generation_node) and `_candidate_exec` (execution
diagnostics read by verification_node) were written to the LangGraph state but
NOT declared in the AEGISState TypedDict. LangGraph only persists declared keys
as channels across super-steps, so the counter reset every iteration and
should_repair (generations > max_repairs) never tripped -> the verify->generate
loop ran until the recursion limit.

This test guards two things:
  1. Structurally: both keys are declared channels in AEGISState.
  2. Behaviourally: the exact repair pattern terminates only when the counter
     is a declared channel (reproduces the bug with an undeclared variant).

Run with: python tests/test_repair_loop_termination.py
"""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # so state.py's `import aegis_types` resolves


def _load_state():
    """Load workflow/state.py directly (its package __init__ pulls torch)."""
    spec = importlib.util.spec_from_file_location(
        "aegis_state", os.path.join(ROOT, "workflow/state.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_required_channels_declared():
    """generation_count and _candidate_exec must be persisted channels."""
    ann = _load_state().AEGISState.__annotations__
    assert "generation_count" in ann, "generation_count must be a declared channel"
    assert "_candidate_exec" in ann, "_candidate_exec must be a declared channel"


def test_repair_loop_terminates_only_when_counter_persisted():
    """Reproduce the loop with the real graph pattern; declared -> terminates."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END
    from langgraph.errors import GraphRecursionError

    max_repairs = 1

    def build(declare_counter: bool):
        if declare_counter:
            class S(TypedDict, total=False):
                generation_count: int
                failed: bool
        else:
            class S(TypedDict, total=False):
                failed: bool  # generation_count NOT a channel

        def generate(state):  # mirrors fslm_generation_node:406
            state["generation_count"] = state.get("generation_count", 0) + 1
            state["failed"] = True  # pretend verification always fails
            return state

        def should_repair(state):  # mirrors graph.should_repair
            if state.get("generation_count", 1) > max_repairs:
                return False
            return state.get("failed", False)

        g = StateGraph(S)
        g.add_node("generate", generate)
        g.add_node("verify", lambda s: s)
        g.set_entry_point("generate")
        g.add_edge("generate", "verify")
        g.add_conditional_edges("verify", should_repair, {True: "generate", False: END})
        return g.compile()

    # Undeclared counter -> never bounded -> hits the recursion limit (the bug).
    raised = False
    try:
        build(False).invoke({}, config={"recursion_limit": 12})
    except GraphRecursionError:
        raised = True
    assert raised, "undeclared counter should loop until the recursion limit"

    # Declared counter -> bounded: 1 initial + max_repairs regenerations.
    out = build(True).invoke({}, config={"recursion_limit": 12})
    assert out["generation_count"] == max_repairs + 1, out.get("generation_count")


if __name__ == "__main__":
    test_required_channels_declared()
    print("PASS  required channels declared")
    test_repair_loop_terminates_only_when_counter_persisted()
    print("PASS  repair loop terminates when counter is a declared channel")
    print("\nAll repair-loop termination tests passed.")

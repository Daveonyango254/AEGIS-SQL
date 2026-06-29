"""AEGIS multi-agent booster harness.

A small, model-agnostic pipeline that wraps a generation model with diverse
candidate generation, execution-guided selection, and execution-feedback
refinement so that *model + harness* beats the *model alone* on text-to-SQL.

It slots inside the paper's committed skeleton — Query Planner → Content-Independent
Router → [Local SLM | Remote Abstraction] → Reviewer — reusing the existing
retriever, generators, verifier, router, and abstraction layers (no model reloads;
everything goes through ``workflow.model_cache``).
"""

from agents.orchestrator import MultiAgentOrchestrator

__all__ = ["MultiAgentOrchestrator"]

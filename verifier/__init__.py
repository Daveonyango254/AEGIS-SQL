"""Neuro-symbolic verification module (Reviewer Agent).

Three-stage verification:
1. Grammar-constrained decoding (PICARD-style)
2. Schema & type checking on parsed AST
3. Execution-aware shape verification on 100-row sample

Generates structured feedback for SLM retry. Both SLM and LLM outputs verified.

Architecture: This module implements the "Reviewer" agent.

References:
    - Build strategy Section 1
    - Paper: NL2SQL-BUGs (Wang et al. 2025)
"""

from verifier.grammar_verifier import GrammarVerifier
from verifier.schema_verifier import SchemaVerifier
from verifier.execution_verifier import ExecutionVerifier
from verifier.feedback_generator import FeedbackGenerator

__all__ = [
    "GrammarVerifier",
    "SchemaVerifier",
    "ExecutionVerifier",
    "FeedbackGenerator",
]

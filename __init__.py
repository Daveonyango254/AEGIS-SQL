"""AEGIS-SQL: Three-Axis Constrained Optimization for Hybrid NL2SQL.

A production-grade implementation of the AEGIS-SQL system for natural language to SQL
translation with formal privacy guarantees, cost optimization, and high accuracy.

References:
    - Paper: "Three-Axis Constrained Optimization for Hybrid NL2SQL"

Architecture:
    Six core components orchestrated via LangGraph StateGraph:
    1. Schema Retriever - Multilingual RAG over schema documentation
    2. DP Abstraction - Token-level ε-differential privacy via exponential mechanism
    3. Content-Independent Router - Feature-based local/remote routing
    4. SLM Generator - Multilingual code models with integrated decomposition
    5. Neuro-Symbolic Verifier - 3-stage validation (grammar, schema, execution)
    6. Reconstruction - Placeholder-to-real-token mapping
"""

__version__ = "0.1.0"
__author__ = "David Onyango, MINDS Lab"

from aegis_types import (
    Query,
    Schema,
    SQL,
    AbstractedPrompt,
    ReconstructionMap,
    VerificationResult,
)
from config import AEGISConfig

__all__ = [
    "Query",
    "Schema",
    "SQL",
    "AbstractedPrompt",
    "ReconstructionMap",
    "VerificationResult",
    "AEGISConfig",
]

"""Schema retrieval module (Query Planner Agent).

Implements RAG over schema documentation using cross-lingual embeddings
(BGE-M3, EmbeddingGemma-300m) with optional LoRA adapters.

Architecture: This module implements the "Query Planner" agent.

References: Build strategy Section 2
"""

from retriever.schema_retriever import SchemaRetriever
from retriever.embedding_models import EmbeddingModelRegistry

__all__ = ["SchemaRetriever", "EmbeddingModelRegistry"]

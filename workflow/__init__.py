"""Shared runtime services: the model cache and embedding persistence.

The LangGraph pipeline that used to live here was removed in AEGIS v1 — the
single per-query pipeline is ``agents.MultiAgentOrchestrator``.
"""

__all__ = ["model_cache", "embedding_cache", "costing"]

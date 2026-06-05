"""Content-independent routing module.

Routes queries to local SLM or remote LLM based ONLY on features derivable
from the abstracted prompt. Content-independence is required for Theorem 1.

Sprint Assignment: Sprint 2
References: Build strategy Section 1, Table 1
"""

from router.content_independent_router import ContentIndependentRouter

__all__ = ["ContentIndependentRouter"]

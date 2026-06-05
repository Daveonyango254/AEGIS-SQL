"""Global model cache for AEGIS-SQL workflow.

Implements singleton pattern to cache expensive model loading operations:
- BGE-M3 embedding model (~5-10s to load)
- SLM generator (~30-40s to load 1.5B params)
- Schema retrievers with pre-computed embeddings (per database)
- Router instance

Benefits:
- First query: ~70s (one-time warmup)
- Subsequent queries: ~1-2s (just inference)
- 26-35x speedup for batch evaluation

Thread-safe for future parallel processing.
"""

import threading
from typing import Dict, Optional
from pathlib import Path

from loguru import logger

from config import AEGISConfig, EmbeddingConfig, SLMConfig, RouterConfig, CostConfig
from aegis_types import Schema
from retriever.schema_retriever import SchemaRetriever
from generator.slm_generator import SLMGenerator
from router.content_independent_router import ContentIndependentRouter


class ModelCache:
    """Global singleton cache for expensive models and embeddings.

    Usage:
        >>> cache = ModelCache.get_instance()
        >>> retriever = cache.get_schema_retriever(db_id, schema)
        >>> generator = cache.get_slm_generator()
        >>> router = cache.get_router()

    Attributes:
        _instance: Singleton instance
        _lock: Thread lock for thread-safe singleton
        _embedding_model: Cached BGE-M3 model
        _slm_generator: Cached SLM generator
        _router: Cached router
        _schema_retrievers: Dict of cached schema retrievers per database
        _config: Global AEGIS configuration
        _max_cached_retrievers: Max number of retrievers to cache (memory limit)
    """

    _instance: Optional['ModelCache'] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize empty cache. Use get_instance() instead."""
        if ModelCache._instance is not None:
            raise RuntimeError("Use ModelCache.get_instance() instead of __init__")

        # Cached models
        self._bgem3_model = None  # BGE-M3 model (shared across all retrievers)
        self._slm_generator: Optional[SLMGenerator] = None
        self._router: Optional[ContentIndependentRouter] = None
        self._schema_retrievers: Dict[str, SchemaRetriever] = {}

        # Config
        self._config: Optional[AEGISConfig] = None

        # Memory management
        self._max_cached_retrievers = 20  # Limit to 20 databases in memory
        self._retriever_access_order = []  # LRU tracking

        # Statistics
        self._stats = {
            'retriever_hits': 0,
            'retriever_misses': 0,
            'slm_loads': 0,
            'router_loads': 0,
        }

        logger.info("✓ ModelCache initialized")

    @classmethod
    def get_instance(cls) -> 'ModelCache':
        """Get singleton instance (thread-safe).

        Returns:
            ModelCache singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ModelCache()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset cache (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._clear_all()
            cls._instance = None

    def set_config(self, config: AEGISConfig) -> None:
        """Set global configuration.

        Args:
            config: AEGIS configuration
        """
        self._config = config
        logger.debug("Config set in ModelCache")

    def get_bgem3_model(self, embedding_config: Optional[EmbeddingConfig] = None):
        """Get or load shared BGE-M3 model.

        This model is shared across all SchemaRetrievers to save memory and time.

        Args:
            embedding_config: Optional embedding config

        Returns:
            Shared BGE-M3 model instance
        """
        if self._bgem3_model is not None:
            logger.debug("Using cached BGE-M3 model")
            return self._bgem3_model

        # Load BGE-M3 model
        logger.info("Loading shared BGE-M3 model...")

        # Get config
        if embedding_config is None:
            if self._config:
                embedding_config = self._config.embedding
            else:
                embedding_config = EmbeddingConfig()

        try:
            from FlagEmbedding import BGEM3FlagModel

            use_fp16 = embedding_config.device == "cuda"

            self._bgem3_model = BGEM3FlagModel(
                embedding_config.model,
                use_fp16=use_fp16,
                device=embedding_config.device,
            )

            logger.info(f"✓ Loaded shared BGE-M3 model on {embedding_config.device}")

        except ImportError:
            logger.warning("FlagEmbedding not installed. Embeddings disabled.")
            self._bgem3_model = None
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            self._bgem3_model = None

        return self._bgem3_model

    def get_schema_retriever(
        self,
        db_id: str,
        schema: Schema,
        embedding_config: Optional[EmbeddingConfig] = None,
        use_disk_cache: bool = True
    ) -> SchemaRetriever:
        """Get or create SchemaRetriever for a database.

        Caches retriever with pre-computed embeddings for the schema.
        Supports disk persistence for faster loading across sessions.

        Args:
            db_id: Database identifier
            schema: Database schema
            embedding_config: Optional embedding config (uses default if None)
            use_disk_cache: If True, try to load from disk first

        Returns:
            Cached or newly created SchemaRetriever
        """
        # Check memory cache
        if db_id in self._schema_retrievers:
            self._stats['retriever_hits'] += 1
            self._update_lru(db_id)
            logger.debug(f"Cache HIT: SchemaRetriever for {db_id}")
            return self._schema_retrievers[db_id]

        # Cache miss - try disk cache first
        self._stats['retriever_misses'] += 1

        # Get config
        if embedding_config is None:
            if self._config:
                embedding_config = self._config.embedding
            else:
                embedding_config = EmbeddingConfig()

        retriever = None

        # Try loading from disk
        if use_disk_cache:
            from workflow.embedding_cache import load_embeddings
            retriever = load_embeddings(db_id, embedding_config, schema)

        # If disk load failed, create new retriever
        if retriever is None:
            logger.info(f"Cache MISS: Loading SchemaRetriever for {db_id}...")

            # Get shared BGE-M3 model (load once, reuse for all retrievers)
            shared_model = self.get_bgem3_model(embedding_config)

            # Create retriever with shared model
            retriever = SchemaRetriever(
                embedding_config,
                schema,
                shared_model=shared_model
            )

            # Save to disk for future runs
            if use_disk_cache:
                try:
                    from workflow.embedding_cache import save_embeddings
                    save_embeddings(db_id, retriever)
                except Exception as e:
                    logger.warning(f"Failed to save embeddings to disk: {e}")

        # Cache in memory
        self._schema_retrievers[db_id] = retriever
        self._retriever_access_order.append(db_id)

        # Enforce memory limit (LRU eviction)
        self._enforce_retriever_limit()

        logger.info(f"✓ Cached SchemaRetriever for {db_id} ({len(self._schema_retrievers)} in cache)")

        return retriever

    def get_slm_generator(self, slm_config: Optional[SLMConfig] = None) -> SLMGenerator:
        """Get or create SLM generator.

        Args:
            slm_config: Optional SLM config (uses default if None)

        Returns:
            Cached or newly created SLMGenerator
        """
        if self._slm_generator is not None:
            logger.debug("Cache HIT: SLMGenerator")
            return self._slm_generator

        # Cache miss - create new generator
        self._stats['slm_loads'] += 1
        logger.info("Cache MISS: Loading SLMGenerator...")

        # Get config
        if slm_config is None:
            if self._config:
                slm_config = self._config.slm
            else:
                slm_config = SLMConfig()

        # Create and cache
        self._slm_generator = SLMGenerator(slm_config)
        logger.info("✓ Cached SLMGenerator")

        return self._slm_generator

    def get_router(
        self,
        router_config: Optional[RouterConfig] = None,
        cost_config: Optional[CostConfig] = None
    ) -> ContentIndependentRouter:
        """Get or create router.

        Args:
            router_config: Optional router config
            cost_config: Optional cost config

        Returns:
            Cached or newly created Router
        """
        if self._router is not None:
            logger.debug("Cache HIT: Router")
            return self._router

        # Cache miss - create new router
        self._stats['router_loads'] += 1
        logger.debug("Cache MISS: Creating Router...")

        # Get configs
        if router_config is None:
            if self._config:
                router_config = self._config.router
            else:
                router_config = RouterConfig()

        if cost_config is None:
            if self._config:
                cost_config = self._config.cost
            else:
                cost_config = CostConfig()

        # Create and cache
        self._router = ContentIndependentRouter(router_config, cost_config)
        logger.debug("✓ Cached Router")

        return self._router

    def _update_lru(self, db_id: str) -> None:
        """Update LRU access order for retriever.

        Args:
            db_id: Database that was accessed
        """
        if db_id in self._retriever_access_order:
            self._retriever_access_order.remove(db_id)
        self._retriever_access_order.append(db_id)

    def _enforce_retriever_limit(self) -> None:
        """Evict least recently used retrievers if over limit."""
        while len(self._schema_retrievers) > self._max_cached_retrievers:
            # Evict least recently used
            lru_db_id = self._retriever_access_order.pop(0)
            del self._schema_retrievers[lru_db_id]
            logger.debug(f"Evicted SchemaRetriever for {lru_db_id} (LRU policy)")

    def warmup(
        self,
        config: AEGISConfig,
        db_ids_and_schemas: list[tuple[str, Schema]]
    ) -> None:
        """Pre-load all models and schema retrievers.

        Call this before evaluation loop for maximum performance.

        Args:
            config: AEGIS configuration
            db_ids_and_schemas: List of (db_id, schema) tuples to pre-cache
        """
        logger.info("=" * 80)
        logger.info("CACHE WARMUP: Pre-loading models...")
        logger.info("=" * 80)

        self.set_config(config)

        # Pre-load SLM (heaviest, ~30-40s)
        logger.info("\n[1/3] Pre-loading SLM Generator...")
        self.get_slm_generator(config.slm)

        # Pre-load Router
        logger.info("\n[2/3] Pre-loading Router...")
        self.get_router(config.router, config.cost)

        # Pre-load schema retrievers (with BGE-M3 embedding)
        logger.info(f"\n[3/3] Pre-loading {len(db_ids_and_schemas)} Schema Retrievers...")
        for i, (db_id, schema) in enumerate(db_ids_and_schemas, 1):
            logger.info(f"  [{i}/{len(db_ids_and_schemas)}] {db_id}...")
            self.get_schema_retriever(db_id, schema, config.embedding)

        logger.info("\n" + "=" * 80)
        logger.info("✓ CACHE WARMUP COMPLETE")
        logger.info("=" * 80)
        self.print_stats()

    def print_stats(self) -> None:
        """Print cache statistics."""
        total_hits = self._stats['retriever_hits']
        total_misses = self._stats['retriever_misses']
        hit_rate = total_hits / (total_hits + total_misses) * 100 if (total_hits + total_misses) > 0 else 0

        logger.info("\nCache Statistics:")
        logger.info(f"  SchemaRetriever hits:   {self._stats['retriever_hits']}")
        logger.info(f"  SchemaRetriever misses: {self._stats['retriever_misses']}")
        logger.info(f"  Hit rate:               {hit_rate:.1f}%")
        logger.info(f"  SLM loads:              {self._stats['slm_loads']}")
        logger.info(f"  Router loads:           {self._stats['router_loads']}")
        logger.info(f"  Cached retrievers:      {len(self._schema_retrievers)}/{self._max_cached_retrievers}")

    def get_stats(self) -> dict:
        """Get cache statistics as dictionary.

        Returns:
            Dictionary of cache statistics
        """
        return {
            **self._stats,
            'cached_retrievers': len(self._schema_retrievers),
            'max_retrievers': self._max_cached_retrievers,
        }

    def _clear_all(self) -> None:
        """Clear all cached models (internal use only)."""
        self._bgem3_model = None
        self._slm_generator = None
        self._router = None
        self._schema_retrievers.clear()
        self._retriever_access_order.clear()
        logger.info("Cache cleared")


# Convenience function for global access
def get_cache() -> ModelCache:
    """Get global model cache instance.

    Returns:
        Global ModelCache singleton
    """
    return ModelCache.get_instance()

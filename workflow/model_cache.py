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

from config import AEGISConfig, EmbeddingConfig, SLMConfig, LLMConfig
from aegis_types import Schema
from retriever.schema_retriever import SchemaRetriever
from generator.slm_generator import SLMGenerator


class ModelCache:
    """Global singleton cache for expensive models and embeddings.

    Usage:
        >>> cache = ModelCache.get_instance()
        >>> retriever = cache.get_schema_retriever(db_id, schema)
        >>> generator = cache.get_slm(model_id)

    Attributes:
        _instance: Singleton instance
        _lock: Thread lock for thread-safe singleton
        _embedding_model: Cached BGE-M3 model
        _slm_generators: Cached SLM generators keyed by model id
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
        # Local SLMs keyed by model id — the CSC pipeline runs TWO checkpoints
        # (generator + merger); identical ids share one instance.
        self._slm_generators: Dict[str, SLMGenerator] = {}
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
                embedding_config = self._config.embedding_config()
            else:
                embedding_config = EmbeddingConfig()

        try:
            from FlagEmbedding import BGEM3FlagModel

            logger.info(f"Loading BGE-M3 model: {embedding_config.model}")
            logger.info(f"  Device: {embedding_config.device}")
            logger.info(f"  Cache dir: {embedding_config.cache_dir}")

            use_fp16 = embedding_config.device == "cuda"

            self._bgem3_model = BGEM3FlagModel(
                embedding_config.model,
                use_fp16=use_fp16,
                device=embedding_config.device,
            )

            logger.info(f"✓ Loaded shared BGE-M3 model on {embedding_config.device}")
            logger.info(f"  Model parameters: {sum(p.numel() for p in self._bgem3_model.model.parameters()) / 1e6:.1f}M")

        except ImportError as e:
            logger.error(f"❌ CRITICAL: FlagEmbedding not installed!")
            logger.error(f"  Install with: pip install -U FlagEmbedding")
            logger.error(f"  Error details: {e}")
            logger.error(f"  Schema retrieval will use PASS-THROUGH mode (poor accuracy)")
            self._bgem3_model = None
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to load BGE-M3 model!")
            logger.error(f"  Model: {embedding_config.model}")
            logger.error(f"  Device: {embedding_config.device}")
            logger.error(f"  Error: {e}")
            logger.error(f"  Schema retrieval will use PASS-THROUGH mode (poor accuracy)")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
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
                embedding_config = self._config.embedding_config()
            else:
                embedding_config = EmbeddingConfig()

        retriever = None

        # Get shared BGE-M3 model (needed even for cached retrievers)
        shared_model = self.get_bgem3_model(embedding_config)

        # Try loading from disk
        if use_disk_cache:
            from workflow.embedding_cache import load_embeddings
            retriever = load_embeddings(db_id, embedding_config, schema, shared_model)

        # If disk load failed, create new retriever
        if retriever is None:
            logger.info(f"Cache MISS: Loading SchemaRetriever for {db_id}...")

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

    def get_slm(self, model_id: Optional[str] = None) -> SLMGenerator:
        """Get or load a local SLM by model id (cached; same id shares one instance).

        ``None`` resolves to the configured candidate GENERATOR. The CSC merge
        model is fetched with ``get_slm(config.models.merger)`` — lazily, so a
        run that never reaches the merge stage never loads it.
        """
        if model_id is None:
            model_id = self._config.models.generator if self._config else SLMConfig().model
        if model_id in self._slm_generators:
            logger.debug(f"Cache HIT: SLM {model_id}")
            return self._slm_generators[model_id]

        self._stats['slm_loads'] += 1
        logger.info(f"Cache MISS: loading SLM {model_id} ...")
        slm_config = (
            self._config.slm_config(model_id) if self._config else SLMConfig(model=model_id)
        )
        self._slm_generators[model_id] = SLMGenerator(slm_config)
        logger.info(f"✓ Cached SLM {model_id}")
        return self._slm_generators[model_id]

    def get_slm_generator(self, slm_config: Optional[SLMConfig] = None) -> SLMGenerator:
        """Back-compat shim: the configured candidate generator."""
        return self.get_slm(slm_config.model if slm_config else None)

    def get_llm_generator(self, llm_config: Optional[LLMConfig] = None):
        """Get LLM fallback generator with proper config.

        Note: Unlike SLM, LLM generator is stateless and lightweight, so we don't cache it.
        Each call creates a new instance with the properly loaded config.

        Args:
            llm_config: Optional LLM config (uses cached config if None)

        Returns:
            New LLMFallback instance with proper config from cache
        """
        # Get config from cache
        if llm_config is None:
            if self._config:
                llm_config = self._config.llm_config()
            else:
                llm_config = LLMConfig()

        # Import here to avoid circular dependency
        from generator.llm_fallback import LLMFallback

        return LLMFallback(llm_config)

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

        # Pre-load the candidate generator (heaviest; the merge model loads
        # lazily on the first query that actually reaches the merge stage).
        logger.info("\n[1/2] Pre-loading candidate generator...")
        self.get_slm(config.models.generator)

        # Pre-load schema retrievers (with BGE-M3 embedding)
        logger.info(f"\n[2/2] Pre-loading {len(db_ids_and_schemas)} Schema Retrievers...")
        for i, (db_id, schema) in enumerate(db_ids_and_schemas, 1):
            logger.info(f"  [{i}/{len(db_ids_and_schemas)}] {db_id}...")
            self.get_schema_retriever(db_id, schema, config.embedding_config())

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
        self._slm_generators = {}
        self._ambiguity_resolver = None
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

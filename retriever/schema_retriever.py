"""Schema retrieval with multilingual embeddings (Query Planner Agent).

Implements cross-lingual RAG over schema documentation using BGE-M3 or
EmbeddingGemma-300m with optional LoRA adapters.

Architecture: This implements the "Query Planner" agent from the system architecture.

References:
    - Build strategy Section 2: Embedding-model selection
    - Build strategy Section 2.3: Schema linking specifically
"""

from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json

import numpy as np
import torch
from loguru import logger

from config import EmbeddingConfig
from aegis_types import Query, Schema, SchemaElement


class SchemaRetriever:
    """Schema retrieval with multilingual embeddings (Query Planner Agent).

    Retrieves top-k schema elements (tables/columns) relevant to the query
    using dense + sparse retrieval (BGE-M3 hybrid mode).

    Attributes:
        config: Embedding configuration
        schema: Database schema
        model: BGE-M3 embedding model
        adapter: Optional LoRA adapter
        schema_texts: Text descriptions for each schema element
        dense_embeddings: Pre-computed dense embeddings for schema
        sparse_embeddings: Pre-computed sparse embeddings (token weights)
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        schema: Schema,
        skip_encoding: bool = False,
        shared_model = None
    ) -> None:
        """Initialize schema retriever.

        Args:
            config: Embedding configuration
            schema: Database schema to index
            skip_encoding: If True, skip model loading and encoding
                          (useful when embeddings already exist)
            shared_model: Pre-loaded BGE-M3 model to share across retrievers
                         (improves performance by avoiding re-loading)
        """
        self.config = config
        self.schema = schema
        self.model = None
        self.adapter = None
        self.schema_texts: List[str] = []
        self.dense_embeddings: Optional[np.ndarray] = None
        self.sparse_embeddings: Optional[Dict] = None
        self._is_encoded = False

        # Set shared model even when skipping encoding (needed for retrieval)
        if shared_model is not None:
            self.model = shared_model
            logger.debug("Using shared BGE-M3 model")

        if skip_encoding:
            logger.debug("Skipping encoding (reusing existing embeddings)")
            return

        # Try to load BGE-M3 model
        try:
            # Use shared model if provided, otherwise load new one
            if shared_model is not None:
                logger.debug("Using shared BGE-M3 model")
                self.model = shared_model
            else:
                self._load_model()

            self._encode_schema()
            self._is_encoded = True
            logger.info(
                f"✓ SchemaRetriever initialized with {config.model} "
                f"({len(self.schema.columns)} columns indexed)"
            )
        except Exception as e:
            logger.error(
                f"❌ CRITICAL: Failed to initialize embedding model!"
            )
            logger.error(f"  Database: {schema.db_id}")
            logger.error(f"  Model: {config.model}")
            logger.error(f"  Error: {e}")
            logger.error(
                f"  ⚠️  Falling back to PASS-THROUGH mode - accuracy will be SEVERELY degraded!"
            )
            logger.error(
                f"  ⚠️  Expected EX accuracy drop: 30-40 percentage points"
            )
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
            self.model = None
            self._is_encoded = False

    def _load_model(self) -> None:
        """Load BGE-M3 embedding model."""
        try:
            from FlagEmbedding import BGEM3FlagModel

            logger.info(f"Loading BGE-M3 model: {self.config.model}")

            # Load model with specified device
            use_fp16 = self.config.device == "cuda"

            self.model = BGEM3FlagModel(
                self.config.model,
                use_fp16=use_fp16,
                device=self.config.device,
            )

            logger.info(f"✓ Loaded BGE-M3 model on {self.config.device}")

            # Load LoRA adapter if specified
            if self.config.adapter_path:
                self.load_adapter(self.config.adapter_path)

        except ImportError:
            logger.warning(
                "FlagEmbedding not installed. Install with: pip install FlagEmbedding"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            raise

    def _encode_schema(self) -> None:
        """Pre-encode all schema elements and build index."""
        if not self.model:
            logger.debug("Model not loaded, skipping schema encoding")
            return

        if not self.schema.columns:
            logger.warning("Schema has no columns to encode")
            return

        # Build text descriptions for each schema element
        self.schema_texts = []
        for col in self.schema.columns:
            # Format: "table_name.column_name (type): description"
            text = f"{col.name}"
            if col.data_type:
                text += f" ({col.data_type})"
            if col.description:
                text += f": {col.description}"
            self.schema_texts.append(text)

        logger.debug(f"Encoding {len(self.schema_texts)} schema elements...")

        try:
            # BGE-M3 encodes with both dense and sparse representations
            embeddings = self.model.encode(
                self.schema_texts,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,  # We don't need ColBERT for this use case
            )

            # Extract dense embeddings (for semantic similarity)
            self.dense_embeddings = embeddings['dense_vecs']

            # Extract sparse embeddings (for keyword matching)
            self.sparse_embeddings = embeddings['lexical_weights']

            logger.info(
                f"✓ Encoded schema: {len(self.schema_texts)} elements, "
                f"embedding dim={self.dense_embeddings.shape[1]}"
            )

        except Exception as e:
            logger.error(f"Failed to encode schema: {e}")
            self.dense_embeddings = None
            self.sparse_embeddings = None

    def retrieve(
        self,
        query: Query,
        top_k: int = 10,
        use_hybrid: bool = True,
        expand_foreign_keys: bool = True,
        max_expanded_tables: int = 3,
    ) -> List[SchemaElement]:
        """Retrieve top-k schema elements relevant to the query.

        Uses BGE-M3 hybrid retrieval (dense + sparse) for multilingual support,
        then recall-oriented FK expansion.

        Args:
            query: Natural language query
            top_k: Number of schema elements to retrieve via semantic search
            use_hybrid: Use both dense and sparse retrieval (default: True)
            expand_foreign_keys: If True, expand retrieval to include FK-related tables
            max_expanded_tables: Max number of additional FK tables to include

        Returns:
            List of retrieved schema elements, ranked by relevance
        """
        # Stage 1: Semantic retrieval
        retrieved = self._semantic_retrieve(query, top_k, use_hybrid)

        # Stage 2: FK expansion (recall).
        if expand_foreign_keys and self.schema.foreign_keys:
            retrieved = self._expand_with_foreign_keys(retrieved, max_expanded_tables)

        return retrieved

    def _semantic_retrieve(
        self, query: Query, top_k: int, use_hybrid: bool
    ) -> List[SchemaElement]:
        """Stage 1: Semantic retrieval using BGE-M3.

        Args:
            query: Natural language query
            top_k: Number of schema elements to retrieve
            use_hybrid: Use hybrid (dense + sparse) retrieval

        Returns:
            List of retrieved schema elements
        """
        # Fallback to pass-through mode if model not loaded
        if not self.model or self.dense_embeddings is None:
            logger.warning(
                f"⚠️  PASS-THROUGH MODE ACTIVE - returning first {top_k} columns WITHOUT semantic matching!"
            )
            logger.warning(
                f"  This will likely result in WRONG table selection and LOW accuracy."
            )
            return self._passthrough_retrieve(top_k)

        try:
            # Encode query
            query_text = query.text
            logger.debug(f"Encoding query: {query_text[:80]}...")

            query_embeddings = self.model.encode(
                [query_text],
                batch_size=1,
                max_length=self.config.max_length,
                return_dense=True,
                return_sparse=use_hybrid,
                return_colbert_vecs=False,
            )

            # Compute similarity scores
            if use_hybrid:
                # Hybrid: combine dense and sparse scores
                dense_scores = self._compute_dense_similarity(
                    query_embeddings['dense_vecs'][0]
                )
                sparse_scores = self._compute_sparse_similarity(
                    query_embeddings['lexical_weights'][0]
                )

                # Combine scores (weighted average)
                # BGE-M3 paper recommends 0.5/0.5 or 0.6/0.4 for dense/sparse
                scores = 0.6 * dense_scores + 0.4 * sparse_scores
            else:
                # Dense-only retrieval
                scores = self._compute_dense_similarity(
                    query_embeddings['dense_vecs'][0]
                )

            # Get top-k indices
            top_k_indices = np.argsort(scores)[::-1][:top_k]

            # Return corresponding schema elements
            retrieved = [self.schema.columns[idx] for idx in top_k_indices]

            # Observability: log top matches with scores + the semantic table set
            # so we can diagnose whether the right tables surface before FK expansion.
            top_n = min(10, len(top_k_indices))
            preview = [
                (self.schema.columns[i].name, round(float(scores[i]), 3))
                for i in top_k_indices[:top_n]
            ]
            sem_tables = sorted({
                self.schema.columns[i].name.split(".", 1)[0]
                for i in top_k_indices if "." in self.schema.columns[i].name
            })
            logger.info(
                "RETRIEVAL_SEMANTIC: %d cols, tables=%s, top%d=%s"
                % (len(retrieved), sem_tables, top_n, preview)
            )

            return retrieved

        except Exception as e:
            logger.error(f"Retrieval failed: {e}, falling back to pass-through")
            return self._passthrough_retrieve(top_k)

    def _expand_with_foreign_keys(
        self, retrieved: List[SchemaElement], max_expanded_tables: int
    ) -> List[SchemaElement]:
        """Stage 2: Expand retrieval with FK-related tables.

        For each table in the retrieved columns, add ALL columns from FK-related tables.
        This ensures JOIN queries have complete table schemas.

        Args:
            retrieved: Columns from semantic retrieval
            max_expanded_tables: Max number of additional tables to include

        Returns:
            Expanded list of schema elements
        """
        # Get unique tables from retrieved columns
        retrieved_tables = set()
        for col in retrieved:
            if '.' in col.name:
                table_name = col.name.split('.', 1)[0]
                retrieved_tables.add(table_name)

        logger.debug(f"Retrieved tables: {retrieved_tables}")

        # Find FK-related tables
        related_tables = set()
        for table in retrieved_tables:
            for fk in self.schema.get_foreign_keys_for_table(table):
                # Add both source and target tables
                related_tables.add(fk.from_table)
                related_tables.add(fk.to_table)

        # Remove already-retrieved tables
        new_tables = related_tables - retrieved_tables

        # Limit expansion
        if len(new_tables) > max_expanded_tables:
            logger.debug(f"Limiting FK expansion from {len(new_tables)} to {max_expanded_tables} tables")
            new_tables = set(list(new_tables)[:max_expanded_tables])

        logger.debug(f"FK expansion: adding tables {new_tables}")

        # Add ALL columns from new tables
        expanded = retrieved.copy()
        for col in self.schema.columns:
            if '.' in col.name:
                table_name = col.name.split('.', 1)[0]
                if table_name in new_tables:
                    expanded.append(col)

        logger.info(
            "RETRIEVAL_FK_EXPANSION: %d → %d columns (+%d tables via FKs: %s)"
            % (len(retrieved), len(expanded), len(new_tables), sorted(new_tables))
        )

        return expanded

    def _compute_dense_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and schema embeddings.

        Args:
            query_embedding: Query dense embedding vector

        Returns:
            Similarity scores for all schema elements
        """
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        schema_norm = self.dense_embeddings / np.linalg.norm(
            self.dense_embeddings, axis=1, keepdims=True
        )

        # Cosine similarity
        scores = np.dot(schema_norm, query_norm)
        return scores

    def _compute_sparse_similarity(
        self, query_weights: Dict[int, float]
    ) -> np.ndarray:
        """Compute sparse similarity (BM25-like) between query and schema.

        Args:
            query_weights: Query sparse token weights

        Returns:
            Similarity scores for all schema elements
        """
        scores = np.zeros(len(self.schema_texts))

        # For each schema element
        for idx, schema_weights in enumerate(self.sparse_embeddings):
            # Compute overlap score
            score = 0.0
            for token_id, query_weight in query_weights.items():
                if token_id in schema_weights:
                    schema_weight = schema_weights[token_id]
                    score += query_weight * schema_weight
            scores[idx] = score

        return scores

    def _passthrough_retrieve(self, top_k: int) -> List[SchemaElement]:
        """Fallback: return first top_k schema elements.

        Args:
            top_k: Number of elements to return

        Returns:
            First top_k schema elements
        """
        logger.debug(f"Pass-through retrieval: returning first {top_k} columns")
        all_columns = self.schema.columns
        return all_columns[:top_k] if top_k else all_columns

    def encode_query(self, text: str) -> np.ndarray:
        """Encode query text into embedding vector.

        Args:
            text: Query text in any supported language

        Returns:
            Dense embedding vector
        """
        if not self.model:
            logger.warning("Model not loaded, returning zero vector")
            return np.zeros(1024)  # BGE-M3 dimension

        try:
            embeddings = self.model.encode(
                [text],
                batch_size=1,
                max_length=self.config.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return embeddings['dense_vecs'][0]

        except Exception as e:
            logger.error(f"Query encoding failed: {e}")
            return np.zeros(1024)

    def load_adapter(self, adapter_path: str) -> None:
        """Load fine-tuned LoRA adapter.

        Args:
            adapter_path: Path to LoRA adapter weights
        """
        try:
            from peft import PeftModel

            logger.info(f"Loading LoRA adapter from: {adapter_path}")

            # Load adapter on top of base model
            # Note: BGE-M3 uses sentence-transformers, which wraps HuggingFace models
            # We need to access the underlying transformer model
            if hasattr(self.model, 'model'):
                base_model = self.model.model
                self.model.model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                )
                logger.info("✓ LoRA adapter loaded successfully")
            else:
                logger.warning("Could not access base model for LoRA loading")

        except ImportError:
            logger.warning("PEFT not installed. Install with: pip install peft")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")

    def add_schema_descriptions(
        self, descriptions: Dict[str, str]
    ) -> None:
        """Add external schema descriptions and re-encode.

        Useful for integrating BIRD database_description CSVs.

        Args:
            descriptions: Dict mapping "table.column" -> description
        """
        # Update schema element descriptions
        for col in self.schema.columns:
            if col.name in descriptions:
                col.description = descriptions[col.name]

        # Re-encode schema with new descriptions
        if self.model:
            logger.info("Re-encoding schema with updated descriptions...")
            self._encode_schema()
            self._is_encoded = True

    def is_encoded(self) -> bool:
        """Check if schema has been encoded.

        Returns:
            True if embeddings are ready, False otherwise
        """
        return self._is_encoded and self.dense_embeddings is not None

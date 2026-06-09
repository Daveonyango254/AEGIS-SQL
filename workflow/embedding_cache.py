"""Disk persistence for schema embeddings.

Saves pre-computed embeddings to disk for faster loading across sessions.

Structure:
    cache/embeddings/{db_id}/
        ├── dense_embeddings.npy       (NumPy array of dense vectors)
        ├── sparse_embeddings.json     (JSON dict of sparse weights)
        ├── schema_texts.json          (List of schema text descriptions)
        └── metadata.json              (DB ID, timestamp, config)

Usage:
    >>> # Save embeddings
    >>> save_embeddings(db_id, retriever)

    >>> # Load embeddings
    >>> retriever = load_embeddings(db_id, config, schema)
    >>> if retriever.is_encoded():
    >>>     print("Loaded from cache!")
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime

from loguru import logger

from config import EmbeddingConfig
from aegis_types import Schema
from retriever.schema_retriever import SchemaRetriever


# Cache directory
CACHE_DIR = Path("cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(db_id: str) -> Path:
    """Get cache directory path for a database.

    Args:
        db_id: Database identifier

    Returns:
        Path to cache directory
    """
    cache_path = CACHE_DIR / db_id
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def save_embeddings(db_id: str, retriever: SchemaRetriever) -> None:
    """Save schema embeddings to disk.

    Args:
        db_id: Database identifier
        retriever: SchemaRetriever with computed embeddings
    """
    if not retriever.is_encoded():
        logger.warning(f"Retriever for {db_id} has no embeddings to save")
        return

    cache_path = get_cache_path(db_id)

    try:
        # Save dense embeddings
        if retriever.dense_embeddings is not None:
            dense_path = cache_path / "dense_embeddings.npy"
            np.save(dense_path, retriever.dense_embeddings)
            logger.debug(f"Saved dense embeddings: {dense_path}")

        # Save sparse embeddings (convert to JSON-serializable format)
        if retriever.sparse_embeddings is not None:
            sparse_path = cache_path / "sparse_embeddings.json"
            # Convert dict of dicts with int keys to JSON
            sparse_json = []
            for weights_dict in retriever.sparse_embeddings:
                # Convert int keys to strings and numpy float32 to Python float
                sparse_json.append({
                    str(k): float(v)  # Convert np.float32 to Python float
                    for k, v in weights_dict.items()
                })

            with open(sparse_path, 'w') as f:
                json.dump(sparse_json, f)
            logger.debug(f"Saved sparse embeddings: {sparse_path}")

        # Save schema texts
        if retriever.schema_texts:
            texts_path = cache_path / "schema_texts.json"
            with open(texts_path, 'w') as f:
                json.dump(retriever.schema_texts, f, indent=2)
            logger.debug(f"Saved schema texts: {texts_path}")

        # Save metadata
        metadata = {
            "db_id": db_id,
            "timestamp": datetime.now().isoformat(),
            "model": retriever.config.model,
            "num_columns": len(retriever.schema.columns),
            "embedding_dim": retriever.dense_embeddings.shape[1] if retriever.dense_embeddings is not None else None,
        }
        metadata_path = cache_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Saved embeddings for {db_id} to {cache_path}")

    except Exception as e:
        logger.error(f"Failed to save embeddings for {db_id}: {e}")


def load_embeddings(
    db_id: str,
    config: EmbeddingConfig,
    schema: Schema,
    shared_model=None
) -> Optional[SchemaRetriever]:
    """Load schema embeddings from disk.

    Args:
        db_id: Database identifier
        config: Embedding configuration
        schema: Database schema
        shared_model: Pre-loaded BGE-M3 model to use for retrieval

    Returns:
        SchemaRetriever with loaded embeddings, or None if not found
    """
    cache_path = get_cache_path(db_id)

    # Check if cache exists
    dense_path = cache_path / "dense_embeddings.npy"
    sparse_path = cache_path / "sparse_embeddings.json"
    texts_path = cache_path / "schema_texts.json"
    metadata_path = cache_path / "metadata.json"

    if not all(p.exists() for p in [dense_path, texts_path, metadata_path]):
        logger.debug(f"Cache not found for {db_id}")
        return None

    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Validate model matches
        if metadata.get("model") != config.model:
            logger.warning(
                f"Cache model mismatch for {db_id}: "
                f"cached={metadata.get('model')}, current={config.model}"
            )
            return None

        # Validate column count matches
        if metadata.get("num_columns") != len(schema.columns):
            logger.warning(
                f"Cache schema mismatch for {db_id}: "
                f"cached={metadata.get('num_columns')} cols, current={len(schema.columns)} cols"
            )
            return None

        # Create retriever without encoding, but WITH shared model for retrieval
        retriever = SchemaRetriever(config, schema, skip_encoding=True, shared_model=shared_model)

        # Load dense embeddings
        retriever.dense_embeddings = np.load(dense_path)
        logger.debug(f"Loaded dense embeddings: {dense_path}")

        # Load sparse embeddings if exists
        if sparse_path.exists():
            with open(sparse_path, 'r') as f:
                sparse_json = json.load(f)
            # Convert string keys back to ints
            retriever.sparse_embeddings = []
            for weights_dict_str in sparse_json:
                weights_dict = {int(k): v for k, v in weights_dict_str.items()}
                retriever.sparse_embeddings.append(weights_dict)
            logger.debug(f"Loaded sparse embeddings: {sparse_path}")

        # Load schema texts
        with open(texts_path, 'r') as f:
            retriever.schema_texts = json.load(f)
        logger.debug(f"Loaded schema texts: {texts_path}")

        # Mark as encoded
        retriever._is_encoded = True

        logger.info(f"✓ Loaded embeddings for {db_id} from cache (saved {metadata.get('timestamp')})")

        return retriever

    except Exception as e:
        logger.error(f"Failed to load embeddings for {db_id}: {e}")
        return None


def clear_cache(db_id: Optional[str] = None) -> None:
    """Clear embedding cache.

    Args:
        db_id: Specific database to clear, or None to clear all
    """
    if db_id:
        # Clear specific database
        cache_path = get_cache_path(db_id)
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            logger.info(f"Cleared cache for {db_id}")
    else:
        # Clear all caches
        if CACHE_DIR.exists():
            import shutil
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared all embedding caches")


def get_cache_info() -> Dict[str, Dict]:
    """Get information about cached embeddings.

    Returns:
        Dict mapping db_id to metadata dict
    """
    cache_info = {}

    if not CACHE_DIR.exists():
        return cache_info

    for db_path in CACHE_DIR.iterdir():
        if not db_path.is_dir():
            continue

        db_id = db_path.name
        metadata_path = db_path / "metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    cache_info[db_id] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read metadata for {db_id}: {e}")

    return cache_info

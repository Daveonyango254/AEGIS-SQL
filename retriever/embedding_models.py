"""Embedding model registry and factory.

Centralizes loading of different embedding models with consistent interface.

Sprint Assignment: Sprint 1
References: Build strategy Section 2.1
"""

from enum import Enum
from typing import Any, Dict

from loguru import logger


class EmbeddingModelType(str, Enum):
    """Supported embedding models."""

    BGE_M3 = "bge-m3"
    EMBEDDING_GEMMA_300M = "embedding-gemma-300m"
    MULTILINGUAL_E5_LARGE = "multilingual-e5-large-instruct"
    SNOWFLAKE_ARCTIC_EMBED = "snowflake-arctic-embed-l-v2.0"
    QWEN3_EMBEDDING_0_6B = "qwen3-embedding-0.6b"
    GTE_MULTILINGUAL_BASE = "gte-multilingual-base"


class EmbeddingModelRegistry:
    """Registry for multilingual embedding models.

    Provides factory methods to load models with consistent interface.
    """

    _MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "bge-m3": {
            "hf_model_id": "BAAI/bge-m3",
            "params": "0.6B",
            "languages": 100,
            "max_length": 8192,
            "features": ["dense", "sparse", "multi-vec"],
        },
        "embedding-gemma-300m": {
            "hf_model_id": "google/embedding-gemma-300m",
            "params": "0.3B",
            "languages": 100,
            "max_length": 512,
            "features": ["dense"],
        },
        "multilingual-e5-large-instruct": {
            "hf_model_id": "intfloat/multilingual-e5-large-instruct",
            "params": "0.6B",
            "languages": 94,
            "max_length": 512,
            "features": ["dense"],
        },
        "snowflake-arctic-embed-l-v2.0": {
            "hf_model_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
            "params": "0.6B",
            "languages": 100,
            "max_length": 512,
            "features": ["dense"],
        },
        "qwen3-embedding-0.6b": {
            "hf_model_id": "Qwen/Qwen3-Embedding-0.6B",
            "params": "0.6B",
            "languages": 100,
            "max_length": 8192,
            "features": ["dense"],
        },
        "gte-multilingual-base": {
            "hf_model_id": "Alibaba-NLP/gte-multilingual-base",
            "params": "0.3B",
            "languages": 70,
            "max_length": 512,
            "features": ["dense"],
        },
    }

    @classmethod
    def load_model(cls, model_name: str, device: str = "cuda") -> Any:
        """Load embedding model by name.

        Args:
            model_name: Model identifier from EmbeddingModelType
            device: Target device (cuda or cpu)

        Returns:
            Loaded embedding model (sentence-transformers interface)

        Raises:
            ValueError: If model name is not supported

        TODO (Sprint 1):
            - Load model from HuggingFace with sentence-transformers
            - Handle device placement (CPU/CUDA)
            - Configure max_length based on model config
            - For BGE-M3: enable hybrid dense+sparse mode
            - Add model caching to avoid re-downloading
        """
        if model_name not in cls._MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. Supported: {list(cls._MODEL_CONFIGS.keys())}"
            )

        config = cls._MODEL_CONFIGS[model_name]
        logger.info(
            f"Loading embedding model: {model_name} "
            f"({config['params']} params, {config['languages']} languages)"
        )
        raise NotImplementedError("Sprint 1 implementation")

    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a model.

        Args:
            model_name: Model identifier

        Returns:
            Model configuration dictionary
        """
        return cls._MODEL_CONFIGS.get(model_name, {})

    @classmethod
    def list_models(cls) -> list[str]:
        """List all supported embedding models.

        Returns:
            List of model names
        """
        return list(cls._MODEL_CONFIGS.keys())

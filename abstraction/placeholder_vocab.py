"""Language-agnostic placeholder vocabulary V_abs.

Defines semantic placeholders used by the DP abstraction mechanism.

References: Build strategy Section 4 (Option C)
"""

from typing import Any, Dict, List

import numpy as np

from aegis_types import SensitivityLevel


class PlaceholderVocabulary:
    """Language-agnostic placeholder vocabulary.

    Placeholders are semantic tags like <PERSON_1>, <LOCATION_1>, <MEDICAL_CODE_1>
    that are language-independent by design. The DP mechanism samples from this
    vocabulary based on embedding distance to sensitive tokens.

    Attributes:
        vocab_size: Number of placeholders (from config)
        placeholders: List of placeholder strings
        embeddings: Pre-computed embeddings for each placeholder
        category_map: Mapping from placeholder to semantic category
    """

    def __init__(self, vocab_size: int = 100, embedding_model: Any = None) -> None:
        """Initialize placeholder vocabulary.

        Args:
            vocab_size: Number of placeholders to generate
            embedding_model: Multilingual embedding model for pre-computing embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_model = embedding_model
        self.placeholders: List[str] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        self.category_map: Dict[str, str] = {}

        # Generate placeholders
        self._generate_placeholders()

        # Pre-compute embeddings if model provided
        if embedding_model:
            self._precompute_embeddings()

    def _generate_placeholders(self) -> None:
        """Generate semantic placeholder vocabulary.

        Creates placeholders for each category:
        - Format: <CATEGORY_N> where N is 1-indexed within category
        - Example: <PERSON_1>, <PERSON_2>, ..., <LOCATION_1>, <LOCATION_2>, ...
        - Distributes vocab_size across categories (weighted by expected frequency)
        """
        # Define categories and their weights (based on expected frequency)
        categories = {
            "PERSON": 20,      # Names, patients, customers
            "LOCATION": 10,    # Addresses, cities, countries
            "ORG": 10,         # Organizations, companies
            "DATE": 10,        # Dates, timestamps
            "NUMBER": 10,      # IDs, codes, numbers
            "MEDICAL": 10,     # Medical codes, diagnoses
            "FINANCIAL": 10,   # Account numbers, transactions
            "PRODUCT": 10,     # Product codes, SKUs
            "TABLE": 10,       # Database table names
            "COLUMN": 10,      # Database column names
        }

        # Calculate how many placeholders per category
        total_weight = sum(categories.values())
        placeholders_per_category = {
            cat: max(1, int(weight / total_weight * self.vocab_size))
            for cat, weight in categories.items()
        }

        # Generate placeholders
        for category, count in placeholders_per_category.items():
            for i in range(1, count + 1):
                placeholder = f"<{category}_{i}>"
                self.placeholders.append(placeholder)
                self.category_map[placeholder] = category

    def _precompute_embeddings(self) -> None:
        """Pre-compute embeddings for all placeholders.

        Encodes each placeholder string with embedding model and stores
        in self.embeddings dict for fast lookup during DP sampling.
        Embeddings are in the same multilingual space as query tokens.
        """
        if not self.embedding_model:
            return

        for placeholder in self.placeholders:
            # Encode placeholder (implementation depends on embedding model)
            # For now, use simple random embeddings as placeholder
            # TODO: Replace with actual embedding model when available
            self.embeddings[placeholder] = np.random.randn(384)  # Typical embedding size

    def get_placeholder_by_category(
        self, category: str, sensitivity: SensitivityLevel
    ) -> List[str]:
        """Get placeholders matching a semantic category.

        Args:
            category: Semantic category (e.g., "PERSON", "LOCATION")
            sensitivity: Sensitivity level (PII, PROPRIETARY, REGULATED)

        Returns:
            List of matching placeholders
        """
        # Filter placeholders by category
        matching = [
            p for p, cat in self.category_map.items()
            if cat.upper() == category.upper()
        ]

        # Sort by index
        matching.sort(key=lambda p: int(p.split("_")[-1].rstrip(">")))

        return matching

    def get_embedding(self, placeholder: str) -> np.ndarray:
        """Get pre-computed embedding for a placeholder.

        Args:
            placeholder: Placeholder string (e.g., "<PERSON_1>")

        Returns:
            Embedding vector

        Raises:
            KeyError: If placeholder not in vocabulary
        """
        if placeholder not in self.embeddings:
            raise KeyError(f"Placeholder not in vocabulary: {placeholder}")
        return self.embeddings[placeholder]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.placeholders)

    def __iter__(self):
        """Iterate over placeholders."""
        return iter(self.placeholders)

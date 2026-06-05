"""Reconstruction module for mapping placeholders back to real tokens.

Lives in the trusted environment only - never sent to remote LLM.

References: Build strategy Section 1 (architectural diagram)
"""

from typing import Dict

from loguru import logger

from aegis_types import ReconstructionMap, SQL


class ReconstructionModule:
    """Reconstructs SQL by replacing placeholders with original tokens.

    Maintains bidirectional mapping created during DP abstraction.
    All operations happen inside the trusted boundary.

    Attributes:
        reconstruction_maps: Cache of reconstruction maps per query
    """

    def __init__(self) -> None:
        """Initialize reconstruction module."""
        self.reconstruction_maps: Dict[str, ReconstructionMap] = {}
        logger.info("Initialized ReconstructionModule")

    def register_map(self, query_id: str, recon_map: ReconstructionMap) -> None:
        """Register a reconstruction map for a query.

        Args:
            query_id: Unique query identifier
            recon_map: Reconstruction map from abstraction phase
        """
        self.reconstruction_maps[query_id] = recon_map
        logger.debug(
            f"Registered reconstruction map for query {query_id} "
            f"with {len(recon_map.placeholder_to_real)} placeholders"
        )

        # TODO: Implement TTL or LRU cache to avoid memory growth
        # For now, simple dictionary storage

    def reconstruct(self, sql: SQL, query_id: str) -> SQL:
        """Reconstruct SQL by replacing placeholders with original tokens.

        Args:
            sql: SQL with placeholders (output from SLM or LLM)
            query_id: Query identifier to look up reconstruction map

        Returns:
            SQL with placeholders replaced by real tokens

        Raises:
            KeyError: If query_id not found in registered maps
        """
        if query_id not in self.reconstruction_maps:
            raise KeyError(f"No reconstruction map found for query: {query_id}")

        logger.debug(f"Reconstructing SQL for query {query_id}")

        recon_map = self.reconstruction_maps[query_id]

        # If no placeholders to replace, return as-is
        if not recon_map.placeholder_to_real:
            logger.debug("No placeholders to reconstruct")
            return sql

        # Replace all placeholders with real tokens
        reconstructed_text = sql.text
        num_replaced = 0

        # Sort by placeholder length (longest first) to avoid partial replacements
        sorted_items = sorted(
            recon_map.placeholder_to_real.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for placeholder, real_token in sorted_items:
            if placeholder in reconstructed_text:
                reconstructed_text = reconstructed_text.replace(placeholder, real_token)
                num_replaced += 1

        logger.debug(f"Reconstructed {num_replaced} placeholders")

        # Create new SQL with reconstructed text
        reconstructed_sql = SQL(
            text=reconstructed_text,
            dialect=sql.dialect,
            source=sql.source,
            verified=sql.verified,
            verification_result=sql.verification_result,
        )

        return reconstructed_sql

    def clear_map(self, query_id: str) -> None:
        """Clear reconstruction map after query completion.

        Args:
            query_id: Query identifier to clear
        """
        if query_id in self.reconstruction_maps:
            del self.reconstruction_maps[query_id]
            logger.debug(f"Cleared reconstruction map for query {query_id}")

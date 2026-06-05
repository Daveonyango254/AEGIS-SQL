"""LangGraph state definition for AEGIS-SQL workflow.

Defines the state machine that tracks query processing through the pipeline.

"""

from typing import Dict, List, Optional, TypedDict

from aegis_types import (
    Query,
    Schema,
    SchemaElement,
    RoutingDecision,
    AbstractedPrompt,
    ReconstructionMap,
    SQL,
    VerificationResult,
)


class AEGISState(TypedDict, total=False):
    """State for AEGIS-SQL workflow.

    Tracks the complete query processing pipeline from natural language input
    to verified SQL output, including all intermediate representations and metrics.

    Attributes:
        query: Original natural language query
        database_id: Target database identifier
        db_path: Path to database file (for verification and evaluation)

        # Schema Extraction
        schema: Full database schema (tables, columns, types)
        schema_elements: Retrieved relevant schema elements

        # Routing Decision
        routing_decision: LOCAL (FSLM) or REMOTE (FLLM)
        router_features: Content-independent features used for routing

        # Abstraction (Remote path only)
        abstracted_prompt: DP-abstracted query with placeholders
        reconstruction_map: Bidirectional mapping for reconstruction

        # Generation
        sql: Generated SQL query
        generation_source: "fslm" or "fllm"

        # Verification
        verification_result: Detailed verification outcome
        verification_attempts: Number of retry attempts

        # Metrics (for evaluation)
        cost_usd: Total cost in USD
        latency_ms: End-to-end latency in milliseconds
        privacy_loss: ε × |prompt| × Pr(r=remote)

        # Error tracking
        error_message: Error description if workflow fails
        error_stage: Stage where error occurred
    """

    # Input
    query: Query
    database_id: str
    db_path: Optional[str]  # Path to database file (for evaluation)

    # Schema Extraction
    schema: Optional[Schema]  # Full database schema
    schema_elements: List[SchemaElement]

    # Routing
    routing_decision: RoutingDecision
    router_features: Optional[Dict]

    # Abstraction (Remote only)
    abstracted_prompt: Optional[AbstractedPrompt]
    reconstruction_map: Optional[ReconstructionMap]

    # Generation
    sql: Optional[SQL]
    generation_source: Optional[str]  # "fslm" | "fllm"

    # Verification
    verification_result: Optional[VerificationResult]
    verification_attempts: int

    # Metrics
    cost_usd: float
    latency_ms: float
    privacy_loss: float

    # Error tracking
    error_message: Optional[str]
    error_stage: Optional[str]

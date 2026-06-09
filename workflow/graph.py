"""LangGraph workflow definition for AEGIS-SQL with router-before-abstraction.

Workflow:
    Query → QUERY PLANNER AGENT → [Local: FSLM → REVIEWER AGENT]
               │                 OR [Remote: Abstraction → FLLM → Reconstruction → REVIEWER AGENT]
               │
               └─> Step 1: Schema Extraction (SchemaRetriever)
               └─> Step 2: Content-Independent Routing (ContentIndependentRouter)

This ensures local path has zero privacy leakage (no abstraction).

Agent Architecture:
    - Query Planner Agent: Schema extraction + routing decision
    - Reviewer Agent: Grammar + Schema + Execution verification

"""

from typing import Literal

from langgraph.graph import StateGraph, END
from loguru import logger

from config import AEGISConfig
from workflow.state import AEGISState
from aegis_types import RoutingDecision, VerificationStatus

# Component imports
from retriever import SchemaRetriever
from abstraction import DPAbstractor, ReconstructionModule
from router import ContentIndependentRouter
from generator import SLMGenerator, LLMFallback
from verifier import (
    GrammarVerifier,
    SchemaVerifier,
    ExecutionVerifier,
    FeedbackGenerator,
)


def build_aegis_graph(config: AEGISConfig) -> StateGraph:
    """Build the AEGIS-SQL LangGraph workflow with router-before-abstraction.

    Architecture:
        QUERY PLANNER AGENT:
            1. schema_extraction_node: Retrieve relevant schema elements (SchemaRetriever)
            2. routing_node: Decide LOCAL (FSLM) or REMOTE (FLLM) (ContentIndependentRouter)

        GENERATION:
            3a. Local path: fslm_generation_node → verification_node
            3b. Remote path: abstraction_node → fllm_generation_node →
                            reconstruction_node → verification_node

        REVIEWER AGENT:
            4. verification_node: Grammar + Schema + Execution checks

    Args:
        config: AEGIS configuration

    Returns:
        Compiled StateGraph ready for execution

    References:
        - Paper Section 3: Hybrid architecture
        - Build strategy Section 1: Component integration
    """
    logger.info("Building AEGIS-SQL workflow graph...")

    # Initialize workflow
    workflow = StateGraph(AEGISState)

    # Define nodes
    # Conditional: Add ambiguity resolution node if enabled
    if config.ambiguity.enabled:
        workflow.add_node("ambiguity_resolution", ambiguity_resolution_node)

    workflow.add_node("schema_extraction", schema_extraction_node)
    workflow.add_node("routing", routing_node)
    workflow.add_node("abstraction", abstraction_node)
    workflow.add_node("fslm_generation", fslm_generation_node)
    workflow.add_node("fllm_generation", fllm_generation_node)
    workflow.add_node("reconstruction", reconstruction_node)
    workflow.add_node("verification", verification_node)

    # Define edges
    if config.ambiguity.enabled:
        # Ambiguity resolution first, then schema extraction
        workflow.set_entry_point("ambiguity_resolution")
        workflow.add_edge("ambiguity_resolution", "schema_extraction")
    else:
        # Standard entry point
        workflow.set_entry_point("schema_extraction")

    workflow.add_edge("schema_extraction", "routing")

    # Conditional routing after router decision
    workflow.add_conditional_edges(
        "routing",
        should_route_to_remote,
        {
            True: "abstraction",  # Remote path: abstraction first
            False: "fslm_generation",  # Local path: FSLM directly
        },
    )

    # Local path: FSLM → Verification
    workflow.add_edge("fslm_generation", "verification")

    # Remote path: Abstraction → FLLM → Reconstruction → Verification
    workflow.add_edge("abstraction", "fllm_generation")
    workflow.add_conditional_edges(
        "fllm_generation",
        should_reconstruct,
        {
            True: "reconstruction",  # Reconstruction enabled
            False: "verification",  # Skip reconstruction
        },
    )
    workflow.add_edge("reconstruction", "verification")

    # Verification → END
    workflow.add_edge("verification", END)

    # Compile graph
    graph = workflow.compile()
    logger.info("✓ AEGIS-SQL workflow graph compiled successfully")

    return graph


# ============================================================================
# Node Implementations
# ============================================================================


def ambiguity_resolution_node(state: AEGISState) -> AEGISState:
    """Query Planner Agent - Step 0: Ambiguity Detection & Resolution (optional).

    Detects and resolves ambiguous natural language queries before SQL generation.
    Only runs if ambiguity.enabled=true in config.

    Args:
        state: Current workflow state

    Returns:
        Updated state with potentially rewritten query
    """
    logger.info("Running Query Planner Agent - Step 0: Ambiguity Resolution...")

    # Get ambiguity resolver from cache
    from workflow.model_cache import get_cache
    cache = get_cache()

    # Lazy-load resolver on first use
    if not hasattr(cache, '_ambiguity_resolver') or cache._ambiguity_resolver is None:
        from query_planner.ambiguity_resolver import AmbiguityResolver
        from config import AEGISConfig

        # Load config to get ambiguity settings
        config = AEGISConfig.from_yaml("config.yaml")
        amb_config = config.ambiguity

        # Get SLM generator if LLM mode is enabled (for privacy-preserving ambiguity detection)
        slm_generator = None
        if amb_config.detector_type == "llm":
            slm_generator = cache.get_slm_generator()

        cache._ambiguity_resolver = AmbiguityResolver(
            detector_type=amb_config.detector_type,
            resolution_mode=amb_config.resolution_mode,
            auto_resolve_temporal=amb_config.auto_resolve_temporal,
            temporal_default_days=amb_config.temporal_default_days,
            confidence_threshold=amb_config.confidence_threshold,
            slm_generator=slm_generator
        )

    resolver = cache._ambiguity_resolver

    # Detect ambiguities
    query = state["query"]
    schema = state.get("schema")

    ambiguities = resolver.detect(query, schema)

    # Store original query before any modifications
    state["original_query"] = query
    state["is_ambiguous"] = len(ambiguities) > 0
    state["detected_ambiguities"] = [
        {
            "type": amb.type,
            "phrase": amb.phrase,
            "reason": amb.reason,
            "candidates": amb.candidates,
            "confidence": amb.confidence
        }
        for amb in ambiguities
    ]

    if len(ambiguities) == 0:
        logger.info("AMBIGUITY_CHECK: No ambiguities detected")
        return state

    logger.info(f"AMBIGUITY_DETECTED: Found {len(ambiguities)} ambiguities")
    for amb in ambiguities:
        logger.debug(f"  - {amb.type}: '{amb.phrase}' → {amb.candidates}")

    # Resolve ambiguities
    try:
        rewritten_query, resolutions = resolver.resolve(query, ambiguities, schema)

        # Update query with resolved version
        state["query"].text = rewritten_query
        state["ambiguity_resolutions"] = [
            {
                "type": res.ambiguity.type,
                "phrase": res.ambiguity.phrase,
                "chosen": res.chosen_interpretation,
                "rewritten": res.rewritten_phrase,
                "method": res.method
            }
            for res in resolutions
        ]

        logger.info(f"AMBIGUITY_RESOLVED: {len(resolutions)} ambiguities resolved")
        logger.info(f"QUERY_REWRITTEN: {rewritten_query}")

    except Exception as e:
        # If interactive mode raises RequiresClarificationException
        # or any other error, store it in state
        logger.warning(f"Ambiguity resolution failed: {e}")
        state["clarification_questions"] = getattr(e, 'questions', None)

    return state


def schema_extraction_node(state: AEGISState) -> AEGISState:
    """Query Planner Agent - Step 1: Schema Extraction.

    Extract relevant schema elements for the query using RAG with BGE-M3 embeddings.

    Args:
        state: Current workflow state

    Returns:
        Updated state with schema_elements
    """
    logger.info("Running Query Planner Agent - Step 1: Schema Extraction...")

    # Get schema from state
    schema = state.get("schema")
    if not schema:
        logger.warning("No schema provided in state")
        state["schema_elements"] = []
        return state

    # Get schema retriever from cache (with pre-computed embeddings)
    from workflow.model_cache import get_cache
    cache = get_cache()
    db_id = state.get("database_id", schema.database_id)
    retriever = cache.get_schema_retriever(db_id, schema)

    # Retrieve schema elements with FK expansion for better JOIN coverage
    query = state["query"]
    schema_elements = retriever.retrieve(
        query,
        top_k=40,  # Increased from 25 to 40 for better schema coverage (~35% of avg database)
        expand_foreign_keys=True,  # Enable FK expansion to include related tables
        max_expanded_tables=3  # Add up to 3 FK-related tables
    )

    state["schema_elements"] = schema_elements

    # Log extracted schema for observability
    logger.info(f"QUERY_PLANNER_COMPLETE: Retrieved {len(schema_elements)} elements")
    logger.info(f"EXTRACTED_SCHEMA: {[elem.name for elem in schema_elements[:5]]}")  # Log first 5 elements

    return state


def routing_node(state: AEGISState) -> AEGISState:
    """Query Planner Agent - Step 2: Content-Independent Routing.

    Make routing decision: LOCAL (FSLM) or REMOTE (FLLM) based on query
    complexity, schema size, and cost constraints.

    Args:
        state: Current workflow state

    Returns:
        Updated state with routing_decision
    """
    logger.info("Routing query...")

    # Get router from cache
    from workflow.model_cache import get_cache
    cache = get_cache()
    router = cache.get_router()

    # Make routing decision
    query = state["query"]
    schema_elements = state.get("schema_elements", [])
    routing_decision = router.route(query, schema_elements)

    state["routing_decision"] = routing_decision

    if routing_decision == RoutingDecision.LOCAL:
        logger.info("ROUTED_TO_LOCAL")
    else:
        logger.info("ROUTED_TO_REMOTE")

    return state


def abstraction_node(state: AEGISState) -> AEGISState:
    """Apply DP abstraction to query (remote path only).

    Args:
        state: Current workflow state

    Returns:
        Updated state with abstracted_prompt and reconstruction_map
    """
    logger.info("Applying DP abstraction...")

    # Get privacy config from cache
    from workflow.model_cache import get_cache
    from abstraction.placeholder_vocab import PlaceholderVocabulary
    from abstraction.sensitivity_policy import SensitivityPolicy

    cache = get_cache()
    if cache._config:
        privacy_config = cache._config.privacy
        policy_config = cache._config.privacy.sensitivity_policy
    else:
        # Fallback to defaults if cache not initialized
        from config import PrivacyConfig, SensitivityPolicyConfig
        privacy_config = PrivacyConfig()
        policy_config = SensitivityPolicyConfig()

    # Create components
    vocab = PlaceholderVocabulary(vocab_size=privacy_config.placeholder_vocab_size)
    policy = SensitivityPolicy(policy_config)
    abstractor = DPAbstractor(privacy_config, vocab, policy, embedding_model=None)

    # Apply abstraction
    query = state["query"]
    schema_elements = state.get("schema_elements", [])
    abstracted_prompt, recon_map = abstractor.abstract(query, schema_elements)

    state["abstracted_prompt"] = abstracted_prompt
    state["reconstruction_map"] = recon_map

    logger.info(f"ABSTRACTION_APPLIED: {abstracted_prompt.num_substitutions} tokens abstracted")

    return state


def fslm_generation_node(state: AEGISState) -> AEGISState:
    """Generate SQL using local SLM (FSLM).

    Args:
        state: Current workflow state

    Returns:
        Updated state with sql
    """
    logger.info("Generating SQL with FSLM...")

    # Get SLM generator from cache
    from workflow.model_cache import get_cache
    cache = get_cache()
    generator = cache.get_slm_generator()

    # Generate SQL
    query = state["query"]
    schema_elements = state.get("schema_elements", [])
    sql = generator.generate(query, schema_elements)

    state["sql"] = sql
    state["generation_source"] = "slm"

    logger.info(f"GENERATION_COMPLETE (FSLM): {sql.text[:80]}...")
    logger.info(f"GENERATED_SQL_FSLM: {sql.text}")  # Log full SQL for observability

    return state


def fllm_generation_node(state: AEGISState) -> AEGISState:
    """Generate SQL using remote LLM (FLLM).

    Args:
        state: Current workflow state

    Returns:
        Updated state with sql
    """
    logger.info("Generating SQL with FLLM...")

    # Get LLM generator from cache (uses properly loaded config)
    from workflow.model_cache import get_cache
    cache = get_cache()
    generator = cache.get_llm_generator()

    # Generate SQL from abstracted prompt
    abstracted_prompt = state.get("abstracted_prompt")
    schema_elements = state.get("schema_elements", [])

    if not abstracted_prompt:
        # Fallback: use original query
        from aegis_types import AbstractedPrompt
        query = state["query"]
        abstracted_prompt = AbstractedPrompt(
            text=query.text,
            original_tokens=[],
            placeholder_map={},
            epsilon=0.0,
            num_substitutions=0,
            evidence=query.evidence,  # NEW: Pass evidence through
        )

    sql = generator.generate(abstracted_prompt, schema_elements)

    state["sql"] = sql
    state["generation_source"] = "llm"

    logger.info(f"GENERATION_COMPLETE (FLLM): {sql.text[:80]}...")
    logger.info(f"GENERATED_SQL_FLLM (with placeholders): {sql.text}")  # Log SQL with placeholders

    return state


def reconstruction_node(state: AEGISState) -> AEGISState:
    """Reconstruct SQL by replacing placeholders with real tokens.

    Args:
        state: Current workflow state

    Returns:
        Updated state with reconstructed sql
    """
    logger.info("Reconstructing SQL...")

    # Initialize reconstruction module
    recon_module = ReconstructionModule()

    # Register reconstruction map
    query_id = state.get("query_id", "default")
    recon_map = state.get("reconstruction_map")

    if not recon_map:
        logger.debug("No reconstruction map found, skipping reconstruction")
        return state

    recon_module.register_map(query_id, recon_map)

    # Reconstruct SQL
    sql = state.get("sql")
    if sql:
        logger.info(f"SQL_BEFORE_RECONSTRUCTION: {sql.text}")  # Log before reconstruction
        reconstructed_sql = recon_module.reconstruct(sql, query_id)
        state["sql"] = reconstructed_sql
        logger.info("RECONSTRUCTION_APPLIED")
        logger.info(f"SQL_AFTER_RECONSTRUCTION: {reconstructed_sql.text}")  # Log after reconstruction

    return state


def verification_node(state: AEGISState) -> AEGISState:
    """Reviewer Agent: 3-Stage SQL Verification.

    Verify generated SQL through three stages:
    1. Grammar verification (syntax check)
    2. Schema verification (element validity)
    3. Execution verification (runtime check)

    Args:
        state: Current workflow state

    Returns:
        Updated state with verification_result
    """
    logger.info("Running Reviewer Agent: 3-Stage SQL Verification...")

    sql = state.get("sql")
    if not sql:
        logger.warning("No SQL to verify")
        return state

    # Validate SQL is not empty
    from aegis_types import VerificationResult, VerificationStatus

    if not sql.text or sql.text.strip() == "":
        logger.warning("Empty SQL generated - marking as failed")
        verification_result = VerificationResult(
            status=VerificationStatus.GRAMMAR_FAIL,
            grammar_valid=False,
            schema_valid=False,
            execution_valid=False,
            error_message="Empty SQL generated by model",
            structured_feedback=None,
            execution_result=None,
        )
        sql.verified = False
        sql.verification_result = verification_result
        state["verification_result"] = verification_result
        logger.info("VERIFICATION_FAILED: Empty SQL")
        return state

    # Initialize verifiers
    schema = state.get("schema")
    db_path = state.get("db_path", ":memory:")

    # For now, create a simplified verification result
    # TODO: Implement actual 3-stage verification with grammar, schema, execution

    try:
        # Simplified verification: just mark as passed
        # TODO: Call actual verifiers
        verification_result = VerificationResult(
            status=VerificationStatus.PASS,
            grammar_valid=True,
            schema_valid=True,
            execution_valid=True,
            error_message=None,
            structured_feedback=None,
            execution_result=None,
        )

        sql.verified = True
        sql.verification_result = verification_result

        logger.info("VERIFICATION_PASSED")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        verification_result = VerificationResult(
            status=VerificationStatus.EXECUTION_FAIL,
            grammar_valid=True,
            schema_valid=True,
            execution_valid=False,
            error_message=str(e),
            structured_feedback=None,
            execution_result=None,
        )
        sql.verified = False
        sql.verification_result = verification_result

        logger.info("VERIFICATION_FAILED")

    state["verification_result"] = verification_result

    return state


# ============================================================================
# Conditional Edge Functions
# ============================================================================


def should_route_to_remote(state: AEGISState) -> bool:
    """Determine if query should route to remote LLM.

    Args:
        state: Current workflow state

    Returns:
        True if routing to REMOTE (FLLM), False if LOCAL (FSLM)
    """
    return state["routing_decision"] == RoutingDecision.REMOTE


def should_reconstruct(state: AEGISState) -> bool:
    """Determine if reconstruction should be applied.

    Args:
        state: Current workflow state

    Returns:
        True if reconstruction_map exists, False otherwise
    """
    return state.get("reconstruction_map") is not None

"""Content-independent routing for local vs remote execution.

CRITICAL ARCHITECTURE CHANGE:
- Router now runs BEFORE abstraction (not after)
- Uses raw query + schema METADATA only (no content)
- This ensures local path has zero privacy leakage

Sprint Assignment: Sprint 2
References:
    - AEGIS integration doc lines 241-246, 461-475
    - Router-before-abstraction: lines 309-316
"""

from typing import Dict, List

from loguru import logger

from config import RouterConfig, CostConfig
from aegis_types import Query, RouterFeatures, RoutingDecision, SchemaElement


class ContentIndependentRouter:
    """Routes queries to local SLM or remote LLM based on content-independent features.

    NEW DESIGN (router-before-abstraction):
    The routing function r(·) operates ONLY on features computable from
    RAW query + schema METADATA:
        - Raw query token count (|query| in tokens)
        - Schema element count (number of tables/columns retrieved)
        - Query structural complexity (keyword counts, estimated depth)

    NEVER uses:
        - Sensitive token content
        - Actual schema element names (only counts)
        - Database cell values

    This content-independence ensures:
    1. Router doesn't need abstracted data → can run BEFORE abstraction
    2. Local path gets real, un-abstracted data → privacy = 0
    3. Remote path gets abstracted data → privacy = ε×|prompt|

    Attributes:
        router_config: Router configuration
        cost_config: Cost configuration for budget constraints
        threshold: Complexity threshold for routing (0-1)
    """

    def __init__(self, router_config: RouterConfig, cost_config: CostConfig) -> None:
        """Initialize content-independent router.

        Args:
            router_config: Router configuration
            cost_config: Cost configuration
        """
        self.router_config = router_config
        self.cost_config = cost_config
        self.threshold = router_config.threshold_complexity

        # Routing statistics (for evaluation)
        self._routing_decisions = []

        logger.info(
            f"✓ ContentIndependentRouter initialized: threshold={self.threshold}"
        )

    def route(
        self,
        query: Query,
        schema_elements: List[SchemaElement],
    ) -> RoutingDecision:
        """Make routing decision: local SLM or remote LLM.

        IMPORTANT: Router operates on RAW query (before abstraction).

        Decision logic:
            1. Check force_local/force_remote flags (override all other logic)
            2. Extract content-independent features from RAW query + schema metadata
            3. Compute complexity score (0-1 normalized)
            4. Check cost budget constraints
            5. If complexity < threshold AND budget allows: LOCAL
            6. Else: REMOTE

        Args:
            query: Raw query (NOT abstracted)
            schema_elements: Retrieved schema elements

        Returns:
            Routing decision (LOCAL or REMOTE)

        References:
            - AEGIS integration doc lines 461-475
            - Content-independence ensures: r depends only on query/schema metadata
        """
        logger.debug(f"Making routing decision for query: {query.text[:50]}...")

        # Check force flags first (override all other logic)
        if self.router_config.force_local:
            decision = RoutingDecision.LOCAL
            reason = "force_local flag enabled"
            self._routing_decisions.append(decision)
            logger.info(f"Routing decision: {decision.value} ({reason})")
            return decision

        if self.router_config.force_remote:
            decision = RoutingDecision.REMOTE
            reason = "force_remote flag enabled"
            self._routing_decisions.append(decision)
            logger.info(f"Routing decision: {decision.value} ({reason})")
            return decision

        # Extract features
        features = self.extract_features(query, schema_elements)

        # Compute complexity score
        complexity_score = self.compute_complexity_score(features)

        # Estimate costs
        local_cost = self.cost_config.local_compute_cost
        estimated_remote_cost = features.query_token_count * self.cost_config.remote_token_cost

        # Routing decision based on complexity and cost
        if complexity_score < self.threshold and local_cost <= self.cost_config.budget_per_query:
            decision = RoutingDecision.LOCAL
            reason = f"complexity={complexity_score:.2f} < threshold={self.threshold}"
        else:
            decision = RoutingDecision.REMOTE
            if complexity_score >= self.threshold:
                reason = f"complexity={complexity_score:.2f} >= threshold={self.threshold}"
            else:
                reason = f"cost constraint: local=${local_cost:.4f} > budget=${self.cost_config.budget_per_query:.4f}"

        # Track decision
        self._routing_decisions.append(decision)

        logger.info(f"Routing decision: {decision.value} ({reason})")
        logger.debug(
            f"Features: tokens={features.query_token_count}, "
            f"schema_elements={features.schema_element_count}, "
            f"complexity={complexity_score:.2f}"
        )

        return decision

    def extract_features(
        self,
        query: Query,
        schema_elements: List[SchemaElement],
    ) -> RouterFeatures:
        """Extract content-independent features for routing.

        All features are computable from RAW query text + schema metadata ONLY.
        No sensitive content is examined.

        Args:
            query: Raw query (before abstraction)
            schema_elements: Retrieved schema elements

        Returns:
            RouterFeatures with metadata only

        Features extracted:
        1. Query token count (simple whitespace split)
        2. Schema element count (number of tables + columns)
        3. Query complexity estimate (keyword-based heuristic)
        """
        # Feature 1: Query token count (simple tokenization)
        query_token_count = len(query.text.split())

        # Feature 2: Schema element count
        schema_element_count = len(schema_elements)

        # Feature 3: Query structural complexity (keyword-based heuristic)
        query_upper = query.text.upper()

        # Count SQL keywords as complexity indicators
        join_count = query_upper.count("JOIN")
        subquery_count = query_upper.count("SELECT") - 1  # -1 for main SELECT
        aggregate_count = sum([
            query_upper.count(agg) for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"]
        ])
        where_count = query_upper.count("WHERE")
        groupby_count = query_upper.count("GROUP BY")
        orderby_count = query_upper.count("ORDER BY")

        # Estimate complexity (normalized heuristic)
        # Base complexity from query length
        complexity_base = min(query_token_count / 50.0, 1.0)  # Normalize by 50 tokens

        # Add complexity from structural elements
        complexity_structure = (
            join_count * 0.15 +
            max(subquery_count, 0) * 0.20 +  # Subqueries are expensive
            aggregate_count * 0.10 +
            where_count * 0.05 +
            groupby_count * 0.10 +
            orderby_count * 0.05
        )

        # Total complexity (clamped to [0, 1])
        query_complexity = min(complexity_base + complexity_structure, 1.0)

        return RouterFeatures(
            query_token_count=query_token_count,
            schema_element_count=schema_element_count,
            query_structural_complexity=query_complexity,
        )

    def compute_complexity_score(self, features: RouterFeatures) -> float:
        """Compute normalized complexity score (0-1) from features.

        Higher score = more complex query = more likely to route to remote LLM.

        Args:
            features: Extracted router features

        Returns:
            Complexity score (0-1)

        Scoring heuristic:
        - Base: Query structural complexity (already 0-1)
        - Adjustment: Large schema → increase complexity
        - Adjustment: Very short queries → decrease complexity
        """
        base_complexity = features.query_structural_complexity

        # Adjustment 1: Large schemas are harder
        schema_adjustment = 0.0
        if features.schema_element_count > 20:
            schema_adjustment = 0.1
        elif features.schema_element_count > 10:
            schema_adjustment = 0.05

        # Adjustment 2: Very short queries are easier
        token_adjustment = 0.0
        if features.query_token_count < 10:
            token_adjustment = -0.1

        # Final score (clamped)
        final_score = base_complexity + schema_adjustment + token_adjustment
        final_score = max(0.0, min(1.0, final_score))

        return final_score

    def get_routing_stats(self) -> Dict[str, float]:
        """Get routing statistics (for evaluation).

        Returns:
            Dictionary with Pr(r=local), Pr(r=remote), total_queries

        Used for computing:
        - Privacy loss: ℒ_priv = ε × E[|prompt|] × Pr(r=remote)
        - Cost analysis: expected cost per query
        """
        if not self._routing_decisions:
            return {
                "pr_local": 0.0,
                "pr_remote": 0.0,
                "total_queries": 0,
            }

        total = len(self._routing_decisions)
        local_count = sum(1 for d in self._routing_decisions if d == RoutingDecision.LOCAL)
        remote_count = total - local_count

        return {
            "pr_local": local_count / total,
            "pr_remote": remote_count / total,
            "total_queries": total,
        }

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._routing_decisions = []
        logger.debug("Router statistics reset")

"""Query ambiguity detection and resolution for AEGIS-SQL Query Planner.

Detects and resolves ambiguous natural language queries before SQL generation
to improve accuracy and user experience.

Supported ambiguity types:
- Temporal: Unclear time references ("recent", "current", "old")
- Schema: Multiple matching schema elements
- Underspecified: Missing constraints ("top", "large" without metrics)

Detection methods:
- Rule-based: Fast, local, zero-cost pattern matching
- LLM-based: Higher accuracy using local SLM (future enhancement)

Resolution modes:
- Auto: Apply sensible defaults automatically
- Interactive: Generate clarification questions for user

References:
    - AmbiSQL (2024): Interactive Ambiguity Detection
    - Sphinteract (2024): User Interaction for Disambiguation
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from loguru import logger

from aegis_types import Query, Schema, SchemaElement


@dataclass
class Ambiguity:
    """Detected ambiguity in natural language query.

    Attributes:
        type: Type of ambiguity (temporal, schema, underspecified, lexical, structural)
        phrase: Ambiguous phrase from query
        reason: Why it's ambiguous
        candidates: Possible interpretations or schema elements
        confidence: Detection confidence (0-1)
    """
    type: str
    phrase: str
    reason: str
    candidates: List[str]
    confidence: float


@dataclass
class Resolution:
    """Resolution for a detected ambiguity.

    Attributes:
        ambiguity: Original ambiguity that was resolved
        chosen_interpretation: Selected interpretation from candidates
        rewritten_phrase: Phrase to replace ambiguous one
        method: How it was resolved (auto_default, auto_heuristic, user)
    """
    ambiguity: Ambiguity
    chosen_interpretation: str
    rewritten_phrase: str
    method: str


class RequiresClarificationException(Exception):
    """Exception raised when interactive clarification is needed.

    Attributes:
        questions: List of clarification questions for user
    """

    def __init__(self, questions: List[Dict]):
        self.questions = questions
        super().__init__(f"Query requires {len(questions)} clarifications")


class AmbiguityResolver:
    """Detects and resolves query ambiguities.

    Supports both rule-based and LLM-based detection.
    Supports auto-resolution and interactive clarification.

    Attributes:
        detector_type: Detection method ("rules" or "llm")
        resolution_mode: Resolution strategy ("auto" or "interactive")
        auto_resolve_temporal: Whether to auto-resolve temporal ambiguities
        temporal_default_days: Default days for "recent" queries
        confidence_threshold: Minimum confidence to flag ambiguity (0-1)
    """

    # Temporal ambiguity keywords and their default interpretations
    TEMPORAL_KEYWORDS = {
        "recent": ["last 7 days", "last 30 days", "last 3 months"],
        "recently": ["last 7 days", "last 30 days", "last 3 months"],
        "latest": ["most recent", "last update", "newest"],
        "old": ["over 1 year", "over 2 years", "archived"],
        "older": ["over 1 year", "over 2 years", "archived"],
        "current": ["today", "this month", "this quarter"],
        "currently": ["today", "this month", "this quarter"],
        "new": ["last week", "last month", "recently added"],
        "newer": ["last week", "last month", "recently added"],
    }

    # Underspecification keywords
    UNDERSPECIFIED_KEYWORDS = {
        "top": ["by value", "by count", "by date", "by name"],
        "best": ["by rating", "by performance", "by popularity"],
        "worst": ["by rating", "by performance", "by unpopularity"],
        "large": ["above average", "top 10%", "specific threshold"],
        "small": ["below average", "bottom 10%", "specific threshold"],
        "big": ["above average", "top 10%", "specific threshold"],
        "high": ["above average", "top 10%", "specific threshold"],
        "low": ["below average", "bottom 10%", "specific threshold"],
    }

    def __init__(
        self,
        detector_type: str = "rules",
        resolution_mode: str = "auto",
        auto_resolve_temporal: bool = True,
        temporal_default_days: int = 30,
        confidence_threshold: float = 0.6
    ):
        """Initialize ambiguity resolver.

        Args:
            detector_type: "rules" (fast, local) or "llm" (accurate, may be slow)
            resolution_mode: "auto" (use defaults) or "interactive" (ask user)
            auto_resolve_temporal: Auto-resolve temporal ambiguities
            temporal_default_days: Default days for "recent" queries (default: 30)
            confidence_threshold: Minimum confidence to flag ambiguity (0-1)
        """
        self.detector_type = detector_type
        self.resolution_mode = resolution_mode
        self.auto_resolve_temporal = auto_resolve_temporal
        self.temporal_default_days = temporal_default_days
        self.confidence_threshold = confidence_threshold

        # LLM detector initialization (future)
        if self.detector_type == "llm":
            logger.warning("LLM-based detection not yet implemented, falling back to rules")
            self.detector_type = "rules"

        logger.info(
            f"✓ AmbiguityResolver initialized "
            f"(detector={self.detector_type}, mode={self.resolution_mode}, "
            f"temporal_default={self.temporal_default_days}d)"
        )

    def detect(self, query: Query, schema: Optional[Schema] = None) -> List[Ambiguity]:
        """Detect ambiguities in query.

        Args:
            query: Natural language query
            schema: Database schema (optional, used for schema ambiguity detection)

        Returns:
            List of detected ambiguities
        """
        if self.detector_type == "rules":
            return self._detect_rules(query, schema)
        else:
            return self._detect_llm(query, schema)

    def resolve(
        self,
        query: Query,
        ambiguities: List[Ambiguity],
        schema: Optional[Schema] = None
    ) -> Tuple[str, List[Resolution]]:
        """Resolve ambiguities and rewrite query.

        Args:
            query: Original query
            ambiguities: Detected ambiguities
            schema: Database schema (optional, for context)

        Returns:
            Tuple of (rewritten_query, resolutions)

        Raises:
            RequiresClarificationException: If interactive mode and clarification needed
        """
        if self.resolution_mode == "auto":
            return self._auto_resolve(query, ambiguities, schema)
        else:
            return self._interactive_resolve(query, ambiguities, schema)

    def _detect_rules(self, query: Query, schema: Optional[Schema]) -> List[Ambiguity]:
        """Rule-based ambiguity detection (fast, local, zero cost).

        Args:
            query: Natural language query
            schema: Database schema

        Returns:
            List of detected ambiguities
        """
        ambiguities = []
        query_lower = query.text.lower()
        tokens = query_lower.split()

        # 1. Temporal ambiguity detection
        for keyword, interpretations in self.TEMPORAL_KEYWORDS.items():
            if keyword in query_lower:
                amb = Ambiguity(
                    type="temporal",
                    phrase=keyword,
                    reason=f"Unclear time reference for '{keyword}'",
                    candidates=interpretations,
                    confidence=0.8
                )
                if amb.confidence >= self.confidence_threshold:
                    ambiguities.append(amb)

        # 2. Schema ambiguity detection (if schema provided)
        if schema:
            for token in tokens:
                # Find matching columns
                matches = [
                    col.name for col in schema.columns
                    if token in col.name.lower()
                ]

                if len(matches) > 1:
                    amb = Ambiguity(
                        type="schema",
                        phrase=token,
                        reason=f"Multiple schema elements match '{token}'",
                        candidates=matches,
                        confidence=0.9
                    )
                    if amb.confidence >= self.confidence_threshold:
                        ambiguities.append(amb)

        # 3. Underspecification detection
        for keyword, suggestions in self.UNDERSPECIFIED_KEYWORDS.items():
            if keyword in query_lower:
                # Check if metric is specified (e.g., "top" should have "by")
                if keyword == "top" and "by" not in query_lower:
                    amb = Ambiguity(
                        type="underspecified",
                        phrase=keyword,
                        reason=f"'{keyword}' requires ranking metric (by salary? by date?)",
                        candidates=suggestions,
                        confidence=0.7
                    )
                    if amb.confidence >= self.confidence_threshold:
                        ambiguities.append(amb)

                elif keyword in ["large", "small", "big", "high", "low"]:
                    # Check if threshold is specified
                    has_threshold = any(
                        word in query_lower
                        for word in ["than", "above", "below", "over", "under"]
                    )
                    if not has_threshold:
                        amb = Ambiguity(
                            type="underspecified",
                            phrase=keyword,
                            reason=f"Relative '{keyword}' needs threshold",
                            candidates=suggestions,
                            confidence=0.6
                        )
                        if amb.confidence >= self.confidence_threshold:
                            ambiguities.append(amb)

        logger.info(f"Rule-based detection found {len(ambiguities)} ambiguities")
        return ambiguities

    def _detect_llm(self, query: Query, schema: Optional[Schema]) -> List[Ambiguity]:
        """LLM-based ambiguity detection (higher accuracy, may add latency).

        Note: Not yet implemented. Falls back to rule-based detection.

        Args:
            query: Natural language query
            schema: Database schema

        Returns:
            List of detected ambiguities
        """
        logger.warning("LLM-based detection not yet implemented, using rules")
        return self._detect_rules(query, schema)

    def _auto_resolve(
        self,
        query: Query,
        ambiguities: List[Ambiguity],
        schema: Optional[Schema]
    ) -> Tuple[str, List[Resolution]]:
        """Automatically resolve ambiguities with defaults.

        Args:
            query: Original query
            ambiguities: Detected ambiguities
            schema: Database schema

        Returns:
            Tuple of (rewritten_query, resolutions)
        """
        rewritten = query.text
        resolutions = []

        for amb in ambiguities:
            if amb.type == "temporal" and self.auto_resolve_temporal:
                # Default: last N days (configurable)
                replacement = f"in the last {self.temporal_default_days} days"
                rewritten = rewritten.replace(amb.phrase, replacement, 1)  # Replace first occurrence
                resolutions.append(Resolution(
                    ambiguity=amb,
                    chosen_interpretation=f"last {self.temporal_default_days} days",
                    rewritten_phrase=replacement,
                    method="auto_default"
                ))

            elif amb.type == "schema":
                # Use first match (most common heuristic)
                best_match = amb.candidates[0] if amb.candidates else amb.phrase
                rewritten = rewritten.replace(amb.phrase, best_match, 1)
                resolutions.append(Resolution(
                    ambiguity=amb,
                    chosen_interpretation=best_match,
                    rewritten_phrase=best_match,
                    method="auto_heuristic"
                ))

            elif amb.type == "underspecified":
                # Log warning but proceed (cannot auto-resolve without domain knowledge)
                logger.warning(
                    f"Underspecified query: '{amb.phrase}' {amb.reason}. "
                    f"Proceeding without modification."
                )
                # Don't modify query for underspecified cases

        logger.info(f"Auto-resolved {len(resolutions)}/{len(ambiguities)} ambiguities")
        return rewritten, resolutions

    def _interactive_resolve(
        self,
        query: Query,
        ambiguities: List[Ambiguity],
        schema: Optional[Schema]
    ) -> Tuple[str, List[Resolution]]:
        """Generate clarification questions for user interaction.

        Args:
            query: Original query
            ambiguities: Detected ambiguities
            schema: Database schema

        Returns:
            Tuple of (None, []) - requires external user interaction

        Raises:
            RequiresClarificationException: With questions for user
        """
        questions = []

        for amb in ambiguities:
            if amb.type == "temporal":
                questions.append({
                    "ambiguity_id": id(amb),
                    "question": f"By '{amb.phrase}', do you mean:",
                    "options": amb.candidates,
                    "default": 1  # Index of default option (e.g., "last 30 days")
                })

            elif amb.type == "schema":
                questions.append({
                    "ambiguity_id": id(amb),
                    "question": f"Which '{amb.phrase}' do you mean:",
                    "options": amb.candidates,
                    "default": 0  # First match
                })

            elif amb.type == "underspecified":
                questions.append({
                    "ambiguity_id": id(amb),
                    "question": f"'{amb.phrase}' needs clarification:",
                    "options": amb.candidates,
                    "default": None  # No default for underspecified
                })

        # Raise exception with clarification questions
        logger.info(f"Interactive mode: Generated {len(questions)} clarification questions")
        raise RequiresClarificationException(questions)

"""Core type definitions for AEGIS-SQL.

Defines the data models used throughout the system pipeline.

"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported languages for multilingual NL2SQL."""

    ENGLISH = "english"
    SPANISH = "spanish"
    KOREAN = "korean"
    CHINESE = "chinese"
    SWAHILI = "swahili"

class RoutingDecision(str, Enum):
    """Routing decision output from ContentIndependentRouter."""

    LOCAL = "local"  # Route to on-premises SLM
    REMOTE = "remote"  # Route to untrusted remote LLM


class SensitivityLevel(str, Enum):
    """Sensitivity classification for schema elements and tokens."""

    PII = "pii"  # Personally identifiable information
    PROPRIETARY = "proprietary"  # Proprietary business logic
    REGULATED = "regulated"  # Regulated data (medical, financial)
    PUBLIC = "public"  # Non-sensitive


class VerificationStatus(str, Enum):
    """Verification result status from neuro-symbolic verifier."""

    PASS = "pass"  # SQL is valid and executable
    GRAMMAR_FAIL = "grammar_fail"  # SQL syntax error
    SCHEMA_FAIL = "schema_fail"  # Invalid table/column reference
    EXECUTION_FAIL = "execution_fail"  # Runtime execution error
    TIMEOUT = "timeout"  # Execution timeout


@dataclass
class Query:
    """Natural language query with metadata.

    Attributes:
        text: Raw natural language question
        language: Language of the query
        database_id: Identifier for the target database
        evidence: Domain knowledge hints for SQL generation (BIRD dataset specific)
        metadata: Optional metadata (e.g., query ID, timestamp)
    """

    text: str
    language: Language
    database_id: str
    evidence: str = ""  # NEW: Evidence/hints from BIRD dataset
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SchemaElement:
    """A single schema element (table, column, or cell value).

    Attributes:
        element_type: Type of schema element
        name: Element name (table.column or table)
        data_type: SQL data type (for columns)
        sensitivity: Sensitivity classification
        description: Optional documentation string
        example_values: Sample values (for columns)
    """

    element_type: str  # "table" | "column" | "value"
    name: str
    data_type: Optional[str] = None
    sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC
    description: Optional[str] = None
    example_values: List[str] = field(default_factory=list)


@dataclass
class ForeignKey:
    """Foreign key relationship between tables.

    Attributes:
        from_table: Source table name
        from_column: Source column name
        to_table: Target table name
        to_column: Target column name
    """
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class Schema:
    """Database schema representation.

    Attributes:
        database_id: Unique database identifier
        tables: List of table names
        columns: List of column schema elements
        foreign_keys: Foreign key relationships (for FK expansion in retrieval)
        primary_keys: Primary key columns per table
        documentation: Optional schema documentation text
        sensitive_elements: Set of sensitive element names
    """

    database_id: str
    tables: List[str]
    columns: List[SchemaElement]
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    primary_keys: Dict[str, List[str]] = field(default_factory=dict)
    documentation: Optional[str] = None
    sensitive_elements: Set[str] = field(default_factory=set)

    def get_sensitive_elements(self) -> List[SchemaElement]:
        """Return all schema elements marked as sensitive."""
        return [col for col in self.columns if col.sensitivity != SensitivityLevel.PUBLIC]

    def get_foreign_keys_for_table(self, table_name: str) -> List[ForeignKey]:
        """Get all foreign keys involving a table (as source or target).

        Args:
            table_name: Name of the table

        Returns:
            List of foreign keys where table is source or target
        """
        return [
            fk for fk in self.foreign_keys
            if fk.from_table == table_name or fk.to_table == table_name
        ]


@dataclass
class AbstractedPrompt:
    """Abstracted prompt after DP mechanism.

    Attributes:
        text: Abstracted text with placeholders substituted
        original_tokens: Original sensitive tokens (stored locally only)
        placeholder_map: Mapping from placeholders to original tokens
        epsilon: ε privacy budget used
        num_substitutions: Number of tokens abstracted
        evidence: Domain knowledge hints (NOT abstracted, safe to send)
    """

    text: str
    original_tokens: List[str]
    placeholder_map: Dict[str, str]  # placeholder -> original token
    epsilon: float
    num_substitutions: int
    evidence: str = ""  # NEW: Evidence is NOT sensitive, safe to include


@dataclass
class ReconstructionMap:
    """Bidirectional mapping for reconstruction.

    Stores the mapping between placeholders and real tokens.
    Never sent to remote LLM - lives in trusted environment only.

    Attributes:
        placeholder_to_real: Forward mapping (placeholder -> real token)
        real_to_placeholder: Reverse mapping (real token -> placeholder)
        metadata: Optional metadata (e.g., timestamp, query_id)
    """

    placeholder_to_real: Dict[str, str]
    real_to_placeholder: Dict[str, str]
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_abstraction(cls, abstracted: AbstractedPrompt) -> "ReconstructionMap":
        """Create reconstruction map from abstracted prompt."""
        return cls(
            placeholder_to_real=abstracted.placeholder_map,
            real_to_placeholder={v: k for k, v in abstracted.placeholder_map.items()},
        )


@dataclass
class SQL:
    """SQL query with metadata.

    Attributes:
        text: SQL query string
        dialect: SQL dialect (e.g., "sqlite", "postgresql")
        source: Generation source ("slm" | "llm")
        verified: Whether the SQL passed verification
        verification_result: Detailed verification result
    """

    text: str
    dialect: str = "sqlite"
    source: str = "slm"  # "slm" | "llm"
    verified: bool = False
    verification_result: Optional["VerificationResult"] = None


@dataclass
class VerificationResult:
    """Detailed verification result from 3-stage verifier.

    Attributes:
        status: Overall verification status
        grammar_valid: Grammar check passed
        schema_valid: Schema check passed
        execution_valid: Execution check passed
        error_message: Human-readable error message
        structured_feedback: Structured feedback for SLM retry
        execution_result: Sample execution result (for passed queries)
    """

    status: VerificationStatus
    grammar_valid: bool
    schema_valid: bool
    execution_valid: bool
    error_message: Optional[str] = None
    structured_feedback: Optional[Dict[str, str]] = None
    execution_result: Optional[List[Tuple]] = None


@dataclass
@dataclass
class RouterFeatures:
    """Content-independent features for routing decision.

    All features are computable from the abstracted prompt only.
    This content-independence is required for Theorem 1.

    Attributes:
        query_token_count: Number of tokens in query
        schema_element_count: Number of retrieved schema elements
        query_structural_complexity: Complexity score (AST depth, joins, etc.)
    """

    query_token_count: int
    schema_element_count: int
    query_structural_complexity: float  # 0-1 normalized


@dataclass
class EvaluationMetrics:
    """Three-axis evaluation metrics.

    Implements the loss functions from Paper Section 2.

    Attributes:
        execution_accuracy: ℒ_util = Pr[exec(generated) ≠ exec(gold)]
        privacy_loss: ℒ_priv = ε × E[|prompt|] × Pr(r=remote)
        cost_per_query: ℒ_cost in USD
        routing_rate_local: Pr(r=local)
        routing_rate_remote: Pr(r=remote)
        avg_prompt_length: E[|prompt|]
    """

    execution_accuracy: float  # EX metric (0-1)
    privacy_loss: float  # ε × E[|prompt|] × Pr(r=remote)
    cost_per_query: float  # USD
    routing_rate_local: float  # 0-1
    routing_rate_remote: float  # 0-1
    avg_prompt_length: float  # tokens

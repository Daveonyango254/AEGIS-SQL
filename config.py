"""Configuration management with Pydantic models.

Loads config.yaml and .env variables with validation.

"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import yaml

from aegis_types import Language


class EmbeddingConfig(BaseModel):
    """Embedding model configuration for Query Planner agent."""

    model: str = Field(default="BAAI/bge-m3", description="HuggingFace model ID for multilingual embeddings")
    device: str = Field(default="cuda", description="Device (cuda/cpu)")
    cache_dir: str = Field(default="~/.cache/huggingface", description="Model cache directory")
    adapter_path: Optional[str] = Field(default=None, description="LoRA adapter path for fine-tuned embeddings")
    batch_size: int = Field(default=32, description="Batch size for embedding computation")
    max_length: int = Field(default=512, description="Maximum sequence length")




class SLMConfig(BaseModel):
    """Small language model configuration (local FSLM)."""

    model: str = Field(default="cycloneboy/SLM-SQL-1.5B", description="HuggingFace model ID for local SLM")
    device: str = Field(default="auto", description="Device map (auto/cuda/cpu)")
    cache_dir: str = Field(default="~/.cache/huggingface", description="Model cache directory")
    hf_token: str = Field(default="${HF_HUB_TOKEN}", description="HuggingFace Hub token from .env")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    torch_dtype: str = Field(default="float16", description="Torch dtype (float16/bfloat16/float32)")
    trust_remote_code: bool = Field(default=True, description="Trust remote code from HuggingFace")
    adapter_path: Optional[str] = Field(default=None, description="LoRA adapter path for fine-tuned SLM")

    # --- Harness optimization knobs (all default-on, individually toggleable) ---
    max_input_length: int = Field(
        default=8192,
        description="Max prompt tokens (raised from 2048 so large schemas don't truncate the question)",
    )
    use_chat_template: bool = Field(
        default=True,
        description="Apply the tokenizer chat template (system+user) for instruct models",
    )
    num_candidates: int = Field(
        default=3,
        description="Number of candidates for execution-guided self-consistency (1 = single greedy decode)",
    )
    selection_temperature: float = Field(
        default=0.8,
        description="Sampling temperature used for the non-greedy candidates",
    )
    enable_value_grounding: bool = Field(
        default=True,
        description="Inject sampled DB values / value-linking hints into the prompt",
    )
    enable_cast_fix: bool = Field(
        default=True,
        description="Wrap division numerators in CAST(... AS REAL) to fix integer-division ratio bugs",
    )
    retrieval_top_k: int = Field(
        default=80,
        description="Number of schema columns retrieved before FK expansion",
    )


class LLMConfig(BaseModel):
    """Large language model configuration for remote fallback (FLLM)."""

    provider: str = Field(default="openai", description="LLM provider (openai/anthropic)")
    model: str = Field(default="gpt-4o", description="LLM model name")
    api_key: str = Field(default="${OPENAI_API_KEY}", description="API key from environment")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries on failure")
    retry_delay: float = Field(default=1.0, description="Initial delay between retries (exponential backoff)")
    # NOTE: the remote path makes exactly one API call per query (single
    # candidate); execution-guided multi-candidate selection is local-SLM-only,
    # so the USD cost is one request per remote query.


class SensitivityPolicyConfig(BaseModel):
    """Sensitivity policy configuration."""

    pii: bool = Field(default=True, description="Protect PII")
    proprietary: bool = Field(default=True, description="Protect proprietary schema")
    regulated: bool = Field(default=True, description="Protect regulated data")


class PrivacyConfig(BaseModel):
    """Privacy configuration for DP abstraction."""

    epsilon: float = Field(default=1.0, description="ε privacy budget (set to 0 to disable abstraction)")
    sensitivity_policy: SensitivityPolicyConfig = Field(
        default_factory=SensitivityPolicyConfig
    )
    placeholder_vocab_size: int = Field(
        default=100, description="Number of semantic placeholders in V_abs"
    )
    abstraction_enabled: bool = Field(
        default=True, description="Enable DP abstraction for remote queries"
    )
    value_aware_abstraction: bool = Field(
        default=True,
        description="Only abstract value-like tokens (proper nouns/literals), never generic "
        "schema-vocabulary words like 'name'/'price'. Prevents over-abstraction that destroys remote accuracy.",
    )
    reconstruction_enabled: bool = Field(
        default=True, description="Enable reconstruction after remote generation (must match abstraction_enabled)"
    )

    @field_validator("epsilon")
    @classmethod
    def validate_epsilon(cls, v: float) -> float:
        """Validate epsilon is non-negative (0 means abstraction disabled)."""
        if v < 0:
            raise ValueError("Epsilon must be non-negative (0 disables abstraction)")
        return v

    @field_validator("reconstruction_enabled")
    @classmethod
    def validate_reconstruction_matches_abstraction(cls, v: bool, info) -> bool:
        """Validate that reconstruction_enabled matches abstraction_enabled."""
        abstraction_enabled = info.data.get("abstraction_enabled", True)
        if v != abstraction_enabled:
            raise ValueError(
                "reconstruction_enabled must match abstraction_enabled "
                "(both on or both off)"
            )
        return v


class CostConfig(BaseModel):
    """Cost configuration and budgets."""

    budget_per_query: float = Field(default=0.01, description="USD budget per query")
    remote_token_cost: float = Field(
        default=0.000015, description="Cost per token for remote LLM"
    )
    local_compute_cost: float = Field(
        default=0.0001, description="Fixed cost per local SLM inference"
    )


class RouterConfig(BaseModel):
    """Content-independent router configuration."""

    threshold_complexity: float = Field(
        default=0.7, description="Routing decision threshold (0-1): queries with complexity >= threshold route to remote"
    )
    features: List[str] = Field(
        default=[
            "query_token_count",
            "schema_element_count",
            "query_structural_complexity",
        ],
        description="Content-independent features for routing",
    )
    force_local: bool = Field(
        default=False, description="Force all queries to local FSLM (overrides threshold)"
    )
    force_remote: bool = Field(
        default=False, description="Force all queries to remote FLLM (overrides threshold)"
    )

    @field_validator("threshold_complexity")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v

    @field_validator("force_remote")
    @classmethod
    def validate_force_flags(cls, v: bool, info) -> bool:
        """Validate that force_local and force_remote are not both True."""
        if v and info.data.get("force_local", False):
            raise ValueError("Cannot set both force_local and force_remote to True")
        return v


class VerifierConfig(BaseModel):
    """Neuro-symbolic verifier configuration."""

    grammar_check: bool = Field(default=True, description="Enable grammar verification")
    schema_check: bool = Field(default=True, description="Enable schema verification")
    execution_check_slm: bool = Field(
        default=True, description="Enable execution verification for SLM outputs"
    )
    execution_check_llm: bool = Field(
        default=True, description="Enable execution verification for LLM outputs"
    )
    sample_size: int = Field(default=100, description="Number of rows for execution verification")
    timeout_seconds: int = Field(default=5, description="Execution timeout per query")
    max_repair_attempts: int = Field(
        default=1,
        description="Max self-correction regenerations when verification fails (0 disables the repair loop)",
    )
    repair_on_empty: bool = Field(
        default=True,
        description="Treat an executed-but-empty result as a soft execution failure "
        "to trigger one value-aware repair (targets literal-mismatch errors).",
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    bird_dev_path: str = Field(default="./data/bird_dev", description="Path to BIRD-dev benchmark")
    output_dir: str = Field(default="./results", description="Evaluation results directory")
    seed: int = Field(default=42, description="Fixed seed for reproducibility")
    metrics: List[str] = Field(
        default=["execution_accuracy", "ves", "privacy_loss", "cost_per_query", "latency"],
        description="Metrics to compute",
    )


class AmbiguityConfig(BaseModel):
    """Query ambiguity resolution configuration.

    Detects and resolves ambiguous queries before SQL generation.

    Attributes:
        enabled: Enable ambiguity detection and resolution (default: False)
        detector_type: Detection method - "rules" (fast, local) or "llm" (accurate)
        resolution_mode: Resolution strategy - "auto" (use defaults) or "interactive" (ask user)
        auto_resolve_temporal: Automatically resolve temporal ambiguities
        temporal_default_days: Default days for "recent" queries (default: 30)
        confidence_threshold: Minimum confidence to flag ambiguity (0-1)
    """

    enabled: bool = Field(default=False, description="Enable ambiguity detection (disabled by default)")
    detector_type: str = Field(default="rules", description="Detection method: 'rules' or 'llm'")
    resolution_mode: str = Field(default="auto", description="Resolution mode: 'auto' or 'interactive'")
    auto_resolve_temporal: bool = Field(default=True, description="Auto-resolve temporal ambiguities")
    temporal_default_days: int = Field(default=30, description="Default days for 'recent' queries")
    confidence_threshold: float = Field(default=0.6, description="Min confidence to flag ambiguity (0-1)")

    @field_validator("detector_type")
    @classmethod
    def validate_detector_type(cls, v: str) -> str:
        """Validate detector type is valid."""
        if v not in ["rules", "llm"]:
            raise ValueError("detector_type must be 'rules' or 'llm'")
        return v

    @field_validator("resolution_mode")
    @classmethod
    def validate_resolution_mode(cls, v: str) -> str:
        """Validate resolution mode is valid."""
        if v not in ["auto", "interactive"]:
            raise ValueError("resolution_mode must be 'auto' or 'interactive'")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validate confidence threshold is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration.

    Structured logging tracks key events:
    - SCHEMA_EXTRACTION_COMPLETE: Schema retrieval finished
    - ROUTED_TO_LOCAL / ROUTED_TO_REMOTE: Routing decision made
    - ABSTRACTION_APPLIED: DP abstraction completed
    - GENERATION_COMPLETE: SQL generated
    - RECONSTRUCTION_APPLIED: Placeholder reconstruction done
    - VERIFICATION_PASSED / VERIFICATION_FAILED: Verification outcome
    - FAILED_GRAMMAR_VERIFICATION: Grammar check failed
    - FAILED_SCHEMA_VERIFICATION: Schema check failed
    - FAILED_EXECUTION_VERIFICATION: Execution check failed
    """

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log format (json or text)")
    file: str = Field(default="./logs/aegis_sql.log", description="Log file path")
    track_routing: bool = Field(default=True, description="Log routing decisions")
    track_privacy: bool = Field(default=True, description="Log privacy metrics")
    track_verification: bool = Field(default=True, description="Log verification outcomes")
    track_latency: bool = Field(default=True, description="Log latency measurements")


class AEGISConfig(BaseSettings):
    """Main AEGIS-SQL configuration.

    Loads from config.yaml and .env files with validation.

    Example:
        >>> config = AEGISConfig.from_yaml("config.yaml")
        >>> print(config.language)
        Language.ENGLISH
    """

    language: Language = Field(default=Language.ENGLISH, description="Query language")
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    slm: SLMConfig = Field(default_factory=SLMConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    ambiguity: AmbiguityConfig = Field(default_factory=AmbiguityConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AEGISConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml

        Returns:
            Validated AEGISConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        # Load .env file first
        from dotenv import load_dotenv
        load_dotenv()

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Substitute environment variables in string values
        config_dict = cls._substitute_env_vars(config_dict)

        return cls(**config_dict)

    @staticmethod
    def _substitute_env_vars(config_dict: dict) -> dict:
        """Recursively substitute ${VAR} with environment variables.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Dictionary with environment variables substituted
        """

        def substitute(value: any) -> any:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: substitute(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute(v) for v in value]
            return value

        return substitute(config_dict)

    def validate_paths(self) -> None:
        """Validate that all specified paths exist or can be created.

        Raises:
            FileNotFoundError: If required paths don't exist
        """
        # Create output directories if they don't exist
        Path(self.evaluation.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.file).parent.mkdir(parents=True, exist_ok=True)

        # Expand cache directories
        embedding_cache = Path(self.embedding.cache_dir).expanduser()
        slm_cache = Path(self.slm.cache_dir).expanduser()
        embedding_cache.mkdir(parents=True, exist_ok=True)
        slm_cache.mkdir(parents=True, exist_ok=True)

        # Check adapter paths if specified
        if self.embedding.adapter_path and not Path(self.embedding.adapter_path).exists():
            raise FileNotFoundError(f"Embedding adapter not found: {self.embedding.adapter_path}")
        if self.slm.adapter_path and not Path(self.slm.adapter_path).exists():
            raise FileNotFoundError(f"SLM adapter not found: {self.slm.adapter_path}")

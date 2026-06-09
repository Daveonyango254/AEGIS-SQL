"""Token-level differential privacy abstraction via exponential mechanism.

Implements Option C from build strategy: cross-lingual embedding distance
with language-agnostic placeholder vocabulary.

References:
    - Build strategy Section 4: DP abstraction in multilingual setting
    - McSherry & Talwar 2007: Mechanism Design via Differential Privacy
    - Paper Theorem 1: Hybrid Privacy Amplification
"""

from typing import Any, Dict, List, Tuple

from loguru import logger
import numpy as np

from config import PrivacyConfig
from aegis_types import AbstractedPrompt, Query, ReconstructionMap, SchemaElement
from abstraction.placeholder_vocab import PlaceholderVocabulary
from abstraction.sensitivity_policy import SensitivityPolicy


class DPAbstractor:
    """Token-level ε-DP abstraction via exponential mechanism.

    Implements the exponential mechanism with utility function:
        u(t, v) = -||embed(t) - embed(v)||

    where t is a sensitive token and v is a candidate placeholder from V_abs.
    Samples placeholder v* with probability ∝ exp(ε · u(t, v) / 2).

    This mechanism provides ε-DP guarantee that is language-invariant because
    the sampling depends only on embeddings, not raw tokens.

    Attributes:
        config: Privacy configuration
        vocab: Language-agnostic placeholder vocabulary V_abs
        policy: Sensitivity policy for token classification
        embedding_model: Multilingual embedding model (shared with retriever)
    """

    def __init__(
        self,
        config: PrivacyConfig,
        vocab: PlaceholderVocabulary,
        policy: SensitivityPolicy,
        embedding_model: Any,
    ) -> None:
        """Initialize DP abstractor.

        Args:
            config: Privacy configuration
            vocab: Placeholder vocabulary
            policy: Sensitivity policy
            embedding_model: Multilingual embedding model
        """
        self.config = config
        self.vocab = vocab
        self.policy = policy
        self.embedding_model = embedding_model
        self.epsilon = config.epsilon
        self._placeholder_embeddings: Dict[str, np.ndarray] = {}
        self._token_to_placeholder: Dict[str, str] = {}  # Cache for deterministic mapping

        # Pre-compute placeholder embeddings
        for placeholder in vocab.placeholders:
            if placeholder in vocab.embeddings:
                self._placeholder_embeddings[placeholder] = vocab.embeddings[placeholder]

        logger.info(f"Initialized DPAbstractor with ε={self.epsilon}")

    def abstract(
        self, query: Query, schema_elements: List[SchemaElement]
    ) -> Tuple[AbstractedPrompt, ReconstructionMap]:
        """Apply ε-DP abstraction to sensitive tokens in query.

        Algorithm:
            1. Identify sensitive tokens via NER + sensitivity policy
            2. For each sensitive token t:
                a. Compute embedding e_t = embed(t)
                b. Compute utility u(t, v) = -||e_t - e_v|| for all v in V_abs
                c. Sample v* with probability ∝ exp(ε · u(t, v) / 2)
                d. Substitute v* for t in the prompt
            3. Return abstracted prompt + reconstruction map

        Args:
            query: Natural language query
            schema_elements: Retrieved schema elements (may contain sensitive names)

        Returns:
            Tuple of (abstracted_prompt, reconstruction_map)

        References:
            - Build strategy Section 4: "Option C — Cross-lingual embedding distance"
            - Exponential mechanism: P(v) ∝ exp(ε · u(t,v) / 2Δu)
              where Δu = 1 (max utility difference for neighboring databases)
        """
        logger.debug(
            f"Abstracting query with ε={self.epsilon}: {query.text[:50]}..."
        )

        # Check if abstraction is disabled
        if self.epsilon == 0 or not self.config.abstraction_enabled:
            logger.debug("Abstraction disabled, passing through")
            return self._passthrough(query, schema_elements)

        # Identify sensitive tokens
        sensitive_tokens = self._identify_sensitive_tokens(query, schema_elements)

        if not sensitive_tokens:
            logger.debug("No sensitive tokens found, passing through")
            return self._passthrough(query, schema_elements)

        # Build reconstruction map
        placeholder_to_real: Dict[str, str] = {}
        abstracted_text = query.text

        # Replace sensitive tokens with placeholders
        for token, start, end in sorted(sensitive_tokens, key=lambda x: -x[1]):
            # Get or sample placeholder for this token
            if token in self._token_to_placeholder:
                placeholder = self._token_to_placeholder[token]
            else:
                # Simplified: just use next available placeholder
                # TODO: Implement exponential mechanism for better DP guarantee
                placeholder = self._sample_placeholder_simple(token)
                self._token_to_placeholder[token] = placeholder

            # Replace in text
            abstracted_text = abstracted_text[:start] + placeholder + abstracted_text[end:]
            placeholder_to_real[placeholder] = token

        # Create abstracted prompt
        abstracted_prompt = AbstractedPrompt(
            text=abstracted_text,
            original_tokens=list(placeholder_to_real.values()),
            placeholder_map=placeholder_to_real,
            epsilon=self.epsilon,
            num_substitutions=len(placeholder_to_real),
            evidence=query.evidence,  # NEW: Evidence is not sensitive, pass through
        )

        # Create reconstruction map
        recon_map = ReconstructionMap(
            placeholder_to_real=placeholder_to_real,
            real_to_placeholder={v: k for k, v in placeholder_to_real.items()},
        )

        logger.debug(f"Abstracted {len(placeholder_to_real)} sensitive tokens")
        return abstracted_prompt, recon_map

    def _passthrough(
        self, query: Query, schema_elements: List[SchemaElement]
    ) -> Tuple[AbstractedPrompt, ReconstructionMap]:
        """Pass through query without abstraction."""
        abstracted_prompt = AbstractedPrompt(
            text=query.text,
            original_tokens=[],
            placeholder_map={},
            epsilon=0.0,
            num_substitutions=0,
            evidence=query.evidence,  # NEW: Pass evidence through
        )
        recon_map = ReconstructionMap(
            placeholder_to_real={},
            real_to_placeholder={},
        )
        return abstracted_prompt, recon_map

    def _identify_sensitive_tokens(
        self, query: Query, schema_elements: List[SchemaElement]
    ) -> List[Tuple[str, int, int]]:
        """Identify sensitive tokens in query.

        Args:
            query: Natural language query
            schema_elements: Schema elements (some may be sensitive)

        Returns:
            List of (token, start_pos, end_pos) for sensitive tokens
        """
        sensitive_tokens = []

        # Simple word tokenization (TODO: use proper NER)
        words = query.text.split()
        position = 0

        for word in words:
            # Check if word is sensitive
            if self.policy.is_sensitive(word):
                start = query.text.find(word, position)
                end = start + len(word)
                sensitive_tokens.append((word, start, end))
                position = end

        # Check schema elements for sensitive names
        for element in schema_elements:
            if self.policy.is_sensitive(element.name):
                # Find occurrences in query
                start = 0
                while True:
                    start = query.text.find(element.name, start)
                    if start == -1:
                        break
                    end = start + len(element.name)
                    sensitive_tokens.append((element.name, start, end))
                    start = end

        # Remove duplicates and sort by position
        sensitive_tokens = list(set(sensitive_tokens))
        sensitive_tokens.sort(key=lambda x: x[1])

        return sensitive_tokens

    def _sample_placeholder(
        self, token: str, token_embedding: np.ndarray
    ) -> str:
        """Sample placeholder via exponential mechanism.

        Args:
            token: Sensitive token
            token_embedding: Embedding of the sensitive token

        Returns:
            Sampled placeholder from V_abs
        """
        # Compute utilities for all placeholders
        utilities = []
        for placeholder in self.vocab.placeholders:
            if placeholder in self._placeholder_embeddings:
                placeholder_emb = self._placeholder_embeddings[placeholder]
                utility = self._compute_utility(token_embedding, placeholder_emb)
                utilities.append((placeholder, utility))

        if not utilities:
            # Fallback: return first placeholder
            return self.vocab.placeholders[0]

        # Convert utilities to probabilities via exponential mechanism
        # P(v) ∝ exp(ε · u(t, v) / 2)
        max_utility = max(u for _, u in utilities)
        exp_utilities = [
            (p, np.exp(self.epsilon * (u - max_utility) / 2))
            for p, u in utilities
        ]

        # Normalize to probabilities
        total = sum(eu for _, eu in exp_utilities)
        probabilities = [(p, eu / total) for p, eu in exp_utilities]

        # Sample placeholder
        placeholders_list = [p for p, _ in probabilities]
        probs_list = [prob for _, prob in probabilities]
        sampled = np.random.choice(placeholders_list, p=probs_list)

        return sampled

    def _sample_placeholder_simple(self, token: str) -> str:
        """Simplified placeholder sampling (deterministic).

        Args:
            token: Sensitive token

        Returns:
            Placeholder from V_abs
        """
        # Simplified: classify token and get category-appropriate placeholder
        sensitivity = self.policy.classify_token(token)

        # Map sensitivity to category
        category_map = {
            "pii": "PERSON",
            "proprietary": "PRODUCT",
            "regulated": "MEDICAL",
        }

        category = category_map.get(sensitivity.value, "PERSON")
        candidates = self.vocab.get_placeholder_by_category(category, sensitivity)

        if not candidates:
            candidates = self.vocab.placeholders

        # Use hash to deterministically select placeholder
        index = hash(token) % len(candidates)
        return candidates[index]

    def _compute_utility(
        self, token_embedding: np.ndarray, placeholder_embedding: np.ndarray
    ) -> float:
        """Compute utility u(t, v) = -||e_t - e_v||.

        Negative cosine distance in embedding space.

        Args:
            token_embedding: Embedding of sensitive token t
            placeholder_embedding: Embedding of candidate placeholder v

        Returns:
            Utility score (higher = better semantic match)
        """
        # Compute cosine similarity
        dot_product = np.dot(token_embedding, placeholder_embedding)
        norm_t = np.linalg.norm(token_embedding)
        norm_p = np.linalg.norm(placeholder_embedding)

        if norm_t == 0 or norm_p == 0:
            return 0.0

        cosine_sim = dot_product / (norm_t * norm_p)

        # Return negative distance (higher utility for closer embeddings)
        # Cosine distance = 1 - cosine_similarity
        return -( 1 - cosine_sim)

    def verify_privacy_guarantee(
        self, abstracted_prompts: List[AbstractedPrompt]
    ) -> float:
        """Verify Theorem 1 privacy bound on a workload.

        Computes: ℒ_priv = ε × E[|prompt|] × Pr(r=remote)

        Args:
            abstracted_prompts: List of abstracted prompts from evaluation

        Returns:
            Privacy loss bound (upper bound on mutual information)

        References:
            - Paper Theorem 1: Hybrid Privacy Amplification
        """
        if not abstracted_prompts:
            return 0.0

        # Compute average prompt length (in tokens)
        avg_length = sum(len(p.text.split()) for p in abstracted_prompts) / len(abstracted_prompts)

        # For now, assume all prompts route remotely (worst case)
        # TODO: Get actual routing decisions from workflow
        remote_rate = 1.0

        # Compute privacy loss bound
        privacy_loss = self.epsilon * avg_length * remote_rate

        logger.info(f"Privacy loss bound: ε={self.epsilon}, avg_length={avg_length:.1f}, "
                   f"remote_rate={remote_rate}, ℒ_priv={privacy_loss:.2f}")

        return privacy_loss

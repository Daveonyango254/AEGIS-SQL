"""Three-axis metrics calculator.

Implements ℒ_util, ℒ_priv, ℒ_cost from Paper Section 2.

Sprint Assignment: Sprint 5-6
References: Paper Section 2 (Problem Setup)
"""

from typing import List

from loguru import logger

from aegis_types import (
    SQL,
    AbstractedPrompt,
    EvaluationMetrics,
    RoutingDecision,
)


class MetricsCalculator:
    """Calculator for the three-axis loss functions.

    Computes:
        - ℒ_util: Execution accuracy (EX metric)
        - ℒ_priv: ε × E[|prompt|] × Pr(r=remote)
        - ℒ_cost: Average cost per query in USD

    Attributes:
        epsilon: Privacy budget
        remote_token_cost: Cost per token for remote LLM
        local_cost: Fixed cost for local SLM
    """

    def __init__(
        self, epsilon: float, remote_token_cost: float, local_cost: float
    ) -> None:
        """Initialize metrics calculator.

        Args:
            epsilon: Privacy budget ε
            remote_token_cost: Cost per token for remote LLM (USD)
            local_cost: Fixed cost for local SLM (USD)
        """
        self.epsilon = epsilon
        self.remote_token_cost = remote_token_cost
        self.local_cost = local_cost
        logger.info(
            f"Initialized MetricsCalculator with ε={epsilon}, "
            f"remote_cost={remote_token_cost}, local_cost={local_cost}"
        )

    def compute_execution_accuracy(
        self, generated_sqls: List[SQL], gold_sqls: List[SQL]
    ) -> float:
        """Compute execution accuracy (ℒ_util).

        Args:
            generated_sqls: Generated SQL queries
            gold_sqls: Ground truth SQL queries

        Returns:
            Execution accuracy (0-1, higher is better)

        TODO (Sprint 5):
            - Execute both generated and gold SQL on database
            - Compare result sets (exact match or set equivalence)
            - Return proportion of queries with matching results
            - Handle execution errors (count as incorrect)

        References:
            - Paper: ℒ_util = Pr[exec(generated) ≠ exec(gold)]
            - Return 1 - ℒ_util (accuracy, not loss)
        """
        logger.debug(
            f"Computing execution accuracy for {len(generated_sqls)} queries"
        )
        raise NotImplementedError("Sprint 5 implementation")

    def compute_privacy_loss(
        self,
        abstracted_prompts: List[AbstractedPrompt],
        routing_decisions: List[RoutingDecision],
    ) -> float:
        """Compute privacy loss (ℒ_priv).

        Args:
            abstracted_prompts: Abstracted prompts from DP mechanism
            routing_decisions: Routing decisions for each query

        Returns:
            Privacy loss bound: ε × E[|prompt|] × Pr(r=remote)

        References:
            - Paper Theorem 1: Hybrid Privacy Amplification
            - This is an upper bound on mutual information I(S_sensitive; LLM_input)
        """
        logger.debug(
            f"Computing privacy loss for {len(abstracted_prompts)} queries"
        )

        if len(abstracted_prompts) == 0 or len(routing_decisions) == 0:
            return 0.0

        # Compute Pr(r=remote): fraction of queries routed to remote
        num_remote = sum(1 for r in routing_decisions if r == RoutingDecision.REMOTE)
        pr_remote = num_remote / len(routing_decisions)

        # If no queries routed to remote, privacy loss is 0
        if pr_remote == 0:
            logger.info("No queries routed to remote → Privacy loss = 0.0")
            return 0.0

        # Compute E[|prompt|]: average prompt length (only for remote queries)
        remote_prompts = [
            p for p, r in zip(abstracted_prompts, routing_decisions)
            if r == RoutingDecision.REMOTE and p is not None
        ]

        if len(remote_prompts) == 0:
            # No abstracted prompts for remote queries (shouldn't happen)
            avg_prompt_len = 0.0
        else:
            # Count tokens (split by whitespace as approximation)
            prompt_lengths = [len(p.text.split()) for p in remote_prompts]
            avg_prompt_len = sum(prompt_lengths) / len(prompt_lengths)

        # Privacy loss: ε × E[|prompt|] × Pr(r=remote)
        privacy_loss = self.epsilon * avg_prompt_len * pr_remote

        logger.info(
            f"Privacy loss: ε={self.epsilon}, E[|prompt|]={avg_prompt_len:.1f}, "
            f"Pr(remote)={pr_remote:.3f} → ℒ_priv={privacy_loss:.4f}"
        )

        return privacy_loss

    def compute_cost_per_query(
        self,
        abstracted_prompts: List[AbstractedPrompt],
        routing_decisions: List[RoutingDecision],
        generated_sqls: List[SQL],
    ) -> float:
        """Compute average cost per query (ℒ_cost).

        Args:
            abstracted_prompts: Abstracted prompts (for token counting)
            routing_decisions: Routing decisions
            generated_sqls: Generated SQLs (for completion token counting)

        Returns:
            Average cost per query in USD

        References:
            - Paper: ℒ_cost = E[c_token · |prompt| · 𝟙[r=remote]] + c_local · 𝟙[r=local]
        """
        logger.debug(
            f"Computing cost for {len(routing_decisions)} queries"
        )

        if len(routing_decisions) == 0:
            return 0.0

        total_cost = 0.0

        for i, routing in enumerate(routing_decisions):
            if routing == RoutingDecision.LOCAL:
                # Local SLM: fixed cost
                cost = self.local_cost
            else:
                # Remote LLM: cost based on token count
                # Prompt tokens
                if i < len(abstracted_prompts) and abstracted_prompts[i] is not None:
                    prompt_tokens = len(abstracted_prompts[i].text.split())
                else:
                    prompt_tokens = 0

                # Completion tokens
                if i < len(generated_sqls) and generated_sqls[i] is not None:
                    completion_tokens = len(generated_sqls[i].text.split())
                else:
                    completion_tokens = 0

                # Total cost for this query
                cost = (prompt_tokens + completion_tokens) * self.remote_token_cost

            total_cost += cost

        avg_cost = total_cost / len(routing_decisions)

        logger.info(
            f"Average cost per query: ${avg_cost:.6f} USD "
            f"(total: ${total_cost:.4f} for {len(routing_decisions)} queries)"
        )

        return avg_cost

    def compute_all_metrics(
        self,
        generated_sqls: List[SQL],
        gold_sqls: List[SQL],
        abstracted_prompts: List[AbstractedPrompt],
        routing_decisions: List[RoutingDecision],
    ) -> EvaluationMetrics:
        """Compute all three-axis metrics.

        Args:
            generated_sqls: Generated SQL queries
            gold_sqls: Ground truth SQL queries
            abstracted_prompts: Abstracted prompts
            routing_decisions: Routing decisions

        Returns:
            EvaluationMetrics with all computed metrics
        """
        logger.info(
            f"Computing all metrics for {len(generated_sqls)} queries"
        )

        # Compute privacy loss
        privacy_loss = self.compute_privacy_loss(abstracted_prompts, routing_decisions)

        # Compute cost
        cost = self.compute_cost_per_query(
            abstracted_prompts, routing_decisions, generated_sqls
        )

        # Compute routing rates
        num_local = sum(1 for r in routing_decisions if r == RoutingDecision.LOCAL)
        num_remote = len(routing_decisions) - num_local
        pr_local = num_local / len(routing_decisions) if routing_decisions else 0.0
        pr_remote = num_remote / len(routing_decisions) if routing_decisions else 0.0

        # Compute average prompt length (for all queries)
        if abstracted_prompts:
            valid_prompts = [p for p in abstracted_prompts if p is not None]
            if valid_prompts:
                avg_prompt_len = sum(len(p.text.split()) for p in valid_prompts) / len(valid_prompts)
            else:
                avg_prompt_len = 0.0
        else:
            avg_prompt_len = 0.0

        # Note: execution_accuracy is computed separately by EX evaluator
        # We don't compute it here to avoid duplicate work

        metrics = EvaluationMetrics(
            execution_accuracy=None,  # Filled by EX evaluator
            privacy_loss=privacy_loss,
            cost_per_query=cost,
            pr_local=pr_local,
            pr_remote=pr_remote,
            avg_prompt_length=avg_prompt_len,
        )

        logger.info("✓ All metrics computed successfully")
        return metrics

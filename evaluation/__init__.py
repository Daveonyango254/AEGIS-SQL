"""Evaluation metrics and benchmarking.

Implements the three-axis loss functions:
- ℒ_util: Execution accuracy (EX metric on BIRD-dev)
- ℒ_priv: Privacy loss bound (ε × E[|prompt|] × Pr(r=remote))
- ℒ_cost: Per-query inference cost

"""

"""Differential privacy abstraction module.

Implements token-level ε-DP via exponential mechanism with cross-lingual
embedding distance utility function (Option C from build strategy).

Sprint Assignment: Sprint 3
References:
    - Build strategy Section 4
    - Paper: McSherry & Talwar 2007 (Exponential Mechanism)
"""

from abstraction.dp_abstractor import DPAbstractor
from abstraction.sensitivity_policy import SensitivityPolicy
from abstraction.placeholder_vocab import PlaceholderVocabulary
from abstraction.reconstruction import ReconstructionModule

__all__ = [
    "DPAbstractor",
    "SensitivityPolicy",
    "PlaceholderVocabulary",
    "ReconstructionModule",
]

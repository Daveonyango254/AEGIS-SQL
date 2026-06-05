"""Sensitivity policy for classifying tokens and schema elements.

Implements policy Π from the paper: PII, proprietary, regulated classifications.

References: Paper Section 2 (Problem Setup)
"""

from typing import List, Set

from config import SensitivityPolicyConfig
from aegis_types import SchemaElement, SensitivityLevel


class SensitivityPolicy:
    """Policy for classifying sensitive tokens and schema elements.

    Attributes:
        config: Sensitivity policy configuration
        pii_patterns: Patterns for PII detection
        proprietary_keywords: Keywords indicating proprietary data
        regulated_keywords: Keywords indicating regulated data
    """

    def __init__(self, config: SensitivityPolicyConfig) -> None:
        """Initialize sensitivity policy.

        Args:
            config: Sensitivity policy configuration
        """
        self.config = config

        # PII patterns (common sensitive column names/patterns)
        self.pii_patterns: List[str] = [
            "ssn", "social_security", "email", "phone", "address",
            "name", "patient", "person", "customer", "user",
            "firstname", "lastname", "birthdate", "dob", "age",
            "gender", "race", "ethnicity", "passport", "license",
            "credit_card", "account_number", "password", "pin"
        ]

        # Proprietary keywords (business-specific data)
        self.proprietary_keywords: Set[str] = {
            "revenue", "profit", "salary", "wage", "bonus", "commission",
            "price", "cost", "margin", "budget", "forecast",
            "product_code", "sku", "inventory", "trade_secret",
            "strategy", "roadmap", "internal", "confidential"
        }

        # Regulated keywords (healthcare, finance)
        self.regulated_keywords: Set[str] = {
            "diagnosis", "prescription", "medication", "treatment",
            "medical", "health", "hipaa", "phi", "icd",
            "account", "balance", "transaction", "payment",
            "loan", "mortgage", "credit", "debit", "banking"
        }

    def classify_token(self, token: str, context: str = "") -> SensitivityLevel:
        """Classify a single token's sensitivity level.

        Args:
            token: Token to classify
            context: Surrounding context (for disambiguation)

        Returns:
            Sensitivity classification
        """
        token_lower = token.lower()

        # Check PII patterns
        if self.config.pii:
            for pattern in self.pii_patterns:
                if pattern in token_lower:
                    return SensitivityLevel.PII

        # Check regulated keywords
        if self.config.regulated:
            for keyword in self.regulated_keywords:
                if keyword in token_lower:
                    return SensitivityLevel.REGULATED

        # Check proprietary keywords
        if self.config.proprietary:
            for keyword in self.proprietary_keywords:
                if keyword in token_lower:
                    return SensitivityLevel.PROPRIETARY

        # Default to public
        return SensitivityLevel.PUBLIC

    def classify_schema_element(self, element: SchemaElement) -> SensitivityLevel:
        """Classify a schema element's sensitivity.

        Args:
            element: Schema element (table, column, or value)

        Returns:
            Sensitivity classification
        """
        # Use existing classification if already marked
        if element.sensitivity != SensitivityLevel.PUBLIC:
            return element.sensitivity

        # Classify based on element name
        name_sensitivity = self.classify_token(element.name)
        if name_sensitivity != SensitivityLevel.PUBLIC:
            return name_sensitivity

        # Check description if available
        if element.description:
            desc_sensitivity = self.classify_token(element.description)
            if desc_sensitivity != SensitivityLevel.PUBLIC:
                return desc_sensitivity

        # Check example values for sensitive patterns
        for value in element.example_values:
            value_sensitivity = self.classify_token(value)
            if value_sensitivity != SensitivityLevel.PUBLIC:
                return value_sensitivity

        return SensitivityLevel.PUBLIC

    def is_sensitive(self, token: str, context: str = "") -> bool:
        """Check if a token is sensitive (shorthand for classify_token).

        Args:
            token: Token to check
            context: Surrounding context

        Returns:
            True if token is PII, proprietary, or regulated
        """
        level = self.classify_token(token, context)
        return level != SensitivityLevel.PUBLIC

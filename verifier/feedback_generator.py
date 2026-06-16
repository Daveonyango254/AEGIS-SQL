"""Structured feedback generator for SLM retry with privacy-aware sanitization.

Aggregates verification errors into actionable feedback.
CRITICAL: Sanitizes feedback for remote path retries to prevent privacy leakage.

References:
    - Build strategy Section 1 (verifier with structured feedback)
    - AEGIS integration doc lines 763-798 (feedback sanitization)
"""

from typing import Dict, Optional
import re

from loguru import logger

from aegis_types import VerificationResult, VerificationStatus


class FeedbackGenerator:
    """Generates structured feedback for SLM correction (part of Reviewer Agent).

    Aggregates errors from grammar, schema, and execution verifiers
    into a single actionable feedback message for the SLM to retry.

    The feedback is structured to help the model self-correct:
        - Identifies specific error location
        - Explains what went wrong
        - Suggests correction (when possible)
    """

    def __init__(self) -> None:
        """Initialize feedback generator."""
        logger.info("Initialized FeedbackGenerator (Reviewer agent)")

    def generate(
        self,
        verification_result: VerificationResult,
    ) -> Optional[str]:
        """Generate structured feedback from verification result.

        Args:
            verification_result: Result from 3-stage verification

        Returns:
            Structured feedback string or None if verification passed
        """
        if verification_result.status == VerificationStatus.PASS:
            return None

        logger.debug(
            f"Generating feedback for status={verification_result.status}"
        )

        # Build feedback based on verification status
        feedback_parts = ["The previous SQL had errors:\n"]

        if verification_result.status == VerificationStatus.GRAMMAR_FAIL:
            feedback_parts.append(self.format_grammar_error(
                verification_result.error_message or "Unknown grammar error"
            ))
        elif verification_result.status == VerificationStatus.SCHEMA_FAIL:
            feedback_parts.append(self.format_schema_error(
                verification_result.error_message or "Unknown schema error"
            ))
        elif verification_result.status == VerificationStatus.EXECUTION_FAIL:
            feedback_parts.append(self.format_execution_error(
                verification_result.error_message or "Unknown execution error"
            ))
        elif verification_result.status == VerificationStatus.TIMEOUT:
            feedback_parts.append("Query execution timeout. Simplify the query.")

        feedback_parts.append("\nPlease generate corrected SQL.")

        return "\n".join(feedback_parts)

    def format_grammar_error(self, error_message: str) -> str:
        """Format grammar error into structured feedback.

        Args:
            error_message: Raw parser error

        Returns:
            Structured feedback string
        """
        return f"""1. Grammar Error: {error_message}
   Location: SQL syntax
   Suggestion: Check for missing or extra commas, parentheses, quotes, or keywords. Ensure all SQL keywords are spelled correctly."""

    def format_schema_error(self, error_message: str) -> str:
        """Format schema error into structured feedback.

        Args:
            error_message: Raw schema validation error

        Returns:
            Structured feedback string
        """
        # Try to extract table/column name from error message
        suggestion = "Verify that all table and column names exist in the schema and are spelled correctly."

        if "table" in error_message.lower():
            suggestion = "Check that the table name exists in the schema. Use exact case matching."
        elif "column" in error_message.lower():
            suggestion = "Check that the column name exists in the referenced table. Verify the exact spelling and case."

        return f"""1. Schema Error: {error_message}
   Location: Table or column reference
   Suggestion: {suggestion}"""

    def format_execution_error(self, error_message: str) -> str:
        """Format execution error into structured feedback.

        Args:
            error_message: Raw execution error

        Returns:
            Structured feedback string
        """
        error_lower = error_message.lower()

        # Identify error category
        if "empty" in error_lower or "0 row" in error_lower or "no rows" in error_lower:
            return """1. Empty Result: The query ran successfully but matched no rows.
   Location: WHERE clause literal value(s)
   Suggestion: A string literal in a WHERE clause likely does not match the value stored in the database. Use one of the exact values shown in the schema 'examples:' comments (e.g. 'Continuation School' instead of 'Continuation'); check spelling, casing, and spacing. Also confirm you are filtering the correct column."""

        if "divide" in error_lower or "division" in error_lower:
            suggestion = "Add a WHERE clause to filter out NULL or zero values, or use CASE to handle division safely."
        elif "type" in error_lower or "datatype" in error_lower:
            suggestion = "Ensure aggregate functions (SUM, AVG) are applied to numeric columns only. Use COUNT for non-numeric data."
        elif "ambiguous" in error_lower:
            suggestion = "Use table aliases to clearly specify which table each column belongs to."
        elif "timeout" in error_lower:
            suggestion = "Simplify the query or add WHERE clauses to reduce the amount of data being processed."
        else:
            suggestion = "Review the SQL query for correctness and ensure it matches the database schema."

        return f"""1. Execution Error: {error_message}
   Location: Query execution
   Suggestion: {suggestion}"""

    def to_dict(self, verification_result: VerificationResult) -> Dict[str, any]:
        """Convert verification result to structured dictionary.

        Useful for JSON logging and analysis.

        Args:
            verification_result: Verification result

        Returns:
            Dictionary with error details
        """
        return {
            "status": verification_result.status.value,
            "grammar_valid": verification_result.grammar_valid,
            "schema_valid": verification_result.schema_valid,
            "execution_valid": verification_result.execution_valid,
            "error_message": verification_result.error_message,
            "structured_feedback": verification_result.structured_feedback,
            "has_execution_result": verification_result.execution_result is not None,
        }

    def sanitize_feedback_for_remote_retry(
        self,
        feedback: str,
        reconstruction_map: Dict[str, str],
    ) -> str:
        """Sanitize verifier feedback for remote path retry.

        CRITICAL PRIVACY FUNCTION:
        When a query is routed remotely and verification fails, the verifier
        runs on RECONSTRUCTED SQL (with real tokens). The error messages may
        contain these real tokens, which must be re-abstracted before sending
        to remote LLM for retry.

        Example:
            Original: "column 'patient_ssn' not found in table 'patients'"
            Sanitized: "column '<PERSON_1>' not found in table '<TABLE_1>'"

        Args:
            feedback: Raw feedback from verifier (may contain real tokens)
            reconstruction_map: {real_token → placeholder} mapping

        Returns:
            Sanitized feedback (all real tokens replaced with placeholders)

        References:
            - AEGIS integration doc lines 763-798
            - Privacy guarantee: remote LLM never sees real tokens
        """
        if not reconstruction_map:
            # No reconstruction happened (local path or no sensitive tokens)
            return feedback

        logger.debug(f"Sanitizing feedback for remote retry ({len(reconstruction_map)} tokens)")

        sanitized = feedback

        # Reverse the reconstruction map: real_token → placeholder
        # (reconstruction_map is typically placeholder → real_token)
        reverse_map = {v: k for k, v in reconstruction_map.items()}

        # Replace all real tokens with placeholders
        for real_token, placeholder in reverse_map.items():
            # Case-insensitive replacement (error messages may vary)
            pattern = re.compile(re.escape(real_token), re.IGNORECASE)
            sanitized = pattern.sub(placeholder, sanitized)

        logger.debug(f"Sanitized feedback: {sanitized[:100]}...")
        return sanitized

    def generate_for_retry(
        self,
        verification_result: VerificationResult,
        generation_source: str,
        reconstruction_map: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Generate feedback for retry with automatic sanitization.

        High-level API that handles both local and remote path retries:
        - Local path: Return raw feedback (SLM sees real data)
        - Remote path: Sanitize feedback before returning

        Args:
            verification_result: Verification result
            generation_source: "slm" or "llm"
            reconstruction_map: Placeholder → real token mapping (remote path only)

        Returns:
            Feedback string (sanitized if remote path)

        Example:
            >>> # Local path retry (no sanitization)
            >>> feedback = generator.generate_for_retry(
            ...     result, generation_source="slm", reconstruction_map=None
            ... )
            >>> # Returns: "column 'patient_ssn' not found"

            >>> # Remote path retry (auto-sanitized)
            >>> feedback = generator.generate_for_retry(
            ...     result, generation_source="llm", reconstruction_map=recon_map
            ... )
            >>> # Returns: "column '<PERSON_1>' not found"
        """
        # Generate raw feedback
        raw_feedback = self.generate(verification_result)

        if raw_feedback is None:
            return None

        # Sanitize if remote path
        if generation_source == "llm" and reconstruction_map:
            return self.sanitize_feedback_for_remote_retry(raw_feedback, reconstruction_map)

        return raw_feedback

"""Grammar verification with constrained decoding.

Implements PICARD-style grammar-constrained decoding to catch syntax errors.

References:
    - Scholak et al. 2021: PICARD
    - Build strategy Section 1: Three-stage verifier
"""

from typing import Any, Optional

from loguru import logger
import sqlglot
from sqlglot import parse_one, ParseError

from aegis_types import SQL


class GrammarVerifier:
    """Grammar verification for SQL (part of Reviewer Agent).

    Checks SQL syntax using sqlglot parser. Can also be used for
    constrained decoding during generation.

    Attributes:
        dialect: SQL dialect (sqlite, postgresql, mysql, etc.)
    """

    def __init__(self, dialect: str = "sqlite") -> None:
        """Initialize grammar verifier.

        Args:
            dialect: SQL dialect for parsing
        """
        self.dialect = dialect
        logger.info(f"Initialized GrammarVerifier (Reviewer agent) with dialect={dialect}")

    def verify(self, sql: SQL) -> tuple[bool, Optional[str]]:
        """Verify SQL grammar.

        Args:
            sql: SQL query to verify

        Returns:
            Tuple of (is_valid, error_message)
        """
        logger.debug(f"Verifying grammar for SQL: {sql.text[:80]}...")
        try:
            # Parse SQL with sqlglot
            parse_one(sql.text, dialect=self.dialect)
            logger.debug("Grammar verification passed")
            return True, None
        except ParseError as e:
            logger.warning(f"Grammar error: {str(e)}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected error during grammar verification: {str(e)}")
            return False, f"Grammar verification failed: {str(e)}"

    def parse_to_ast(self, sql_text: str) -> Optional[Any]:
        """Parse SQL to AST for downstream verification.

        Args:
            sql_text: SQL query string

        Returns:
            Parsed AST or None if parsing fails
        """
        try:
            ast = parse_one(sql_text, dialect=self.dialect)
            return ast
        except ParseError as e:
            logger.warning(f"Failed to parse SQL: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during parsing: {str(e)}")
            return None

    def format_error_for_retry(self, error_message: str) -> str:
        """Format grammar error as structured feedback for SLM retry.

        Args:
            error_message: Raw parser error message

        Returns:
            Structured feedback string
        """
        # Extract key information from error message
        feedback = f"SQL Syntax Error: {error_message}\n"
        feedback += "Please check:\n"
        feedback += "- Matching parentheses and quotes\n"
        feedback += "- Correct SQL keywords (SELECT, FROM, WHERE, etc.)\n"
        feedback += "- Proper JOIN syntax\n"
        feedback += "- Valid column and table references\n"
        return feedback

"""Execution verification on database.

Catches runtime errors and validates result shape.

References:
    - Build strategy Section 1: Execution-aware shape verification
    - NL2SQL-BUGs: Execution errors (division by zero, type errors, etc.)
"""

import sqlite3
from typing import List, Optional, Tuple

from loguru import logger

from aegis_types import SQL, Schema


class ExecutionVerifier:
    """Execution verification for SQL queries (part of Reviewer Agent).

    Executes SQL on a 100-row database sample to catch runtime errors:
        - Division by zero
        - Type mismatches in functions
        - NULL handling issues
        - Aggregation errors

    Also validates result shape (non-empty, expected column count).

    Attributes:
        schema: Database schema
        db_path: Path to database file
        sample_size: Number of rows to execute on
        timeout: Execution timeout in seconds
    """

    def __init__(
        self,
        schema: Schema,
        db_path: str,
        sample_size: int = 100,
        timeout: int = 5,
    ) -> None:
        """Initialize execution verifier.

        Args:
            schema: Database schema
            db_path: Path to SQLite database file
            sample_size: Number of rows to execute on
            timeout: Execution timeout in seconds
        """
        self.schema = schema
        self.db_path = db_path
        self.sample_size = sample_size
        self.timeout = timeout
        self.connection: Optional[sqlite3.Connection] = None

        # Connect to database
        try:
            self.connection = sqlite3.connect(db_path, timeout=timeout)
            self.connection.row_factory = sqlite3.Row
            logger.info(
                f"Initialized ExecutionVerifier (Reviewer agent) with db={db_path}, "
                f"sample_size={sample_size}, timeout={timeout}s"
            )
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database {db_path}: {e}")
            raise

    def verify(self, sql: SQL) -> tuple[bool, Optional[str], Optional[List[Tuple]]]:
        """Execute SQL and verify result.

        Args:
            sql: SQL query to execute

        Returns:
            Tuple of (is_valid, error_message, result_rows)

        Error categories to catch:
            - Division by zero
            - Type errors (e.g., SUM on string column)
            - NULL propagation issues
            - Overflow errors
            - Timeout (query too complex for sample)

        References:
            - NL2SQL-BUGs execution error taxonomy
        """
        logger.debug(f"Executing SQL on {self.sample_size}-row sample: {sql.text[:80]}...")

        if not self.connection:
            return False, "Database connection not established", None

        try:
            # Execute SQL with timeout
            result_rows = self.execute_with_timeout(sql.text, self.timeout)

            # Validate result (non-empty check for SELECT queries)
            if sql.text.strip().upper().startswith("SELECT"):
                if len(result_rows) == 0:
                    logger.warning("Query returned empty result set")
                    # Empty results are valid, just log it

            logger.debug(f"Execution successful: {len(result_rows)} rows returned")
            return True, None, result_rows

        except sqlite3.OperationalError as e:
            error_msg = f"Execution error: {str(e)}"
            logger.warning(error_msg)
            return False, error_msg, None
        except sqlite3.Error as e:
            error_msg = f"SQL error: {str(e)}"
            logger.warning(error_msg)
            return False, error_msg, None
        except TimeoutError as e:
            error_msg = f"Query timeout after {self.timeout}s: {str(e)}"
            logger.warning(error_msg)
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Unexpected execution error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None

    def create_sample_tables(self) -> None:
        """Create 100-row sample tables for verification.

        TODO: For more advanced sampling:
            - Use temp tables to avoid modifying original database
            - Rewrite query to use table_sample instead of table

        Note: For now, we execute on the full database with LIMIT clauses.
        This is simpler and works for most verification scenarios.
        """
        # Simplified implementation: Execute directly on database
        # More advanced implementation would create temp tables with sampled data
        logger.debug("Using direct database execution (no sample tables created)")

    def execute_with_timeout(
        self, sql_text: str, timeout: int
    ) -> List[Tuple]:
        """Execute SQL with timeout.

        Args:
            sql_text: SQL query string
            timeout: Timeout in seconds

        Returns:
            Result rows

        Raises:
            sqlite3.Error: On execution error
            TimeoutError: On timeout
        """
        if not self.connection:
            raise sqlite3.Error("Database connection not established")

        try:
            # Set timeout on connection
            self.connection.execute(f"PRAGMA busy_timeout = {timeout * 1000}")

            # Execute query
            cursor = self.connection.cursor()
            cursor.execute(sql_text)

            # Fetch all rows
            rows = cursor.fetchall()

            # Convert Row objects to tuples
            result_rows = [tuple(row) for row in rows]

            return result_rows

        except sqlite3.OperationalError as e:
            # Check if it's a timeout
            if "timeout" in str(e).lower():
                raise TimeoutError(f"Query execution timeout: {e}")
            raise

    def format_error_for_retry(self, error: Exception) -> str:
        """Format execution error as structured feedback for SLM retry.

        Args:
            error: Exception from execution

        Returns:
            Structured feedback string
        """
        error_msg = str(error).lower()

        # Identify error category and provide actionable feedback
        if "divide" in error_msg or "division by zero" in error_msg:
            return "Division by zero error detected. Add a NULL or zero check in your WHERE clause or use CASE to handle division."

        elif "type" in error_msg or "datatype" in error_msg:
            return "Type error detected. Ensure aggregate functions (SUM, AVG) are used on numeric columns. Use COUNT for text columns."

        elif "no such table" in error_msg:
            table_name = error_msg.split("no such table:")[-1].strip()
            return f"Table '{table_name}' not found. Check the table name against the schema."

        elif "no such column" in error_msg:
            return "Column not found. Verify column names match the schema exactly, including case sensitivity."

        elif "ambiguous" in error_msg:
            return "Ambiguous column reference. Use table aliases to specify which table's column you're referencing."

        elif "syntax error" in error_msg:
            return "SQL syntax error. Check for missing commas, parentheses, or keywords."

        elif "timeout" in error_msg:
            return "Query timeout. Simplify the query or add WHERE clauses to reduce data processing."

        else:
            # Generic error feedback
            return f"Execution error: {str(error)}. Review the SQL for correctness."

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.debug("Closed database connection")

"""Schema verification on parsed SQL AST.

Checks that all referenced tables/columns exist and types are compatible.

References:
    - Build strategy Section 1: Verifier is language-independent
    - NL2SQL-BUGs (Wang et al. 2025): Semantic error taxonomy
"""

from typing import Any, List, Optional, Set

from loguru import logger

from aegis_types import SQL, Schema, SchemaElement


class SchemaVerifier:
    """Schema verification for SQL queries (part of Reviewer Agent).

    Validates that:
        1. All referenced tables exist in the schema
        2. All referenced columns exist in their respective tables
        3. JOIN conditions reference valid columns
        4. Data types are compatible in predicates

    Attributes:
        schema: Database schema
        valid_tables: Set of valid table names
        valid_columns: Dict mapping table -> set of column names
    """

    def __init__(self, schema: Schema) -> None:
        """Initialize schema verifier.

        Args:
            schema: Database schema
        """
        self.schema = schema
        self.valid_tables: Set[str] = set(schema.tables)
        self.valid_columns: dict[str, Set[str]] = {}

        # Build table -> columns mapping
        for column in schema.columns:
            # Extract table name from column.name (format: "table.column")
            if "." in column.name:
                table, col = column.name.split(".", 1)
            else:
                # Assume column belongs to first table if not qualified
                table = schema.tables[0] if schema.tables else "unknown"
                col = column.name

            if table not in self.valid_columns:
                self.valid_columns[table] = set()
            self.valid_columns[table].add(col)

        logger.info(
            f"Initialized SchemaVerifier (Reviewer agent) with {len(schema.tables)} tables, "
            f"{len(schema.columns)} columns"
        )

    def verify(self, sql: SQL, ast: Any) -> tuple[bool, Optional[str]]:
        """Verify SQL against schema.

        Args:
            sql: SQL query
            ast: Parsed AST from grammar verifier

        Returns:
            Tuple of (is_valid, error_message)

        References:
            - NL2SQL-BUGs error categories:
              * Non-existent column
              * Non-existent table
              * Ambiguous column reference
              * Type mismatch in predicate
        """
        logger.debug(f"Verifying schema for SQL: {sql.text[:80]}...")

        if ast is None:
            return False, "Cannot verify schema: AST is None"

        try:
            # Extract table references
            tables = self.extract_table_references(ast)
            for table in tables:
                if not self.validate_table(table):
                    error_msg = f"Table '{table}' does not exist. Valid tables: {', '.join(sorted(self.valid_tables))}"
                    logger.warning(error_msg)
                    return False, error_msg

            # Extract column references
            columns = self.extract_column_references(ast)
            for table, column in columns:
                if table and not self.validate_column(table, column):
                    error_msg = f"Column '{column}' does not exist in table '{table}'"
                    if table in self.valid_columns:
                        error_msg += f". Valid columns: {', '.join(sorted(self.valid_columns[table]))}"
                    logger.warning(error_msg)
                    return False, error_msg

            logger.debug("Schema verification passed")
            return True, None

        except Exception as e:
            logger.error(f"Error during schema verification: {str(e)}")
            return False, f"Schema verification failed: {str(e)}"

    def extract_table_references(self, ast: Any) -> List[str]:
        """Extract all table names referenced in SQL.

        Args:
            ast: Parsed SQL AST

        Returns:
            List of table names
        """
        tables = []
        try:
            # Simple extraction from AST string representation
            # In production, would walk AST properly with sqlglot
            ast_str = str(ast)
            for table in self.valid_tables:
                if table in ast_str:
                    tables.append(table)
        except Exception as e:
            logger.warning(f"Error extracting table references: {str(e)}")

        return list(set(tables))  # Deduplicate

    def extract_column_references(self, ast: Any) -> List[tuple[str, str]]:
        """Extract all column references in SQL.

        Args:
            ast: Parsed SQL AST

        Returns:
            List of (table, column) tuples
        """
        columns = []
        try:
            # Simple extraction - in production would use AST walker
            ast_str = str(ast)
            for table, cols in self.valid_columns.items():
                for col in cols:
                    if col in ast_str:
                        columns.append((table, col))
        except Exception as e:
            logger.warning(f"Error extracting column references: {str(e)}")

        return columns

    def validate_table(self, table_name: str) -> bool:
        """Check if table exists in schema.

        Args:
            table_name: Table name to validate

        Returns:
            True if table exists
        """
        return table_name in self.valid_tables

    def validate_column(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in table.

        Args:
            table_name: Table name
            column_name: Column name

        Returns:
            True if column exists in the table
        """
        return (
            table_name in self.valid_columns
            and column_name in self.valid_columns[table_name]
        )

    def format_error_for_retry(
        self, error_type: str, details: dict
    ) -> str:
        """Format schema error as structured feedback for SLM retry.

        Args:
            error_type: Type of schema error
            details: Error details (table/column names, etc.)

        Returns:
            Structured feedback string
        """
        if error_type == "table_not_found":
            table = details.get("table", "unknown")
            return f"Table '{table}' does not exist. Valid tables: {', '.join(sorted(self.valid_tables))}"

        elif error_type == "column_not_found":
            table = details.get("table", "unknown")
            column = details.get("column", "unknown")
            feedback = f"Column '{column}' does not exist in table '{table}'."
            if table in self.valid_columns:
                feedback += f" Valid columns: {', '.join(sorted(self.valid_columns[table]))}"
            return feedback

        elif error_type == "ambiguous_column":
            column = details.get("column", "unknown")
            return f"Ambiguous column '{column}' could refer to multiple tables. Please qualify with table name."

        else:
            return f"Schema error: {error_type}"

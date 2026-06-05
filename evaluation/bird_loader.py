"""BIRD dataset loader for AEGIS-SQL evaluation.

Loads queries, schemas, and databases from BIRD-dev benchmark.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger

from aegis_types import Query, Language, Schema, SchemaElement, SensitivityLevel


class BIRDLoader:
    """Loader for BIRD-dev benchmark dataset.

    Attributes:
        bird_path: Path to BIRD data directory
        queries: List of loaded queries
        schemas: Dict mapping db_id to Schema objects
    """

    def __init__(self, bird_path: str | Path = "data/bird"):
        """Initialize BIRD loader.

        Args:
            bird_path: Path to BIRD data directory
        """
        self.bird_path = Path(bird_path)
        self.dev_json_path = self.bird_path / "dev.json"
        self.db_root = self.bird_path / "dev_databases"
        self.tables_json_path = self.bird_path / "dev_tables.json"

        # Validate paths
        if not self.dev_json_path.exists():
            raise FileNotFoundError(f"BIRD dev.json not found: {self.dev_json_path}")
        if not self.db_root.exists():
            raise FileNotFoundError(f"BIRD databases not found: {self.db_root}")

        logger.info(f"Initialized BIRDLoader with path: {self.bird_path}")

    def load_queries(self) -> List[Dict[str, Any]]:
        """Load all queries from dev.json.

        Returns:
            List of query dictionaries

        Example:
            >>> loader = BIRDLoader()
            >>> queries = loader.load_queries()
            >>> len(queries)
            1534
        """
        with open(self.dev_json_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        logger.info(f"Loaded {len(queries)} queries from {self.dev_json_path}")
        return queries

    def load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load schema information from dev_tables.json.

        Returns:
            Dict mapping db_id to schema dictionary
        """
        if not self.tables_json_path.exists():
            logger.warning(f"Schema file not found: {self.tables_json_path}")
            return {}

        with open(self.tables_json_path, 'r', encoding='utf-8') as f:
            schemas_list = json.load(f)

        schemas = {schema['db_id']: schema for schema in schemas_list}
        logger.info(f"Loaded schemas for {len(schemas)} databases")
        return schemas

    def get_database_path(self, db_id: str) -> Path:
        """Get path to SQLite database file.

        Args:
            db_id: Database identifier

        Returns:
            Path to .sqlite file
        """
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        return db_path

    def query_to_aegis_query(self, query_dict: Dict[str, Any]) -> Query:
        """Convert BIRD query dict to AEGIS Query object.

        Args:
            query_dict: Query dictionary from dev.json

        Returns:
            AEGIS Query object
        """
        return Query(
            text=query_dict['question'],
            language=Language.ENGLISH,
            database_id=query_dict['db_id'],
        )

    def load_schema_for_db(self, db_id: str) -> Schema:
        """Load schema for a specific database.

        Extracts schema from SQLite database file.

        Args:
            db_id: Database identifier

        Returns:
            AEGIS Schema object
        """
        db_path = self.get_database_path(db_id)

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            # Get columns for each table
            schema_elements = []
            for table in tables:
                # Quote table name to handle reserved keywords like "order"
                cursor.execute(f'PRAGMA table_info("{table}");')
                columns = cursor.fetchall()

                for col in columns:
                    col_name = col[1]
                    col_type = col[2]

                    schema_elements.append(
                        SchemaElement(
                            element_type="column",
                            name=f"{table}.{col_name}",
                            data_type=col_type,
                            sensitivity=SensitivityLevel.PUBLIC,  # Default
                        )
                    )

            conn.close()

            logger.debug(f"Loaded schema for {db_id}: {len(tables)} tables, {len(schema_elements)} columns")

            # Create schema object
            schema = Schema(
                database_id=db_id,
                tables=tables,
                columns=schema_elements,
                documentation=None,
                sensitive_elements=set(),  # BIRD doesn't mark sensitive data
            )

            # Enhance with BIRD descriptions if available
            try:
                from evaluation.schema_description_loader import enhance_schema_with_descriptions
                enhance_schema_with_descriptions(schema, db_id, self.bird_path)
            except Exception as e:
                logger.warning(f"Could not load BIRD descriptions for {db_id}: {e}")

            return schema

        except Exception as e:
            logger.error(f"Failed to load schema for {db_id}: {e}")
            # Return empty schema as fallback
            return Schema(
                database_id=db_id,
                tables=[],
                columns=[],
                documentation=None,
                sensitive_elements=set(),
            )

    def prepare_evaluation_batch(
        self,
        queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare a batch of queries for evaluation.

        Adds database paths and schemas to query dictionaries.

        Args:
            queries: List of query dictionaries

        Returns:
            List of queries enriched with db_path and schema info
        """
        prepared = []

        for query_dict in queries:
            db_id = query_dict['db_id']

            try:
                # Get database path
                db_path = self.get_database_path(db_id)

                # Load schema
                schema = self.load_schema_for_db(db_id)

                # Enrich query dict
                enriched = query_dict.copy()
                enriched['db_path'] = str(db_path)
                enriched['schema'] = schema

                prepared.append(enriched)

            except Exception as e:
                logger.error(f"Failed to prepare query {query_dict['question_id']}: {e}")
                # Skip this query
                continue

        logger.info(f"Prepared {len(prepared)}/{len(queries)} queries for evaluation")
        return prepared


def load_bird_dev(
    bird_path: str | Path = "data/bird",
    num_queries: Optional[int] = None,
    seed: int = 42,
    stratify: bool = True
) -> List[Dict[str, Any]]:
    """Convenience function to load and optionally sample BIRD-dev queries.

    Args:
        bird_path: Path to BIRD data directory
        num_queries: Number of queries to sample (None = all queries)
        seed: Random seed for sampling
        stratify: Stratified sampling by difficulty

    Returns:
        List of query dictionaries (sampled and prepared)

    Example:
        >>> queries = load_bird_dev(num_queries=100, seed=42)
        >>> len(queries)
        100
    """
    loader = BIRDLoader(bird_path)
    queries = loader.load_queries()

    # Sample if requested
    if num_queries is not None and num_queries < len(queries):
        from evaluation.sampling import sample_queries
        queries = sample_queries(queries, num_queries, seed, stratify)

    # Prepare for evaluation
    prepared = loader.prepare_evaluation_batch(queries)

    return prepared

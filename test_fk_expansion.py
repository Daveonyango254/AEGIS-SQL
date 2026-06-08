"""Test FK extraction and expansion for Query 42."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from evaluation.bird_loader import BIRDLoader
from aegis_types import Query, Language

logger.remove()
logger.add(sys.stderr, level="INFO")


def test_fk_extraction():
    """Test FK extraction from california_schools database."""
    logger.info("=" * 80)
    logger.info("TEST 1: FK Extraction from california_schools")
    logger.info("=" * 80)

    loader = BIRDLoader("data/bird")
    schema = loader.load_schema_for_db("california_schools")

    logger.info(f"\nDatabase: {schema.database_id}")
    logger.info(f"Tables: {schema.tables}")
    logger.info(f"Total columns: {len(schema.columns)}")
    logger.info(f"Total foreign keys: {len(schema.foreign_keys)}")
    logger.info(f"Primary keys: {schema.primary_keys}")

    logger.info("\nForeign Key Relationships:")
    for fk in schema.foreign_keys:
        logger.info(f"  {fk.from_table}.{fk.from_column} → {fk.to_table}.{fk.to_column}")

    # Expected: satscores.cds → schools.CDSCode, frpm.CDSCode → schools.CDSCode
    assert len(schema.foreign_keys) >= 2, f"Expected at least 2 FKs, got {len(schema.foreign_keys)}"

    logger.info("\n✓ FK extraction working correctly")


def test_fk_expansion_query_42():
    """Test FK expansion on Query 42 (JOIN query that was failing)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: FK Expansion for Query 42")
    logger.info("=" * 80)

    # Load schema with FKs
    loader = BIRDLoader("data/bird")
    schema = loader.load_schema_for_db("california_schools")

    # Create query 42
    query = Query(
        text="What is the type of education offered in the school who scored the highest average in Math?",
        language=Language.ENGLISH,
        database_id="california_schools"
    )

    # Create retriever with shared model
    from workflow.model_cache import get_cache
    from config import AEGISConfig

    cache = get_cache()
    config = AEGISConfig.from_yaml("config.yaml")
    cache.set_config(config)

    retriever = cache.get_schema_retriever("california_schools", schema)

    logger.info(f"\nQuery: {query.text}")

    # Test retrieval WITHOUT FK expansion
    logger.info("\n--- WITHOUT FK expansion ---")
    retrieved_no_fk = retriever.retrieve(query, top_k=25, expand_foreign_keys=False)
    tables_no_fk = set()
    for col in retrieved_no_fk:
        if '.' in col.name:
            table = col.name.split('.', 1)[0]
            tables_no_fk.add(table)

    logger.info(f"Retrieved {len(retrieved_no_fk)} columns from tables: {tables_no_fk}")

    # Test retrieval WITH FK expansion
    logger.info("\n--- WITH FK expansion ---")
    retrieved_with_fk = retriever.retrieve(query, top_k=25, expand_foreign_keys=True, max_expanded_tables=3)
    tables_with_fk = set()
    for col in retrieved_with_fk:
        if '.' in col.name:
            table = col.name.split('.', 1)[0]
            tables_with_fk.add(table)

    logger.info(f"Retrieved {len(retrieved_with_fk)} columns from tables: {tables_with_fk}")

    # Expected: should include both satscores and schools (or frpm and schools)
    # Ground truth query uses: satscores JOIN schools
    assert len(retrieved_with_fk) > len(retrieved_no_fk), "FK expansion should add columns"
    assert len(tables_with_fk) >= 2, f"Expected at least 2 tables with FK expansion, got {len(tables_with_fk)}"

    logger.info(f"\n✓ FK expansion added {len(retrieved_with_fk) - len(retrieved_no_fk)} columns")
    logger.info(f"✓ FK expansion added {len(tables_with_fk) - len(tables_no_fk)} tables")

    # Check if satscores and schools are both present
    has_satscores = 'satscores' in tables_with_fk
    has_schools = 'schools' in tables_with_fk

    logger.info(f"\nTable coverage for JOIN:")
    logger.info(f"  satscores: {'✓' if has_satscores else '✗'}")
    logger.info(f"  schools: {'✓' if has_schools else '✗'}")

    if has_satscores and has_schools:
        logger.info("\n✓ Both tables needed for JOIN are present!")
    else:
        logger.warning("\n⚠ Missing tables - may need to increase top_k or max_expanded_tables")


if __name__ == "__main__":
    try:
        test_fk_extraction()
        test_fk_expansion_query_42()

        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""Test script for multilingual schema retrieval with BGE-M3.

Tests:
1. English query retrieval
2. Multilingual query retrieval (Chinese, Spanish, French)
3. Hybrid vs dense-only retrieval comparison
4. Schema description enhancement
"""

from loguru import logger
from config import EmbeddingConfig
from aegis_types import Query, Language
from evaluation.bird_loader import BIRDLoader

# Configure logging
logger.add("test_multilingual_retrieval.log", level="DEBUG")


def test_english_retrieval():
    """Test schema retrieval with English queries."""
    print("\n" + "=" * 80)
    print("TEST 1: English Query Retrieval")
    print("=" * 80)

    # Load schema for california_schools
    loader = BIRDLoader("data/bird")
    schema = loader.load_schema_for_db("california_schools")

    print(f"\nLoaded schema: {schema.database_id}")
    print(f"  - Tables: {len(schema.tables)}")
    print(f"  - Columns: {len(schema.columns)}")

    # Sample first few columns
    print(f"\nSample columns:")
    for col in schema.columns[:5]:
        desc = col.description[:80] if col.description else "(no description)"
        print(f"  - {col.name} ({col.data_type}): {desc}")

    # Create retriever
    from retriever.schema_retriever import SchemaRetriever

    config = EmbeddingConfig(device="cuda")  # Use GPU if available
    retriever = SchemaRetriever(config, schema)

    # Test query
    query = Query(
        text="What is the highest eligible free rate for K-12 students?",
        language=Language.ENGLISH,
        database_id="california_schools",
    )

    print(f"\n\nQuery: {query.text}")

    # Retrieve with hybrid mode
    results = retriever.retrieve(query, top_k=5, use_hybrid=True)

    print(f"\nTop-5 Retrieved Columns (Hybrid):")
    for i, col in enumerate(results, 1):
        desc = col.description[:100] if col.description else "(no description)"
        print(f"  {i}. {col.name} ({col.data_type})")
        print(f"     {desc}")

    return retriever, schema


def test_multilingual_retrieval(retriever, schema):
    """Test schema retrieval with multilingual queries."""
    print("\n\n" + "=" * 80)
    print("TEST 2: Multilingual Query Retrieval")
    print("=" * 80)

    # Test queries in different languages
    test_queries = [
        ("English", "Which schools have the highest test scores?"),
        ("Spanish", "¿Cuáles escuelas tienen las calificaciones más altas?"),
        ("Chinese", "哪些学校的考试成绩最高?"),
        ("French", "Quelles écoles ont les meilleurs résultats aux tests?"),
    ]

    for language_name, query_text in test_queries:
        print(f"\n\n{language_name} Query: {query_text}")
        print("-" * 80)

        query = Query(
            text=query_text,
            language=Language.ENGLISH,  # BGE-M3 handles language detection
            database_id="california_schools",
        )

        results = retriever.retrieve(query, top_k=5, use_hybrid=True)

        print(f"Top-5 Retrieved Columns:")
        for i, col in enumerate(results, 1):
            print(f"  {i}. {col.name}")


def test_hybrid_vs_dense():
    """Compare hybrid retrieval vs dense-only."""
    print("\n\n" + "=" * 80)
    print("TEST 3: Hybrid vs Dense-Only Retrieval")
    print("=" * 80)

    loader = BIRDLoader("data/bird")
    schema = loader.load_schema_for_db("superhero")

    from retriever.schema_retriever import SchemaRetriever

    config = EmbeddingConfig(device="cuda")
    retriever = SchemaRetriever(config, schema)

    query = Query(
        text="What percentage of female superheroes are bad?",
        language=Language.ENGLISH,
        database_id="superhero",
    )

    print(f"\nQuery: {query.text}")

    # Hybrid retrieval
    print("\n\nHybrid Retrieval (Dense + Sparse):")
    results_hybrid = retriever.retrieve(query, top_k=5, use_hybrid=True)
    for i, col in enumerate(results_hybrid, 1):
        print(f"  {i}. {col.name}")

    # Dense-only retrieval
    print("\n\nDense-Only Retrieval:")
    results_dense = retriever.retrieve(query, top_k=5, use_hybrid=False)
    for i, col in enumerate(results_dense, 1):
        print(f"  {i}. {col.name}")


def test_description_enhancement():
    """Test schema description loading from BIRD CSVs."""
    print("\n\n" + "=" * 80)
    print("TEST 4: Schema Description Enhancement")
    print("=" * 80)

    from evaluation.schema_description_loader import load_bird_descriptions

    # Load descriptions for california_schools
    descriptions = load_bird_descriptions("california_schools", "data/bird")

    print(f"\nLoaded {len(descriptions)} column descriptions")

    # Sample some descriptions
    print("\nSample descriptions:")
    for i, (col_name, description) in enumerate(list(descriptions.items())[:5], 1):
        print(f"\n{i}. {col_name}:")
        print(f"   {description[:200]}...")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MULTILINGUAL SCHEMA RETRIEVAL TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: English retrieval
        retriever, schema = test_english_retrieval()

        # Test 2: Multilingual retrieval
        test_multilingual_retrieval(retriever, schema)

        # Test 3: Hybrid vs dense-only
        test_hybrid_vs_dense()

        # Test 4: Description enhancement
        test_description_enhancement()

        print("\n\n" + "=" * 80)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except Exception as e:
        print(f"\n\n✗ TEST FAILED: {e}")
        logger.exception("Test suite failed")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

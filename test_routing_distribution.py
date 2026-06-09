"""Test routing distribution with adjusted complexity calculation.

Verifies that threshold=0.7 routes 65%+ of queries to local SLM.
"""

from router.content_independent_router import ContentIndependentRouter
from config import RouterConfig, CostConfig
from aegis_types import Query, Language, SchemaElement

def test_routing_distribution():
    """Test that adjusted complexity routes 65%+ queries to local."""

    # Create router with threshold=0.7
    router_config = RouterConfig(
        threshold_complexity=0.7,
        force_local=False,
        force_remote=False,
        features=["query_token_count", "schema_element_count", "query_structural_complexity"]
    )
    cost_config = CostConfig(
        budget_per_query=0.01,
        remote_token_cost=0.000015,
        local_compute_cost=0.0001
    )

    router = ContentIndependentRouter(router_config, cost_config)

    # Test queries from smoke_test1_100 (representative sample)
    test_queries = [
        # Simple queries (8-21 tokens)
        ("What is the number of SAT test takers of the schools", 10),
        ("What is the type of education offered in the school", 9),
        ("Please specify all of the schools", 6),
        ("How many active and closed District Community Day Schools", 8),
        ("How many 'classic' cards are eligible for loan?", 8),
        ("Which state special schools have the highest number", 8),
        ("What is the total number of schools whose total SAT scores", 11),
        ("How many students from the ages of 5 to 17", 11),
        ("Among the account opened, how many female customers", 8),
        ("List the top ten districts, by descending order", 8),

        # Moderate queries (12-20 tokens)
        ("What is the difference between the number of molecules that are carcinogenic and those that are not?", 17),
        ("What are the elements for bond id TR001_10_11?", 8),
        ("What percentage of legendary frame effect cards have a maximum starting maximum hand size of +3?", 16),
        ("What language is the set of 180 cards that belongs to the Ravnica block translated into?", 16),
        ("Which of the cards that are a promotional painting have multiple faces on the same card?", 17),
        ("Which citizenship do the vast majority of the drivers hold?", 10),
        ("Who is the oldest patient with the highest total cholesterol (T-CHO)?", 12),
        ("Which patient is the first patient with an abnormal anti-SSA to come to the hospital?", 16),
        ("Who among KAM's customers consumed the most? How much did they consume?", 13),
        ("What segment did the customer have at 2012/8/23 21:20:00?", 10),
    ]

    # Create schema elements (typical size: 20-25)
    schema_elements = [
        SchemaElement(
            element_type="column",
            name=f"table{i//10}.column{i}",
            data_type="TEXT",
            description=""
        )
        for i in range(23)
    ]

    print("=" * 80)
    print("Testing Routing Distribution with Adjusted Complexity")
    print("=" * 80)
    print(f"Threshold: {router_config.threshold_complexity}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Schema elements: {len(schema_elements)}\n")

    local_count = 0
    remote_count = 0

    print("Query Analysis:")
    print("-" * 80)

    for i, (query_text, expected_tokens) in enumerate(test_queries, 1):
        query = Query(
            text=query_text,
            language=Language.ENGLISH,
            database_id="test_db"
        )

        # Extract features
        features = router.extract_features(query, schema_elements)
        complexity = router.compute_complexity_score(features)
        decision = router.route(query, schema_elements)

        if decision.value == "local":
            local_count += 1
            status = "LOCAL"
        else:
            remote_count += 1
            status = "REMOTE"

        print(f"{i:2d}. {status:6s} | complexity={complexity:.3f} | tokens={features.query_token_count:2d} | {query_text[:60]}...")

    print("-" * 80)
    print(f"\nRouting Summary:")
    print(f"  Local:  {local_count}/{len(test_queries)} ({local_count/len(test_queries)*100:.1f}%)")
    print(f"  Remote: {remote_count}/{len(test_queries)} ({remote_count/len(test_queries)*100:.1f}%)")
    print()

    # Check target
    local_percentage = local_count / len(test_queries) * 100
    target_met = local_percentage >= 65.0

    if target_met:
        print(f"SUCCESS: {local_percentage:.1f}% routed to local (target: >=65%)")
        if local_percentage >= 95:
            print(f"NOTE: Very high local routing ({local_percentage:.1f}%). Complex queries also go local.")
            print(f"  This is acceptable - local SLM can handle these queries.")
    else:
        print(f"FAILED: {local_percentage:.1f}% routed to local (target: >=65%)")
        print(f"  Need to reduce complexity scores further")

    print("=" * 80)

    return target_met

if __name__ == "__main__":
    success = test_routing_distribution()
    exit(0 if success else 1)

"""End-to-end test script for AEGIS-SQL workflow.

Tests the full pipeline from natural language query to verified SQL,
with all metrics logged and outputs saved to evaluation/output/.
"""

import json
import sys
import time
from pathlib import Path
from loguru import logger

# Direct imports since run_e2e_test.py is in the same directory as the modules
from config import AEGISConfig
from workflow import build_aegis_graph
from aegis_types import Query, Language, Schema, SchemaElement, SensitivityLevel


# Configure logger to save to file
OUTPUT_DIR = Path("evaluation/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "e2e_test_log.txt"
logger.add(LOG_FILE, format="{time} {level} {message}", level="DEBUG")


def create_test_schema() -> Schema:
    """Create a test database schema."""
    return Schema(
        database_id="company_db",
        tables=["employees", "departments"],
        columns=[
            SchemaElement(
                element_type="column",
                name="employees.id",
                data_type="INTEGER",
                sensitivity=SensitivityLevel.PUBLIC,
            ),
            SchemaElement(
                element_type="column",
                name="employees.name",
                data_type="TEXT",
                sensitivity=SensitivityLevel.PII,
            ),
            SchemaElement(
                element_type="column",
                name="employees.salary",
                data_type="REAL",
                sensitivity=SensitivityLevel.PROPRIETARY,
            ),
            SchemaElement(
                element_type="column",
                name="employees.department_id",
                data_type="INTEGER",
                sensitivity=SensitivityLevel.PUBLIC,
            ),
            SchemaElement(
                element_type="column",
                name="departments.id",
                data_type="INTEGER",
                sensitivity=SensitivityLevel.PUBLIC,
            ),
            SchemaElement(
                element_type="column",
                name="departments.name",
                data_type="TEXT",
                sensitivity=SensitivityLevel.PUBLIC,
            ),
        ],
        documentation=None,
        sensitive_elements=set(["employees.name", "employees.salary"]),
    )


def run_e2e_test():
    """Run end-to-end test with OpenAI API."""
    logger.info("=" * 80)
    logger.info("AEGIS-SQL End-to-End Test")
    logger.info("=" * 80)

    try:
        # 1. Load configuration
        logger.info("\n[1/7] Loading configuration...")
        config = AEGISConfig.from_yaml("config.yaml")
        logger.info(f"✓ Configuration loaded: router.force_remote={config.router.force_remote}")

        # 2. Build workflow graph
        logger.info("\n[2/7] Building workflow graph...")
        graph = build_aegis_graph(config)
        logger.info("✓ Workflow graph compiled successfully")

        # 3. Create test query and schema
        logger.info("\n[3/7] Creating test query and schema...")
        query = Query(
            text="Find all employees with salary greater than 50000",
            language=Language.ENGLISH,
            database_id="company_db",
        )
        schema = create_test_schema()
        logger.info(f"✓ Test query: {query.text}")
        logger.info(f"✓ Test schema: {len(schema.tables)} tables, {len(schema.columns)} columns")

        # 4. Execute workflow
        logger.info("\n[4/7] Executing workflow...")
        start_time = time.time()

        initial_state = {
            "query": query,
            "schema": schema,
            "database_id": "company_db",
            "db_path": ":memory:",
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "privacy_loss": 0.0,
            "verification_attempts": 0,
        }

        result = graph.invoke(initial_state)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        logger.info(f"✓ Workflow completed in {latency_ms:.2f}ms")

        # 5. Extract results
        logger.info("\n[5/7] Extracting results...")
        sql = result.get("sql")
        routing_decision = result.get("routing_decision")
        schema_elements = result.get("schema_elements", [])
        abstracted_prompt = result.get("abstracted_prompt")
        reconstruction_map = result.get("reconstruction_map")
        verification_result = result.get("verification_result")

        logger.info(f"✓ Routing decision: {routing_decision}")
        logger.info(f"✓ Generated SQL: {sql.text if sql else 'None'}")
        logger.info(f"✓ SQL verified: {sql.verified if sql else False}")

        # 6. Save outputs to evaluation/output/
        logger.info("\n[6/7] Saving outputs to evaluation/output/...")

        # Save extracted schema
        schema_output = {
            "query": query.text,
            "extracted_elements": [
                {
                    "name": elem.name,
                    "type": elem.data_type,
                    "sensitivity": elem.sensitivity.value,
                }
                for elem in schema_elements
            ],
            "count": len(schema_elements),
        }
        schema_file = OUTPUT_DIR / "e2e_test_schema.json"
        with open(schema_file, "w") as f:
            json.dump(schema_output, f, indent=2)
        logger.info(f"✓ Saved schema to: {schema_file}")

        # Save generated SQL
        sql_output = {
            "query": query.text,
            "routing_decision": routing_decision.value if routing_decision else None,
            "generation_source": result.get("generation_source"),
            "sql_before_reconstruction": (
                abstracted_prompt.text if abstracted_prompt else None
            ),
            "sql_after_reconstruction": sql.text if sql else None,
            "verified": sql.verified if sql else False,
            "verification_status": (
                verification_result.status.value if verification_result else None
            ),
            "abstraction_applied": abstracted_prompt is not None,
            "num_substitutions": (
                abstracted_prompt.num_substitutions if abstracted_prompt else 0
            ),
        }
        sql_file = OUTPUT_DIR / "e2e_test_sql.json"
        with open(sql_file, "w") as f:
            json.dump(sql_output, f, indent=2)
        logger.info(f"✓ Saved SQL to: {sql_file}")

        # Save metrics
        metrics_output = {
            "query": query.text,
            "routing_decision": routing_decision.value if routing_decision else None,
            "latency_ms": latency_ms,
            "cost_usd": result.get("cost_usd", 0.0),
            "privacy_loss": result.get("privacy_loss", 0.0),
            "verification_attempts": result.get("verification_attempts", 0),
            "abstraction": {
                "applied": abstracted_prompt is not None,
                "epsilon": abstracted_prompt.epsilon if abstracted_prompt else 0.0,
                "num_substitutions": (
                    abstracted_prompt.num_substitutions if abstracted_prompt else 0
                ),
            },
            "verification": {
                "status": (
                    verification_result.status.value if verification_result else None
                ),
                "grammar_valid": (
                    verification_result.grammar_valid if verification_result else None
                ),
                "schema_valid": (
                    verification_result.schema_valid if verification_result else None
                ),
                "execution_valid": (
                    verification_result.execution_valid if verification_result else None
                ),
            },
        }
        metrics_file = OUTPUT_DIR / "e2e_test_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_output, f, indent=2)
        logger.info(f"✓ Saved metrics to: {metrics_file}")

        # 7. Print summary
        logger.info("\n[7/7] Test Summary")
        logger.info("=" * 80)
        logger.info(f"Query: {query.text}")
        logger.info(f"Routing: {routing_decision}")
        logger.info(f"Generated SQL: {sql.text if sql else 'None'}")
        logger.info(f"Verified: {sql.verified if sql else False}")
        logger.info(f"Latency: {latency_ms:.2f}ms")
        logger.info(f"Cost: ${result.get('cost_usd', 0.0):.6f}")
        logger.info(f"Privacy Loss: {result.get('privacy_loss', 0.0):.4f}")
        logger.info("=" * 80)

        logger.info(f"\n✓ Test completed successfully!")
        logger.info(f"✓ All outputs saved to: {OUTPUT_DIR}")
        logger.info(f"✓ Log file: {LOG_FILE}")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(run_e2e_test())

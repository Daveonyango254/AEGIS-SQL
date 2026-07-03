"""Schema-Linking agent (the Query-Planner stage).

Produces the focused schema slice the generators see. It reuses the existing
hybrid retriever (BGE-M3 dense+sparse + one-hop FK closure — table recall is
already ~99%, so we add no new retrieval logic) and the value sampler (grounds
text columns with real DB values so the model matches exact string literals).
"""

from typing import List, Optional, Tuple

from loguru import logger


class SchemaLinkerAgent:
    """Retrieve the relevant schema slice and attach grounded value hints."""

    def __init__(self, config) -> None:
        self.config = config

    def link(
        self, retriever, query, schema, db_path: Optional[str]
    ) -> Tuple[List, List[str], int]:
        """Return ``(schema_elements, retrieved_tables, num_columns)``.

        Args:
            retriever: a ready ``SchemaRetriever`` for this database (from the cache).
            query: the natural-language query.
            schema: the full ``Schema`` (used for the optional full-schema mode).
            db_path: SQLite path for value grounding (skipped if absent/in-memory).
        """
        slm = self.config.slm

        # Full-schema mode inlines every column when the DB is small enough;
        # otherwise use top-k hybrid retrieval + FK closure (the stable baseline).
        all_columns = [c for c in schema.columns if "." in c.name]
        if slm.full_schema and 0 < len(all_columns) <= slm.full_schema_max_columns:
            schema_elements = list(all_columns)
            logger.info(f"SchemaLinker: full schema ({len(schema_elements)} columns)")
        else:
            schema_elements = retriever.retrieve(
                query,
                top_k=slm.retrieval_top_k,
                expand_foreign_keys=True,
                max_expanded_tables=slm.max_expanded_tables,
            )

        # Value grounding: attach sampled DB values to text columns so the model
        # uses real literals (e.g. 'Continuation School', not 'Continuation').
        if slm.enable_value_grounding and db_path and db_path != ":memory:":
            try:
                from retriever.value_sampler import get_value_hints

                hints = get_value_hints(db_path, schema_elements, query.text)
                for elem in schema_elements:
                    vals = hints.get(elem.name)
                    if vals:
                        elem.example_values = vals
            except Exception as e:  # value grounding is best-effort, never fatal
                logger.warning(f"SchemaLinker: value grounding skipped ({e})")

        tables = sorted({e.name.split(".", 1)[0] for e in schema_elements if "." in e.name})
        logger.info(
            f"SchemaLinker: {len(schema_elements)} columns across {len(tables)} tables"
        )
        return schema_elements, tables, len(schema_elements)

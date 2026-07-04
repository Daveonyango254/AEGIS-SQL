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
        rag = getattr(self.config, "rag", None)

        # Full-schema mode inlines every column when the DB is small enough;
        # multi-step RAG (decompose -> fuse -> value-ground -> budget) is the
        # default; legacy single-shot top-k + FK closure remains the A/B control.
        all_columns = [c for c in schema.columns if "." in c.name]
        used_pipeline = False
        if slm.full_schema and 0 < len(all_columns) <= slm.full_schema_max_columns:
            schema_elements = list(all_columns)
            logger.info(f"SchemaLinker: full schema ({len(schema_elements)} columns)")
        elif rag is not None and rag.multi_step:
            from retriever.pipeline import MultiStepRetriever

            pipeline = MultiStepRetriever(retriever, schema, self.config)
            schema_elements = pipeline.retrieve(query, db_path)
            used_pipeline = bool(schema_elements)
            if not schema_elements:  # pass-through/model-less fallback
                schema_elements = retriever.retrieve(
                    query,
                    top_k=slm.retrieval_top_k,
                    expand_foreign_keys=True,
                    max_expanded_tables=slm.max_expanded_tables,
                )
        else:
            schema_elements = retriever.retrieve(
                query,
                top_k=slm.retrieval_top_k,
                expand_foreign_keys=True,
                max_expanded_tables=slm.max_expanded_tables,
            )

        # Value grounding: attach sampled DB values to text columns so the model
        # uses real literals (e.g. 'Continuation School', not 'Continuation').
        # IMPORTANT: the retriever returns the CACHED schema's element objects,
        # shared across queries — mutate copies (dataclasses.replace), never the
        # originals, or one query's value hints leak into the next query's prompt.
        if slm.enable_value_grounding and db_path and db_path != ":memory:":
            try:
                from dataclasses import replace

                from retriever.value_sampler import get_value_hints

                hints = get_value_hints(db_path, schema_elements, query.text)
                if used_pipeline:
                    # The pipeline attached EXACT stored literals from value
                    # retrieval — higher fidelity than sampled hints, so grounding
                    # only fills columns that have no values yet.
                    schema_elements = [
                        replace(e, example_values=hints[e.name])
                        if not e.example_values and e.name in hints else e
                        for e in schema_elements
                    ]
                else:
                    # Legacy semantics: query-linked hints replace stale examples.
                    schema_elements = [
                        replace(e, example_values=hints.get(e.name, e.example_values))
                        for e in schema_elements
                    ]
            except Exception as e:  # value grounding is best-effort, never fatal
                logger.warning(f"SchemaLinker: value grounding skipped ({e})")

        tables = sorted({e.name.split(".", 1)[0] for e in schema_elements if "." in e.name})
        logger.info(
            f"SchemaLinker: {len(schema_elements)} columns across {len(tables)} tables"
        )
        return schema_elements, tables, len(schema_elements)

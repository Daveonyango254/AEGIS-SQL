"""Schema stage for AEGIS v1 — recall-first.

Measured lesson (aegis_feat_2 vs the full-run baseline): table recall is
everything — pruning that drops one needed table caps that query's EX at 0,
and the literature agrees ("The Death of Schema Linking", CHASE-SQL's light
linking). BIRD-dev schemas are small, so the default is the FULL schema (also
the exact setting behind CSC-SQL's published 69.19% dev EX — its table-linking
stage is disabled). The multi-step linked retriever is the fallback for schemas
too large to inline, and it only caps COLUMNS per table, never tables.

Value grounding runs in both modes: question-linked literals are grounded to
their EXACT stored DB forms (value index) and low-cardinality text columns get
sampled values — both shown as example values in the DDL.
"""

from dataclasses import replace
from typing import List, Optional, Tuple

from loguru import logger


class SchemaLinkerAgent:
    """Produce the schema slice (elements + tables) for prompt rendering."""

    def __init__(self, config) -> None:
        self.config = config
        self.retrieval = config.retrieval

    def link(
        self, retriever, query, schema, db_path: Optional[str]
    ) -> Tuple[List, List[str], int]:
        """Return ``(schema_elements, retrieved_tables, num_columns)``."""
        all_columns = [c for c in schema.columns if "." in c.name]
        mode = self.retrieval.schema_mode
        use_full = mode == "full" or (
            mode == "auto" and len(all_columns) <= self.retrieval.full_max_columns
        )

        if use_full:
            schema_elements = list(all_columns)
            logger.info(f"SchemaLinker: FULL schema ({len(schema_elements)} columns)")
        else:
            from retriever.pipeline import MultiStepRetriever

            pipeline = MultiStepRetriever(retriever, schema, self.config)
            schema_elements = pipeline.retrieve(query, db_path)
            if not schema_elements:  # model-less/pass-through fallback
                schema_elements = list(all_columns)
                logger.warning("SchemaLinker: linked retrieval empty; using full schema")

        schema_elements = self._ground_values(schema_elements, query, db_path)

        tables = sorted({e.name.split(".", 1)[0] for e in schema_elements})
        logger.info(
            f"SchemaLinker: {len(schema_elements)} columns across {len(tables)} tables"
        )
        return schema_elements, tables, len(schema_elements)

    def _ground_values(self, elements: List, query, db_path: Optional[str]) -> List:
        """Attach value hints (copies only — cached elements are shared across queries).

        Exact stored literals from the value index take priority; generic
        query-linked / low-cardinality samples fill the remaining columns.
        """
        if not db_path or db_path == ":memory:":
            return elements
        try:
            exact = {}
            if self.retrieval.value_retrieval:
                from retriever.query_decompose import decompose
                from retriever.value_index import find_value_columns

                dq = decompose(query.text, getattr(query, "evidence", "") or "")
                hits = find_value_columns(
                    db_path, elements, dq.literals,
                    max_probes=self.retrieval.max_value_probes,
                )
                for pairs in hits.values():
                    for col, stored in pairs:
                        exact.setdefault(col, [])
                        if stored not in exact[col]:
                            exact[col].append(stored)

            from retriever.value_sampler import get_value_hints

            hints = get_value_hints(db_path, elements, query.text)
            out = []
            for e in elements:
                vals = exact.get(e.name) or hints.get(e.name) or e.example_values
                out.append(replace(e, example_values=vals) if vals is not e.example_values else e)
            return out
        except Exception as e:  # grounding is best-effort, never fatal
            logger.warning(f"SchemaLinker: value grounding skipped ({e})")
            return elements

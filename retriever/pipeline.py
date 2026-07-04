"""Multi-step, agent-driven schema retrieval (the RAG v2 pipeline).

Replaces single-shot top-k retrieval with a bounded agentic loop:

    decompose ──> per-sub-query hybrid search ──> RRF fusion + metadata boosts
        │                                              │
        └── literals ──> value retrieval (DB probes) ──┤
                                                       ▼
                     adaptive budget (evidence tables + FK bridges + column caps)
                                                       │
                     coverage evaluation ── unmatched entity? ──> relaxed round 2
                                                       │
                     optional ColBERT re-rank ─────────┘

Design rationale (measured on the 1,534-query BIRD run):
* Simple questions received 7.5 tables for the 1.76 they needed — the noise that
  inverted simple-vs-moderate accuracy. The adaptive budget fixes the ratio.
* 247 predictions used the wrong table SET, concentrated in FK-maze databases
  (financial 43%). FK bridge insertion keeps every slice join-connected.
* Literal mismatches dominate single-table failures; value retrieval grounds the
  exact stored literal deterministically.

The pipeline consumes the existing ``SchemaRetriever`` scoring machinery (one
encode per sub-query on precomputed embeddings) — no second embedding model, no
new heavy dependencies.
"""

from typing import Dict, List, Set, Tuple

from loguru import logger

from retriever.fusion import adaptive_budget, apply_boosts, rrf_fuse
from retriever.query_decompose import decompose
from retriever.value_index import find_value_columns

# Table-card hits raise every member column before budgeting: a whole-table
# semantic match ("orders placed by clients" ~ table `order`) is table-level
# evidence that individual column embeddings can miss.
TABLE_CARD_WEIGHT = 0.5

_NUMERIC_TYPES = ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")


class MultiStepRetriever:
    """Agentic multi-step retrieval over one database's schema."""

    def __init__(self, retriever, schema, config) -> None:
        """
        Args:
            retriever: a ready ``SchemaRetriever`` (cached embeddings) for this DB.
            schema: the populated ``Schema`` (FKs / PKs for bridging and budget).
            config: the loaded AEGIS config (``config.rag`` holds the knobs).
        """
        self.retriever = retriever
        self.schema = schema
        self.rag = config.rag
        self.by_name = {c.name: c for c in schema.columns if "." in c.name}

    def retrieve(self, query, db_path: str) -> List:
        """Run the multi-step loop; returns the final ordered SchemaElement slice.

        Returns ``[]`` when the retriever cannot produce scored rankings (legacy
        retriever, pass-through mode) — the caller falls back to single-shot
        retrieval rather than failing the query.
        """
        if not hasattr(self.retriever, "retrieve_scored"):
            return []
        dq = decompose(query.text, getattr(query, "evidence", "") or "",
                       max_sub_queries=self.rag.max_sub_queries)

        value_hits: Dict[str, List[Tuple[str, str]]] = {}
        chosen: List[str] = []
        for round_idx in range(max(1, self.rag.max_rounds)):
            relaxed = round_idx > 0

            # --- Step 1: per-sub-query hybrid search (dense+sparse, scored) ----
            rankings: List[List[str]] = []
            for sub in dq.sub_queries:
                ranked = self.retriever.retrieve_scored(sub, top_k=self.rag.per_query_top_k)
                rankings.append([e.name for e, _ in ranked])
            if not rankings or not rankings[0]:
                break  # retriever in pass-through mode; caller falls back

            # --- Step 2: rank fusion across sub-queries ------------------------
            scores = rrf_fuse(rankings)

            # Table-card evidence lifts every column of a matched table.
            if self.rag.table_cards:
                card_scores = self.retriever.score_table_cards(dq.sub_queries[0])
                for name in scores:
                    t = name.split(".", 1)[0]
                    if t in card_scores:
                        scores[name] += TABLE_CARD_WEIGHT * card_scores[t]

            # --- Step 3: value retrieval (deterministic literal -> column) -----
            if self.rag.value_retrieval and dq.literals:
                pool = [self.by_name[n] for n in scores if n in self.by_name]
                value_hits = find_value_columns(
                    db_path, pool, dq.literals,
                    max_probes=self.rag.max_value_probes, relaxed=relaxed,
                )
            hit_columns: Set[str] = {
                col for pairs in value_hits.values() for col, _ in pairs
            }

            # --- Step 4: metadata boosts + adaptive budget ----------------------
            question_tokens = {
                t.lower().strip(".,;:!?'\"") for t in query.text.split() if len(t) > 2
            }
            numeric_columns = {
                n for n in scores
                if n in self.by_name
                and any(t in (self.by_name[n].data_type or "").upper() for t in _NUMERIC_TYPES)
            }
            boosted = apply_boosts(
                scores, question_tokens, hit_columns,
                dq.wants_aggregate, bool(dq.numbers), numeric_columns,
            )
            chosen = adaptive_budget(
                boosted, hit_columns, self.schema,
                max_tables=self.rag.max_tables,
                per_table_columns=self.rag.per_table_columns,
            )

            # --- Step 5: coverage evaluation (drives the recursive round) ------
            if not self._uncovered(dq.literals, value_hits, chosen) or relaxed:
                break
            logger.debug("MultiStepRetriever: uncovered entities -> relaxed round")

        # --- Step 6: optional ColBERT re-rank of the final slice ---------------
        if self.rag.rerank and chosen:
            chosen = self._rerank(query.text, chosen)

        elements = [self.by_name[n] for n in chosen if n in self.by_name]

        # Attach the EXACT stored literal for every value hit — the strongest
        # value-grounding hint available (copies; never mutate cached elements).
        if value_hits:
            from dataclasses import replace

            exact: Dict[str, List[str]] = {}
            for pairs in value_hits.values():
                for col, stored in pairs:
                    exact.setdefault(col, [])
                    if stored not in exact[col]:
                        exact[col].append(stored)
            elements = [
                replace(e, example_values=exact.get(e.name, e.example_values))
                if e.name in exact else e
                for e in elements
            ]

        logger.info(
            f"MultiStepRetriever: {len(elements)} columns / "
            f"{len({e.name.split('.', 1)[0] for e in elements})} tables, "
            f"{len(value_hits)} literals grounded, {len(dq.sub_queries)} sub-queries"
        )
        return elements

    @staticmethod
    def _uncovered(literals, value_hits, chosen) -> bool:
        """True when some literal has neither a value hit nor a name-matched column."""
        chosen_tokens = " ".join(chosen).lower()
        for lit in literals:
            if lit in value_hits:
                continue
            head = lit.split()[0].lower()
            if head not in chosen_tokens:
                return True
        return False

    def _rerank(self, question: str, chosen: List[str]) -> List[str]:
        """Re-order the final slice with BGE-M3 ColBERT interaction scores.

        Uses the SAME embedding model (no new dependency): multi-vector scoring
        captures token-level interaction that pooled dense vectors blur. Guarded:
        any failure (older FlagEmbedding, CPU fallback) keeps the fused order.
        """
        try:
            texts = []
            for name in chosen:
                e = self.by_name.get(name)
                desc = f": {e.description}" if e is not None and e.description else ""
                texts.append(f"{name}{desc}")
            model = self.retriever.model
            q = model.encode([question], return_dense=False, return_sparse=False,
                             return_colbert_vecs=True)["colbert_vecs"][0]
            docs = model.encode(texts, return_dense=False, return_sparse=False,
                                return_colbert_vecs=True)["colbert_vecs"]
            scored = sorted(
                zip(chosen, (float(model.colbert_score(q, d)) for d in docs)),
                key=lambda t: -t[1],
            )
            return [name for name, _ in scored]
        except Exception as e:
            logger.debug(f"MultiStepRetriever: rerank skipped ({e})")
            return chosen

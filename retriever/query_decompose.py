"""Query decomposition for multi-step schema retrieval (stdlib only).

A single embedding of the whole question under-weights individual entities: in
"List the phones of schools in Fresno with SAT scores above 500", the tokens
"phone" and "Fresno" each need to find their own column, but the full-sentence
embedding is dominated by the overall topic. We therefore decompose the question
into (a) grounded artefacts — quoted strings, proper-noun spans, numbers — and
(b) focused sub-queries, and retrieve per sub-query before fusing the rankings.

Evidence strings (BIRD's analyst hints like "K-12 refers to Enrollment (K-12)")
are mined for explicit column references, which become high-precision sub-queries.
"""

import re
from dataclasses import dataclass, field
from typing import List

# Words that start questions/clauses and should never be treated as entities.
_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "list", "give", "show", "find", "name", "state", "please", "for", "the",
    "of", "in", "on", "at", "by", "and", "or", "with", "from", "to", "between",
    "please", "calculate", "identify", "indicate", "mention", "provide", "write",
    "among", "please", "tell", "count", "many", "much", "average", "total",
}

_QUOTED_RE = re.compile(r"""["'`]([^"'`]{1,60})["'`]""")
# Proper-noun span: 1-4 consecutive Capitalized/ALLCAPS words (may include digits),
# e.g. "Fresno County Office", "K-12", "OPEC". Requires a letter to avoid years.
_PROPER_RE = re.compile(r"\b([A-Z][\w\-']*(?:\s+[A-Z][\w\-']*){0,3})\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
# Evidence patterns: "X refers to Y", "X = Y", "X means Y"
_EVIDENCE_REF_RE = re.compile(
    r"([\w\s().%/-]{2,50}?)\s*(?:refers to|means|=)\s*([^;,.]{2,80})", re.IGNORECASE
)


@dataclass
class DecomposedQuery:
    """The retrieval-relevant artefacts extracted from a question + evidence.

    Attributes:
        literals: value-like strings to locate in the DB (quoted text, proper nouns).
        numbers: numeric literals (for type-expectation boosting, not value lookup).
        sub_queries: focused retrieval strings, first entry is always the full
            question (+ evidence) so decomposition can only ADD recall, not lose it.
        wants_aggregate: question asks for count/avg/sum -> boost numeric columns.
    """

    literals: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    wants_aggregate: bool = False


def decompose(question: str, evidence: str = "", max_sub_queries: int = 6) -> DecomposedQuery:
    """Decompose a question (+ evidence) into literals and focused sub-queries."""
    question = (question or "").strip()
    evidence = (evidence or "").strip()

    literals: List[str] = []
    seen = set()

    def add_literal(text: str) -> None:
        text = text.strip().strip(".,;:!?")
        key = text.lower()
        # Skip stopwords, bare numbers, and one-letter fragments.
        if len(text) < 2 or key in _STOPWORDS or text.isdigit():
            return
        if key not in seen:
            seen.add(key)
            literals.append(text)

    # Quoted strings are the strongest value signal (exact literals).
    for src in (question, evidence):
        for m in _QUOTED_RE.findall(src):
            add_literal(m)

    # Proper-noun spans, skipping the sentence-initial word (capitalized by
    # grammar, not because it names an entity).
    for src in (question, evidence):
        for m in _PROPER_RE.finditer(src):
            if m.start() == 0:
                span = m.group(1).split()
                # keep the tail if the initial word is a question word
                if span and span[0].lower() in _STOPWORDS:
                    rest = " ".join(span[1:])
                    if rest:
                        add_literal(rest)
                    continue
            add_literal(m.group(1))

    numbers = list(dict.fromkeys(_NUMBER_RE.findall(question)))

    # Sub-queries: full question first (baseline recall), then evidence column
    # references, then one per entity (helps each entity find its own column).
    subs: List[str] = [f"{question} {evidence}".strip()]
    for lhs, rhs in _EVIDENCE_REF_RE.findall(evidence):
        subs.append(f"{lhs.strip()} {rhs.strip()}")
    for lit in literals:
        subs.append(lit)

    # Dedup preserving order, cap for bounded retrieval cost.
    deduped: List[str] = []
    sseen = set()
    for s in subs:
        k = s.lower()
        if s and k not in sseen:
            sseen.add(k)
            deduped.append(s)

    wants_aggregate = bool(
        re.search(r"\b(how many|count|number of|average|avg|total|sum|ratio|percentage|highest|lowest|most|least)\b",
                  question, re.IGNORECASE)
    )
    return DecomposedQuery(
        literals=literals,
        numbers=numbers,
        sub_queries=deduped[:max_sub_queries],
        wants_aggregate=wants_aggregate,
    )

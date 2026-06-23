"""Retrieval diagnostics for BIRD predictions (table recall vs noise).

Separates *retrieval* failures from *generation/prompt* failures by comparing,
per query:
  - GT tables (parsed from the ground-truth SQL),
  - retrieved tables (recorded by the schema retriever, `retrieved_tables`), and
  - predicted tables (parsed from the predicted SQL).

Key metrics:
  * table_recall  = fraction of queries where GT tables ⊆ retrieved tables
                    (did retrieval even contain what the query needs?).
  * gen_table_match = fraction where predicted tables == GT tables
                    (did the model pick the right tables, given retrieval?).
  * avg_extra_tables / avg_retrieved_cols = retrieval noise.

If table_recall is high but gen_table_match is low, the bottleneck is the
prompt/model (the tables were retrieved but not used) — not retrieval.

Stdlib only. Usage:  python evaluation/analyze_retrieval.py PREDICTIONS.jsonl
"""

import json
import re
import sys
from collections import defaultdict

_TABLE_RE = re.compile(r'(?:FROM|JOIN)\s+["`\[]?([A-Za-z_][\w]*)["`\]]?', re.IGNORECASE)
# SQL keywords that can follow FROM but are not tables (e.g. subqueries).
_NON_TABLES = {"select", "lateral"}


def extract_tables(sql):
    """Return the lower-cased set of table names referenced in a SQL string."""
    if not sql:
        return set()
    return {t.lower() for t in _TABLE_RE.findall(sql) if t.lower() not in _NON_TABLES}


def analyze(rows):
    """Compute retrieval/generation table metrics overall and by difficulty."""
    buckets = defaultdict(lambda: {
        "n": 0, "ret_n": 0, "recall_hit": 0, "gen_match": 0,
        "extra_tables": 0, "retrieved_cols": 0, "retrieval_misses": [],
    })

    for r in rows:
        diff = r.get("difficulty", "unknown")
        gt = extract_tables(r.get("ground_truth_sql"))
        pred = extract_tables(r.get("predicted_sql"))
        retrieved = {t.lower() for t in (r.get("retrieved_tables") or [])}
        if not gt:
            continue
        for key in (diff, "OVERALL"):
            b = buckets[key]
            b["n"] += 1
            if retrieved:  # only count recall/noise where retrieval data exists
                b["ret_n"] += 1
                if gt <= retrieved:
                    b["recall_hit"] += 1
                else:
                    b["retrieval_misses"].append(
                        (r.get("question_id"), sorted(gt - retrieved))
                    )
                b["extra_tables"] += max(0, len(retrieved - gt))
                b["retrieved_cols"] += r.get("num_retrieved_columns", 0) or 0
            if pred and pred == gt:
                b["gen_match"] += 1
    return buckets


def _pct(num, den):
    return f"{100.0 * num / den:5.1f}%" if den else "  n/a"


def main(path):
    rows = [json.loads(l) for l in open(path)]
    buckets = analyze(rows)
    order = ["simple", "moderate", "challenging", "unknown", "OVERALL"]
    print(f"\nRetrieval diagnostics for {path}  (n={len(rows)})")
    print("=" * 78)
    print(f"{'bucket':<12} {'n':>4} {'table_recall':>13} {'gen_match':>10} "
          f"{'avg_extra_tbl':>13} {'avg_ret_cols':>12}")
    for key in order:
        if key not in buckets:
            continue
        b = buckets[key]
        n = b["n"]
        if not n:
            continue
        rn = b["ret_n"]  # rows with retrieval data (recall/noise denominator)
        extra = f"{b['extra_tables']/rn:>13.1f}" if rn else f"{'n/a':>13}"
        cols = f"{b['retrieved_cols']/rn:>12.1f}" if rn else f"{'n/a':>12}"
        print(f"{key:<12} {n:>4} {_pct(b['recall_hit'], rn):>13} "
              f"{_pct(b['gen_match'], n):>10} {extra} {cols}")
    print("=" * 78)
    misses = buckets["OVERALL"]["retrieval_misses"]
    print(f"\nRETRIEVAL MISSES (GT table not retrieved): {len(misses)}")
    for qid, missing in misses[:20]:
        print(f"  q{qid}: missing {missing}")
    print(
        "\nInterpretation: high table_recall + low gen_match => prompt/model issue "
        "(tables retrieved but unused). Low table_recall => true retrieval gap."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python evaluation/analyze_retrieval.py PREDICTIONS.jsonl")
        raise SystemExit(2)
    main(sys.argv[1])

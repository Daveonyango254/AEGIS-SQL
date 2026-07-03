"""Reviewer agent — the 3-stage neuro-symbolic verifier.

A thin wrapper over the shared ``verifier.review.run_verification`` so the booster
records exactly the same grammar/schema/execution verdict as the LangGraph
pipeline. Refinement has already happened by the time we review, so the
empty-result soft-fail is disabled (a query that runs is reported as passing).
"""

from verifier.review import run_verification

# Large sentinel so flag_empty_for_repair never marks an already-refined,
# executed-but-empty result as a failure.
_NO_MORE_REPAIRS = 10_000


class ReviewerAgent:
    """Verify the final SQL and produce a VerificationResult for the record."""

    def __init__(self, config) -> None:
        self.vcfg = getattr(config, "verifier", None)

    def review(self, sql, schema, db_path):
        return run_verification(
            sql, schema, db_path, vcfg=self.vcfg, generation_count=_NO_MORE_REPAIRS
        )

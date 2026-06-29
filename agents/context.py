"""Per-query working state passed between the booster's agents.

Keeping the cross-agent state in one explicit, well-documented object (rather than
a mutable global or a god-graph) is what makes the pipeline easy to read and test:
each agent takes a ``RunContext`` plus its own inputs and returns plain values.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional


# A model-agnostic generation function. Both the local SLM and the remote LLM
# expose a ``complete(prompt, n, temperature, system_prompt) -> List[str]`` method;
# the orchestrator adapts whichever model is on the active path to this signature.
GenerateFn = Callable[[str, int, float, Optional[str]], List[str]]


@dataclass
class RunContext:
    """Everything an agent needs to act on a single query.

    Attributes:
        query: the real natural-language query (used for execution, verification,
            and the trusted local judge — never abstracted).
        gen_query: the query used to *build prompts*. Identical to ``query`` on the
            local path; on the remote path its ``.text`` carries DP placeholders so
            no sensitive value crosses the trust boundary.
        schema: the full populated ``Schema`` (real foreign/primary keys).
        schema_elements: the retrieved column slice (with grounded value hints).
        db_path: path to the SQLite database for execution (None / ":memory:" disables it).
        route: LOCAL or REMOTE routing decision.
        source: "slm" (local) or "llm" (remote) — recorded in the prediction.
        generate_fn: model-agnostic generator for the active path.
        reconstruct_fn: maps a generated SQL string back to real tokens. Identity on
            the local path; placeholder→real reconstruction on the remote path.
        recon_map: the abstraction reconstruction map (remote only) — used to
            sanitize verifier feedback before it is shown to the remote model.
        expose_keys: render FK/PK hints in prompts.
        config: the loaded AEGIS configuration.
    """

    query: object
    gen_query: object
    schema: object
    schema_elements: List
    db_path: Optional[str]
    config: object
    route: object = None
    source: str = "slm"
    generate_fn: Optional[GenerateFn] = None
    reconstruct_fn: Callable[[str], str] = field(default=lambda s: s)
    recon_map: object = None
    expose_keys: bool = True

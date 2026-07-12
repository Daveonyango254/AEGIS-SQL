"""SQL generation: local SLM, remote LLM, and the CSC selection engine.

Imports are lazy (PEP 562) so pure modules (csc, sql_postprocess,
candidate_selector) are importable without torch/transformers/openai — needed
for offline unit tests and lightweight tooling.
"""


def __getattr__(name):
    if name == "SLMGenerator":
        from generator.slm_generator import SLMGenerator
        return SLMGenerator
    if name == "LLMFallback":
        from generator.llm_fallback import LLMFallback
        return LLMFallback
    raise AttributeError(f"module 'generator' has no attribute {name!r}")


__all__ = ["SLMGenerator", "LLMFallback"]

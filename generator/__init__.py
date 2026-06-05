"""SQL generation modules for local SLM and remote LLM.

Implements FSLM (fine-tuned small language model) for local generation
and FLLM (foundation large language model) for remote fallback.

"""

from generator.slm_generator import SLMGenerator
from generator.llm_fallback import LLMFallback

__all__ = ["SLMGenerator", "LLMFallback"]

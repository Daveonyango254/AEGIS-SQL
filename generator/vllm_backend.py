"""Optional vLLM backend for local SLM sampling (5-10x faster than HF generate).

vLLM's continuous batching + paged KV cache makes n-sample generation roughly an
order of magnitude faster than sequential/naive-batched ``model.generate`` — the
difference between a 5-hour and a 30-minute 100-query evaluation.

Exposes the same ``complete(prompt, n, temperature, max_tokens, system_prompt,
raw)`` contract as ``SLMGenerator`` so the orchestrator is backend-agnostic.
Guarded import: when vLLM is not installed the model cache falls back to the HF
path automatically (``models.backend: auto``).

Two engines can share one GPU: `gpu_memory_utilization` is per-engine and the
config splits it (generator 0.55 / merger 0.35 by default on a 48GB card).
"""

from typing import List, Optional

from loguru import logger

from generator.sql_postprocess import extract_sql, finalize_sql


class VllmGenerator:
    """A local SLM served by vLLM, matching SLMGenerator's complete() contract."""

    def __init__(self, model_id: str, gpu_fraction: float, config) -> None:
        """
        Args:
            model_id: HF model id to serve.
            gpu_fraction: vLLM ``gpu_memory_utilization`` for THIS engine (engines
                sharing a GPU must not sum above ~0.92).
            config: internal SLMConfig (dtype / max_tokens / cache dir).
        """
        from vllm import LLM  # guarded by the caller

        self.config = config
        self.model_id = model_id
        logger.info(f"Loading vLLM engine for {model_id} (gpu_fraction={gpu_fraction})")
        self.llm = LLM(
            model=model_id,
            dtype=config.torch_dtype,
            gpu_memory_utilization=gpu_fraction,
            download_dir=str(config.cache_dir) if config.cache_dir else None,
            trust_remote_code=config.trust_remote_code,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def complete(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        raw: bool = False,
    ) -> List[str]:
        """One greedy + (n-1) sampled completions in a single continuous batch."""
        from vllm import SamplingParams

        max_tokens = max_tokens or self.config.max_tokens
        temperature = (
            temperature if temperature is not None
            else getattr(self.config, "selection_temperature", 0.8)
        )
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Greedy anchor + temperature samples, submitted together — vLLM batches
        # them in one pass (top_p left at 1.0, the CSC-SQL setting).
        params = [SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens)]
        if n > 1 and temperature > 0:
            params.append(SamplingParams(n=n - 1, temperature=temperature,
                                         max_tokens=max_tokens))
        outputs = []
        for p in params:
            for out in self.llm.generate([text], p, use_tqdm=False):
                outputs.extend(o.text for o in out.outputs)

        if raw:
            return [t.strip() for t in outputs if t and t.strip()]
        finalized = (
            finalize_sql(extract_sql(t),
                         enable_cast_fix=getattr(self.config, "enable_cast_fix", True))
            for t in outputs
        )
        return [s for s in finalized if s]


def vllm_available() -> bool:
    """True when the vllm package is importable (backend 'auto' probe)."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False

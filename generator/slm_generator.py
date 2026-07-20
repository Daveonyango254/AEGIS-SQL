"""Local small language model (FSLM) for SQL generation.

Implements on-premises SQL generation using fine-tuned code models like
CodeLlama, DeepSeek-Coder, or similar SLMs optimized for text-to-SQL.

References:
    - Build strategy Section 2.1: SLM selection and fine-tuning
    - Paper Section 3: Local path has zero privacy leakage
"""

import os
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

from config import SLMConfig
from aegis_types import Query, SchemaElement, SQL
from prompts.prompt_manager import get_prompt_manager
from generator.sql_postprocess import finalize_sql

# Default system prompt used when templates.yaml does not define one.
_DEFAULT_SLM_SYSTEM_PROMPT = (
    "You are an expert text-to-SQL generator for the SQLite/BIRD benchmark. "
    "Given a database schema and a question, output a single valid SQLite query. "
    "Use the exact column names and literal values shown in the schema "
    "(prefer values listed under 'examples:'). For ratios or averages of "
    "integer columns, cast the numerator with CAST(... AS REAL) to avoid "
    "integer division. Return only the SQL query."
)


class SLMGenerator:
    """Local SLM generator for SQL (FSLM).

    Uses a fine-tuned small language model for on-premises SQL generation.
    Operates on real, non-abstracted data with zero privacy leakage.

    Recommended models:
        - CodeLlama-7B/13B-Instruct (fine-tuned on BIRD-train)
        - DeepSeek-Coder-6.7B-Instruct
        - StarCoder-15B
        - Phi-3-medium (3.8B) for code

    Attributes:
        config: SLM configuration
        model: Loaded language model instance
        tokenizer: Model tokenizer
        device: Compute device (cuda/cpu)
    """

    def __init__(self, config: SLMConfig) -> None:
        """Initialize SLM generator.

        Args:
            config: SLM configuration
        """
        self.config = config

        # Get HuggingFace token from environment if using ${} syntax
        hf_token = config.hf_token
        if hf_token.startswith("${") and hf_token.endswith("}"):
            env_var = hf_token[2:-1]
            hf_token = os.getenv(env_var)
            if not hf_token:
                logger.warning(f"Environment variable {env_var} not set, proceeding without token")

        # Expand cache directory
        cache_dir = Path(config.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading SLM from HuggingFace: {config.model}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model,
                token=hf_token,
                cache_dir=str(cache_dir),
                trust_remote_code=config.trust_remote_code,
            )

            # Resolve dtype. transformers >= 4.56 renamed the `torch_dtype`
            # argument to `dtype` (and deprecated the old name); pick whichever
            # the installed version accepts.
            dtype = getattr(torch, config.torch_dtype)
            import transformers
            _tf_ver = tuple(int(p) for p in transformers.__version__.split(".")[:2])
            dtype_kwarg = "dtype" if _tf_ver >= (4, 56) else "torch_dtype"

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model,
                device_map=config.device,
                token=hf_token,
                cache_dir=str(cache_dir),
                trust_remote_code=config.trust_remote_code,
                **{dtype_kwarg: dtype},
            )

            # Load LoRA adapter if specified
            if config.adapter_path:
                from peft import PeftModel
                logger.info(f"Loading LoRA adapter from {config.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, config.adapter_path)

            self.device = config.device
            logger.info(f"✓ SLMGenerator initialized with {config.model}")

        except Exception as e:
            logger.error(f"Failed to load SLM: {e}")
            logger.warning("Falling back to stub mode")
            self.model = None
            self.tokenizer = None
            self.device = "cpu"

    def generate(
        self,
        query: Query,
        schema_elements: List[SchemaElement],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        schema=None,
    ) -> SQL:
        """Generate SQL from natural language query using local SLM.

        Args:
            query: Natural language query (NOT abstracted - real data)
            schema_elements: Retrieved schema elements (real names)
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)

        Returns:
            Generated SQL query
        """
        # Use config defaults if not specified
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        # If model not loaded, use stub mode
        if self.model is None or self.tokenizer is None:
            logger.warning("SLM not loaded, using stub mode")
            return self._generate_stub(schema_elements)

        logger.debug(f"Generating SQL with FSLM: {self.config.model}")

        try:
            inputs = self._build_inputs(query, schema_elements, schema=schema)
            texts = self._run_generation(
                inputs,
                max_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                num_return_sequences=1,
            )
            generated_text = texts[0] if texts else ""
            logger.debug(f"Raw model output (first 500 chars): {generated_text[:500]}")
            sql_text = self._finalize(generated_text)
            logger.debug(f"Extracted SQL: {sql_text}")

            return SQL(text=sql_text, dialect="sqlite", source="slm", verified=False)

        except Exception as e:
            logger.error(f"SLM generation failed: {e}")
            logger.warning("Falling back to stub mode")
            return self._generate_stub(schema_elements)

    def generate_candidates(
        self,
        query: Query,
        schema_elements: List[SchemaElement],
        n: int = 1,
        temperature: Optional[float] = None,
        feedback: Optional[str] = None,
        max_tokens: Optional[int] = None,
        schema=None,
    ) -> List[SQL]:
        """Generate N candidate SQL queries for execution-guided selection.

        The first candidate is always a deterministic greedy decode (used as the
        tie-breaker downstream); the remaining ``n-1`` candidates are temperature
        samples. Optional ``feedback`` is appended for self-correction retries.

        Args:
            query: Natural language query (evidence already folded into text)
            schema_elements: Retrieved schema elements (with optional value hints)
            n: Total number of candidates to return
            temperature: Sampling temperature for non-greedy candidates
            feedback: Structured verifier feedback for a repair attempt
            max_tokens: Optional override for max new tokens

        Returns:
            List of SQL candidates (greedy first). Falls back to a single stub
            candidate if the model is unavailable.
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = (
            temperature if temperature is not None
            else getattr(self.config, "selection_temperature", 0.8)
        )

        if self.model is None or self.tokenizer is None:
            logger.warning("SLM not loaded, using stub mode")
            return [self._generate_stub(schema_elements)]

        try:
            inputs = self._build_inputs(query, schema_elements, feedback=feedback, schema=schema)

            texts: List[str] = []
            # Candidate 1: deterministic greedy decode.
            texts.extend(
                self._run_generation(
                    inputs, max_tokens=max_tokens, do_sample=False,
                    temperature=0.0, num_return_sequences=1,
                )
            )
            # Candidates 2..n: temperature samples (memory-bounded chunks).
            if n > 1 and temperature > 0:
                texts.extend(
                    self._sample_chunked(
                        inputs, max_tokens=max_tokens,
                        temperature=temperature, n_samples=n - 1,
                    )
                )

            candidates = []
            for t in texts:
                sql_text = self._finalize(t)
                if sql_text:
                    candidates.append(
                        SQL(text=sql_text, dialect="sqlite", source="slm", verified=False)
                    )

            if not candidates:
                candidates = [self._generate_stub(schema_elements)]
            return candidates

        except Exception as e:
            logger.error(f"SLM candidate generation failed: {e}")
            logger.warning("Falling back to stub mode")
            return [self._generate_stub(schema_elements)]

    def complete(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        raw: bool = False,
    ) -> List[str]:
        """Model-agnostic completion from a prebuilt prompt (booster interface).

        Returns up to ``n`` finalized SQL strings: one greedy decode plus
        ``n-1`` temperature samples. Used by the multi-agent generator to drive
        arbitrary reasoning strategies through the same model plumbing as
        ``generate_candidates``.

        ``raw=True`` skips SQL extraction and returns the decoded text verbatim —
        required for non-SQL replies such as the selection judge's candidate
        index (a bare "2" has no SELECT, so the extractor reduced it to "" and
        the judge silently never fired). Returns ``[]`` if the model is
        unavailable so the caller can fall back.
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = (
            temperature if temperature is not None
            else getattr(self.config, "selection_temperature", 0.8)
        )
        if self.model is None or self.tokenizer is None:
            logger.warning("SLM not loaded; complete() returns no candidates")
            return []

        try:
            inputs = self._build_inputs(
                None, None, user_content=prompt, system_prompt=system_prompt
            )
            texts = self._run_generation(
                inputs, max_tokens=max_tokens, do_sample=False,
                temperature=0.0, num_return_sequences=1,
            )
            if n > 1 and temperature > 0:
                texts.extend(self._sample_chunked(
                    inputs, max_tokens=max_tokens,
                    temperature=temperature, n_samples=n - 1,
                ))
            if raw:
                return [t.strip() for t in texts if t and t.strip()]
            return [s for s in (self._finalize(t) for t in texts) if s]
        except Exception as e:
            logger.error(f"SLM complete() failed: {e}")
            return []

    def _build_inputs(
        self,
        query: Query,
        schema_elements: List[SchemaElement],
        feedback: Optional[str] = None,
        schema=None,
        user_content: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """Build tokenized model inputs, applying the chat template when available.

        ``user_content`` overrides the default DDL prompt (used by the multi-agent
        booster to drive alternative reasoning strategies); ``system_prompt``
        overrides the default system message (e.g. a chain-of-thought system prompt).
        """
        if user_content is None:
            user_content = self._format_prompt(query, schema_elements, schema=schema)
        if feedback:
            user_content += (
                f"\n\n-- The previous attempt was rejected by the verifier:\n"
                f"-- {feedback}\n-- Generate a corrected SQL query.\n-- SQL:"
            )

        prompt_text = user_content
        chat_applied = False
        use_chat = getattr(self.config, "use_chat_template", True)
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if use_chat and chat_template:
            try:
                prompt_mgr = get_prompt_manager()
                system_prompt = (
                    system_prompt
                    or prompt_mgr.get_slm_system_prompt()
                    or _DEFAULT_SLM_SYSTEM_PROMPT
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                chat_applied = True
            except Exception as e:
                logger.warning(f"Chat template failed ({e}); using raw prompt")
                prompt_text = user_content

        max_len = getattr(self.config, "max_input_length", 8192)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            # The chat template already injects special tokens; don't double them.
            add_special_tokens=not chat_applied,
        )

        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def _sample_chunked(self, inputs, max_tokens: int, temperature: float,
                        n_samples: int) -> List[str]:
        """Decode ``n_samples`` temperature samples in memory-bounded chunks.

        Decoding all samples in one ``model.generate`` batch OOMs small GPUs at
        higher ``num_candidates`` — and an OOM here used to silently collapse the
        whole candidate pool into a stub (the 3%-EX disaster mode). Instead:

        * at most ``config.local_chunk_size`` sequences per call (identical
          sampling params, so outputs are statistically unchanged);
        * a DISTINCT seed per chunk (``generation_seed + produced``) so seeded
          runs stay reproducible without chunks duplicating each other;
        * on CUDA OOM: empty the cache, halve the chunk (floor 1) and retry —
          LOUDLY; if even a single sequence cannot decode, return what we have
          (the greedy candidate still competes) rather than raising into stub.

        Model-agnostic: plain HF generate over whatever ``slm.model`` is loaded.
        """
        texts: List[str] = []
        chunk = max(1, int(getattr(self.config, "local_chunk_size", 2)))
        base_seed = getattr(self.config, "generation_seed", None)
        produced = 0
        while produced < n_samples:
            size = min(chunk, n_samples - produced)
            seed = (base_seed + produced) if base_seed is not None else None
            try:
                texts.extend(
                    self._run_generation(
                        inputs, max_tokens=max_tokens, do_sample=True,
                        temperature=temperature, num_return_sequences=size,
                        seed=seed,
                    )
                )
                produced += size
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    logger.error(
                        "CUDA OOM decoding even a SINGLE sampled sequence — "
                        f"returning {produced}/{n_samples} samples. Free GPU "
                        "memory (check for stale processes with nvidia-smi) or "
                        "lower slm.max_tokens / num_candidates."
                    )
                    break
                chunk = max(1, size // 2)
                logger.warning(
                    f"CUDA OOM at chunk={size}; halving to {chunk} and retrying "
                    "(candidates are preserved, not dropped)"
                )
        return texts

    def _run_generation(
        self,
        inputs,
        max_tokens: int,
        do_sample: bool,
        temperature: float,
        num_return_sequences: int = 1,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Run model.generate and decode only the newly generated tokens.

        ``seed`` (sampling only) makes the call reproducible; chunked sampling
        passes a distinct seed per chunk so chunks don't duplicate each other.
        Falls back to ``config.generation_seed`` when not given.
        """
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
            # Optional reproducibility: seed the RNG before sampling so the
            # temperature candidates (and thus EX) are stable run-to-run.
            if seed is None:
                seed = getattr(self.config, "generation_seed", None)
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        texts = []
        for seq in outputs:
            generated_tokens = seq[input_length:]
            texts.append(
                self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            )
        return texts

    def _finalize(self, generated_text: str) -> str:
        """Extract SQL from raw model output and apply deterministic fixes."""
        sql_text = self._extract_sql_from_output(generated_text, "")
        return finalize_sql(
            sql_text, enable_cast_fix=getattr(self.config, "enable_cast_fix", True)
        )

    def _generate_stub(self, schema_elements: List[SchemaElement]) -> SQL:
        """Generate stub SQL query (fallback mode).

        Args:
            schema_elements: Schema elements

        Returns:
            Stub SQL query
        """
        if schema_elements:
            table_names = set()
            for elem in schema_elements:
                if "." in elem.name:
                    table = elem.name.split(".")[0]
                    table_names.add(table)

            if table_names:
                table = list(table_names)[0]
                sql_text = f"SELECT * FROM {table} LIMIT 10"
            else:
                sql_text = "SELECT 1"
        else:
            sql_text = "SELECT 1"

        return SQL(
            text=sql_text,
            dialect="sqlite",
            source="slm",
            verified=False,
        )

    def _format_prompt(
        self, query: Query, schema_elements: List[SchemaElement], schema=None
    ) -> str:
        """Build the DDL prompt (CREATE TABLE + FK/PK hints + few-shot)."""
        return self._format_prompt_ddl(query, schema_elements, schema=schema)

    def _format_prompt_ddl(
        self, query: Query, schema_elements: List[SchemaElement], schema=None
    ) -> str:
        """Format the DDL prompt (CREATE TABLE + FK/PK hints + few-shot + question).

        Delegates to the shared ``build_direct_prompt`` so the local SLM prompt and
        the multi-agent ``direct`` strategy stay identical by construction.

        Args:
            query: Natural language query
            schema_elements: Schema elements
            schema: Full Schema (for real foreign keys / primary keys)
        """
        from prompts.sql_strategies import build_direct_prompt

        return build_direct_prompt(
            query,
            schema_elements,
            schema=schema,
            expose_keys=getattr(self.config, "expose_keys", True),
        )

    def _extract_sql_from_output(self, output: str, prompt: str) -> str:
        """Extract SQL from model output via the shared extractor.

        Delegates to :func:`generator.sql_postprocess.extract_sql` so the SLM and
        LLM paths handle fenced blocks, reasoning-then-SQL, and truncated fences
        identically.

        Args:
            output: Raw model output
            prompt: Original prompt (unused; new tokens are already isolated)
        """
        from generator.sql_postprocess import extract_sql

        sql = extract_sql(output)
        if not sql:
            logger.warning("Could not extract SQL from output")
        return sql

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
            # Candidates 2..n: temperature samples (batched in one call).
            if n > 1 and temperature > 0:
                texts.extend(
                    self._run_generation(
                        inputs, max_tokens=max_tokens, do_sample=True,
                        temperature=temperature, num_return_sequences=n - 1,
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
    ) -> List[str]:
        """Model-agnostic completion from a prebuilt prompt (booster interface).

        Returns up to ``n`` finalized SQL strings: one greedy decode plus
        ``n-1`` temperature samples. Used by the multi-agent generator to drive
        arbitrary reasoning strategies through the same model plumbing as
        ``generate_candidates``. Returns ``[]`` if the model is unavailable so the
        caller can fall back to other strategies.
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
                texts.extend(self._run_generation(
                    inputs, max_tokens=max_tokens, do_sample=True,
                    temperature=temperature, num_return_sequences=n - 1,
                ))
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

    def _run_generation(
        self,
        inputs,
        max_tokens: int,
        do_sample: bool,
        temperature: float,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Run model.generate and decode only the newly generated tokens."""
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

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
        """Extract SQL from model output.

        The model generates SQL first, then may add explanations. We extract just the SQL.

        Args:
            output: Raw model output
            prompt: Original prompt (not used since we already stripped it)

        Returns:
            Extracted SQL query string
        """
        sql = output.strip()

        # Strip any surrounding markdown code fences (leading ```sql/``` and
        # everything after a closing ```), otherwise a trailing fence leaks into
        # the SQL and fails grammar verification.
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if "```" in sql:
            sql = sql.split("```", 1)[0]
        sql = sql.strip()

        # Priority 1: Extract from ```sql ... ``` code blocks
        if "```sql" in sql:
            start = sql.find("```sql") + 6
            end = sql.find("```", start)
            if end != -1:
                result = sql[start:end].strip()
                logger.debug(f"Extracted from ```sql block")
                return result

        # Priority 2: SQL is at the beginning, stop at question markers
        # The model outputs: "SELECT ... ; \nWhat is the SQL query..."
        # We want just the "SELECT ... ;"
        if sql.upper().startswith("SELECT"):
            # Find the first semicolon
            if ";" in sql:
                # Take everything up to the first semicolon
                sql_part = sql.split(";")[0].strip() + ";"

                # Additional cleanup: stop at newline followed by "What is" or "Instructions"
                lines = sql_part.split("\n")
                result_lines = []
                for line in lines:
                    stripped = line.strip()
                    # Stop at explanation/question markers
                    if stripped.startswith(("What is", "Instructions:", "Note:", "Explanation:")):
                        break
                    if stripped:
                        result_lines.append(stripped)

                result = " ".join(result_lines)
                # Ensure ends with semicolon
                if not result.endswith(";"):
                    if ";" in result:
                        result = result.split(";")[0].strip() + ";"

                logger.debug(f"Extracted SQL from beginning")
                return result

        # Priority 3: Search for SQL statement in the output
        lines = sql.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
                # Found SQL start, collect until semicolon
                sql_lines = [stripped]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    # Stop at explanations
                    if next_line.startswith(("What is", "Instructions:", "Note:", "Explanation:")):
                        break
                    if next_line:
                        sql_lines.append(next_line)
                    if ";" in next_line:
                        break
                    j += 1

                result = " ".join(sql_lines)
                if ";" in result:
                    result = result.split(";")[0].strip() + ";"
                logger.debug(f"Extracted SQL from line search")
                return result

        logger.warning(f"Could not extract SQL from output")
        return ""

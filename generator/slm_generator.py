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

            # Get torch dtype
            torch_dtype = getattr(torch, config.torch_dtype)

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model,
                device_map=config.device,
                torch_dtype=torch_dtype,
                token=hf_token,
                cache_dir=str(cache_dir),
                trust_remote_code=config.trust_remote_code,
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
            # Format prompt
            prompt = self._format_prompt(query, schema_elements)

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

            # Move inputs to device
            if self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate SQL
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (skip the input prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Log raw output for debugging
            logger.debug(f"Raw model output (first 500 chars): {generated_text[:500]}")

            # Extract SQL from output
            sql_text = self._extract_sql_from_output(generated_text, "")  # No need to remove prompt now
            logger.debug(f"Extracted SQL: {sql_text}")

            return SQL(
                text=sql_text,
                dialect="sqlite",
                source="slm",
                verified=False,
            )

        except Exception as e:
            logger.error(f"SLM generation failed: {e}")
            logger.warning("Falling back to stub mode")
            return self._generate_stub(schema_elements)

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
        self, query: Query, schema_elements: List[SchemaElement]
    ) -> str:
        """Format prompt for SLM generation with CREATE TABLE syntax, FK hints, and examples.

        Args:
            query: Natural language query
            schema_elements: Schema elements

        Returns:
            Formatted prompt string with CREATE TABLE structure, FK relationships, and few-shot examples
        """
        # Group schema elements by table
        tables = {}
        for elem in schema_elements:
            if '.' in elem.name:
                table, col = elem.name.split('.', 1)
                if table not in tables:
                    tables[table] = []
                tables[table].append(elem)

        # Extract FK relationships from schema (if available)
        fk_hints = ""
        if schema_elements and hasattr(schema_elements[0], '__dict__'):
            # Try to get schema from first element (hacky but works)
            # In practice, we'd pass schema separately, but this avoids breaking API
            pass  # FKs will be added in a future enhancement

        # Format as CREATE TABLE statements with backticks for special characters
        schema_str = ""
        for table, cols in tables.items():
            schema_str += f"CREATE TABLE {table} (\n"
            for col in cols:
                col_name = col.name.split('.', 1)[1]
                # Add backticks for columns with spaces, parentheses, or special chars
                if ' ' in col_name or '(' in col_name or '-' in col_name:
                    col_name = f"`{col_name}`"
                col_type = col.data_type if col.data_type else "TEXT"
                schema_str += f"  {col_name} {col_type}"
                if col.description:
                    schema_str += f" -- {col.description}"
                schema_str += ",\n"
            schema_str = schema_str.rstrip(",\n") + "\n);\n\n"

        # Load prompt manager for centralized templates
        prompt_mgr = get_prompt_manager()

        # Get few-shot examples from templates
        examples = prompt_mgr.format_slm_examples()

        # Get FK hints if multiple tables
        table_names_list = list(tables.keys())
        fk_hints = prompt_mgr.get_slm_fk_hint(table_names_list)

        # Get instructions
        instructions = prompt_mgr.get_slm_instructions()

        # Get question format (with evidence if available)
        question_section = prompt_mgr.format_slm_question(query.text, query.evidence)

        return f"{schema_str}{fk_hints}{examples}{instructions}{question_section}"

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

"""Remote large language model (FLLM) fallback for SQL generation.

Implements remote SQL generation using foundation LLMs like GPT-4, Claude,
or Gemini via API. Operates on abstracted data for privacy protection.

References:
    - Build strategy Section 2.2: LLM API integration
    - Paper Section 3: Remote path requires DP abstraction
"""

import time
from typing import List, Optional

from loguru import logger
import openai
import anthropic

from config import LLMConfig
from aegis_types import Query, SchemaElement, SQL, AbstractedPrompt
from prompts.prompt_manager import get_prompt_manager


class LLMFallback:
    """Remote LLM fallback for SQL (FLLM).

    Uses foundation LLMs via API for complex queries that exceed
    local SLM capabilities. Operates on abstracted data.

    Supported providers:
        - OpenAI (GPT-4, GPT-4-Turbo, GPT-4o)
        - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
        - Google (Gemini Pro)

    Attributes:
        config: LLM configuration
        client: API client (openai.Client or anthropic.Anthropic)
        provider: Provider name
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLM fallback.

        Args:
            config: LLM configuration

        Initializes API client based on provider configuration.
        """
        self.config = config
        self.provider = config.provider.lower()
        self.client = None

        if self.provider == "openai":
            self.client = openai.Client(api_key=config.api_key)
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=config.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        logger.info(
            f"Initialized LLMFallback with provider={config.provider}, model={config.model}"
        )

    def generate(
        self,
        abstracted_query: AbstractedPrompt,
        schema_elements: List[SchemaElement],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        schema=None,
        feedback: Optional[str] = None,
    ) -> SQL:
        """Generate SQL from abstracted query using remote LLM.

        IMPORTANT: This method receives ABSTRACTED data with placeholders.
        Real sensitive tokens have been replaced. Reconstruction happens later.

        Args:
            abstracted_query: DP-abstracted query with placeholders
            schema_elements: Schema elements (may also be abstracted)
            max_tokens: Maximum tokens (defaults to config)
            temperature: Temperature (defaults to config)
            schema: Full Schema (for real foreign keys / primary keys), mirroring
                the local SLM prompt so both paths get identical join grounding
            feedback: Verifier feedback on a repair pass (appended to the prompt;
                at temperature 0 a feedback-free regen would be a no-op)

        Returns:
            Generated SQL query (with placeholders, needs reconstruction)

        Implementation:
            - Formats prompt with abstracted query + schema
            - Calls LLM API (OpenAI or Anthropic)
            - Handles API errors and retries
            - Extracts SQL from response
            - Returns SQL object with source="llm"

        References:
            - Build strategy Section 2.2: LLM API integration
            - Paper Theorem 1: Privacy guarantee holds because abstraction applied
        """
        logger.debug(
            f"Generating SQL with FLLM for abstracted query: {abstracted_query.text[:50]}..."
        )

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        # Format prompt
        prompt = self._format_prompt(
            abstracted_query, schema_elements, schema=schema, feedback=feedback
        )

        # Call LLM API based on provider
        if self.provider == "openai":
            sql_text, token_usage = self._call_openai(prompt, max_tokens, temperature)
        elif self.provider == "anthropic":
            sql_text, token_usage = self._call_anthropic(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Return SQL (still has placeholders - needs reconstruction). token_usage
        # is carried on the returned object (thread-safe) for cost accounting.
        return SQL(
            text=sql_text,
            dialect="sqlite",  # TODO: Get from config
            source="fllm",
            verified=False,
            token_usage=token_usage,
        )

    def _call_openai(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> tuple:
        """Call OpenAI API with retry logic.

        Args:
            prompt: Formatted prompt
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            (generated SQL text, total tokens consumed)

        Calls OpenAI API with exponential backoff retry logic.
        """
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                # Get system prompt from template
                prompt_mgr = get_prompt_manager()
                system_prompt = prompt_mgr.get_llm_system_prompt("openai")

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self.config.timeout,
                )
                sql_text = response.choices[0].message.content.strip()
                usage = getattr(response, "usage", None)
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                return self._extract_sql_from_output(sql_text), total_tokens

            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
                last_error = e
                retry_delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.config.max_retries}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"OpenAI API failed after {self.config.max_retries} attempts")
                    raise

            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                raise

        # Should not reach here, but for completeness
        raise last_error

    def _call_anthropic(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> tuple:
        """Call Anthropic API with retry logic.

        Args:
            prompt: Formatted prompt
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            (generated SQL text, total tokens consumed)

        Calls Anthropic API with exponential backoff retry logic.
        """
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                # Get system prompt from template
                prompt_mgr = get_prompt_manager()
                system_prompt = prompt_mgr.get_llm_system_prompt("anthropic")

                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{system_prompt} {prompt}",
                        }
                    ],
                    timeout=self.config.timeout,
                )
                sql_text = response.content[0].text.strip()
                usage = getattr(response, "usage", None)
                total_tokens = (
                    getattr(usage, "input_tokens", 0) or 0
                ) + (getattr(usage, "output_tokens", 0) or 0)
                return self._extract_sql_from_output(sql_text), total_tokens

            except (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
                last_error = e
                retry_delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Anthropic API error (attempt {attempt + 1}/{self.config.max_retries}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Anthropic API failed after {self.config.max_retries} attempts")
                    raise

            except Exception as e:
                logger.error(f"Anthropic API error: {str(e)}")
                raise

        # Should not reach here, but for completeness
        raise last_error

    def _format_prompt(
        self,
        abstracted_query: AbstractedPrompt,
        schema_elements: List[SchemaElement],
        schema=None,
        feedback: Optional[str] = None,
    ) -> str:
        """Format prompt for LLM generation with CREATE TABLE syntax and FK/PK hints.

        Mirrors the local SLM's DDL prompt so a remote-vs-local comparison swaps
        only the model, not the schema grounding: real foreign keys (JOIN hints)
        and PRIMARY KEY markers are rendered from the populated ``schema`` rather
        than the never-set ``abstracted_query.schema``.

        Args:
            abstracted_query: Abstracted query
            schema_elements: Schema elements
            schema: Full Schema (for real foreign keys / primary keys)
            feedback: Verifier feedback to append on a repair pass

        Returns:
            Formatted prompt string
        """
        # Group schema elements by table
        tables = {}
        for elem in schema_elements:
            if '.' in elem.name:
                table, col = elem.name.split('.', 1)
                if table not in tables:
                    tables[table] = []
                tables[table].append(elem)

        # Real FK/PK from the populated Schema (abstracted_query.schema is never
        # set). FKs are filtered to retrieved tables on BOTH endpoints so JOIN
        # hints only reference tables actually present in the prompt.
        primary_keys = getattr(schema, "primary_keys", None) or {}
        fk_relationships = []
        if getattr(schema, "foreign_keys", None):
            table_names_set = set(tables.keys())
            for fk in schema.foreign_keys:
                if fk.from_table in table_names_set and fk.to_table in table_names_set:
                    fk_relationships.append(fk)

        # Format as CREATE TABLE statements
        schema_str = ""
        for table, cols in tables.items():
            pk_cols = set(primary_keys.get(table, []))
            schema_str += f"CREATE TABLE {table} (\n"
            for col in cols:
                raw_col = col.name.split('.', 1)[1]
                col_name = raw_col
                # Add backticks for special characters
                if ' ' in col_name or '(' in col_name or '-' in col_name:
                    col_name = f"`{col_name}`"
                col_type = col.data_type if col.data_type else "TEXT"
                schema_str += f"  {col_name} {col_type}"
                if raw_col in pk_cols:
                    schema_str += " PRIMARY KEY"
                if col.description:
                    schema_str += f" -- {col.description}"
                schema_str += ",\n"
            schema_str = schema_str.rstrip(",\n") + "\n);\n\n"

        # Add explicit FK relationship hints
        fk_hints = ""
        if fk_relationships:
            fk_hints = "\nFOREIGN KEY RELATIONSHIPS (Use these exact columns for JOINs):\n"
            for fk in fk_relationships:
                fk_hints += f"  {fk.from_table}.{fk.from_column} = {fk.to_table}.{fk.to_column}\n"
        elif len(tables) > 1:
            # Generic hint if no explicit FKs available
            table_names = list(tables.keys())
            fk_hints = f"\nNote: Tables {', '.join(table_names)} may be related via foreign keys. Use JOIN when the query requires combining data from multiple tables.\n"

        # Add evidence section if available
        evidence_section = ""
        if abstracted_query.evidence:
            evidence_section = f"\nDomain Knowledge:\n{abstracted_query.evidence}\n"

        # On a repair pass, surface why the previous attempt was rejected.
        feedback_section = ""
        if feedback:
            feedback_section = (
                f"\nThe previous attempt was rejected by the verifier:\n"
                f"{feedback}\nGenerate a corrected SQL query.\n"
            )

        prompt = f"""Given the following SQLite database schema:

{schema_str}{fk_hints}{evidence_section}{feedback_section}
Generate a valid SQLite query to answer this question:
{abstracted_query.text}

Return only the SQL query without explanation. Use proper JOINs if multiple tables are needed."""
        return prompt

    def _extract_sql_from_output(self, output: str) -> str:
        """Extract SQL from LLM output.

        Args:
            output: Raw LLM output

        Returns:
            Extracted SQL query string

        Removes markdown code fences and extracts clean SQL.
        """
        # Remove code fences
        sql = output.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        return sql.strip()

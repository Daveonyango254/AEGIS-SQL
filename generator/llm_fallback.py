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
    ) -> SQL:
        """Generate SQL from abstracted query using remote LLM.

        IMPORTANT: This method receives ABSTRACTED data with placeholders.
        Real sensitive tokens have been replaced. Reconstruction happens later.

        Args:
            abstracted_query: DP-abstracted query with placeholders
            schema_elements: Schema elements (may also be abstracted)
            max_tokens: Maximum tokens (defaults to config)
            temperature: Temperature (defaults to config)

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
        prompt = self._format_prompt(abstracted_query, schema_elements)

        # Call LLM API based on provider
        if self.provider == "openai":
            sql_text = self._call_openai(prompt, max_tokens, temperature)
        elif self.provider == "anthropic":
            sql_text = self._call_anthropic(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Return SQL (still has placeholders - needs reconstruction)
        return SQL(
            text=sql_text,
            dialect="sqlite",  # TODO: Get from config
            source="fllm",
            verified=False,
        )

    def _call_openai(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Call OpenAI API with retry logic.

        Args:
            prompt: Formatted prompt
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            Generated SQL text

        Calls OpenAI API with exponential backoff retry logic.
        """
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert SQL generator. Generate valid SQL queries based on natural language questions and database schemas.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self.config.timeout,
                )
                sql_text = response.choices[0].message.content.strip()
                return self._extract_sql_from_output(sql_text)

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
    ) -> str:
        """Call Anthropic API with retry logic.

        Args:
            prompt: Formatted prompt
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            Generated SQL text

        Calls Anthropic API with exponential backoff retry logic.
        """
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": f"You are an expert SQL generator. {prompt}",
                        }
                    ],
                    timeout=self.config.timeout,
                )
                sql_text = response.content[0].text.strip()
                return self._extract_sql_from_output(sql_text)

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
        self, abstracted_query: AbstractedPrompt, schema_elements: List[SchemaElement]
    ) -> str:
        """Format prompt for LLM generation.

        Args:
            abstracted_query: Abstracted query
            schema_elements: Schema elements

        Returns:
            Formatted prompt string

        Formats prompt with abstracted query and schema for the LLM.
        """
        # Simple prompt template
        schema_str = "\n".join(
            [f"- {elem.name} ({elem.data_type})" for elem in schema_elements]
        )
        prompt = f"""Given the following database schema:

{schema_str}

Generate a valid SQLite query to answer this question:
{abstracted_query.text}

Return only the SQL query without explanation."""
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

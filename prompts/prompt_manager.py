"""Prompt template manager for AEGIS-SQL.

Loads and formats prompts from templates.yaml with variable substitution.
"""

from pathlib import Path
from typing import Dict, List, Any
import yaml
from loguru import logger


class PromptManager:
    """Manages prompt templates and formatting.

    Loads templates from YAML and provides methods for formatting
    with variable substitution.

    Attributes:
        templates: Loaded template dictionary
        template_path: Path to templates.yaml file
    """

    def __init__(self, template_path: str | Path = None):
        """Initialize PromptManager.

        Args:
            template_path: Path to templates.yaml (defaults to prompts/templates.yaml)
        """
        if template_path is None:
            # Default to prompts/templates.yaml relative to this file
            template_path = Path(__file__).parent / "templates.yaml"

        self.template_path = Path(template_path)
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from YAML file.

        Returns:
            Dictionary of templates

        Raises:
            FileNotFoundError: If templates.yaml doesn't exist
        """
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file not found: {self.template_path}")

        with open(self.template_path, "r", encoding="utf-8") as f:
            templates = yaml.safe_load(f)

        logger.debug(f"Loaded prompt templates from {self.template_path}")
        return templates

    def get_slm_examples(self) -> List[Dict[str, str]]:
        """Get SLM few-shot examples.

        Returns:
            List of example dictionaries with 'question', 'sql', 'description'
        """
        return self.templates.get("slm", {}).get("examples", [])

    def format_slm_examples(self) -> str:
        """Format SLM examples as prompt string.

        Returns:
            Formatted examples string
        """
        examples = self.get_slm_examples()
        formatted = []

        for i, example in enumerate(examples, 1):
            formatted.append(f"-- Example {i}: {example['description']}")
            formatted.append(f"-- Question: {example['question']}")
            formatted.append(f"-- SQL: {example['sql']}")
            formatted.append("")  # Empty line

        return "\n".join(formatted)

    def get_slm_fk_hint(self, table_names: List[str]) -> str:
        """Get foreign key hint for SLM prompt.

        Args:
            table_names: List of table names in the query

        Returns:
            Formatted FK hint string (empty if single table)
        """
        if len(table_names) <= 1:
            return ""

        template = self.templates.get("slm", {}).get("fk_hint", "")
        return template.format(table_names=", ".join(table_names))

    def get_slm_instructions(self) -> str:
        """Get SLM instructions.

        Returns:
            Instructions string
        """
        return self.templates.get("slm", {}).get("instructions", "")

    def format_slm_question(self, query: str) -> str:
        """Format question section for SLM prompt.

        Args:
            query: Natural language query

        Returns:
            Formatted question string
        """
        template = self.templates.get("slm", {}).get("question_format", "")
        return template.format(query=query)

    def get_llm_system_prompt(self, provider: str = "openai") -> str:
        """Get LLM system prompt.

        Args:
            provider: LLM provider ('openai' or 'anthropic')

        Returns:
            System prompt string
        """
        llm_templates = self.templates.get("llm", {})
        return llm_templates.get(provider, {}).get("system_prompt", "")

    def format_llm_user_prompt(
        self, schema: str, query: str, fk_hints: str = ""
    ) -> str:
        """Format LLM user prompt.

        Args:
            schema: Formatted schema string
            query: Natural language query
            fk_hints: Foreign key hints (optional)

        Returns:
            Formatted user prompt
        """
        template = self.templates.get("llm", {}).get("user_prompt", "")
        return template.format(schema=schema, query=query, fk_hints=fk_hints)

    def reload(self):
        """Reload templates from file.

        Useful for hot-reloading templates during development.
        """
        self.templates = self._load_templates()
        logger.info(f"Reloaded prompt templates from {self.template_path}")


# Global instance for easy access
_global_manager: PromptManager | None = None


def get_prompt_manager(template_path: str | Path = None) -> PromptManager:
    """Get global PromptManager instance.

    Args:
        template_path: Optional path to templates.yaml

    Returns:
        Global PromptManager singleton
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = PromptManager(template_path)

    return _global_manager

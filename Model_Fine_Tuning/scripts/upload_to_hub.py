"""Upload fine-tuned LoRA adapter to HuggingFace Hub.

Uploads the trained adapter with model card and metadata.

Usage:
    python upload_to_hub.py --adapter_path ../checkpoints/phi4 --repo_id Daveonyango254/aegis-sql-phi4-lora
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from loguru import logger


def create_model_card(
    base_model: str,
    training_data: str,
    metrics: dict = None
) -> str:
    """Create a model card in Markdown format.

    Args:
        base_model: Base model name
        training_data: Training dataset description
        metrics: Optional evaluation metrics

    Returns:
        Model card content
    """
    metrics_section = ""
    if metrics:
        metrics_section = "## Evaluation Metrics\n\n"
        for key, value in metrics.items():
            metrics_section += f"- **{key}**: {value}\n"

    model_card = f"""---
language:
- en
- es
- ko
- zh
- sw
license: mit
tags:
- text-to-sql
- sql-generation
- bird-benchmark
- lora
- peft
base_model: {base_model}
datasets:
- birdsql/bird23-train-filtered
---

# AEGIS-SQL Fine-Tuned LoRA Adapter

This is a LoRA (Low-Rank Adaptation) adapter for **{base_model}** fine-tuned on the BIRD benchmark for Text-to-SQL generation.

## Model Description

- **Base Model**: {base_model}
- **Training Data**: {training_data}
- **Task**: Natural Language to SQL (NL2SQL)
- **Method**: LoRA fine-tuning (Parameter-Efficient Fine-Tuning)
- **Framework**: AEGIS-SQL (Privacy-Preserving Hybrid NL2SQL)

## Training Details

The model was fine-tuned using:
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.05
- **Training Epochs**: 3
- **Learning Rate**: 2e-4 with cosine decay
- **Batch Size**: 16 (effective)
- **Optimization**: AdamW with gradient accumulation
- **Regularization**: Weight decay (0.01), early stopping

{metrics_section}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "YOUR_REPO_ID_HERE")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Generate SQL
prompt = \"\"\"Generate a valid SQLite query to answer the following question.

Schema:
CREATE TABLE students (
  student_id INTEGER,
  name TEXT,
  age INTEGER
);

Question: How many students are older than 18?

SQL Query:
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(sql)
```

## Integration with AEGIS-SQL

To use this adapter in the AEGIS-SQL framework:

1. Update `config.yaml`:
```yaml
slm:
  model: "{base_model}"
  adapter_path: "YOUR_REPO_ID_HERE"
```

2. Run AEGIS-SQL as normal - it will automatically load the adapter.

## Citation

If you use this model, please cite:

```bibtex
@software{{aegis_sql_2025,
  title = {{AEGIS-SQL: Privacy-Preserving Hybrid NL2SQL}},
  author = {{AEGIS Team}},
  year = {{2025}},
  url = {{https://github.com/yourusername/aegis_SQL}}
}}
```

## License

MIT License - See base model license for additional restrictions.
"""
    return model_card


def upload_adapter_to_hub(
    adapter_path: str,
    repo_id: str,
    base_model: str = "microsoft/Phi-4-mini-instruct",
    private: bool = False
) -> None:
    """Upload LoRA adapter to HuggingFace Hub.

    Args:
        adapter_path: Local path to adapter
        repo_id: HuggingFace repository ID (username/repo-name)
        base_model: Base model name
        private: Whether to create a private repository
    """
    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    # Check for HF token
    hf_token = os.getenv('HF_HUB_TOKEN')
    if not hf_token:
        raise ValueError(
            "HF_HUB_TOKEN not found in environment. "
            "Please set it in your .env file or environment."
        )

    logger.info(f"Uploading adapter to HuggingFace Hub: {repo_id}")
    logger.info(f"  Adapter path: {adapter_path}")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Private repo: {private}")

    # Create API client
    api = HfApi(token=hf_token)

    # Create repository
    try:
        logger.info(f"Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=private,
            exist_ok=True
        )
        logger.success(f"Repository created/verified: {repo_id}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise

    # Create model card
    logger.info("Creating model card...")
    model_card = create_model_card(
        base_model=base_model,
        training_data="BIRD23-train-filtered (6,601 samples)"
    )

    readme_path = adapter_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card)

    logger.success("Model card created")

    # Upload all files in the adapter directory
    logger.info("Uploading adapter files...")
    try:
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            token=hf_token,
            commit_message="Upload AEGIS-SQL fine-tuned LoRA adapter"
        )
        logger.success(f"Adapter uploaded successfully!")
        logger.info(f"View at: https://huggingface.co/{repo_id}")

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload LoRA adapter to HuggingFace Hub"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (username/repo-name)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Base model name (default: microsoft/Phi-4-mini-instruct)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )

    args = parser.parse_args()

    # Resolve adapter path
    script_dir = Path(__file__).parent
    if not Path(args.adapter_path).is_absolute():
        adapter_path = (script_dir / args.adapter_path).resolve()
    else:
        adapter_path = Path(args.adapter_path)

    # Upload
    upload_adapter_to_hub(
        adapter_path=str(adapter_path),
        repo_id=args.repo_id,
        base_model=args.base_model,
        private=args.private
    )

    logger.success("All done!")


if __name__ == "__main__":
    main()

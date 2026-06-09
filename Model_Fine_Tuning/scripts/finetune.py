"""Generic fine-tuning script for AEGIS-SQL models.

Fine-tunes code models on BIRD benchmark using LoRA (Low-Rank Adaptation).
Supports multiple model architectures with unified training pipeline.

Best Practices Implemented:
1. LoRA for parameter-efficient fine-tuning (prevents catastrophic forgetting)
2. Gradient accumulation for stable training
3. Mixed precision training (fp16/bf16)
4. Learning rate warmup + cosine decay
5. Early stopping to prevent overfitting
6. Database-based train/eval split for generalization
7. Gradient checkpointing for memory efficiency
8. Regular evaluation and checkpoint saving

Usage:
    python finetune.py --config ../config/phi4_config.yaml
    python finetune.py --config ../config/qwen_config.yaml
    python finetune.py --config ../config/deepseek_config.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)


class FineT

uner:
    """Fine-tuner for code models on Text-to-SQL task.

    Implements best practices for fine-tuning:
    - LoRA adapter training (preserves base model knowledge)
    - Gradient accumulation and checkpointing
    - Early stopping based on eval loss
    - Mixed precision training
    """

    def __init__(self, config_path: str):
        """Initialize fine-tuner with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from: {config_path}")
        logger.info(f"Model: {config['model']['name']}")

        return config

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer from HuggingFace."""
        model_name = self.config['model']['name']
        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config['model'].get('trust_remote_code', True),
            use_fast=True
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        # Load model with quantization configuration
        torch_dtype = getattr(torch, self.config['model']['torch_dtype'])

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.config['model'].get('device_map', 'auto'),
            trust_remote_code=self.config['model'].get('trust_remote_code', True),
        )

        logger.success(f"Model loaded successfully")
        logger.info(f"Model dtype: {self.model.dtype}")
        logger.info(f"Model device: {self.model.device}")

    def setup_lora(self):
        """Setup LoRA adapter for parameter-efficient fine-tuning.

        LoRA Benefits:
        - Trains only ~0.1% of parameters (much faster)
        - Prevents catastrophic forgetting of base knowledge
        - Lower memory footprint
        - Can be merged or removed later
        """
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias=self.config['lora'].get('bias', 'none'),
            task_type=self.config['lora'].get('task_type', 'CAUSAL_LM'),
        )

        logger.info("Setting up LoRA adapter...")
        logger.info(f"  Rank (r): {lora_config.r}")
        logger.info(f"  Alpha: {lora_config.lora_alpha}")
        logger.info(f"  Dropout: {lora_config.lora_dropout}")
        logger.info(f"  Target modules: {lora_config.target_modules}")

        # Prepare model for k-bit training (if using quantization)
        # self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = 100 * trainable_params / total_params

        logger.success(f"LoRA adapter applied successfully")
        logger.info(f"  Trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")
        logger.info(f"  Total params: {total_params:,}")

    def load_datasets(self):
        """Load and preprocess training and evaluation datasets."""
        train_file = Path(self.config['data']['train_file'])
        eval_file = Path(self.config['data']['eval_file'])

        # Resolve paths relative to config file
        script_dir = Path(__file__).parent
        train_file = (script_dir / train_file).resolve()
        eval_file = (script_dir / eval_file).resolve()

        logger.info(f"Loading training data from: {train_file}")
        logger.info(f"Loading evaluation data from: {eval_file}")

        # Load JSON files
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        logger.info(f"Loaded {len(train_data)} training samples")
        logger.info(f"Loaded {len(eval_data)} evaluation samples")

        # Convert to HuggingFace datasets
        self.train_dataset = Dataset.from_list(train_data)
        self.eval_dataset = Dataset.from_list(eval_data)

        # Preprocess datasets
        max_length = self.config['data']['max_seq_length']
        num_workers = self.config['data'].get('preprocessing_num_workers', 4)

        logger.info("Tokenizing datasets...")
        self.train_dataset = self.train_dataset.map(
            lambda x: self._tokenize_function(x, max_length),
            batched=False,
            num_proc=num_workers,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing train dataset"
        )

        self.eval_dataset = self.eval_dataset.map(
            lambda x: self._tokenize_function(x, max_length),
            batched=False,
            num_proc=num_workers,
            remove_columns=self.eval_dataset.column_names,
            desc="Tokenizing eval dataset"
        )

        logger.success("Datasets prepared successfully")

    def _tokenize_function(self, example: Dict, max_length: int) -> Dict:
        """Tokenize a single training example.

        Formats as: [instruction][input][output]

        Args:
            example: Training example with instruction/input/output
            max_length: Maximum sequence length

        Returns:
            Tokenized example
        """
        # Format the prompt
        prompt = f"{example['instruction']}\n\n{example['input']}\n\nSQL Query:\n"
        completion = example['output']

        # Tokenize prompt and completion separately
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=False,
            add_special_tokens=True
        )

        completion_tokens = self.tokenizer(
            completion,
            truncation=False,
            add_special_tokens=False
        )

        # Combine tokens
        input_ids = prompt_tokens['input_ids'] + completion_tokens['input_ids'] + [self.tokenizer.eos_token_id]
        attention_mask = prompt_tokens['attention_mask'] + completion_tokens['attention_mask'] + [1]

        # Create labels (mask prompt, only train on completion)
        labels = [-100] * len(prompt_tokens['input_ids']) + completion_tokens['input_ids'] + [self.tokenizer.eos_token_id]

        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def train(self):
        """Run fine-tuning with best practices."""
        # Get training configuration
        training_config = self.config['training']
        output_config = self.config['output']

        # Resolve output directory
        script_dir = Path(__file__).parent
        output_dir = (script_dir / output_config['output_dir']).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Starting fine-tuning...")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Num epochs: {training_config['num_epochs']}")
        logger.info(f"  Batch size: {training_config['per_device_train_batch_size']}")
        logger.info(f"  Gradient accumulation: {training_config['gradient_accumulation_steps']}")
        logger.info(f"  Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}")
        logger.info(f"  Learning rate: {training_config['learning_rate']}")
        logger.info("=" * 60)

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config['num_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_ratio=training_config['warmup_ratio'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            max_grad_norm=training_config['max_grad_norm'],
            logging_steps=training_config['logging_steps'],
            eval_steps=training_config['eval_steps'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            evaluation_strategy="steps",
            fp16=training_config.get('fp16', False),
            bf16=training_config.get('bf16', False),
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            optim=training_config.get('optim', 'adamw_torch'),
            seed=training_config.get('seed', 42),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            push_to_hub=output_config.get('push_to_hub', False),
            hub_model_id=output_config.get('hub_model_id'),
            hub_strategy=output_config.get('hub_strategy', 'checkpoint'),
            hub_private_repo=output_config.get('hub_private_repo', False),
        )

        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM (not masked LM)
        )

        # Setup callbacks
        callbacks = []

        # Early stopping callback
        early_stop_config = self.config.get('early_stopping', {})
        if early_stop_config.get('enabled', True):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stop_config.get('patience', 3)
                )
            )
            logger.info(f"Early stopping enabled (patience={early_stop_config.get('patience', 3)})")

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Train!
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logger.success("Fine-tuning complete!")
        logger.info(f"Final model saved to: {output_dir}")

        # Print final metrics
        logger.info("=" * 60)
        logger.info("Final Training Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tune code models for Text-to-SQL"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )

    args = parser.parse_args()

    # Check for HuggingFace token
    hf_token = os.getenv('HF_HUB_TOKEN')
    if not hf_token:
        logger.warning("HF_HUB_TOKEN not found in environment. Push to Hub will fail.")
    else:
        logger.info("HF_HUB_TOKEN found, Hub upload enabled")

    # Create fine-tuner and run
    finetuner = FineTuner(args.config)

    logger.info("Step 1/5: Loading model and tokenizer...")
    finetuner.load_model_and_tokenizer()

    logger.info("Step 2/5: Setting up LoRA adapter...")
    finetuner.setup_lora()

    logger.info("Step 3/5: Loading and preprocessing datasets...")
    finetuner.load_datasets()

    logger.info("Step 4/5: Starting training...")
    finetuner.train()

    logger.success("All done! Model fine-tuning complete.")


if __name__ == "__main__":
    main()

"""Full fine-tuning pipeline orchestrator for AEGIS-SQL.

Runs the complete pipeline:
1. Download BIRD training data
2. Prepare and format data
3. Fine-tune models (optionally all three)
4. Evaluate on BIRD dev set
5. Upload to HuggingFace Hub

Usage:
    # Fine-tune a specific model
    python run_full_pipeline.py --model phi4

    # Fine-tune all models
    python run_full_pipeline.py --model all

    # Just download and prepare data
    python run_full_pipeline.py --data_only

    # Skip upload to Hub
    python run_full_pipeline.py --model qwen --no_upload
"""

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger


def run_command(cmd: list, description: str) -> bool:
    """Run a command and handle errors.

    Args:
        cmd: Command list
        description: Description for logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.success(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e}")
        return False


def main():
    """Main orchestrator."""
    parser = argparse.ArgumentParser(
        description="AEGIS-SQL Fine-Tuning Pipeline Orchestrator"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["phi4", "qwen", "deepseek", "all"],
        default="phi4",
        help="Model to fine-tune (default: phi4)"
    )
    parser.add_argument(
        "--data_only",
        action="store_true",
        help="Only download and prepare data, don't train"
    )
    parser.add_argument(
        "--skip_data",
        action="store_true",
        help="Skip data download/preparation (assume data is ready)"
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip evaluation after training"
    )
    parser.add_argument(
        "--no_upload",
        action="store_true",
        help="Skip uploading to HuggingFace Hub"
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=100,
        help="Number of samples for quick evaluation (default: 100)"
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent / "scripts"
    python = sys.executable

    logger.info("=" * 60)
    logger.info("AEGIS-SQL Fine-Tuning Pipeline")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Data only: {args.data_only}")
    logger.info(f"  Skip data: {args.skip_data}")
    logger.info("=" * 60)

    # Step 1: Download data
    if not args.skip_data:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Downloading BIRD training data")
        logger.info("=" * 60)

        cmd = [python, str(script_dir / "download_data.py")]
        if not run_command(cmd, "Download BIRD data"):
            logger.error("Data download failed. Exiting.")
            return 1

        # Step 2: Prepare data
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Preparing and formatting data")
        logger.info("=" * 60)

        cmd = [python, str(script_dir / "prepare_data.py")]
        if not run_command(cmd, "Prepare data"):
            logger.error("Data preparation failed. Exiting.")
            return 1

    if args.data_only:
        logger.success("\nData preparation complete!")
        return 0

    # Determine which models to train
    if args.model == "all":
        models_to_train = ["phi4", "qwen", "deepseek"]
    else:
        models_to_train = [args.model]

    # Model configurations
    model_configs = {
        "phi4": {
            "script": "finetune_phi4.py",
            "base_model": "microsoft/Phi-4-mini-instruct",
            "repo_id": "Daveonyango254/aegis-sql-phi4-lora",
            "checkpoint_dir": "../checkpoints/phi4"
        },
        "qwen": {
            "script": "finetune_qwen.py",
            "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "repo_id": "Daveonyango254/aegis-sql-qwen25-coder-7b-lora",
            "checkpoint_dir": "../checkpoints/qwen"
        },
        "deepseek": {
            "script": "finetune_deepseek.py",
            "base_model": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "repo_id": "Daveonyango254/aegis-sql-deepseek-coder-67b-lora",
            "checkpoint_dir": "../checkpoints/deepseek"
        }
    }

    # Train each model
    for model_name in models_to_train:
        config = model_configs[model_name]

        # Step 3: Fine-tune
        logger.info("\n" + "=" * 60)
        logger.info(f"STEP 3: Fine-tuning {model_name}")
        logger.info("=" * 60)

        cmd = [python, str(script_dir / config["script"])]
        if not run_command(cmd, f"Fine-tune {model_name}"):
            logger.warning(f"Fine-tuning {model_name} failed. Continuing to next model...")
            continue

        # Step 4: Evaluate
        if not args.no_eval:
            logger.info("\n" + "=" * 60)
            logger.info(f"STEP 4: Evaluating {model_name}")
            logger.info("=" * 60)

            cmd = [
                python,
                str(script_dir / "evaluate_model.py"),
                "--adapter_path", config["checkpoint_dir"],
                "--base_model", config["base_model"],
                "--num_queries", str(args.eval_samples)
            ]

            if not run_command(cmd, f"Evaluate {model_name}"):
                logger.warning(f"Evaluation of {model_name} failed. Continuing...")

        # Step 5: Upload to Hub
        if not args.no_upload:
            logger.info("\n" + "=" * 60)
            logger.info(f"STEP 5: Uploading {model_name} to HuggingFace Hub")
            logger.info("=" * 60)

            cmd = [
                python,
                str(script_dir / "upload_to_hub.py"),
                "--adapter_path", config["checkpoint_dir"],
                "--repo_id", config["repo_id"],
                "--base_model", config["base_model"]
            ]

            if not run_command(cmd, f"Upload {model_name}"):
                logger.warning(f"Upload of {model_name} failed. Continuing...")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.success("Pipeline complete!")
    logger.info("=" * 60)
    logger.info("\nModels trained:")
    for model_name in models_to_train:
        logger.info(f"  - {model_name}: {model_configs[model_name]['repo_id']}")

    logger.info("\nNext steps:")
    logger.info("  1. Check evaluation results in Model_Fine_Tuning/output/")
    logger.info("  2. View models on HuggingFace Hub")
    logger.info("  3. Update config.yaml to use the fine-tuned adapter")

    return 0


if __name__ == "__main__":
    sys.exit(main())

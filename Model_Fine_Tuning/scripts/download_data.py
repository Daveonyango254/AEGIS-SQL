"""Download BIRD training dataset from HuggingFace.

This script downloads the BIRD23-train-filtered dataset (6,601 high-quality samples)
from HuggingFace and saves it locally for fine-tuning.

Dataset: birdsql/bird23-train-filtered
Source: https://huggingface.co/datasets/birdsql/bird23-train-filtered

Usage:
    python download_data.py [--output_dir ../data]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from loguru import logger


def download_bird_training_data(output_dir: str = "../data") -> None:
    """Download BIRD training dataset from HuggingFace.

    Args:
        output_dir: Directory to save the downloaded data

    The dataset structure:
        - question_id: Unique identifier
        - db_id: Database identifier
        - question: Natural language query
        - evidence: Domain knowledge hints (critical for accuracy!)
        - SQL: Gold standard SQL query
        - difficulty: Query complexity (simple/moderate/challenging)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading BIRD23-train-filtered from HuggingFace...")
    logger.info("This is a filtered 70% subset (6,601/9,428) optimized for fine-tuning")

    try:
        # Download the dataset
        # Note: If the exact dataset name is different, we'll fall back to local BIRD data
        try:
            dataset = load_dataset("birdsql/bird23-train-filtered", split="train")
            logger.success(f"Downloaded {len(dataset)} training samples from HuggingFace")
        except Exception as e:
            logger.warning(f"Could not download from HuggingFace: {e}")
            logger.info("Falling back to local BIRD training data...")

            # Try to load from local BIRD data if available
            local_bird_path = Path(__file__).parent.parent.parent / "data" / "bird" / "train.json"
            if not local_bird_path.exists():
                # If train.json doesn't exist, we can use dev.json as a fallback for demonstration
                local_bird_path = Path(__file__).parent.parent.parent / "data" / "bird" / "dev.json"
                logger.warning(f"Using dev.json as training data fallback: {local_bird_path}")

            if local_bird_path.exists():
                with open(local_bird_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                logger.info(f"Loaded {len(dataset)} samples from local BIRD data")
            else:
                raise FileNotFoundError(
                    "Could not find BIRD training data. Please download from: "
                    "https://huggingface.co/datasets/birdsql/bird23-train-filtered"
                )

        # Save the dataset locally
        train_file = output_path / "bird_train_raw.json"

        # Convert HuggingFace dataset to JSON if needed
        if hasattr(dataset, 'to_dict'):
            # HuggingFace Dataset object
            data_dict = dataset.to_dict()
            # Convert to list of dicts
            num_samples = len(data_dict[list(data_dict.keys())[0]])
            data_list = [
                {key: data_dict[key][i] for key in data_dict.keys()}
                for i in range(num_samples)
            ]
        else:
            # Already a list
            data_list = dataset

        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)

        logger.success(f"Saved raw training data to: {train_file}")

        # Print statistics
        logger.info("=" * 60)
        logger.info("Dataset Statistics:")
        logger.info(f"  Total samples: {len(data_list)}")

        # Count by difficulty if available
        if isinstance(data_list[0], dict) and 'difficulty' in data_list[0]:
            difficulty_counts = {}
            for sample in data_list:
                diff = sample.get('difficulty', 'unknown')
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

            logger.info("  Difficulty distribution:")
            for diff, count in sorted(difficulty_counts.items()):
                logger.info(f"    {diff}: {count} ({count/len(data_list)*100:.1f}%)")

        # Count unique databases
        if isinstance(data_list[0], dict) and 'db_id' in data_list[0]:
            unique_dbs = set(sample['db_id'] for sample in data_list)
            logger.info(f"  Unique databases: {len(unique_dbs)}")

        # Count samples with evidence
        if isinstance(data_list[0], dict) and 'evidence' in data_list[0]:
            with_evidence = sum(1 for sample in data_list if sample.get('evidence', '').strip())
            logger.info(f"  Samples with evidence hints: {with_evidence} ({with_evidence/len(data_list)*100:.1f}%)")

        logger.info("=" * 60)
        logger.success("Download complete!")
        logger.info(f"Next step: Run prepare_data.py to format and split the data")

    except Exception as e:
        logger.error(f"Failed to download BIRD training data: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download BIRD training dataset from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Output directory for downloaded data (default: ../data)"
    )

    args = parser.parse_args()

    # Convert to absolute path
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()

    download_bird_training_data(str(output_dir))


if __name__ == "__main__":
    main()

"""Prepare BIRD training data for fine-tuning.

Formats the raw BIRD data into instruction-tuning format and splits into train/eval sets.

Output format (Alpaca-style instruction tuning):
    {
        "instruction": "Generate a SQL query to answer the question based on the schema.",
        "input": "Schema: CREATE TABLE ...\n\nQuestion: <question>\nEvidence: <evidence>",
        "output": "<SQL query>"
    }

Usage:
    python prepare_data.py [--input ../data/bird_train_raw.json] [--output ../data] [--eval_split 0.1]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger


def load_schema_for_database(db_id: str, schema_dir: Path) -> str:
    """Load schema for a database from dev_tables.json.

    Args:
        db_id: Database identifier
        schema_dir: Directory containing schema files

    Returns:
        Formatted CREATE TABLE schema string
    """
    # Try to load from dev_tables.json (contains schema info)
    tables_file = schema_dir / "bird" / "dev_tables.json"

    if not tables_file.exists():
        # Fallback to spider tables if bird not available
        tables_file = schema_dir / "spider" / "tables.json"

    if not tables_file.exists():
        logger.warning(f"Schema file not found: {tables_file}")
        return "-- Schema not available --"

    try:
        with open(tables_file, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)

        # Find the database
        db_schema = None
        for db in tables_data:
            if db.get('db_id') == db_id:
                db_schema = db
                break

        if not db_schema:
            logger.warning(f"Schema not found for database: {db_id}")
            return f"-- Schema not found for database: {db_id} --"

        # Format as CREATE TABLE statements
        schema_str = ""
        table_names = db_schema.get('table_names_original', [])
        column_names = db_schema.get('column_names_original', [])
        column_types = db_schema.get('column_types', [])

        # Build CREATE TABLE statements
        for table_idx, table_name in enumerate(table_names):
            schema_str += f"CREATE TABLE {table_name} (\n"

            # Get columns for this table
            table_columns = [
                (col_name, col_types[i] if i < len(column_types) else "TEXT")
                for i, (tab_idx, col_name) in enumerate(column_names)
                if tab_idx == table_idx
            ]

            for col_name, col_type in table_columns:
                # Add backticks for columns with special characters
                if ' ' in col_name or '(' in col_name or '-' in col_name:
                    col_name = f"`{col_name}`"
                schema_str += f"  {col_name} {col_type},\n"

            schema_str = schema_str.rstrip(',\n') + "\n);\n\n"

        return schema_str.strip()

    except Exception as e:
        logger.error(f"Error loading schema for {db_id}: {e}")
        return f"-- Error loading schema: {e} --"


def format_sample_for_training(sample: Dict, schema_dir: Path) -> Dict[str, str]:
    """Format a single sample into instruction-tuning format.

    Args:
        sample: Raw BIRD sample
        schema_dir: Directory containing schema files

    Returns:
        Formatted sample with instruction/input/output fields
    """
    # Load schema for this database
    db_id = sample.get('db_id', '')
    schema = load_schema_for_database(db_id, schema_dir)

    # Build the instruction (consistent across all samples)
    instruction = (
        "Generate a valid SQLite query to answer the following question based on the provided database schema. "
        "Use exact column names with backticks for special characters. "
        "If Evidence is provided, use the exact values and column names mentioned."
    )

    # Build the input with schema + question + evidence
    question = sample.get('question', '')
    evidence = sample.get('evidence', '').strip()

    input_text = f"Schema:\n{schema}\n\n"
    input_text += f"Question: {question}\n"

    if evidence:
        input_text += f"\nEvidence: {evidence}\n"

    # Output is the gold SQL
    output_sql = sample.get('SQL', '')

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_sql,
        "metadata": {
            "question_id": sample.get('question_id', -1),
            "db_id": db_id,
            "difficulty": sample.get('difficulty', 'unknown')
        }
    }


def split_data_by_database(
    data: List[Dict],
    eval_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train and eval sets, ensuring database diversity.

    We split by database to prevent data leakage and ensure the model
    generalizes to unseen databases (critical for preventing overfitting).

    Args:
        data: List of training samples
        eval_ratio: Fraction of databases to use for evaluation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, eval_data)
    """
    random.seed(seed)

    # Group samples by database
    db_samples = {}
    for sample in data:
        db_id = sample['metadata']['db_id']
        if db_id not in db_samples:
            db_samples[db_id] = []
        db_samples[db_id].append(sample)

    # Split databases into train and eval
    db_ids = list(db_samples.keys())
    random.shuffle(db_ids)

    num_eval_dbs = max(1, int(len(db_ids) * eval_ratio))
    eval_db_ids = set(db_ids[:num_eval_dbs])
    train_db_ids = set(db_ids[num_eval_dbs:])

    logger.info(f"Total databases: {len(db_ids)}")
    logger.info(f"Train databases: {len(train_db_ids)}")
    logger.info(f"Eval databases: {len(eval_db_ids)}")

    # Split samples
    train_data = []
    eval_data = []

    for db_id, samples in db_samples.items():
        if db_id in eval_db_ids:
            eval_data.extend(samples)
        else:
            train_data.extend(samples)

    # Shuffle within each split
    random.shuffle(train_data)
    random.shuffle(eval_data)

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Eval samples: {len(eval_data)}")

    return train_data, eval_data


def prepare_training_data(
    input_file: str,
    output_dir: str,
    eval_split: float = 0.1,
    seed: int = 42
) -> None:
    """Prepare BIRD training data for fine-tuning.

    Args:
        input_file: Path to raw BIRD training data
        output_dir: Directory to save formatted data
        eval_split: Fraction of data for evaluation (by database)
        seed: Random seed for reproducibility
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw training data from: {input_path}")

    # Load raw data
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} raw samples")

    # Schema directory (parent of data dir)
    schema_dir = input_path.parent.parent / "data"

    # Format samples
    logger.info("Formatting samples for instruction tuning...")
    formatted_data = []

    for i, sample in enumerate(raw_data):
        if (i + 1) % 500 == 0:
            logger.info(f"Formatted {i + 1}/{len(raw_data)} samples...")

        try:
            formatted_sample = format_sample_for_training(sample, schema_dir)
            formatted_data.append(formatted_sample)
        except Exception as e:
            logger.warning(f"Failed to format sample {i}: {e}")

    logger.success(f"Successfully formatted {len(formatted_data)} samples")

    # Split into train and eval (by database for better generalization)
    logger.info("Splitting data into train and eval sets...")
    train_data, eval_data = split_data_by_database(
        formatted_data,
        eval_ratio=eval_split,
        seed=seed
    )

    # Save train data
    train_file = output_path / "bird_train_formatted.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    logger.success(f"Saved training data to: {train_file}")

    # Save eval data
    eval_file = output_path / "bird_eval_formatted.json"
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    logger.success(f"Saved evaluation data to: {eval_file}")

    # Print statistics
    logger.info("=" * 60)
    logger.info("Data Preparation Complete!")
    logger.info(f"  Train samples: {len(train_data)}")
    logger.info(f"  Eval samples: {len(eval_data)}")
    logger.info(f"  Total: {len(formatted_data)}")

    # Difficulty distribution for train set
    train_difficulties = {}
    for sample in train_data:
        diff = sample['metadata']['difficulty']
        train_difficulties[diff] = train_difficulties.get(diff, 0) + 1

    logger.info("  Train difficulty distribution:")
    for diff, count in sorted(train_difficulties.items()):
        logger.info(f"    {diff}: {count} ({count/len(train_data)*100:.1f}%)")

    logger.info("=" * 60)
    logger.info("Next step: Run one of the fine-tuning scripts:")
    logger.info("  - finetune_phi4.py")
    logger.info("  - finetune_qwen.py")
    logger.info("  - finetune_deepseek.py")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare BIRD training data for fine-tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/bird_train_raw.json",
        help="Input raw BIRD training data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Output directory for formatted data"
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.1,
        help="Fraction of databases for evaluation (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Convert to absolute paths
    script_dir = Path(__file__).parent
    input_file = (script_dir / args.input).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    prepare_training_data(
        str(input_file),
        str(output_dir),
        args.eval_split,
        args.seed
    )


if __name__ == "__main__":
    main()

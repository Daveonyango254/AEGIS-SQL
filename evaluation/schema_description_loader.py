"""Load schema descriptions from BIRD database_description CSVs.

BIRD datasets include detailed column descriptions in CSV files:
    data/bird/dev_databases/{db_id}/database_description/{table}.csv

This module loads these descriptions to enhance semantic retrieval.
"""

import csv
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


def load_bird_descriptions(
    db_id: str, bird_path: str | Path = "data/bird"
) -> Dict[str, str]:
    """Load schema descriptions from BIRD database_description CSVs.

    Args:
        db_id: Database identifier (e.g., "california_schools")
        bird_path: Path to BIRD data directory

    Returns:
        Dict mapping "table.column" -> description

    Example:
        >>> descriptions = load_bird_descriptions("california_schools")
        >>> descriptions["schools.CDSCode"]
        'CDSCode'
        >>> descriptions["schools.NCESDist"]
        'This field represents the 7-digit National Center for...'
    """
    bird_path = Path(bird_path)
    descriptions = {}

    # Path to database_description directory
    desc_dir = bird_path / "dev_databases" / db_id / "database_description"

    if not desc_dir.exists():
        logger.warning(f"No database_description directory found for {db_id}")
        return descriptions

    # Load all CSV files in description directory
    csv_files = list(desc_dir.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No description CSV files found in {desc_dir}")
        return descriptions

    logger.debug(f"Loading descriptions for {db_id} from {len(csv_files)} CSV files")

    for csv_file in csv_files:
        table_name = csv_file.stem  # filename without .csv

        try:
            # Try multiple encodings to handle BIRD dataset encoding issues
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            file_content = None

            for encoding in encodings:
                try:
                    with open(csv_file, 'r', encoding=encoding) as f:
                        file_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if file_content is None:
                logger.warning(f"Failed to read {csv_file} with all attempted encodings")
                continue

            # Parse CSV from decoded content
            import io
            reader = csv.DictReader(io.StringIO(file_content))

            for row in reader:
                # BIRD CSV format:
                # - column_name: The actual column name
                # - column_description: Human-readable description
                # - value_description: Additional value semantics
                # - data_format: Data type information

                column_name = row.get('column_name', '').strip()
                if not column_name:
                    continue

                # Build comprehensive description
                description_parts = []

                # Main description
                col_desc = row.get('column_description', '').strip()
                if col_desc:
                    description_parts.append(col_desc)

                # Value description (often contains important semantics)
                val_desc = row.get('value_description', '').strip()
                if val_desc and val_desc != col_desc:
                    description_parts.append(f"Values: {val_desc}")

                # Data format
                data_format = row.get('data_format', '').strip()
                if data_format and data_format.lower() not in ['text', 'integer', 'real']:
                    description_parts.append(f"Format: {data_format}")

                # Combine all parts
                full_description = " | ".join(description_parts)

                # Store as "table.column" -> description
                key = f"{table_name}.{column_name}"
                descriptions[key] = full_description if full_description else column_name

        except Exception as e:
            logger.warning(f"Failed to load descriptions from {csv_file}: {e}")
            continue

    logger.info(f"✓ Loaded {len(descriptions)} column descriptions for {db_id}")

    return descriptions


def enhance_schema_with_descriptions(
    schema, db_id: str, bird_path: str | Path = "data/bird"
):
    """Enhance Schema object with BIRD descriptions in-place.

    Args:
        schema: Schema object to enhance
        db_id: Database identifier
        bird_path: Path to BIRD data directory

    Returns:
        Number of columns enhanced with descriptions
    """
    descriptions = load_bird_descriptions(db_id, bird_path)

    if not descriptions:
        return 0

    enhanced_count = 0

    for col in schema.columns:
        # Schema columns have format "table.column"
        if col.name in descriptions:
            col.description = descriptions[col.name]
            enhanced_count += 1

    logger.info(f"Enhanced {enhanced_count}/{len(schema.columns)} columns with descriptions")

    return enhanced_count

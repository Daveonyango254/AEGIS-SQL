"""Configuration for evaluation modules.

Default settings for EX and VES evaluators.
"""

from pathlib import Path

# Output directory for evaluation results
OUTPUT_DIR = Path("evaluation/output")

# Default query timeout in seconds
DEFAULT_QUERY_TIMEOUT = 5

# Maximum number of rows to fetch for comparison
MAX_ROWS = 10000

# Database configuration
DB_TIMEOUT = 10  # seconds

# Multiprocessing settings
NUM_WORKERS = 4  # Number of parallel workers for evaluation

# VES-specific settings
VES_TIME_LIMIT = 30  # seconds
VES_SAMPLE_SIZE = 100  # rows to sample for efficiency comparison

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time} {level} {message}"

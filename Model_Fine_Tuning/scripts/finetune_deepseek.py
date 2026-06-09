"""Fine-tune deepseek-ai/deepseek-coder-6.7b-instruct on BIRD benchmark.

Wrapper script for easy DeepSeek fine-tuning.

Usage:
    python finetune_deepseek.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run DeepSeek fine-tuning."""
    # Get config path
    script_dir = Path(__file__).parent
    config_path = script_dir / "../config/deepseek_config.yaml"

    # Run generic fine-tuning script
    cmd = [
        sys.executable,
        str(script_dir / "finetune.py"),
        "--config",
        str(config_path)
    ]

    print(f"Starting DeepSeek-Coder fine-tuning...")
    print(f"Config: {config_path}")
    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

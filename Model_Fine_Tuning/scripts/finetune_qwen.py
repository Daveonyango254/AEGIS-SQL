"""Fine-tune Qwen/Qwen2.5-Coder-7B-Instruct on BIRD benchmark.

Wrapper script for easy Qwen fine-tuning.

Usage:
    python finetune_qwen.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run Qwen fine-tuning."""
    # Get config path
    script_dir = Path(__file__).parent
    config_path = script_dir / "../config/qwen_config.yaml"

    # Run generic fine-tuning script
    cmd = [
        sys.executable,
        str(script_dir / "finetune.py"),
        "--config",
        str(config_path)
    ]

    print(f"Starting Qwen2.5-Coder fine-tuning...")
    print(f"Config: {config_path}")
    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

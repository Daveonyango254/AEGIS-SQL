"""Fine-tune microsoft/Phi-4-mini-instruct on BIRD benchmark.

Wrapper script for easy Phi-4 fine-tuning.

Usage:
    python finetune_phi4.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run Phi-4 fine-tuning."""
    # Get config path
    script_dir = Path(__file__).parent
    config_path = script_dir / "../config/phi4_config.yaml"

    # Run generic fine-tuning script
    cmd = [
        sys.executable,
        str(script_dir / "finetune.py"),
        "--config",
        str(config_path)
    ]

    print(f"Starting Phi-4 fine-tuning...")
    print(f"Config: {config_path}")
    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

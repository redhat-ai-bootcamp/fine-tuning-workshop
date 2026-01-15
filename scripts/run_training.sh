#!/usr/bin/env bash
set -euo pipefail

python scripts/download_dataset.py --output_dir data/raw
python scripts/prepare_jsonl.py --input_dir data/raw/maildir --output_dir data/processed
python scripts/train.py --data_dir data/processed --output_dir outputs

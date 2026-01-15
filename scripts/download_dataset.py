#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Enron dataset from Kaggle")
    parser.add_argument(
        "--dataset",
        default="wcukierski/enron-email-dataset",
        help="Kaggle dataset slug",
    )
    parser.add_argument(
        "--output_dir",
        default="data/raw",
        help="Directory to download and unzip the dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    maildir = output_dir / "maildir"
    if maildir.exists():
        print(f"Dataset already present at {maildir}")
        return

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit(
            "KAGGLE_USERNAME/KAGGLE_KEY not set. Export them before downloading."
        )

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {args.dataset} to {output_dir}...")
    api.dataset_download_files(args.dataset, path=str(output_dir), unzip=True)
    if not maildir.exists():
        print("Download completed, but maildir was not found. Check the output directory.")


if __name__ == "__main__":
    main()

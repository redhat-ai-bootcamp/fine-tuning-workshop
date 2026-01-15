#!/usr/bin/env python3
import argparse
import json
import random
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

PHISH_KEYWORDS = [
    "password",
    "verify",
    "urgent",
    "bank",
    "account",
    "login",
    "click",
    "link",
    "reset",
    "wire",
    "transfer",
    "ssn",
    "payment",
    "invoice",
    "confirm",
    "security",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Enron emails into JSONL")
    parser.add_argument(
        "--input_dir",
        default="data/raw/maildir",
        help="Path to the maildir dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed",
        help="Directory to write JSONL files",
    )
    parser.add_argument(
        "--max_emails",
        type=int,
        default=50000,
        help="Maximum number of emails to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--phish_threshold",
        type=int,
        default=2,
        help="Keyword hits required to label as phishing",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=4000,
        help="Maximum characters to keep from the email body",
    )
    return parser.parse_args()


def iter_email_files(maildir: Path) -> Iterable[Path]:
    for path in maildir.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            yield path


def extract_subject_body(path: Path, max_chars: int) -> Tuple[str, str]:
    with path.open("rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)

    subject = message.get("subject") or ""
    body_parts = []
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body_parts.append(part.get_content())
                except Exception:
                    continue
    else:
        try:
            body_parts.append(message.get_content())
        except Exception:
            body_parts.append("")

    body = "\n".join(body_parts)
    subject = subject.strip()
    body = body.strip()

    # Normalize to ASCII to keep JSONL portable.
    subject = subject.encode("ascii", "ignore").decode("ascii")
    body = body.encode("ascii", "ignore").decode("ascii")

    body = re.sub(r"\s+", " ", body)
    if len(body) > max_chars:
        body = body[:max_chars]

    return subject, body


def label_email(text: str, threshold: int) -> str:
    lowered = text.lower()
    score = sum(1 for kw in PHISH_KEYWORDS if kw in lowered)
    if "http://" in lowered or "https://" in lowered:
        score += 1
    return "phishing" if score >= threshold else "benign"


def build_training_text(subject: str, body: str, label: str) -> str:
    return (
        "### Instruction:\n"
        "Classify the email as phishing or benign. Reply with only the label.\n"
        "### Email:\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n"
        "### Response:\n"
        f"{label}"
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = list(iter_email_files(input_dir))
    if not files:
        raise SystemExit("No email files found under maildir")

    random.Random(args.seed).shuffle(files)
    files = files[: args.max_emails]

    records = []
    for path in tqdm(files, desc="Parsing emails"):
        try:
            subject, body = extract_subject_body(path, args.max_chars)
        except Exception:
            continue
        if not body:
            continue
        text_for_label = f"{subject} {body}"
        label = label_email(text_for_label, args.phish_threshold)
        records.append(
            {
                "subject": subject,
                "body": body,
                "label": label,
                "text": build_training_text(subject, body, label),
            }
        )

    if not records:
        raise SystemExit("No usable emails parsed")

    labels = [r["label"] for r in records]
    stratify = labels if len(set(labels)) > 1 else None
    train_records, temp_records = train_test_split(
        records, test_size=0.2, random_state=args.seed, stratify=stratify
    )
    val_records, test_records = train_test_split(
        temp_records,
        test_size=0.5,
        random_state=args.seed,
        stratify=[r["label"] for r in temp_records] if stratify else None,
    )

    def write_jsonl(path: Path, rows) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)
    write_jsonl(output_dir / "test.jsonl", test_records)

    stats = {
        "total": len(records),
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
        "phishing": sum(1 for r in records if r["label"] == "phishing"),
        "benign": sum(1 for r in records if r["label"] == "benign"),
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Wrote JSONL files to {output_dir}")


if __name__ == "__main__":
    main()

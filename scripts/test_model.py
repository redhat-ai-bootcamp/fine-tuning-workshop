#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the phishing detector endpoint")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/predict",
        help="Prediction endpoint",
    )
    parser.add_argument(
        "--model_name",
        default="nvidia/Nemotron-4-Mini-HF",
        help="Base model name for local inference",
    )
    parser.add_argument(
        "--adapter_dir",
        default="",
        help="Optional LoRA adapter path for local inference",
    )
    parser.add_argument(
        "--test_file",
        default="",
        help="Optional JSONL test file with subject/body/label",
    )
    parser.add_argument("--max_samples", type=int, default=20)
    return parser.parse_args()


def build_prompt(subject: str, body: str) -> str:
    return (
        "### Instruction:\n"
        "Classify the email as phishing or benign. Reply with only the label.\n"
        "### Email:\n"
        f"Subject: {subject.strip()}\n"
        f"Body: {body.strip()}\n"
        "### Response:\n"
    )


def load_local_model(model_name: str, adapter_dir: Path):
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=quant_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer


def local_predict(model, tokenizer, subject: str, body: str):
    prompt = build_prompt(subject, body)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    raw = decoded[len(prompt) :].strip()
    label = "phishing" if "phish" in raw.lower() else "benign"
    return {"label": label, "raw_response": raw}


def remote_predict(endpoint: str, subject: str, body: str):
    payload = {"subject": subject, "body": body}
    response = requests.post(endpoint, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def run_examples(endpoint: str, model=None, tokenizer=None) -> None:
    samples = [
        {
            "subject": "Verify your payroll account",
            "body": "Please click the link to verify your payroll account before payroll runs.",
        },
        {
            "subject": "Lunch plans",
            "body": "Are we still on for lunch tomorrow at 12?",
        },
    ]
    for sample in samples:
        if model is not None:
            result = local_predict(model, tokenizer, sample["subject"], sample["body"])
        else:
            result = remote_predict(endpoint, sample["subject"], sample["body"])
        print(f"Subject: {sample['subject']}")
        print(f"Label: {result['label']} (raw: {result['raw_response']})")
        print("-")


def run_test_file(
    endpoint: str,
    test_file: Path,
    max_samples: int,
    model=None,
    tokenizer=None,
) -> None:
    correct = 0
    total = 0
    with test_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if total >= max_samples:
                break
            row = json.loads(line)
            if model is not None:
                result = local_predict(model, tokenizer, row.get("subject", ""), row.get("body", ""))
            else:
                result = remote_predict(endpoint, row.get("subject", ""), row.get("body", ""))
            pred = result.get("label")
            label = row.get("label")
            if pred == label:
                correct += 1
            total += 1
    if total:
        print(f"Accuracy on {total} samples: {correct / total:.2%}")
    else:
        print("No samples evaluated.")


def main() -> None:
    args = parse_args()
    model = tokenizer = None
    if args.adapter_dir:
        model, tokenizer = load_local_model(args.model_name, Path(args.adapter_dir))
    if args.test_file:
        run_test_file(args.endpoint, Path(args.test_file), args.max_samples, model, tokenizer)
    else:
        run_examples(args.endpoint, model, tokenizer)


if __name__ == "__main__":
    main()

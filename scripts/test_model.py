#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
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
        "--api",
        choices=["predict", "openai"],
        default="predict",
        help="Endpoint type: custom /predict or OpenAI-compatible completions",
    )
    parser.add_argument(
        "--model_name",
        default="nvidia/Nemotron-Mini-4B-Instruct",
        help="Base model name for local inference",
    )
    parser.add_argument(
        "--openai_model",
        default="",
        help="Model name to send to the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--adapter_dir",
        default="",
        help="Optional LoRA adapter path for local inference",
    )
    parser.add_argument(
        "--sft_model_dir",
        default="",
        help="Optional full SFT model path for local inference",
    )
    parser.add_argument(
        "--test_file",
        default="",
        help="Optional JSONL test file with subject/body/label",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="Optional JSON output file for accuracy results",
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


def load_local_model(model_name: str, adapter_dir: Optional[Path]):
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
    if adapter_dir:
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    else:
        model = base
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


def openai_predict(endpoint: str, model: str, prompt: str) -> Optional[str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 6,
        "temperature": 0.0,
    }
    response = requests.post(endpoint, json=payload, timeout=60)
    if response.status_code == 400:
        try:
            message = response.json().get("error", {}).get("message", "")
        except ValueError:
            message = response.text or ""
        if "maximum context length" in message.lower():
            return None
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI response missing choices")
    text = choices[0].get("text")
    if text is None:
        text = choices[0].get("message", {}).get("content", "")
    return text.strip()


def remote_predict(endpoint: str, subject: str, body: str):
    payload = {"subject": subject, "body": body}
    response = requests.post(endpoint, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_label(text: str) -> str:
    lowered = text.lower()
    if "phish" in lowered:
        return "phishing"
    if "benign" in lowered or "ham" in lowered:
        return "benign"
    return "unknown"


def run_examples(
    endpoint: str,
    api: str,
    openai_model: str,
    model=None,
    tokenizer=None,
) -> None:
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
        elif api == "openai":
            prompt = build_prompt(sample["subject"], sample["body"])
            raw = openai_predict(endpoint, openai_model, prompt)
            if raw is None:
                print("Skipped example: prompt too long for model context.")
                continue
            result = {"label": normalize_label(raw), "raw_response": raw}
        else:
            result = remote_predict(endpoint, sample["subject"], sample["body"])
        print(f"Subject: {sample['subject']}")
        print(f"Label: {result['label']} (raw: {result['raw_response']})")
        print("-")


def run_test_file(
    endpoint: str,
    api: str,
    openai_model: str,
    test_file: Path,
    max_samples: int,
    model=None,
    tokenizer=None,
) -> dict:
    correct = 0
    total = 0
    skipped = 0
    with test_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if total >= max_samples:
                break
            row = json.loads(line)
            if model is not None:
                result = local_predict(model, tokenizer, row.get("subject", ""), row.get("body", ""))
            elif api == "openai":
                prompt = build_prompt(row.get("subject", ""), row.get("body", ""))
                raw = openai_predict(endpoint, openai_model, prompt)
                if raw is None:
                    skipped += 1
                    continue
                result = {"label": normalize_label(raw), "raw_response": raw}
            else:
                result = remote_predict(endpoint, row.get("subject", ""), row.get("body", ""))
            pred = result.get("label")
            label = row.get("label")
            if pred == label:
                correct += 1
            total += 1
    if skipped:
        print(f"Skipped {skipped} samples over max context length.")
    if total:
        print(f"Accuracy on {total} samples: {correct / total:.2%}")
    else:
        print("No samples evaluated.")
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "skipped": skipped,
    }


def main() -> None:
    args = parse_args()
    model = tokenizer = None
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    model_name = args.model_name
    if args.sft_model_dir:
        sft_dir = Path(args.sft_model_dir)
        if not sft_dir.exists():
            raise SystemExit(f"SFT model directory not found: {sft_dir}")
        model_name = str(sft_dir)
        adapter_dir = None
    if adapter_dir or args.sft_model_dir:
        model, tokenizer = load_local_model(model_name, adapter_dir)
    openai_model = args.openai_model or args.model_name
    if args.test_file:
        result = run_test_file(
            args.endpoint,
            args.api,
            openai_model,
            Path(args.test_file),
            args.max_samples,
            model,
            tokenizer,
        )
        if args.output_file:
            payload = {
                "correct": result["correct"],
                "total": result["total"],
                "accuracy": result["accuracy"],
                "endpoint": args.endpoint,
                "api": args.api,
                "model_name": args.model_name,
                "openai_model": openai_model,
                "adapter_dir": args.adapter_dir,
                "sft_model_dir": args.sft_model_dir,
                "test_file": args.test_file,
                "max_samples": args.max_samples,
                "skipped": result.get("skipped", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2))
    else:
        run_examples(args.endpoint, args.api, openai_model, model, tokenizer)


if __name__ == "__main__":
    main()

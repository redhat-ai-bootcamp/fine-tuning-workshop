#!/usr/bin/env python3
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the phishing detector endpoint")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/predict",
        help="Prediction endpoint",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run local inference instead of calling the endpoint",
    )
    parser.add_argument(
        "--model_name",
        default="nvidia/Nemotron-4-Mini-HF",
        help="Base model name for local inference",
    )
    parser.add_argument(
        "--adapter_dir",
        default="",
        help="LoRA adapter path for local inference",
    )
    parser.add_argument("--num_requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def build_prompt() -> str:
    return (
        "### Instruction:\n"
        "Classify the email as phishing or benign. Reply with only the label.\n"
        "### Email:\n"
        "Subject: Account verification required\n"
        "Body: Please verify your account to avoid suspension. Click the link.\n"
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


def send_request(endpoint: str) -> float:
    payload = {
        "subject": "Account verification required",
        "body": "Please verify your account to avoid suspension. Click the link.",
    }
    start = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=60)
    response.raise_for_status()
    end = time.perf_counter()
    return end - start


def send_local(model, tokenizer) -> float:
    prompt = build_prompt()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    end = time.perf_counter()
    return end - start


def main() -> None:
    args = parse_args()

    model = tokenizer = None
    if args.local:
        if not args.adapter_dir:
            raise SystemExit("--adapter_dir is required for local benchmarking")
        model, tokenizer = load_local_model(args.model_name, Path(args.adapter_dir))

    for _ in range(args.warmup):
        if args.local:
            send_local(model, tokenizer)
        else:
            send_request(args.endpoint)

    latencies = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.local:
            futures = [pool.submit(send_local, model, tokenizer) for _ in range(args.num_requests)]
        else:
            futures = [pool.submit(send_request, args.endpoint) for _ in range(args.num_requests)]
        for future in as_completed(futures):
            latencies.append(future.result())
    end = time.perf_counter()

    latencies.sort()
    total_time = end - start
    rps = args.num_requests / total_time if total_time else 0
    p50 = statistics.median(latencies) if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99) - 1] if latencies else 0

    print(f"Requests: {args.num_requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Total time: {total_time:.2f}s")
    print(f"RPS: {rps:.2f}")
    print(f"p50 latency: {p50:.3f}s")
    print(f"p95 latency: {p95:.3f}s")
    print(f"p99 latency: {p99:.3f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a fine-tuned Nemotron model")
    parser.add_argument(
        "--model_name",
        default="nvidia/Nemotron-4-Mini-HF",
        help="Base model name",
    )
    parser.add_argument(
        "--adapter_dir",
        default="outputs/adapter",
        help="Path to LoRA adapter",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def build_prompt(subject: str, body: str) -> str:
    subject = subject.strip()
    body = body.strip()
    return (
        "### Instruction:\n"
        "Classify the email as phishing or benign. Reply with only the label.\n"
        "### Email:\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n"
        "### Response:\n"
    )


class PredictRequest(BaseModel):
    subject: Optional[str] = ""
    body: str


class PredictResponse(BaseModel):
    label: str
    raw_response: str


def load_model(model_name: str, adapter_dir: Path):
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


def infer(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip()


def normalize_label(text: str) -> str:
    lowered = text.lower()
    if "phish" in lowered:
        return "phishing"
    if "benign" in lowered or "ham" in lowered:
        return "benign"
    return "unknown"


def create_app(args: argparse.Namespace) -> FastAPI:
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise SystemExit(f"Adapter directory not found: {adapter_dir}")

    model, tokenizer = load_model(args.model_name, adapter_dir)
    app = FastAPI(title="Nemotron Phishing Detector")

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        prompt = build_prompt(request.subject or "", request.body)
        raw = infer(model, tokenizer, prompt, args.max_new_tokens, args.temperature)
        label = normalize_label(raw)
        return PredictResponse(label=label, raw_response=raw)

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

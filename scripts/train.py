#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron with LoRA")
    parser.add_argument(
        "--model_name",
        default="nvidia/Nemotron-4-Mini-HF",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--data_dir",
        default="data/processed",
        help="Directory containing train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Output directory for adapters and logs",
    )
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules",
    )
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--bf16", action="store_true", help="Force bfloat16 compute")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16 compute")
    return parser.parse_args()


def resolve_bf16(force_bf16: bool, disable_bf16: bool) -> bool:
    if disable_bf16:
        return False
    if force_bf16:
        return True
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def build_quant_config(use_4bit: bool, bf16: bool) -> BitsAndBytesConfig:
    if not use_4bit:
        return None
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bf16 = resolve_bf16(args.bf16, args.no_bf16)
    use_4bit = not args.no_4bit

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    if not train_file.exists() or not val_file.exists():
        raise SystemExit("Missing train.jsonl or val.jsonl. Run prepare_jsonl.py first.")

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )

    quant_config = build_quant_config(use_4bit, bf16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="none",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        peft_config=lora_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        packing=True,
    )

    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()

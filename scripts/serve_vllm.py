#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Nemotron with vLLM (OpenAI API)")
    parser.add_argument(
        "--model_name",
        default="nvidia/Nemotron-Mini-4B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter_dir",
        default="",
        help="Optional LoRA adapter directory",
    )
    parser.add_argument(
        "--adapter_name",
        default="phishing",
        help="LoRA adapter name exposed to the OpenAI API",
    )
    parser.add_argument(
        "--served_model_name",
        default="",
        help="Override the model name exposed by the OpenAI server",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=0,
        help="Override the model context length (0 uses vLLM default)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.0,
        help="Target GPU memory utilization, e.g. 0.8 (0 uses vLLM default)",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=0,
        help="Upper bound for total tokens per batch (0 uses vLLM default)",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=0,
        help="Upper bound for concurrent sequences (0 uses vLLM default)",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graph capture to reduce startup memory pressure",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        if not adapter_dir.exists():
            raise SystemExit(f"Adapter directory not found: {adapter_dir}")
        cmd += ["--enable-lora", "--lora-modules", f"{args.adapter_name}={adapter_dir}"]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.gpu_memory_utilization:
        cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.max_num_seqs:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.enforce_eager:
        cmd += ["--enforce-eager"]
    return cmd


def main() -> None:
    args = parse_args()
    command = build_command(args)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

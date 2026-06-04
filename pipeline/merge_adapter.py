"""
Stage 4c step 1 — merge a LoRA adapter back into Qwen2.5-1.5B-Instruct at fp16.

The 4-bit base used during training is incompatible with GGUF conversion. We need
a clean fp16 merged checkpoint to feed into llama.cpp's convert_hf_to_gguf.py.

Usage:
  E:/coglang/envs/coglang-train/python.exe E:/coglang/pipeline/merge_adapter.py --run-id drug-v3

Output: E:/coglang/merged/<run-id>/  (full fp16 model, ~3GB)
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "E:/coglang/hf-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True,
                    help="adapter dir name under E:/coglang/adapters/")
    args = ap.parse_args()

    adapter = Path("E:/coglang/adapters") / args.run_id
    out_dir = Path("E:/coglang/merged") / args.run_id
    if not adapter.exists():
        sys.exit(f"adapter dir not found: {adapter}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading base {MODEL_ID} in fp16 ...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    print(f"  VRAM after base load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print(f"attaching adapter {adapter} ...")
    model = PeftModel.from_pretrained(base, str(adapter))

    print("merging adapter into base weights ...")
    model = model.merge_and_unload()
    print(f"  VRAM after merge: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print(f"saving merged model -> {out_dir}")
    model.save_pretrained(str(out_dir), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.save_pretrained(str(out_dir))

    files = sorted(out_dir.glob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024**2
    print(f"DONE. {len(files)} files, total {total_mb:.0f} MB -> {out_dir}")


if __name__ == "__main__":
    main()

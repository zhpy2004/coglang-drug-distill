"""
Stage 4b — QLoRA fine-tune Qwen2.5-1.5B-Instruct on drug-domain CogLang SFT data.

Usage (from the coglang-train env):
  set HF_HOME=E:\\coglang\\hf-cache
  E:\\coglang\\envs\\coglang-train\\python.exe E:\\coglang\\pipeline\\train_qlora.py

The script:
  1. Loads pipeline/sft_positive.jsonl (116 chat-format samples).
  2. Stratified split by level → ~100 train / ~16 val (deterministic SEED=42).
  3. Loads Qwen2.5-1.5B-Instruct in 4-bit (nf4, double-quant, fp16 compute).
  4. Wraps with LoRA r=16 alpha=32 on q/k/v/o + gate/up/down (all attn + MLP).
  5. Trains 3 epochs, batch=1, grad_accum=4, lr=2e-4 cosine, max_seq_len=512.
  6. Saves the adapter to E:\\coglang\\adapters\\drug-v0\\.

Tuning knobs at top of file — keep changes there so the rest mirrors a clean recipe.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

# E: cache redirects — set BEFORE any HF import so they take effect
os.environ.setdefault("HF_HOME", "E:/coglang/hf-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- fixed knobs (rarely changed across runs) ---
MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_PATH     = Path("E:/coglang/pipeline/sft_positive.jsonl")
ADAPTERS_DIR = Path("E:/coglang/adapters")
VAL_SIZE     = 16
SEED         = 42

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODS  = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

BATCH        = 1
GRAD_ACC     = 4
MAX_SEQ_LEN  = 1024


def stratified_split(samples: list[dict], val_size: int, seed: int):
    """Per-level deterministic split: val takes (val_size * n_lv / N) per level."""
    rng = random.Random(seed)
    by_level: dict[str, list[dict]] = {}
    for s in samples:
        by_level.setdefault(s["level"], []).append(s)
    train, val = [], []
    n_total = len(samples)
    for lv, items in by_level.items():
        items = items.copy()
        rng.shuffle(items)
        n_val = max(1, round(val_size * len(items) / n_total))
        val += items[:n_val]
        train += items[n_val:]
    rng.shuffle(train)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True,
                    help="adapter dir name (e.g. drug-v1) — output lands at "
                         "E:/coglang/adapters/<run-id>/")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--completion-only", action="store_true", default=True,
                    help="mask loss on system+user tokens (default: TRUE — see drug-v0 lessons)")
    ap.add_argument("--no-completion-only", dest="completion_only",
                    action="store_false",
                    help="DISABLE completion-only loss (train on full sequence) — used to reproduce drug-v0 baseline")
    args = ap.parse_args()

    out_dir = ADAPTERS_DIR / args.run_id
    if not SFT_PATH.exists():
        sys.exit(f"missing SFT file: {SFT_PATH}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== run: {args.run_id}  epochs={args.epochs}  lr={args.lr}  "
          f"completion_only={args.completion_only} ===")

    raw = [json.loads(line) for line in SFT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    train, val = stratified_split(raw, VAL_SIZE, SEED)
    train_lv = {lv: sum(1 for s in train if s["level"] == lv) for lv in sorted({s["level"] for s in raw})}
    val_lv = {lv: sum(1 for s in val if s["level"] == lv) for lv in sorted({s["level"] for s in raw})}
    print(f"data: {len(raw)} total -> train={len(train)} {train_lv}  val={len(val)} {val_lv}")

    train_ds = Dataset.from_list([{"messages": s["messages"]} for s in train])
    val_ds   = Dataset.from_list([{"messages": s["messages"]} for s in val])

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # match bf16=True training dtype
        bnb_4bit_use_double_quant=True,
    )
    print("loading base model (4-bit) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    print(f"  VRAM after model load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    peft_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODS,
    )

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",        # only final adapter via trainer.save_model()
        eval_strategy="epoch",
        bf16=True,                 # Ada+ (4060) native bf16 — no GradScaler needed
        optim="paged_adamw_8bit",  # bnb 8-bit optimizer to save VRAM
        max_length=MAX_SEQ_LEN,
        packing=False,
        completion_only_loss=args.completion_only,
        report_to="none",
        seed=SEED,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tok,
    )
    print("starting training ...")
    trainer.train()
    print(f"PEAK VRAM during training: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"\nDONE. adapter + tokenizer saved -> {out_dir}")


if __name__ == "__main__":
    main()

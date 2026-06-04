"""
Stage 4b verification — compare baseline vs LoRA-adapted Qwen2.5-1.5B
on the held-out val split.

Reproduces the train/val split from train_qlora.py (same SEED=42, same stratified
logic), then runs both the bare base model and the base+adapter through:
  1. the 'warfarin interacts' showcase prompt
  2. all val cases (~16)

Writes responses_base.jsonl + responses_lora.jsonl, then scores both files via
`coglang generation-eval --responses-file --fixture drug_expanded_fixture.json`.
"""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HF_HOME", "E:/coglang/hf-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

HERE     = Path(__file__).resolve().parent
REPO     = HERE.parent / "coglang"
# coglang package is installed in the SYSTEM python (3.14), not the train env (3.12)
COGLANG_PY = r"C:\Python314\python.exe"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_PATH = HERE / "sft_positive.jsonl"
FIXTURE  = HERE / "drug_expanded_fixture.json"
SEED, VAL_SIZE = 42, 16


def stratified_split(samples, val_size, seed):
    rng = random.Random(seed)
    by_level = {}
    for s in samples:
        by_level.setdefault(s["level"], []).append(s)
    train, val = [], []
    n_total = len(samples)
    for lv, items in by_level.items():
        items = items.copy(); rng.shuffle(items)
        n_val = max(1, round(val_size * len(items) / n_total))
        val += items[:n_val]; train += items[n_val:]
    rng.shuffle(train)
    return train, val


def load_base(bnb_cfg):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_cfg, device_map="auto",
    )


def gen(model, tok, system, user, max_new=192):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    txt = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return txt, dt, out.shape[1] - inp.input_ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True,
                    help="run id under E:/coglang/adapters/ (e.g. drug-v1)")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the BASE pass — only score the LoRA adapter "
                         "(use when comparing several adapters in a row)")
    ap.add_argument("--ood-fixture", default="",
                    help="optional hand-written OOD fixture (JSON, same shape as "
                         "drug_seed_fixture.json). If given, also generates + "
                         "scores those prompts. Writes responses_*_ood.jsonl.")
    args = ap.parse_args()

    adapter = Path("E:/coglang/adapters") / args.run_id
    if not adapter.exists():
        sys.exit(f"adapter dir not found: {adapter}")
    resp_base = HERE / "responses_base.jsonl"
    resp_lora = HERE / f"responses_{args.run_id}.jsonl"
    resp_base_ood = HERE / "responses_base_ood.jsonl"
    resp_lora_ood = HERE / f"responses_{args.run_id}_ood.jsonl"

    ood_cases = []
    ood_fixture_path = None
    if args.ood_fixture:
        ood_fixture_path = Path(args.ood_fixture).resolve()
        if not ood_fixture_path.exists():
            sys.exit(f"OOD fixture not found: {ood_fixture_path}")
        ood_cases = json.loads(ood_fixture_path.read_text(encoding="utf-8"))["cases"]
        print(f"OOD fixture: {ood_fixture_path.name}  ({len(ood_cases)} cases)")

    # Reproduce val split from training
    raw = [json.loads(l) for l in SFT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    _, val = stratified_split(raw, VAL_SIZE, SEED)
    print(f"=== eval {args.run_id} === val split: {len(val)} cases  "
          f"(levels: { {lv: sum(1 for s in val if s['level']==lv) for lv in sorted({s['level'] for s in val})} })")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)

    # Use the SYSTEM stored in the SFT samples (so both runs see identical context)
    SYSTEM = val[0]["messages"][0]["content"]
    SHOWCASE = "Return a CogLang expression that lists all drugs that warfarin is known to interact with."

    runs = []
    if not args.skip_base:
        runs.append(("BASE", False, resp_base, resp_base_ood))
    runs.append((args.run_id.upper(), True, resp_lora, resp_lora_ood))
    for tag, with_adapter, out_path, ood_out_path in runs:
        print(f"\n===== {tag} =====")
        model = load_base(bnb)
        if with_adapter:
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()

        sc_out, sc_dt, sc_n = gen(model, tok, SYSTEM, SHOWCASE)
        print(f"[showcase] warfarin interacts -> ({sc_n} toks / {sc_dt:.1f}s)")
        print(f"  {sc_out!r}")

        # Generate on val
        with out_path.open("w", encoding="utf-8") as f:
            for i, s in enumerate(val, 1):
                user = s["messages"][1]["content"]
                txt, dt, ntok = gen(model, tok, SYSTEM, user)
                f.write(json.dumps({"case_id": s["case_id"], "output": txt}, ensure_ascii=False) + "\n")
                print(f"  {i:>2}/{len(val)}  {s['case_id']:<13} ({ntok:>3}t/{dt:>4.1f}s) {txt[:90]}")
        print(f"-> wrote {out_path.name}")

        if ood_cases:
            with ood_out_path.open("w", encoding="utf-8") as f:
                for i, c in enumerate(ood_cases, 1):
                    txt, dt, ntok = gen(model, tok, SYSTEM, c["prompt"])
                    f.write(json.dumps({"case_id": c["case_id"], "output": txt}, ensure_ascii=False) + "\n")
                    print(f"  OOD {i:>2}/{len(ood_cases)}  {c['case_id']:<10} ({ntok:>3}t/{dt:>4.1f}s) {txt[:90]}")
            print(f"-> wrote {ood_out_path.name}")

        # Free the model before loading the next one
        del model
        torch.cuda.empty_cache()

    # Score both response files via coglang generation-eval
    print("\n===== SCORING =====")

    def score(tag, resp, fixture_path, label):
        proc = subprocess.run(
            [COGLANG_PY, "-m", "coglang", "generation-eval",
             "--responses-file", str(resp), "--fixture", str(fixture_path),
             "--format", "json"],
            cwd=str(REPO), capture_output=True, text=True,
        )
        try:
            scored = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(f"[{tag} {label}] generation-eval produced no JSON:\n{proc.stdout}\n{proc.stderr}")
            return
        answered = {s["case_id"] for s in [json.loads(l) for l in resp.read_text(encoding="utf-8").splitlines()]}
        cases = [c for c in scored["cases"] if c["case_id"] in answered]
        n = len(cases) or 1
        rate = lambda k: sum(1 for c in cases if c.get(k)) / n
        fails = sum(1 for c in cases if c["failure_categories"])
        hall = sum(len(c["hallucinated_heads"]) for c in cases)
        print(f"[{tag} {label}] n={len(cases)}  pass={n-fails}/{n} ({100*(n-fails)//n}%)  "
              f"parse_ok={rate('parse_ok'):.0%}  validate_ok={rate('validate_ok'):.0%}  "
              f"top_head_ok={rate('expected_top_level_head_ok'):.0%}  hallucinated={hall}")

    score_list = []
    if not args.skip_base:
        score_list.append(("BASE", resp_base, resp_base_ood))
    score_list.append((args.run_id.upper(), resp_lora, resp_lora_ood))
    for tag, resp_val, resp_ood in score_list:
        score(tag, resp_val, FIXTURE, "val")
        if ood_cases:
            score(tag, resp_ood, ood_fixture_path, "OOD")


if __name__ == "__main__":
    main()

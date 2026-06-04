"""
Stage 4c step 3 — run the Q4_K_M GGUF model through llama-cli with GBNF
grammar constraint, score against the OOD eval set.

For each prompt this measures: prompt eval tok/s, generation tok/s.
Peak RSS is measured externally (look at the llama-cli wall-time + psutil snapshot).

Usage:
  PYTHONUTF8=1 E:/coglang/envs/coglang-train/python.exe \
    E:/coglang/pipeline/eval_gguf.py \
    --gguf E:/coglang/gguf/drug-v3-q4_k_m.gguf \
    --fixture E:/coglang/pipeline/ood_eval.json \
    --tag drug-v3-q4_k_m
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "coglang"
SFT_PATH = HERE / "sft_positive.jsonl"
LLAMA_CLI = Path("E:/coglang/llama.cpp/bin/llama-cli.exe")
GRAMMAR = REPO / "examples" / "grammar" / "coglang.gbnf"
COGLANG_PY = r"C:\Python314\python.exe"

# llama-cli emits an ASCII art banner + info + a "> {user_prompt}" echo + reply + timings + "Exiting..."
# We need to extract just the reply text between the echoed user line and the [Prompt: ... t/s] line.
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TIMING_RE = re.compile(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def run_one(gguf: Path, system: str, user: str, max_tokens: int, threads: int) -> dict:
    cmd = [
        str(LLAMA_CLI),
        "-m", str(gguf),
        "--grammar-file", str(GRAMMAR),
        "-st",
        "-sys", system,
        "-p", user,
        "-n", str(max_tokens),
        "--temp", "0",
        "-t", str(threads),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    out = strip_ansi(proc.stdout)
    # find timing line + last user-prompt echo line
    m = TIMING_RE.search(out)
    prompt_tps = float(m.group(1)) if m else None
    gen_tps = float(m.group(2)) if m else None
    # reply = text between "> {user}" line and the timing line
    timing_pos = m.start() if m else len(out)
    head = out[:timing_pos]
    user_echo = "> " + user
    if user_echo in head:
        reply = head.split(user_echo, 1)[1]
    else:
        # fallback: take everything after the last "> " up to timing
        reply = head.rsplit("> ", 1)[-1]
    reply = reply.strip()
    return {"reply": reply, "prompt_tps": prompt_tps, "gen_tps": gen_tps, "wall_s": dt,
            "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.stderr else []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="path to GGUF model")
    ap.add_argument("--fixture", required=True, help="OOD fixture JSON")
    ap.add_argument("--tag", required=True, help="output tag, e.g. drug-v3-q4_k_m")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--system-file", default="",
                    help="optional text file with override SYSTEM prompt; "
                         "default uses SYSTEM from sft_positive.jsonl[0]")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    gguf = Path(args.gguf).resolve()
    fixture = Path(args.fixture).resolve()
    if not gguf.exists(): sys.exit(f"missing GGUF: {gguf}")
    if not fixture.exists(): sys.exit(f"missing fixture: {fixture}")
    if not LLAMA_CLI.exists(): sys.exit(f"missing llama-cli: {LLAMA_CLI}")
    if not GRAMMAR.exists(): sys.exit(f"missing grammar: {GRAMMAR}")

    # Use the same SYSTEM the adapter was trained with — unless --system-file overrides
    if args.system_file:
        SYSTEM = Path(args.system_file).read_text(encoding="utf-8")
        print(f"SYSTEM len={len(SYSTEM)} chars from {args.system_file} (OVERRIDE)")
    else:
        raw = [json.loads(l) for l in SFT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        SYSTEM = raw[0]["messages"][0]["content"]
        print(f"SYSTEM len={len(SYSTEM)} chars from sft_positive.jsonl[0]")
    print(f"loaded {len(json.loads(fixture.read_text(encoding='utf-8'))['cases'])} OOD cases")

    cases = json.loads(fixture.read_text(encoding="utf-8"))["cases"]
    resp_path = HERE / f"responses_{args.tag}_ood.jsonl"
    tps_prompt = []
    tps_gen = []
    wall_total = 0.0
    with resp_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(cases, 1):
            r = run_one(gguf, SYSTEM, c["prompt"], args.max_tokens, args.threads)
            f.write(json.dumps({"case_id": c["case_id"], "output": r["reply"]},
                               ensure_ascii=False) + "\n")
            if r["prompt_tps"] is not None: tps_prompt.append(r["prompt_tps"])
            if r["gen_tps"] is not None: tps_gen.append(r["gen_tps"])
            wall_total += r["wall_s"]
            print(f"  {i:>2}/{len(cases)}  {c['case_id']:<10}  "
                  f"prompt={r['prompt_tps']!s:>6}t/s  gen={r['gen_tps']!s:>6}t/s  "
                  f"wall={r['wall_s']:>4.1f}s  reply={r['reply'][:80]}")
    print(f"\n-> wrote {resp_path.name}")
    if tps_prompt:
        print(f"avg prompt tok/s: {sum(tps_prompt)/len(tps_prompt):.1f}")
    if tps_gen:
        print(f"avg gen    tok/s: {sum(tps_gen)/len(tps_gen):.1f}")
    print(f"total wall:       {wall_total:.1f}s ({wall_total/len(cases):.1f}s per case)")

    # Score via coglang generation-eval
    print("\n===== SCORING =====")
    proc = subprocess.run(
        [COGLANG_PY, "-m", "coglang", "generation-eval",
         "--responses-file", str(resp_path), "--fixture", str(fixture),
         "--format", "json"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    try:
        scored = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit("generation-eval produced no JSON:\n" + proc.stdout + proc.stderr)
    answered = {s["case_id"] for s in [json.loads(l) for l in resp_path.read_text(encoding="utf-8").splitlines()]}
    sc = [c for c in scored["cases"] if c["case_id"] in answered]
    n = len(sc) or 1
    rate = lambda k: sum(1 for c in sc if c.get(k)) / n
    fails = sum(1 for c in sc if c["failure_categories"])
    hall = sum(len(c["hallucinated_heads"]) for c in sc)
    print(f"[{args.tag}] n={len(sc)}  pass={n-fails}/{n} ({100*(n-fails)//n}%)  "
          f"parse_ok={rate('parse_ok'):.0%}  validate_ok={rate('validate_ok'):.0%}  "
          f"top_head_ok={rate('expected_top_level_head_ok'):.0%}  hallucinated={hall}")


if __name__ == "__main__":
    main()

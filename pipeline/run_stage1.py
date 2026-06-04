"""
CogLang distillation — Stage 1 data engine.

Pipeline:  requests.jsonl  ->  teacher (OpenAI)  ->  responses.jsonl
           ->  coglang generation-eval  ->  split into SFT positives + error pool

Env vars:
  OPENAI_API_KEY   (required)
  OPENAI_BASE_URL  (optional, for OpenAI-compatible endpoints)
  TEACHER_MODEL    (optional, default "gpt-4o-mini")

Usage:
  python run_stage1.py                  # full run on requests.jsonl
  python run_stage1.py --limit 8        # only first N cases (cheap smoke test)
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "coglang"          # the cloned coglang checkout
REQUESTS = HERE / "requests.jsonl"
RESPONSES = HERE / "responses.jsonl"
SCORED = HERE / "scored.json"
SFT_POS = HERE / "sft_positive.jsonl"
ERR_POOL = HERE / "error_pool.jsonl"

# Compact teaching prompt — this is the "skill 教材".
# The DOMAIN section is calibrated against the smoke-test error pool: every
# bullet there fixes a systematic mistake DeepSeek made cold on drug prompts
# (see error_pool.jsonl). Keep this block tight — every token here is paid
# again at on-device inference.
SYSTEM = """You write CogLang, a graph-first M-expression language for a medication-safety knowledge graph.
Rules:
- Output EXACTLY one CogLang M-expression. No markdown fences, no prose, no alternatives.
- Syntax is Head[arg, arg, ...]. Strings use double quotes. Variables end with underscore, e.g. n_.
- Dicts use {"key": value}. Lists use List[a, b].

DOMAIN schema:
- Node `category` is one of: Drug, Ingredient, DrugClass, Condition, AdverseEffect, Population.
- Edge `relation_type` is one of: contains, belongs_to, interacts_with, contraindicated_for, causes, treats.

Operators (pick the one the task asks for as top-level head):
- One-hop walk along a relation: Traverse[node_id, relation_type]. e.g. Traverse["warfarin", "interacts_with"].
- Read an attribute: Get[node_id, attr]. e.g. Get["fluoxetine", "confidence"].
- Filter nodes: Query[n_, Equal[Get[n_, attr], value]]. Optional limit: append {"k": 3, "mode": "default"} as 3rd arg.
- Iterate: ForEach[List[a, b], var_, expr_using_var_]. e.g. ForEach[List["warfarin", "aspirin"], d_, Get[d_, "name"]].
- Create node: Create["Entity", {"id": ..., "category": "Drug", "name": ...}] — first arg is always Entity|Concept|Rule|Meta|Edge, NEVER the category name.
- Create edge: Create["Edge", {"from": <id>, "to": <id>, "relation_type": <rel>}]. Keys exactly "from"/"to"/"relation_type" (NOT "source"/"target"/"type").
- Update attribute: Update[node_id, {"attr": value}]. e.g. Update["aspirin", {"name": "Aspirin"}].
- Soft-delete: Delete[node_id]. Do NOT use Update to set a deleted/status flag.
- Audit one write: Trace[Create[...]] / Trace[Update[...]] / Trace[Delete[...]].
- Two ordered writes: Do[Update[...], Create[...]].
- Conditional: If[cond, then, else] (3 args); IfFound[expr, var_, then, else] (4 args).
- For "list X if NON-EMPTY else fallback" use If[Equal[Traverse[...], List[]], fallback, Traverse[...]]. IfFound's else fires ONLY on error heads; an empty List[] enters the THEN branch.
- CogLang has no And/Or/Not, no Filter, no Contains — combine predicates by nesting If[Equal[...], ...].

VERB → OPERATOR RULES (apply when the trigger phrase matches):
1. TRAVERSE: 'list/show/find/return what node_id points to via relation_type' → Traverse["node_id", "relation_type"]. Do NOT use Query or Get.
   - 'list the adverse effects caused by ibuprofen' → Traverse["ibuprofen", "causes"]
   - 'list the populations for which naproxen is contraindicated' → Traverse["naproxen", "contraindicated_for"]
2. FOREACH: 'for each X in the list of A, B, C' → ForEach[List["A", "B", "C"], var_, <expr using var_>]. Do NOT wrap with Do — ForEach is the top-level.
   - 'for each drug in warfarin, aspirin, return its name' → ForEach[List["warfarin", "aspirin"], d_, Get[d_, "name"]]
3. TRACE: prompt verb is audit/log/trace → wrap with Trace[...].
   - 'audit an update that sets X name' → Trace[Update[X, {"name": ...}]]
4. DO: prompt contains 'and then' linking two actions → wrap with Do[<first>, <second>]. Do NOT trigger Do for 'and' inside a list (e.g. 'A, B and C' is a list).
   - 'update X and then create edge Y' → Do[Update[X,...], Create["Edge",{...Y...}]]"""

FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def clean(text: str) -> str:
    """Strip markdown fences from a model answer, preserving multi-line
    expressions (CogLang may span lines; taking only line 1 truncates them)."""
    t = FENCE.sub("", text).strip()
    # collapse internal newlines to spaces so a multi-line M-expression parses
    return " ".join(line.strip() for line in t.splitlines() if line.strip())


def call_teacher(client, model, req) -> str:
    msg = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": req["prompt"] + "\n\n" + req["instructions"]},
    ]
    resp = client.chat.completions.create(model=model, messages=msg, temperature=0)
    return clean(resp.choices[0].message.content or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N cases")
    ap.add_argument("--fixture", default="", help="question-bank JSON to use "
                    "(default: packaged fixture via the static requests.jsonl)")
    args = ap.parse_args()

    if args.fixture:
        # resolve to absolute: subprocesses run with cwd=REPO, not here
        args.fixture = str(Path(args.fixture).resolve())

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set. Set it first:\n"
                 '  $env:OPENAI_API_KEY = "sk-..."   (PowerShell)')

    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL from env
    model = os.environ.get("TEACHER_MODEL", "gpt-4o-mini")

    # When a custom fixture is given, export its requests fresh; else use the
    # static requests.jsonl exported from the packaged fixture.
    if args.fixture:
        exp = subprocess.run(
            [sys.executable, "-m", "coglang", "generation-eval", "--fixture", args.fixture,
             "--export-requests", "--request-format", "jsonl"],
            cwd=str(REPO), capture_output=True, text=True,
        )
        if not exp.stdout.strip():
            sys.exit("could not export requests from fixture:\n" + exp.stderr)
        REQUESTS.write_text(exp.stdout, encoding="utf-8")

    reqs = [json.loads(l) for l in REQUESTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        reqs = reqs[: args.limit]
    print(f"[1/3] Teacher = {model} | answering {len(reqs)} cases ...")

    with RESPONSES.open("w", encoding="utf-8") as f:
        for i, r in enumerate(reqs, 1):
            out = call_teacher(client, model, r)
            f.write(json.dumps({"case_id": r["case_id"], "output": out}, ensure_ascii=False) + "\n")
            print(f"   {i:>3}/{len(reqs)}  {r['case_id']:<7} {out}")

    print("[2/3] Scoring with coglang generation-eval ...")
    score_cmd = [sys.executable, "-m", "coglang", "generation-eval",
                 "--responses-file", str(RESPONSES), "--format", "json"]
    if args.fixture:
        score_cmd[4:4] = ["--fixture", args.fixture]
    proc = subprocess.run(score_cmd, cwd=str(REPO), capture_output=True, text=True)
    # NOTE: generation-eval exits 1 whenever any case fails; that is expected
    # here (failures are the whole point). Only treat unparseable stdout as fatal.
    try:
        scored = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit("generation-eval produced no JSON.\nstdout:\n" + proc.stdout
                 + "\nstderr:\n" + proc.stderr)
    SCORED.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[3/3] Splitting into SFT positives + error pool ...")
    answered = {r["case_id"] for r in reqs}
    cases = [c for c in scored["cases"] if c["case_id"] in answered]
    by_id = {r["case_id"]: r for r in reqs}

    n_pos = n_err = 0
    with SFT_POS.open("w", encoding="utf-8") as fp, ERR_POOL.open("w", encoding="utf-8") as fe:
        for c in cases:
            req = by_id[c["case_id"]]
            if not c["failure_categories"]:
                # chat-format SFT sample, target = canonicalized correct answer
                fp.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": req["prompt"]},
                    {"role": "assistant", "content": c.get("canonical") or c["output"]},
                ], "case_id": c["case_id"], "level": c["level"]}, ensure_ascii=False) + "\n")
                n_pos += 1
            else:
                fe.write(json.dumps({
                    "case_id": c["case_id"], "level": c["level"],
                    "prompt": req["prompt"],
                    "rejected": c["output"],
                    "failure_categories": c["failure_categories"],
                    "hallucinated_heads": c["hallucinated_heads"],
                    "parse_error": c.get("parse_error"),
                }, ensure_ascii=False) + "\n")
                n_err += 1

    print(f"\nDONE.  SFT positives: {n_pos} -> {SFT_POS.name}"
          f"   |   error pool: {n_err} -> {ERR_POOL.name}")
    # rates over ANSWERED cases only (not the full fixture)
    n = len(cases) or 1
    rate = lambda key: sum(1 for c in cases if c.get(key)) / n
    hall = sum(len(c["hallucinated_heads"]) for c in cases)
    print(f"(answered {len(cases)})  parse_ok={rate('parse_ok'):.0%}  "
          f"validate_ok={rate('validate_ok'):.0%}  "
          f"top_head_ok={rate('expected_top_level_head_ok'):.0%}  "
          f"hallucinated={hall}")


if __name__ == "__main__":
    main()

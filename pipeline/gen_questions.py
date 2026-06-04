"""
CogLang distillation — question-bank expansion (self-instruct).

Flow:
  teacher (DeepSeek) invents new {prompt, reference_expr, expected_top_level_heads}
  -> write candidate fixture
  -> score candidates with `coglang generation-eval --fixture` (references ARE the answers)
  -> keep only cases whose reference parses, validates, matches its declared head,
     and uses no hallucinated operator
  -> write a clean expanded fixture (same schema as the packaged one)

The question bank is therefore self-verified by the same tooling that scores teachers.

Env: OPENAI_API_KEY (required), OPENAI_BASE_URL, TEACHER_MODEL (default gpt-4o-mini)
Usage:
  python gen_questions.py --per-level 20            # ~60 candidates -> filtered bank
  python gen_questions.py --per-level 20 --rounds 3 # 3 batches per level (more, dedup'd)
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "coglang"
CAND = HERE / "candidate_fixture.json"
CAND_SCORED = HERE / "candidate_scored.json"
EXPANDED = HERE / "expanded_fixture.json"

OPERATORS = {
    "L1": ["Equal", "Compare", "Get", "If", "IfFound", "Do", "Trace", "Assert",
           "Unify", "Match", "List", "Explain", "Defer", "Estimate", "Inspect"],
    "L2": ["Query", "Traverse", "AllNodes", "ForEach", "IfFound", "Do", "Compare",
           "Trace", "Assert", "Abstract", "Probe", "Explore", "Resume", "Merge", "Instantiate"],
    "L3": ["Create", "Update", "Delete", "Do", "If", "IfFound", "ForEach", "Send", "Trace", "Assert"],
}
LEVEL_HINT = {
    "L1": "pure / local operations on literals, dicts, lists, simple control flow",
    "L2": "graph reads: queries, traversals, iteration, exploration over a knowledge graph",
    "L3": "graph writes & sequences: create/update/delete entities and edges, write intents",
}
FEWSHOT = {
    "L1": 'Equal[1, 1]  |  Get[{"name": "Ada"}, "name"]  |  Do[Equal[1, 1], Equal[2, 2]]',
    "L2": 'Query[n_, Equal[Get[n_, "category"], "Person"]]  |  Traverse["einstein", "born_in"]  |  ForEach[List["ada", "grace"], x_, Get[x_, "label"]]',
    "L3": 'Create["Entity", {"id": "ada", "category": "Person"}]  |  Update["ada", {"label": "Ada"}]  |  Do[Delete["old"], Create["Entity", {"id": "new"}]]',
}
FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def gen_prompt(level: str, n: int) -> str:
    return f"""Generate {n} NEW, diverse CogLang training cases for difficulty {level} ({LEVEL_HINT[level]}).

Use ONLY these top-level operators for {level}: {", ".join(OPERATORS[level])}.
Reference syntax examples: {FEWSHOT[level]}

Each case is an object with:
- "prompt": a natural-language instruction (vary entities, domains, fields; do NOT copy the examples)
- "reference_expr": one correct CogLang M-expression answering the prompt
- "expected_top_level_heads": list with the single top-level operator of reference_expr

Rules: strings use double quotes, variables end with underscore (n_), dicts {{"k": v}}.
Return ONLY a JSON array of {n} objects. No markdown fences, no prose."""


def call_json_array(client, model, prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,  # diversity for question generation
    )
    raw = FENCE.sub("", resp.choices[0].message.content or "").strip()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("cases") or data.get("items") or []
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit('ERROR: OPENAI_API_KEY not set.')
    from openai import OpenAI
    client = OpenAI()
    model = os.environ.get("TEACHER_MODEL", "gpt-4o-mini")

    # 1) generate candidates
    cands, seen = [], set()
    for level in ("L1", "L2", "L3"):
        for r in range(args.rounds):
            try:
                items = call_json_array(client, model, gen_prompt(level, args.per_level))
            except Exception as e:
                print(f"  [{level} round {r+1}] generation/parse failed: {e}")
                continue
            kept = 0
            for it in items:
                p = (it.get("prompt") or "").strip()
                expr = (it.get("reference_expr") or "").strip()
                heads = it.get("expected_top_level_heads") or []
                if not p or not expr or not heads:
                    continue
                key = (level, expr)
                if key in seen:           # dedup identical references
                    continue
                seen.add(key)
                cands.append({"level": level, "prompt": p,
                              "reference_expr": expr, "expected_top_level_heads": heads})
                kept += 1
            print(f"  [{level} round {r+1}] got {kept} candidates")

    # assign provisional ids and write candidate fixture
    by_lvl = {}
    for c in cands:
        by_lvl.setdefault(c["level"], 0)
        by_lvl[c["level"]] += 1
        c["case_id"] = f'{c["level"]}-G{by_lvl[c["level"]]:03d}'
    CAND.write_text(json.dumps({
        "schema_version": "coglang-generation-eval-fixture/v0.1",
        "name": "expanded-candidates", "description": "self-instruct candidates (unverified)",
        "defined_levels": ["L1", "L2", "L3", "L4", "L5", "L6"], "cases": cands,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal candidates: {len(cands)} -> {CAND.name}")

    # 2) verify candidates by scoring their OWN references
    proc = subprocess.run(
        [sys.executable, "-m", "coglang", "generation-eval",
         "--fixture", str(CAND), "--format", "json"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    try:
        scored = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit("verify step produced no JSON:\n" + proc.stdout + proc.stderr)
    CAND_SCORED.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) keep only fully-passing cases
    status = {c["case_id"]: c for c in scored["cases"]}
    good = [c for c in cands if not status[c["case_id"]]["failure_categories"]]

    # 4) renumber cleanly and write the verified expanded bank
    cnt = {}
    for c in good:
        cnt[c["level"]] = cnt.get(c["level"], 0) + 1
        c["case_id"] = f'{c["level"]}-E{cnt[c["level"]]:03d}'
        c.pop("level_tmp", None)
    EXPANDED.write_text(json.dumps({
        "schema_version": "coglang-generation-eval-fixture/v0.1",
        "name": "coglang-expanded-question-bank",
        "description": "Self-instruct generated, self-verified CogLang question bank.",
        "defined_levels": ["L1", "L2", "L3", "L4", "L5", "L6"],
        "cases": [{"case_id": c["case_id"], "level": c["level"], "prompt": c["prompt"],
                   "reference_expr": c["reference_expr"],
                   "expected_top_level_heads": c["expected_top_level_heads"]} for c in good],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    kept_by = {}
    for c in good:
        kept_by[c["level"]] = kept_by.get(c["level"], 0) + 1
    print(f"\nVERIFIED bank: kept {len(good)}/{len(cands)} "
          f"({100*len(good)//max(len(cands),1)}%) -> {EXPANDED.name}")
    print("by level:", dict(sorted(kept_by.items())))


if __name__ == "__main__":
    main()

"""
CogLang distillation — drug-safety question-bank expansion (self-instruct).

Domain-specific fork of gen_questions.py. Schema and few-shot anchors come from
drug_seed_fixture.json (see step1_survey.md §5):

  category     : Drug | DrugClass | Ingredient | Condition | AdverseEffect | Population
  relation_type: contains | belongs_to | interacts_with | contraindicated_for | causes | treats

Output: drug_expanded_fixture.json (only cases whose reference passes
`coglang generation-eval`).

Env: OPENAI_API_KEY (required), OPENAI_BASE_URL, TEACHER_MODEL.
Usage:
  python gen_questions_drug.py --per-level 30            # ~60 candidates -> filtered bank
  python gen_questions_drug.py --per-level 30 --rounds 3 # 3 batches per level
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "coglang"
SEED = HERE / "drug_seed_fixture_v4.json"
CAND = HERE / "drug_candidate_fixture.json"
CAND_SCORED = HERE / "drug_candidate_scored.json"
EXPANDED = HERE / "drug_expanded_fixture.json"

# Only Core / non-reserved operators. Reserved heads (Defer/Resume/Merge/Probe/
# Explore/Estimate/Decompose/Inspect/Instantiate/Send/Explain) pass static
# generation-eval but return StubError at runtime — exclude from training data.
# L1 is omitted: pure-literal control flow is non-domain. The general bank
# already covers it; the drug specialist learns L1 syntax inside L2 predicates.
OPERATORS = {
    "L2": ["Query", "Traverse", "AllNodes", "ForEach", "IfFound", "If", "Do",
           "Get", "Equal", "Compare", "Trace", "Assert", "List"],
    "L3": ["Create", "Update", "Delete", "Do", "If", "IfFound", "ForEach",
           "Trace", "Assert", "Get", "Equal", "List"],
}

LEVEL_HINT = {
    "L2": "graph reads on the medication-safety graph: queries by category, "
          "one-hop traversals over relation_type edges, iteration, null-guarded lookups",
    "L3": "graph writes on the medication-safety graph: create entities/edges, "
          "update node attributes, soft-delete, optionally wrapped in Trace/Assert",
}

# Per-operator quotas calibrated to the v2 seed distribution. These are what
# drug-v2/v3 failed to learn: too many Query-by-category samples crowded out
# Traverse/Get/Update/Trace/Do. The quota is enforced in the gen prompt; if
# DeepSeek under-delivers on one op, we still get coverage from the seeds.
OP_QUOTAS = {
    "L2": {
        # op_name : (target fraction of cases, brief hint shown to teacher)
        "Traverse": (0.32, 'Traverse[node_id, relation_type]  — one-hop walk'),
        "Query":    (0.24, 'Query[n_, Equal[Get[n_, attr], value]]  — filter; VARY attr (category, name, confidence)'),
        "Get":      (0.20, 'Get[node_id, attr]  — read a node attribute'),
        "ForEach":  (0.12, 'ForEach[List[a, b], var_, expr_using_var_]'),
        "If":       (0.08, 'If[Equal[Traverse[...], List[]], fallback, Traverse[...]]  — for "non-empty else fallback"'),
        "IfFound":  (0.04, 'IfFound[expr, var_, var_, fallback]  — fallback ONLY on error head (rare)'),
    },
    "L3": {
        "Create-Entity": (0.19, 'Create["Entity", {"id": id, "category": cat, "name": Name}]'),
        "Create-Edge":   (0.19, 'Create["Edge", {"from": src, "to": dst, "relation_type": rel}]'),
        "Update":        (0.19, 'Update[node_id, {"attr": value}]'),
        "Trace":         (0.19, 'Trace[Create[...]] / Trace[Update[...]] / Trace[Delete[...]]  — audit ONE write'),
        "Do":            (0.14, 'Do[<write1>, <write2>]  — two ordered writes'),
        "Delete":        (0.10, 'Delete[node_id]'),
    },
}


def quota_block(level: str, n: int) -> str:
    """Render the per-operator quota lines for the gen prompt."""
    targets = {op: max(1, round(frac * n)) for op, (frac, _) in OP_QUOTAS[level].items()}
    # If rounding sums to != n, leave it — DeepSeek doesn't need to hit it exactly.
    lines = []
    for op, (_, hint) in OP_QUOTAS[level].items():
        lines.append(f'  {targets[op]:>3} cases  with top-level head `{op}`: {hint}')
    return "\n".join(lines)

# Known IDs in the seed graph (see drug_graph.py). Encourages the model to
# reference real nodes instead of inventing IDs that will mismatch at exec time.
KNOWN_IDS = {
    "Drug": ["warfarin", "aspirin", "ibuprofen", "naproxen", "fluoxetine",
             "sertraline", "tramadol", "acetaminophen"],
    "Ingredient": ["acetylsalicylic_acid"],
    "DrugClass": ["nsaid", "ssri"],
    "Condition": ["hypertension", "depression", "headache"],
    "AdverseEffect": ["gi_bleed", "serotonin_syndrome"],
    "Population": ["pregnancy", "children"],
}

CATEGORY_ENUM = list(KNOWN_IDS.keys())
RELATION_ENUM = ["contains", "belongs_to", "interacts_with",
                 "contraindicated_for", "causes", "treats"]
CREATE_TYPES = {"Entity", "Concept", "Rule", "Meta", "Edge"}
RESERVED_OPS = {"Defer", "Resume", "Merge", "Probe", "Explore", "Estimate",
                "Decompose", "Inspect", "Instantiate", "Send", "Explain"}

FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)

# Schema post-filters — generation-eval parses the reference but does NOT
# execute it against the drug graph, so Create["Drug",...] / bogus
# relation_type / nested reserved ops slip through. These regex checks catch
# the obvious schema violations before the case enters training data.
_CREATE_TYPE_RE = re.compile(r'Create\[\s*"([^"]+)"')
_RELATION_RE = re.compile(r'"relation_type"\s*:\s*"([^"]+)"')
_CATEGORY_RE = re.compile(r'"category"\s*:\s*"([^"]+)"')
_OP_HEAD_RE = re.compile(r'(?<![A-Za-z0-9_])([A-Z][A-Za-z0-9]*)\[')


def schema_violations(expr: str) -> list[str]:
    """Return human-readable schema violations; empty list = clean."""
    issues = []
    for t in _CREATE_TYPE_RE.findall(expr):
        if t not in CREATE_TYPES:
            issues.append(f'Create["{t}",...] not in {sorted(CREATE_TYPES)}')
    for r in _RELATION_RE.findall(expr):
        if r not in RELATION_ENUM:
            issues.append(f'relation_type "{r}" not in {RELATION_ENUM}')
    for c in _CATEGORY_RE.findall(expr):
        if c not in CATEGORY_ENUM:
            issues.append(f'category "{c}" not in {CATEGORY_ENUM}')
    for op in _OP_HEAD_RE.findall(expr):
        if op in RESERVED_OPS:
            issues.append(f'reserved/stub op `{op}`')
    return issues


def _load_seed_fewshot() -> dict[str, str]:
    """Build per-level few-shot strings from the verified seed fixture."""
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    by_level: dict[str, list[str]] = {}
    for c in seed["cases"]:
        line = f'{c["prompt"]}\n  -> {c["reference_expr"]}'
        by_level.setdefault(c["level"], []).append(line)
    return {lvl: "\n".join(lines) for lvl, lines in by_level.items()}


def gen_prompt(level: str, n: int, fewshot: str) -> str:
    ids_block = "\n".join(f'  {cat}: {", ".join(ids)}'
                          for cat, ids in KNOWN_IDS.items())
    return f"""Generate {n} NEW, diverse CogLang training cases for the medication-safety \
knowledge graph at difficulty {level} ({LEVEL_HINT[level]}).

GRAPH SCHEMA (use ONLY these values):
- node `category` ∈ {{{", ".join(CATEGORY_ENUM)}}}
- edge `relation_type` ∈ {{{", ".join(RELATION_ENUM)}}}
- known node IDs (prefer these so questions ground in the seed graph):
{ids_block}

OPERATOR QUOTA — distribute the {n} cases roughly as follows. Do NOT cluster \
everything on one operator. Each line = how many cases should have that top-level head:
{quota_block(level, n)}

DO NOT use: Defer, Resume, Merge, Probe, Explore, Estimate, Decompose, Inspect, \
Instantiate, Send, Explain (these return StubError at runtime).
DO NOT use: And, Or, Not, Filter, Contains — CogLang Core has no boolean \
combinators; inside Query predicates use a single Equal[...] or nest If[Equal[...], ...].

Reference cases (DO NOT copy — invent new ones, vary the entities and verbs):
{fewshot}

Each case is one JSON object:
- "prompt": natural-language instruction grounded in the drug-safety domain. \
Vary the verb (list/show/find/read/iterate/create/record/update/audit/then-and-then).
- "reference_expr": one correct CogLang M-expression answering the prompt
- "expected_top_level_heads": list with the single top-level operator of reference_expr

Syntax rules: strings use double quotes, variables end with underscore (n_, drug_),
dicts {{"k": v}}, Create node types are exactly Entity|Concept|Rule|Meta|Edge \
(category goes in attrs, NOT in the first arg).

Return ONLY a JSON array of {n} objects. No markdown fences, no prose."""


def call_json_array(client, model, prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = FENCE.sub("", resp.choices[0].message.content or "").strip()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("cases") or data.get("items") or []
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set.")
    from openai import OpenAI
    # Bound per-request latency: a hung connection should fail in 2 min, not 30+.
    # max_retries=1 keeps one cheap retry on transient failures.
    client = OpenAI(timeout=120.0, max_retries=1)
    model = os.environ.get("TEACHER_MODEL", "deepseek-chat")

    fewshot = _load_seed_fewshot()

    # 1) generate candidates
    cands, seen = [], set()
    schema_dropped = 0
    for level in ("L2", "L3"):
        for r in range(args.rounds):
            try:
                items = call_json_array(client, model,
                                        gen_prompt(level, args.per_level, fewshot[level]))
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
                if key in seen:
                    continue
                seen.add(key)
                bad = schema_violations(expr)
                if bad:
                    schema_dropped += 1
                    continue
                cands.append({"level": level, "prompt": p,
                              "reference_expr": expr,
                              "expected_top_level_heads": heads})
                kept += 1
            print(f"  [{level} round {r+1}] got {kept} candidates "
                  f"(after dedup + schema pre-filter)")
    if schema_dropped:
        print(f"  schema pre-filter rejected {schema_dropped} candidate(s)")

    by_lvl = {}
    for c in cands:
        by_lvl.setdefault(c["level"], 0)
        by_lvl[c["level"]] += 1
        c["case_id"] = f'DRUG-{c["level"]}-G{by_lvl[c["level"]]:03d}'
    CAND.write_text(json.dumps({
        "schema_version": "coglang-generation-eval-fixture/v0.1",
        "name": "drug-safety-candidates",
        "description": "self-instruct drug-domain candidates (unverified)",
        "defined_levels": ["L1", "L2", "L3", "L4", "L5", "L6"],
        "cases": cands,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal candidates: {len(cands)} -> {CAND.name}")

    # 2) self-verify by scoring references through generation-eval
    proc = subprocess.run(
        [sys.executable, "-m", "coglang", "generation-eval",
         "--fixture", str(CAND), "--format", "json"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    try:
        scored = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit("verify step produced no JSON:\n" + proc.stdout + proc.stderr)
    CAND_SCORED.write_text(json.dumps(scored, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    status = {c["case_id"]: c for c in scored["cases"]}
    good = [c for c in cands if not status[c["case_id"]]["failure_categories"]]

    # 3) merge seed (already verified) + new verified cases, renumber
    seed_cases = json.loads(SEED.read_text(encoding="utf-8"))["cases"]
    all_good = list(seed_cases) + good

    cnt = {}
    out_cases = []
    for c in all_good:
        cnt[c["level"]] = cnt.get(c["level"], 0) + 1
        out_cases.append({
            "case_id": f'DRUG-{c["level"]}-E{cnt[c["level"]]:03d}',
            "level": c["level"],
            "prompt": c["prompt"],
            "reference_expr": c["reference_expr"],
            "expected_top_level_heads": c["expected_top_level_heads"],
        })

    EXPANDED.write_text(json.dumps({
        "schema_version": "coglang-generation-eval-fixture/v0.1",
        "name": "coglang-drug-safety-question-bank",
        "description": "Drug-safety verified CogLang question bank "
                       "(12 hand-crafted seeds + self-instruct expansion).",
        "defined_levels": ["L1", "L2", "L3", "L4", "L5", "L6"],
        "cases": out_cases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    kept_by = {}
    for c in good:
        kept_by[c["level"]] = kept_by.get(c["level"], 0) + 1
    print(f"\nVERIFIED expansion: kept {len(good)}/{len(cands)} "
          f"({100*len(good)//max(len(cands),1)}%) new cases.")
    print(f"+ {len(seed_cases)} seed cases = {len(out_cases)} total -> {EXPANDED.name}")
    print("new-cases by level:", dict(sorted(kept_by.items())))


if __name__ == "__main__":
    main()

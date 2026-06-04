"""Per-operator + entity-novelty breakdown of an OOD eval run.

Re-scores an eval response file via `coglang generation-eval` and groups the
pass rate by (a) top-level operator and (b) whether the case references any
entity that exists in the seed graph ("in-graph") or only novel entities.

Usage (PYTHONUTF8=1, from repo root):
  python pipeline/eval_breakdown.py --responses pipeline/responses_drug-v6-oodv1_ood.jsonl \
      --fixture pipeline/ood_eval.json
"""
from __future__ import annotations
import argparse, json, math, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "coglang"
COGLANG_PY = r"C:\Python314\python.exe"
QUOTED = re.compile(r'"([^"]+)"')


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-m)/d, (c+m)/d)


def graph_node_ids() -> set:
    sys.path.insert(0, str(HERE))
    from drug_graph import build_seed_drug_graph
    return set(build_seed_drug_graph().nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--fixture", required=True)
    args = ap.parse_args()

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in fixture["cases"]}
    nodes = graph_node_ids()

    proc = subprocess.run(
        [COGLANG_PY, "-m", "coglang", "generation-eval",
         "--responses-file", str(Path(args.responses).resolve()),
         "--fixture", str(Path(args.fixture).resolve()), "--format", "json"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    try:
        scored = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit("generation-eval produced no JSON:\n" + proc.stdout + proc.stderr)

    op_tot, op_pass = defaultdict(int), defaultdict(int)
    nov_tot, nov_pass = defaultdict(int), defaultdict(int)
    overall_pass = overall = 0
    for c in scored["cases"]:
        case = by_id.get(c["case_id"])
        if not case:
            continue
        passed = not c["failure_categories"]
        op = (case.get("expected_top_level_heads") or ["?"])[0]
        toks = set(QUOTED.findall(case["reference_expr"]))
        novelty = "in-graph" if (toks & nodes) else "novel"
        op_tot[op] += 1; op_pass[op] += passed
        nov_tot[novelty] += 1; nov_pass[novelty] += passed
        overall += 1; overall_pass += passed

    lo, hi = wilson(overall_pass, overall)
    print(f"OVERALL: {overall_pass}/{overall} ({100*overall_pass//max(overall,1)}%)  "
          f"95% CI [{100*lo:.0f}%, {100*hi:.0f}%]\n")
    print("by operator:")
    for op in sorted(op_tot, key=lambda o: -op_tot[o]):
        print(f"  {op:<10} {op_pass[op]:>2}/{op_tot[op]:<2}")
    print("\nby entity novelty:")
    for k in ("in-graph", "novel"):
        if nov_tot[k]:
            print(f"  {k:<10} {nov_pass[k]:>2}/{nov_tot[k]:<2}")


if __name__ == "__main__":
    main()

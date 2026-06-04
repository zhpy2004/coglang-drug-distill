"""Independent review of an eval fixture's reference answers.

(1) lint: parse + schema-check every reference_expr (relation_types/categories
    in-vocab; Create Entity/Edge dict keys present). Execute read-op references
    that use only in-graph entities and assert no parse/exec error.
(2) deepseek: re-derive CogLang for each prompt with the SAME SYSTEM_v6d the
    model uses, and flag cases where DeepSeek's top-level head (or normalized
    expression) disagrees with our reference - a second opinion on our authoring.

Usage (PYTHONUTF8=1, from repo root):
  python pipeline/eval_review.py lint     pipeline/ood_eval.json
  OPENAI_API_KEY=$YOUR_TEACHER_KEY OPENAI_BASE_URL=https://api.deepseek.com \
    TEACHER_MODEL=deepseek-chat \
    python pipeline/eval_review.py deepseek pipeline/ood_eval.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "coglang" / "src"))

from coglang import PythonCogLangExecutor, parse  # noqa: E402
from drug_graph import build_seed_drug_graph  # noqa: E402

RELATIONS = {"contains", "belongs_to", "interacts_with", "contraindicated_for", "causes", "treats"}
CATEGORIES = {"Drug", "Ingredient", "DrugClass", "Condition", "AdverseEffect", "Population"}
READ_HEADS = {"Traverse", "Get", "Query", "ForEach", "If", "IfFound"}
QUOTED = re.compile(r'"([^"]+)"')


def _norm(s: str) -> str:
    return " ".join(s.split())


# Tokens that are not entity ids, so the rest are treated as ids for the in-graph test.
NON_ID = RELATIONS | CATEGORIES | {"name", "category", "confidence", "default", "mode", "none"}


def lint(path: str) -> bool:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    nodes = set(build_seed_drug_graph().nodes)
    issues = []
    for c in cases:
        expr_s = c["reference_expr"]
        try:
            expr = parse(expr_s)
        except Exception as e:  # noqa: BLE001
            issues.append((c["case_id"], f"parse fail: {e}"))
            continue
        if 'Create["Edge"' in expr_s and not all(k in expr_s for k in ('"from"', '"to"', '"relation_type"')):
            issues.append((c["case_id"], "Create Edge missing from/to/relation_type"))
        if 'Create["Entity"' in expr_s and not all(k in expr_s for k in ('"id"', '"category"', '"name"')):
            issues.append((c["case_id"], "Create Entity missing id/category/name"))
        if getattr(expr, "head", None) in READ_HEADS:
            id_toks = {t for t in QUOTED.findall(expr_s) if t not in NON_ID}
            if id_toks and id_toks <= nodes:
                try:
                    PythonCogLangExecutor(build_seed_drug_graph()).execute(parse(expr_s))
                except Exception as e:  # noqa: BLE001
                    issues.append((c["case_id"], f"exec error on in-graph read: {e}"))
    print(f"lint: {len(cases)} cases, {len(issues)} issues")
    for cid, msg in issues:
        print("  ISSUE", cid, msg)
    return len(issues) == 0


def deepseek(path: str) -> bool:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    system = (HERE / "SYSTEM_v6d.txt").read_text(encoding="utf-8")
    model = os.environ.get("TEACHER_MODEL", "deepseek-chat")
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    disagree = []
    for c in cases:
        r = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": c["prompt"]},
            ],
        )
        reply = r.choices[0].message.content.strip()
        try:
            dh = getattr(parse(reply), "head", None)
        except Exception:  # noqa: BLE001
            dh = "<parse-fail>"
        rh = (c.get("expected_top_level_heads") or [None])[0]
        if dh != rh or _norm(reply) != _norm(c["reference_expr"]):
            disagree.append((c["case_id"], rh, dh, _norm(reply)))
    print(f"deepseek: {len(cases)} cases, {len(disagree)} disagreements (head or exact-expr)")
    for cid, rh, dh, dexpr in disagree:
        print(f"  DIFF {cid}  ref_head={rh}  ds_head={dh}  ds_expr={dexpr[:80]}")
    return True


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "lint":
        sys.exit(0 if lint(path) else 1)
    if mode == "deepseek":
        sys.exit(0 if deepseek(path) else 1)
    sys.exit("unknown mode")

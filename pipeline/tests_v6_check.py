"""Lightweight v6 data checks (run with PYTHONUTF8=1 from repo root).
Usage: python pipeline/tests_v6_check.py seeds <fixture.json>
       python pipeline/tests_v6_check.py sft   <sft.jsonl> <expected_system_substr>
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coglang" / "src"))
from coglang import parse  # noqa: E402


def check_seeds(path):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    bad = []
    for c in cases:
        try:
            expr = parse(c["reference_expr"])
            head = getattr(expr, "head", None)
            if c.get("expected_top_level_heads") and head not in c["expected_top_level_heads"]:
                bad.append((c["case_id"], f"head {head} not in {c['expected_top_level_heads']}"))
        except Exception as e:  # noqa: BLE001
            bad.append((c["case_id"], f"parse fail: {e}"))
    print(f"{len(cases)} cases, {len(bad)} bad")
    for cid, msg in bad:
        print("  BAD", cid, msg)
    return len(bad) == 0


def check_sft(path, expected_system_substr):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    sys_msgs = {r["messages"][0]["content"] for r in rows}
    ok = all(expected_system_substr in s for s in sys_msgs)
    print(f"{len(rows)} SFT rows, {len(sys_msgs)} distinct system msgs, "
          f"expected-substr-present={ok}")
    return ok


def _norm(s):
    return " ".join(s.split())


def check_dedup(fixture_path, sft_path):
    """Hard fail on PROMPT collisions (the real contamination signal — an exact
    prompt seen in training). reference_expr matches are reported as info only:
    common queries legitimately share an expression with training (e.g.
    Traverse["warfarin","interacts_with"]); that is not contamination."""
    cases = json.loads(Path(fixture_path).read_text(encoding="utf-8"))["cases"]
    rows = [json.loads(l) for l in Path(sft_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    train_prompts = {_norm(r["messages"][1]["content"].lower()) for r in rows}
    train_exprs = {_norm(r["messages"][2]["content"]) for r in rows}
    prompt_hits, expr_hits = [], []
    for c in cases:
        if _norm(c["prompt"].lower()) in train_prompts:
            prompt_hits.append(c["case_id"])
        if _norm(c["reference_expr"]) in train_exprs:
            expr_hits.append(c["case_id"])
    print(f"{len(cases)} eval cases vs {len(rows)} training rows: "
          f"{len(prompt_hits)} PROMPT collisions (hard), {len(expr_hits)} expr matches (info)")
    for cid in prompt_hits:
        print("  PROMPT-COLLISION", cid)
    if expr_hits:
        print(f"  (info) expr-shared cases: {', '.join(expr_hits)}")
    return len(prompt_hits) == 0


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "seeds":
        sys.exit(0 if check_seeds(sys.argv[2]) else 1)
    if mode == "sft":
        sys.exit(0 if check_sft(sys.argv[2], sys.argv[3]) else 1)
    if mode == "dedup":
        sys.exit(0 if check_dedup(sys.argv[2], sys.argv[3]) else 1)
    sys.exit("unknown mode")

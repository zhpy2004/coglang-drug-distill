"""
Few-shot ICL A/B test on the user's actual wild prompts.

For each prompt, runs drug-v6 Q4+GBNF twice:
  (1) without few-shot examples (baseline = current production behavior)
  (2) with fewshot_v0.txt prepended as ICL

Prints side-by-side and computes simple stats:
  - top-level head distribution
  - parse success rate
  - "junk write op" rate (Create/Update/Delete that mutate graph for a read prompt)

Each prompt = 2 llama-cli subprocesses (~8s each) = ~16s per prompt.
30 prompts = ~8 minutes.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drug_chat import llama_generate, collect_heads, WRITE_HEADS, SYSTEM_PATH, FEWSHOT_PATH
from coglang import parse as cog_parse

PROMPTS = [
    # 1. Telegraphic
    ("telegraphic", "warfarin interactions",                                 "Traverse"),
    ("telegraphic", "ibuprofen side effects",                                "Traverse"),
    ("telegraphic", "aspirin ingredients",                                   "Traverse"),
    ("telegraphic", "fluoxetine treats?",                                    "Traverse"),
    ("telegraphic", "tramadol classes",                                      "Traverse"),
    # 2. Natural / casual
    ("casual",      "what's bad about ibuprofen",                            "Traverse"),
    ("casual",      "what should I avoid mixing with warfarin",              "Traverse"),
    ("casual",      "show me everything aspirin connects to",                "Traverse"),
    ("casual",      "tell me about tramadol",                                "Get"),
    ("casual",      "who treats depression",                                 "Query"),
    # 3. Concept / safety
    ("concept",     "Can I give ibuprofen to a pregnant patient?",           "Traverse"),
    ("concept",     "Is aspirin safe for children?",                         "Traverse"),
    ("concept",     "Does fluoxetine cause serotonin syndrome?",             "Traverse"),
    ("concept",     "What drugs are unsafe in pregnancy?",                   "Query"),
    ("concept",     "Is tramadol an opioid?",                                "Get"),
    # 4. Case/format noise
    ("noise",       "WHAT INTERACTS WITH WARFARIN",                          "Traverse"),
    ("noise",       "warfarin   interact   with    what",                    "Traverse"),
    ("noise",       "list adverse effects of:ibuprofen",                     "Traverse"),
    ("noise",       "warfarin->interacts_with->?",                           "Traverse"),
    # 5. Write ops (allowed when prompt requests them)
    ("write",       "delete tramadol",                                       "Delete"),
    ("write",       "remove all NSAIDs from the graph",                      "Delete"),
    ("write",       "create a new drug called clopidogrel",                  "Create"),
    ("write",       "update aspirin to be safe in children",                 "Update"),
    # 6. Reverse queries — hard
    ("reverse",     "which drugs treat depression",                          "Query"),
    ("reverse",     "which drugs cause gi_bleed",                            "Query"),
    ("reverse",     "who's contraindicated for pregnancy",                   "Query"),
    ("reverse",     "list everything that belongs to nsaid",                 "Query"),
    # 7. Multi-step reasoning — known impossible
    ("reason",      "What's safer between ibuprofen and naproxen for headache?", "?"),
    ("reason",      "Find a drug that treats depression but doesn't cause serotonin syndrome.", "?"),
    ("reason",      "Which NSAID is OK during pregnancy?",                   "?"),
]


def head_of(reply: str) -> str:
    try:
        expr = cog_parse(reply)
        return getattr(expr, "head", "?")
    except Exception:
        return "<parse-fail>"


def main():
    system = SYSTEM_PATH.read_text(encoding="utf-8")
    fewshot = FEWSHOT_PATH.read_text(encoding="utf-8")
    print(f"baseline SYSTEM: {len(system)} chars")
    print(f"few-shot block:  {len(fewshot)} chars")
    print(f"running {len(PROMPTS)} prompts × 2 = {2*len(PROMPTS)} subprocess calls...")
    print("=" * 80)

    rows = []
    for category, prompt, expected_head in PROMPTS:
        reply_b, dt_b = llama_generate(system, prompt, fewshot="")
        head_b = head_of(reply_b)
        write_b = head_b in WRITE_HEADS and expected_head not in WRITE_HEADS
        ok_b = head_b == expected_head

        reply_f, dt_f = llama_generate(system, prompt, fewshot=fewshot)
        head_f = head_of(reply_f)
        write_f = head_f in WRITE_HEADS and expected_head not in WRITE_HEADS
        ok_f = head_f == expected_head

        rows.append({
            "category": category, "prompt": prompt, "expected": expected_head,
            "head_b": head_b, "write_b": write_b, "ok_b": ok_b, "reply_b": reply_b,
            "head_f": head_f, "write_f": write_f, "ok_f": ok_f, "reply_f": reply_f,
            "dt_b": dt_b, "dt_f": dt_f,
        })

        flag = ""
        if ok_f and not ok_b:    flag = " 🟢 FEW-SHOT WIN"
        elif ok_b and not ok_f:  flag = " 🔴 FEW-SHOT REGRESS"
        elif write_b and not write_f: flag = " 🟡 fewer junk writes"
        print(f"\n[{category}] {prompt!r}{flag}")
        print(f"   exp:{expected_head:<10}  base→{head_b:<14}  +few→{head_f}")
        print(f"   base: {reply_b[:90]}")
        print(f"   +few: {reply_f[:90]}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    print(f"\n{'category':<14}  {'total':>6}  {'base ✓':>8}  {'+few ✓':>8}  {'base junk writes':>18}  {'+few junk writes':>18}")
    print("-" * 84)
    for cat, items in by_cat.items():
        n = len(items)
        ok_b = sum(1 for r in items if r["ok_b"])
        ok_f = sum(1 for r in items if r["ok_f"])
        jw_b = sum(1 for r in items if r["write_b"])
        jw_f = sum(1 for r in items if r["write_f"])
        print(f"{cat:<14}  {n:>6}  {ok_b:>8}  {ok_f:>8}  {jw_b:>18}  {jw_f:>18}")
    total = len(rows)
    print("-" * 84)
    print(f"{'TOTAL':<14}  {total:>6}  "
          f"{sum(1 for r in rows if r['ok_b']):>8}  "
          f"{sum(1 for r in rows if r['ok_f']):>8}  "
          f"{sum(1 for r in rows if r['write_b']):>18}  "
          f"{sum(1 for r in rows if r['write_f']):>18}")
    print(f"\navg wall: base={sum(r['dt_b'] for r in rows)/total:.1f}s  +few={sum(r['dt_f'] for r in rows)/total:.1f}s")


if __name__ == "__main__":
    main()

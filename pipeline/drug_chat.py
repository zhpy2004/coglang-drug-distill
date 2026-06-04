"""
Interactive demo of drug-v6 + SYSv6d + GBNF.

Loop: user types a medication-safety question in NL → llama-cli generates a
CogLang expression under GBNF grammar → PythonCogLangExecutor runs it against
the seed drug graph → result is formatted back to the user.

This wraps the 80%-OOD-accuracy model in something tangible: you can actually
ask "what does warfarin interact with?" and get a concrete answer from the
graph (not from model memory — the model only translates to CogLang).

Usage:
  PYTHONUTF8=1 C:/Python314/python.exe E:/coglang/pipeline/drug_chat.py
  (Python 3.14 because coglang package is installed there)

Type a question, or one of:
  /help        show this help
  /show <id>   print a node's attributes from the graph
  /graph       summarize the graph (counts by category)
  /reset       rebuild the graph from drug_graph.py (drops any mutations)
  /yolo        toggle write-op confirmation (default: ON — asks before any Create/Update/Delete)
  /quit        exit
"""
from __future__ import annotations
import os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "coglang"
LLAMA_CLI = Path("E:/coglang/llama.cpp/bin/llama-cli.exe")
GRAMMAR = REPO / "examples" / "grammar" / "coglang.gbnf"
GGUF = Path(os.environ.get("COGLANG_MODEL", "E:/coglang/gguf/drug-v6-q4_k_m.gguf"))
SYSTEM_PATH = HERE / "SYSTEM_v6d.txt"
FEWSHOT_PATH = HERE / "fewshot_v1.txt"  # optional ICL examples; loaded if exists

sys.path.insert(0, str(HERE))
from drug_graph import build_seed_drug_graph
from coglang import PythonCogLangExecutor, parse as cog_parse, canonicalize

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TIMING_RE = re.compile(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]")
WRITE_HEADS = {"Create", "Update", "Delete"}
HEAD_RE = re.compile(r"[A-Z]\w*\[")


def extract_last_mexpr(text: str) -> str:
    """Find the LAST TOP-LEVEL balanced Head[...] M-expression in text.

    Robust to llama-cli prompt echo (which may contain other Head[] examples
    from a few-shot block). Only top-level matches are collected — nested heads
    inside an outer expression are skipped via end-of-expression jump.
    """
    last = ""
    i = 0
    while True:
        m = HEAD_RE.search(text, i)
        if not m:
            break
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        end = -1
        for j in range(start, len(text)):
            c = text[j]
            if esc:
                esc = False
                continue
            if in_str:
                if c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            break  # unbalanced; bail
        last = text[start:end+1]
        i = end + 1  # JUMP past this top-level expr — don't recurse inside
    return last.strip() if last else text.strip()


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def llama_generate(system: str, user: str, max_tokens: int = 256, threads: int = 8,
                   fewshot: str = "") -> tuple[str, float]:
    """Run llama-cli single-shot with GBNF; return (reply_text, wall_seconds).

    If fewshot is non-empty, prepend it to the user message as in-context examples.
    """
    user_msg = f"{fewshot}\n---\nUser question: {user}" if fewshot else user
    cmd = [str(LLAMA_CLI), "-m", str(GGUF),
           "--grammar-file", str(GRAMMAR), "-st",
           "-sys", system, "-p", user_msg,
           "-n", str(max_tokens), "--temp", "0", "-t", str(threads)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    dt = time.time() - t0
    out = strip_ansi(proc.stdout)
    m = TIMING_RE.search(out)
    head = out[:m.start() if m else len(out)]
    # llama-cli echoes the prompt (incl. few-shot block); the model's actual
    # answer is the LAST balanced M-expression in the buffer.
    reply = extract_last_mexpr(head)
    return reply, dt


def collect_heads(expr, acc=None) -> set:
    """Recursively collect every operator head in an expression."""
    acc = acc if acc is not None else set()
    if hasattr(expr, "head"):
        acc.add(expr.head)
    if hasattr(expr, "args"):
        for a in expr.args:
            collect_heads(a, acc)
    return acc


def format_result(result) -> str:
    """Pretty-print a CogLang execution result for the terminal."""
    if hasattr(result, "head") and result.head == "List":
        items = list(result.args)
        if not items:
            return "  (empty list — no matches in the graph)"
        return "\n".join(f"  - {item!r}" for item in items)
    if isinstance(result, list):
        if not result:
            return "  (empty list)"
        return "\n".join(f"  - {item!r}" for item in result)
    if hasattr(result, "head"):
        return f"  {canonicalize(result)}"
    return f"  {result!r}"


def show_node(graph, node_id: str) -> None:
    if node_id not in graph.nodes:
        print(f"  node '{node_id}' not in graph.")
        return
    attrs = dict(graph.nodes[node_id])
    print(f"  node {node_id!r}: {attrs}")
    out_edges = list(graph.out_edges(node_id, data=True))
    if out_edges:
        print("  out-edges:")
        for _, dst, edata in out_edges:
            print(f"    --[{edata.get('relation_type')}]--> {dst}")
    in_edges = list(graph.in_edges(node_id, data=True))
    if in_edges:
        print("  in-edges:")
        for src, _, edata in in_edges:
            print(f"    {src} --[{edata.get('relation_type')}]-->")


def summarize_graph(graph) -> None:
    from collections import Counter
    cats = Counter(graph.nodes[n].get("category") for n in graph.nodes)
    rels = Counter(graph.edges[e].get("relation_type") for e in graph.edges)
    print(f"  {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges")
    print(f"  by category: {dict(cats)}")
    print(f"  by relation: {dict(rels)}")


def main():
    for p in (LLAMA_CLI, GRAMMAR, GGUF, SYSTEM_PATH):
        if not p.exists():
            sys.exit(f"missing: {p}")

    system = SYSTEM_PATH.read_text(encoding="utf-8")
    fewshot = FEWSHOT_PATH.read_text(encoding="utf-8") if FEWSHOT_PATH.exists() else ""
    graph = build_seed_drug_graph()
    executor = PythonCogLangExecutor(graph)
    confirm_writes = True  # safety default — toggle with /yolo

    print("=" * 72)
    print(" drug-v6-q4_k_m + SYSv6d + GBNF interactive demo")
    print(f" model: {GGUF.name}  ({GGUF.stat().st_size // 1024 // 1024} MiB)")
    print(f" graph: {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges")
    if fewshot:
        print(f" few-shot ICL: {FEWSHOT_PATH.name} ({len(fewshot)} chars)")
    print(" Type a question, /help for commands, /quit to exit.")
    print(" Write operations (Create/Update/Delete) will ask y/N before executing.")
    print("=" * 72)

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/help":
            print(__doc__)
            continue
        if user == "/graph":
            summarize_graph(graph)
            continue
        if user.startswith("/show "):
            show_node(graph, user.split(" ", 1)[1].strip())
            continue
        if user == "/reset":
            graph = build_seed_drug_graph()
            executor = PythonCogLangExecutor(graph)
            print(f"  graph reset — back to {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges.")
            continue
        if user == "/yolo":
            confirm_writes = not confirm_writes
            print(f"  write confirmation is now {'ON (default)' if confirm_writes else 'OFF (YOLO mode — writes happen silently)'}")
            continue

        print("  [model generating...]")
        try:
            reply, dt = llama_generate(system, user, fewshot=fewshot)
        except subprocess.SubprocessError as e:
            print(f"  llama-cli failed: {e}")
            continue
        print(f"\nCogLang ({dt:.1f}s): {reply}")

        try:
            expr = cog_parse(reply)
        except Exception as e:
            print(f"  parse failed: {e}")
            continue

        # Safety: write ops (Create/Update/Delete, possibly nested inside Trace/Do) need confirmation
        heads = collect_heads(expr)
        write_heads = heads & WRITE_HEADS
        if write_heads and confirm_writes:
            print(f"  ⚠️  This is a WRITE operation ({', '.join(sorted(write_heads))}). "
                  f"It will MUTATE the in-memory graph.")
            ans = input("  Execute? [y/N] ").strip().lower()
            if ans != "y":
                print("  cancelled — graph unchanged.")
                continue

        try:
            result = executor.execute(expr)
        except Exception as e:
            print(f"  execution error: {e}")
            continue
        print("Result:")
        print(format_result(result))


if __name__ == "__main__":
    main()

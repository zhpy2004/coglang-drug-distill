# app/coglang_runtime.py
"""CogLang parse/execute/format helpers, lifted from pipeline/drug_chat.py.

The model's output is constrained by GBNF, but we still extract the last
top-level M-expression defensively, then parse + execute via the coglang package.
"""
from __future__ import annotations

import re

from coglang import PythonCogLangExecutor, parse, canonicalize  # noqa: F401

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
HEAD_RE = re.compile(r"[A-Z]\w*\[")
WRITE_HEADS = {"Create", "Update", "Delete"}


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def extract_last_mexpr(text: str) -> str:
    """Find the LAST top-level balanced Head[...] M-expression in text."""
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
            break
        last = text[start:end + 1]
        i = end + 1
    return last.strip() if last else text.strip()


def parse_expr(text: str):
    return parse(text)


def collect_heads(expr, acc=None) -> set:
    acc = acc if acc is not None else set()
    if hasattr(expr, "head"):
        acc.add(expr.head)
    if hasattr(expr, "args"):
        for a in expr.args:
            collect_heads(a, acc)
    return acc


def is_write(expr) -> bool:
    return bool(collect_heads(expr) & WRITE_HEADS)


def format_result(result) -> str:
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

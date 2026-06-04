# app/server.py
"""Flask app: NL -> CogLang (via injected llm_call) -> execute on in-memory graph.

create_app() is a factory so tests can inject a fake llm_call. main() (Task 7)
wires the real llama-server client + lifecycle.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from coglang import PythonCogLangExecutor

from app import coglang_runtime as cr
from app.domains import Domain
from app.graph_loader import build_graph
from app.llm import build_user_message

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(domain: Domain, grammar: str, llm_call) -> Flask:
    app = Flask(__name__, static_folder=None)
    state = {"graph": build_graph(domain.graph_data)}
    state["executor"] = PythonCogLangExecutor(state["graph"])

    def _execute(coglang_text: str) -> str:
        expr = cr.parse_expr(coglang_text)
        result = state["executor"].execute(expr)
        return cr.format_result(result)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:fname>")
    def static_files(fname):
        return send_from_directory(STATIC_DIR, fname)

    @app.get("/domain")
    def domain_info():
        # CogLang writes are soft-deletes (confidence -> 0), so count only live
        # elements (confidence > 0), matching the executor's AllNodes semantics.
        g = state["graph"]
        live_nodes = sum(1 for _, d in g.nodes(data=True)
                         if d.get("confidence", 1.0) > 0)
        live_edges = sum(1 for *_, d in g.edges(data=True)
                         if d.get("confidence", 1.0) > 0)
        return jsonify(name=domain.name, description=domain.description,
                       nodes=live_nodes, edges=live_edges)

    @app.post("/ask")
    def ask():
        q = (request.get_json(force=True) or {}).get("q", "").strip()
        if not q:
            return jsonify(error="empty question"), 400
        user_message = build_user_message(domain.fewshot, q)
        raw = llm_call(domain.system, user_message, grammar)
        coglang_text = cr.extract_last_mexpr(cr.strip_ansi(raw))
        try:
            expr = cr.parse_expr(coglang_text)
        except Exception as e:  # noqa: BLE001
            return jsonify(coglang=coglang_text, result="", is_write=False,
                           needs_confirm=False, error=f"parse failed: {e}")
        is_write = cr.is_write(expr)
        if is_write:
            return jsonify(coglang=coglang_text, result="", is_write=True,
                           needs_confirm=True)
        try:
            result = cr.format_result(state["executor"].execute(expr))
        except Exception as e:  # noqa: BLE001
            return jsonify(coglang=coglang_text, result=f"execution error: {e}",
                           is_write=False, needs_confirm=False)
        return jsonify(coglang=coglang_text, result=result, is_write=False,
                       needs_confirm=False)

    @app.post("/execute")
    def execute():
        coglang_text = (request.get_json(force=True) or {}).get("coglang", "").strip()
        try:
            result = _execute(coglang_text)
        except Exception as e:  # noqa: BLE001
            return jsonify(error=f"execution error: {e}"), 400
        return jsonify(result=result)

    @app.post("/reset")
    def reset():
        state["graph"] = build_graph(domain.graph_data)
        state["executor"] = PythonCogLangExecutor(state["graph"])
        return jsonify(ok=True)

    return app


import argparse
import atexit
import os
import subprocess
import sys

from app import llm as llmclient
from app.domains import load_config, load_domain


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def main():
    root = Path(__file__).resolve().parent
    cfg = load_config(root / "config.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=cfg["active_domain"])
    args = ap.parse_args()

    domain = load_domain(root / "domains", args.domain)
    grammar = Path(_expand(cfg["grammar_path"])).read_text(encoding="utf-8")
    model_path = Path(_expand(cfg["models_dir"])) / domain.model
    server_url = f"http://127.0.0.1:{cfg['llama_server_port']}"

    if not model_path.is_file():
        sys.exit(f"model not found: {model_path}")

    proc = subprocess.Popen([
        _expand(cfg.get("llama_server_bin", "llama-server")),
        "-m", str(model_path),
        "--port", str(cfg["llama_server_port"]),
        "-t", str(cfg["threads"]),
        "-c", "2048",
    ])
    atexit.register(proc.terminate)

    print(f"waiting for llama-server on {server_url} ...")
    if not llmclient.wait_for_health(server_url, timeout=180):
        proc.terminate()
        sys.exit("llama-server did not become healthy in time")

    def llm_call(system, user_message, grammar_str):
        return llmclient.generate(server_url, system, user_message, grammar_str,
                                  max_tokens=domain.max_tokens,
                                  temperature=domain.temperature)

    app = create_app(domain, grammar=grammar, llm_call=llm_call)
    print(f"serving {domain.name} on http://127.0.0.1:{cfg['port']}")
    app.run(host="127.0.0.1", port=cfg["port"])


if __name__ == "__main__":
    main()

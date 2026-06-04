# CogLang Domain Web App

A modular local web app that turns natural language → [CogLang](https://github.com/jaysinailabs/coglang) → an executed result against a knowledge graph, rendered in a browser chat UI. Pure Python + vanilla JS; runs on desktop **and** on a phone under Termux.

> ⚠️ **Medical disclaimer.** Learning/research demo only. **Not a medical device, not clinical decision support.** The bundled drug graph is a tiny illustrative seed. Never use any output for real medical decisions. See [`../NOTICE`](../NOTICE).

## What it is

```
Browser ──HTTP──▶ Flask (127.0.0.1:8080) ──┬──▶ llama-server (resident, GBNF) → CogLang text
                                            └──▶ coglang.PythonCogLangExecutor → result
```

Flask holds an in-memory `networkx` graph built from the active **domain pack** and executes the model's CogLang against it. The model **only writes queries** — facts live in the graph, never in the weights. Write operations (`Create`/`Update`/`Delete`) are gated behind an explicit browser confirmation; the graph is in-memory only and `Reset graph` restores the seed.

## Run on desktop

```bash
pip install -r app/requirements.txt
# point app/config.yaml at this machine (do not commit machine-specific paths):
#   llama_server_bin: "E:/coglang/llama.cpp/bin/llama-server.exe"
#   models_dir:       "E:/coglang/gguf"
#   grammar_path:     "E:/coglang/coglang/examples/grammar/coglang.gbnf"
python -m app.server                 # add --domain <name> to override active_domain
# open http://127.0.0.1:8080 and ask "warfarin interactions"
```

On Chinese-locale Windows, prefix Python invocations with `PYTHONUTF8=1`.

## Verified on-device

Confirmed end-to-end in Termux on an Android phone (`pkg install python git`, deps installed including `coglang`, model + grammar in `~/coglang`, config defaults unchanged):

- `/domain` → `18 nodes / 22 edges`
- `/ask "warfarin interactions"` → `Traverse["warfarin", "interacts_with"]` → `aspirin / ibuprofen / naproxen` (identical to desktop)
- On-phone CPU latency for that query: prompt eval ~2.85 tok/s, generation ~5.06 tok/s (≈7 min for a 1240-token prompt). Functional, slow — desktop CPU is ~27 tok/s.

This resolves the deployment risks: `coglang` installs cleanly on Termux (R1), and GBNF passed via llama-server's `/completion` `grammar` field works on-device with no fallback (R3).

## Run in Termux (on a phone)

```sh
pkg install -y python
git clone https://github.com/zhpy2004/coglang-drug-distill
cd coglang-drug-distill
pip install -r app/requirements.txt
# config.yaml already defaults to ~/coglang for models_dir + grammar_path;
# put drug-v6-q4_k_m.gguf and coglang.gbnf there, and have llama-server on PATH.
python -m app.server
# phone browser → http://localhost:8080
```

## Authoring a new domain pack

A domain is a folder under `app/domains/<name>/` with four files — **no code change** needed to add one:

| File | Purpose |
|------|---------|
| `meta.yaml` | `name`, `description`, `model` (.gguf filename), `max_tokens`, `temperature` |
| `graph.json` | the knowledge graph (schema below) |
| `system.txt` | system prompt describing the domain's categories + relations |
| `fewshot.txt` | in-context-learning examples (`question -> CogLang`) |

Then set `active_domain: <name>` in `config.yaml` (or run `--domain <name>`). See `app/domains/toy/` for a minimal example.

### `graph.json` schema

```json
{
  "schema": {"categories": ["Drug", "..."], "relation_types": ["interacts_with", "..."]},
  "nodes": [{"id": "warfarin", "category": "Drug", "name": "Warfarin", "confidence": 1.0}],
  "edges": [{"from": "warfarin", "to": "aspirin", "relation_type": "interacts_with", "confidence": 1.0}]
}
```

Required node keys: `id`, `category`, `name`, `confidence`. Required edge keys: `from`, `to`, `relation_type`, `confidence`. Extra keys (e.g. `severity`) pass through as attributes.

## Safety guard

- **Write confirmation.** `/ask` classifies the generated CogLang; any expression containing `Create`/`Update`/`Delete` returns `needs_confirm: true` and is **not** executed until the user confirms via `/execute`.
- **Soft-delete semantics.** CogLang `Delete` sets `confidence → 0` rather than removing nodes; `/domain` reports only live elements (`confidence > 0`), so a confirmed delete is visible and `Reset graph` restores the seed.
- **Local only.** Binds to `127.0.0.1`; the graph is in-memory and never persisted.

## HTTP API

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | chat UI |
| `/domain` | GET | active domain name + live node/edge counts |
| `/ask` | POST `{q}` | NL → CogLang; executes reads, gates writes |
| `/execute` | POST `{coglang}` | run a (confirmed) CogLang expression |
| `/reset` | POST | rebuild the in-memory graph from the seed |

## Layout

```
app/
  config.yaml          # active_domain + ports + machine paths
  server.py            # Flask factory (create_app) + main() lifecycle
  llm.py               # llama-server ChatML + GBNF client
  coglang_runtime.py   # extract / parse / write-detect / format helpers
  graph_loader.py      # graph.json → networkx DiGraph
  domains.py           # config + domain-pack loader
  domains/<name>/      # a domain pack (meta.yaml + graph.json + system.txt + fewshot.txt)
  static/              # index.html + app.js + style.css
  tests/               # pytest suite (18 tests)
```

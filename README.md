# coglang-drug-distill

Small-model CogLang generation for a medication-safety knowledge graph.

This repository contains the public demo, domain pack, and lightweight evaluation
tools for a distilled graph-query model. The model translates natural-language
questions into [CogLang](https://github.com/jaysinailabs/coglang), then the
runtime executes the query against a small illustrative drug-interaction graph.
The model writes query programs; drug facts live in the graph, not in model
weights.

> Medical disclaimer: research/demo only. This is not a medical device and not
> clinical decision support. The bundled graph is tiny and illustrative. Never
> use outputs for real medical decisions.

## Model

Recommended public artifact:

- `drug-v6-q4_k_m.gguf`
- Hugging Face: [zhpy2004/coglang-drug-distill](https://huggingface.co/zhpy2004/coglang-drug-distill)
- Runtime prompt: `pipeline/SYSTEM_v6d.txt`
- Few-shot prompt block: `pipeline/fewshot_v1.txt`

Reviewed OOD eval-v2 result for `drug-v6-q4_k_m.gguf`:

| Fixture | Pass | Wilson 95% CI | Notes |
|---|---:|---:|---|
| `pipeline/ood_eval.json` | 118/156 (75%) | [68%, 82%] | GBNF constrained decoding, 0 hallucinated heads |

The GGUF model file is not stored in Git. Download it from Hugging Face and put
it in the local model directory configured by `app/config.yaml`.

## What Is Included

```text
app/                       local Flask chat UI and domain-pack loader
mobile/                    Termux/mobile run notes
scripts/                   small export/helper scripts
pipeline/
  SYSTEM_v6d.txt           deployed system prompt
  fewshot_v1.txt           deployed few-shot block
  ood_eval.json            156-case public OOD fixture
  drug_graph.py            seed graph builder
  eval_*.py                lightweight evaluation helpers
  train_qlora.py           training entrypoint, without bundled training data
  merge_adapter.py         adapter merge helper
  drug_chat.py             local CLI demo
```

Raw training data, request/response batches, intermediate logs, and internal
experiment journals are intentionally not part of this public GitHub repository.

## Run The Web App

```bash
pip install -r app/requirements.txt
python -m app.server
```

Then open `http://127.0.0.1:8080`.

`app/config.yaml` defaults to portable paths under `~/coglang`. Put the GGUF
model and CogLang grammar there, or edit the config locally without committing
machine-specific paths.

## Run Evaluation Helpers

Install the Python dependencies and the upstream CogLang package:

```bash
pip install -r requirements.txt
pip install coglang
```

Example checks:

```bash
python pipeline/tests_v6_check.py seeds pipeline/ood_eval.json
python pipeline/eval_breakdown.py --responses <model-outputs-jsonl> --fixture pipeline/ood_eval.json
```

To regenerate model responses, use `pipeline/eval_gguf.py` with a local GGUF
path and the CogLang GBNF grammar from the upstream CogLang project.

## Repository Notes

- The app is domain-pack based. See `app/domains/drug/` and `app/domains/toy/`.
- Write operations produced by the model are gated by browser confirmation.
- The graph is in-memory; reset restores the seed graph.
- The upstream CogLang runtime is a dependency and is not vendored here.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

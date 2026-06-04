# Phase B — on-device deployment on a phone (Android, llama.cpp + GBNF)

The distilled `drug-v6` model runs **natively on an Android phone**, with GBNF
grammar-constrained decoding forcing every output to be **valid CogLang** — the
same syntax-100%-guaranteed story as the desktop loop, now on the edge device
that is the actual point of this project.

This reuses the exact desktop artifact (`drug-v6-q4_k_m.gguf` + `SYSTEM_v6d.txt`
+ `fewshot_v1.txt` + `coglang.gbnf`) — **zero reconversion, zero retraining.**

> ⚠️ Same [medical disclaimer](../README.md) as the rest of the repo. Learning/
> research demo only — not a medical device.

## Why llama.cpp on Android (not MLC-LLM)

The original plan was MLC-LLM. It was abandoned for this milestone because:

- **The prebuilt MLC wheels are mutually incompatible right now.** `mlc-llm`'s
  native module imports a *unified* `tvm.dll`, while the only available
  `mlc-ai` nightly ships the *split* `tvm_runtime.dll` / `tvm_compiler.dll`
  (new tvm-ffi layout). The wheel index offers exactly one version of each and
  they don't pair — no config or DLL-path fix bridges the C++ ABI gap.
- **llama.cpp's GBNF is native and confirmed on-device.** CogLang's whole
  syntax guarantee depends on grammar-constrained decoding. llama.cpp enforces
  GBNF on Android out of the box; MLC's mobile-runtime grammar exposure was
  still unconfirmed.
- **Reuses the proven artifact** — no MLC convert/compile toolchain, no NDK
  build to get a first working demo.

MLC-via-WSL2 (for mobile-GPU acceleration via TVM) remains a possible future
sub-project, not a prerequisite.

## What runs on-device

| | |
|---|---|
| Model | `drug-v6-q4_k_m.gguf` (Qwen2.5-1.5B-Instruct QLoRA -> Q4_K_M) |
| Constraints | `coglang.gbnf` (grammar), `SYSTEM_v6d.txt` (4 verb→operator rules), `fewshot_v1.txt` (12 ICL pairs) |
| Decoding | greedy (`--temp 0`), `-n 256`, single-turn |
| Runtime | llama.cpp `llama-cli`, built for Android aarch64 |

## Measured results

**Device:** MediaTek **Dimensity 7300**, 8 GB RAM (+8 GB extended), Android,
Termux (`llama-cli`, Clang 21, aarch64). Mid-range SoC — a flagship will be
several times faster.

| Metric | Value |
|---|---|
| Model on disk | 940 MiB (986 MB) |
| RAM while running | ~1 GB, no noticeable lag |
| Prompt eval | ~18–23 tok/s |
| Generation | ~3.6–5.7 tok/s |

Generation is slower than the desktop CPU's ~27 tok/s — expected: phone CPU,
plus the GBNF grammar + the ~3 KB system prompt and few-shot context add per-token cost.

### On-device examples (verbatim output, GBNF-constrained)

| Prompt | Output | Notes |
|---|---|---|
| `warfarin interactions` | `Traverse["warfarin", "interacts_with"]` | ✅ exact-correct |
| `what are aspirin ingredients` | `Get["aspirin", "contains"]` | ⚠️ valid syntax, but the known **Get-vs-Traverse** semantic miss (`contains` is a relation → should be `Traverse`) |
| `can I give ibuprofen to a pregnant patient` | `Query[n_, Equal[Traverse[n_, "contraindicated_for"], List["pregnancy"]]]` | 🟡 valid + reasonable ("which drugs are pregnancy-contraindicated"), not a direct yes/no |

All three **parse as valid CogLang, with zero hallucinated operators and zero
dangerous writes** — the GBNF + distillation payoff. The semantic misses are the
same borderline cases tracked on desktop (≈50–68% end-to-end), not new on-device
regressions.

## The GBNF difference (why this matters)

Loading the *same* GGUF in a generic no-grammar app (PocketPal) with no system
prompt, the bare model free-writes a medical essay **and hallucinates** — e.g.
"warfarin also known as coumarin" (wrong) and "warfarin interacts with …
warfarin" (nonsense). That is exactly the from-memory hallucination this project
exists to eliminate.

With `SYSTEM_v6d` + `coglang.gbnf`, the model can only emit a single, checkable
CogLang query to be executed against the curated graph. **That is the on-device
safety story.**

## Reproduce on your own Android phone

1. Install **Termux** (from F-Droid or GitHub releases — *not* the Play Store build).
2. In Termux:
   ```sh
   termux-setup-storage
   # If the default mirrors are unreachable (e.g. in CN), switch to a local one:
   #   echo "deb https://mirrors.tuna.tsinghua.edu.cn/termux/apt/termux-main stable main" > $PREFIX/etc/apt/sources.list
   pkg update -y && pkg upgrade -y
   pkg install -y llama-cpp wget
   ```
3. Assemble these four files in one directory on the phone:
   - `coglang.sh` — this directory
   - `SYSTEM_v6d.txt`, `fewshot_v1.txt` — [`../pipeline/`](../pipeline)
   - `coglang.gbnf` — upstream CogLang's `examples/grammar/` (not vendored here)
   - `drug-v6-q4_k_m.gguf` (940 MiB) — Hugging Face
   ```sh
   mkdir -p ~/coglang && cd ~/coglang
   wget -O drug-v6-q4_k_m.gguf \
     https://huggingface.co/zhpy2004/coglang-drug-distill/resolve/main/drug-v6-q4_k_m.gguf
   sed -i 's/\r$//' coglang.sh && chmod +x coglang.sh
   ```
4. Run:
   ```sh
   export COGLANG_MODEL=~/coglang/drug-v6-q4_k_m.gguf
   ./coglang.sh "warfarin interactions"
   ```

`SYSTEM_v6d.txt`, `fewshot_v1.txt`, and `coglang.gbnf` are the same files used by
the desktop demo (in [`../pipeline/`](../pipeline) and CogLang's
`examples/grammar/`); copies are not duplicated here — fetch them from there or
from the upstream grammar.

## Beyond the CLI

For a browser chat UI on the phone (instead of `coglang.sh`'s one-shot CLI), see
the modular web app in [`../app/`](../app) — same on-device `llama-server` + GBNF,
with swappable domain packs and a write-confirmation safety gate. Run
`python -m app.server` in Termux and open `http://localhost:8080`.

## Files

- [`coglang.sh`](./coglang.sh) — Termux wrapper; assembles few-shot + system
  prompt and calls `llama-cli` with the grammar, mirroring `drug_chat.py`.

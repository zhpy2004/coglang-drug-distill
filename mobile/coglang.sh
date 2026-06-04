#!/data/data/com.termux/files/usr/bin/sh
# coglang.sh — on-device NL -> CogLang via llama.cpp + GBNF grammar (Android/Termux).
# Mirrors the desktop pipeline/drug_chat.py invocation:
#   llama-cli -m <gguf> --grammar-file coglang.gbnf -st -sys <SYSTEM>
#             -p "<fewshot>\n---\nUser question: <q>" -n 256 --temp 0
#
# Usage:   ./coglang.sh "warfarin interactions"
# Files (SYSTEM_v6d.txt, fewshot_v1.txt, coglang.gbnf) live next to this script.
# Model path defaults to the Downloads folder; override with COGLANG_MODEL=...

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${COGLANG_MODEL:-$HOME/storage/downloads/drug-v6-q4_k_m.gguf}"
SYS="$DIR/SYSTEM_v6d.txt"
FEW="$DIR/fewshot_v1.txt"
GBNF="$DIR/coglang.gbnf"

if [ -z "$1" ]; then
  echo "usage: ./coglang.sh \"your medication-safety question\""
  exit 1
fi

for f in "$MODEL" "$SYS" "$FEW" "$GBNF"; do
  if [ ! -f "$f" ]; then echo "missing file: $f"; exit 1; fi
done

USERMSG="$(cat "$FEW")
---
User question: $1"

llama-cli -m "$MODEL" \
  --grammar-file "$GBNF" \
  -st \
  -sys "$(cat "$SYS")" \
  -p "$USERMSG" \
  -n 256 --temp 0

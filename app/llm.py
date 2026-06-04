# app/llm.py
"""Client for a resident llama.cpp llama-server. Builds Qwen2.5 ChatML prompts
and constrains generation with a GBNF grammar string via /completion."""
from __future__ import annotations

import time

import requests

CHATML = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def build_user_message(fewshot: str, question: str) -> str:
    fewshot = (fewshot or "").strip()
    if fewshot:
        return f"{fewshot}\n---\nUser question: {question}"
    return f"User question: {question}"


def build_prompt(system: str, user_message: str) -> str:
    return CHATML.format(system=system, user=user_message)


def _post_completion(url: str, payload: dict) -> dict:
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def generate(server_url: str, system: str, user_message: str, grammar: str,
             max_tokens: int = 256, temperature: float = 0.0) -> str:
    prompt = build_prompt(system, user_message)
    payload = {
        "prompt": prompt,
        "grammar": grammar,
        "n_predict": max_tokens,
        "temperature": temperature,
        "cache_prompt": True,
        "stop": ["<|im_end|>"],
    }
    data = _post_completion(f"{server_url.rstrip('/')}/completion", payload)
    return data.get("content", "").strip()


def wait_for_health(server_url: str, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    url = f"{server_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return False

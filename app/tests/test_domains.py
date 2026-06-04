# app/tests/test_domains.py
from pathlib import Path

from app.domains import load_domain, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_load_drug_domain():
    d = load_domain(ROOT / "app" / "domains", "drug")
    assert d.name == "drug"
    assert d.model == "drug-v6-q4_k_m.gguf"
    assert d.max_tokens == 256
    assert "warfarin" in d.graph_data_node_ids()      # graph.json loaded
    assert d.system.strip() != ""                      # system.txt loaded
    assert d.fewshot.strip() != ""                     # fewshot.txt loaded


def test_load_config():
    cfg = load_config(ROOT / "app" / "config.yaml")
    assert cfg["active_domain"] == "drug"
    assert cfg["port"] == 8080

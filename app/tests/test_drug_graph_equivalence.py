# app/tests/test_drug_graph_equivalence.py
"""The exported graph.json must rebuild to the exact same graph as drug_graph.py."""
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from drug_graph import build_seed_drug_graph  # noqa: E402
from app.graph_loader import load_graph  # noqa: E402


def test_exported_graph_matches_source():
    src = build_seed_drug_graph()
    loaded = load_graph(ROOT / "app" / "domains" / "drug" / "graph.json")
    assert set(src.nodes) == set(loaded.nodes)
    assert set(src.edges) == set(loaded.edges)
    for n in src.nodes:
        assert dict(src.nodes[n]) == dict(loaded.nodes[n])
    for e in src.edges:
        assert dict(src.edges[e]) == dict(loaded.edges[e])

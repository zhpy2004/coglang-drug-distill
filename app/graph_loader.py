"""Build a networkx DiGraph from a domain pack's graph.json (generalizes
pipeline/drug_graph.py). Required node keys: id, category, name, confidence.
Required edge keys: from, to, relation_type, confidence. Extra keys pass through
as attributes."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def build_graph(data: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for node in data.get("nodes", []):
        node = dict(node)
        node_id = node.pop("id")
        g.add_node(node_id, **node)
    for edge in data.get("edges", []):
        edge = dict(edge)
        src = edge.pop("from")
        dst = edge.pop("to")
        g.add_edge(src, dst, **edge)
    return g


def load_graph(path: str | Path) -> nx.DiGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return build_graph(data)

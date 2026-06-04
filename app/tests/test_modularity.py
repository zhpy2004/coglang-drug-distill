# app/tests/test_modularity.py
from pathlib import Path

from app.domains import load_domain

ROOT = Path(__file__).resolve().parents[2]


def test_switching_domain_changes_graph_and_system():
    drug = load_domain(ROOT / "app" / "domains", "drug")
    toy = load_domain(ROOT / "app" / "domains", "toy")
    assert drug.graph_data_node_ids() != toy.graph_data_node_ids()
    assert "cat" in toy.graph_data_node_ids()
    assert drug.system != toy.system
    assert toy.name == "toy"

# scripts/export_drug_graph.py
"""One-shot: dump pipeline/drug_graph.py's seed graph to app/domains/drug/graph.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from drug_graph import build_seed_drug_graph  # noqa: E402

SCHEMA = {
    "categories": ["Drug", "DrugClass", "Ingredient", "Condition", "AdverseEffect", "Population"],
    "relation_types": ["contains", "belongs_to", "interacts_with", "contraindicated_for", "causes", "treats"],
}


def main():
    g = build_seed_drug_graph()
    nodes = [{"id": n, **g.nodes[n]} for n in g.nodes]
    edges = [{"from": u, "to": v, **g.edges[u, v]} for u, v in g.edges]
    out = {"schema": SCHEMA, "nodes": nodes, "edges": edges}
    dest = ROOT / "app" / "domains" / "drug" / "graph.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest} ({len(nodes)} nodes / {len(edges)} edges)")


if __name__ == "__main__":
    main()

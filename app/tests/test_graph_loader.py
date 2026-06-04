from app.graph_loader import build_graph


def _sample():
    return {
        "nodes": [
            {"id": "warfarin", "category": "Drug", "name": "Warfarin", "confidence": 1.0},
            {"id": "aspirin", "category": "Drug", "name": "Aspirin", "confidence": 1.0},
        ],
        "edges": [
            {"from": "warfarin", "to": "aspirin", "relation_type": "interacts_with",
             "confidence": 1.0, "severity": "major"},
        ],
    }


def test_build_graph_nodes_and_attrs():
    g = build_graph(_sample())
    assert g.number_of_nodes() == 2
    assert g.nodes["warfarin"]["category"] == "Drug"
    assert g.nodes["warfarin"]["name"] == "Warfarin"
    assert g.nodes["warfarin"]["confidence"] == 1.0


def test_build_graph_edge_attrs_passthrough():
    g = build_graph(_sample())
    assert g.number_of_edges() == 1
    edata = g.edges["warfarin", "aspirin"]
    assert edata["relation_type"] == "interacts_with"
    assert edata["severity"] == "major"      # extra key passed through
    assert edata["confidence"] == 1.0

"""Seed medication-safety knowledge graph for CogLang.

Schema (lock these — every prompt/expression in the training set must match):
  Node attribute  `category` ∈ {Drug, DrugClass, Ingredient, Condition,
                                 AdverseEffect, Population}
  Edge `relation_type` ∈ {contains, belongs_to, interacts_with,
                          contraindicated_for, causes, treats}
  Severity (informational, on interacts_with edges only): minor | moderate | major | contraindicated

The graph is small (~20 nodes) and intentionally illustrative. Not a clinical
source. Real distillation should swap this for a curated subset of
RxNorm + DDInter — see step1_survey.md §7.
"""
from __future__ import annotations

import networkx as nx


_DRUGS = [
    ("warfarin",   "Warfarin"),
    ("aspirin",    "Aspirin"),
    ("ibuprofen",  "Ibuprofen"),
    ("naproxen",   "Naproxen"),
    ("fluoxetine", "Fluoxetine"),
    ("sertraline", "Sertraline"),
    ("tramadol",   "Tramadol"),
    ("acetaminophen", "Acetaminophen"),
]

_INGREDIENTS = [
    ("acetylsalicylic_acid", "Acetylsalicylic acid"),
]

_CLASSES = [
    ("nsaid", "NSAID"),
    ("ssri",  "SSRI"),
]

_CONDITIONS = [
    ("hypertension", "Hypertension"),
    ("depression",   "Major depressive disorder"),
    ("headache",     "Headache"),
]

_ADVERSE = [
    ("gi_bleed",       "Gastrointestinal bleeding"),
    ("serotonin_syndrome", "Serotonin syndrome"),
]

_POPULATIONS = [
    ("pregnancy", "Pregnancy"),
    ("children",  "Children under 12"),
]

# (from, to, relation_type, optional extra attrs)
_EDGES: list[tuple[str, str, str, dict]] = [
    # ingredient containment
    ("aspirin", "acetylsalicylic_acid", "contains", {}),

    # drug class membership
    ("ibuprofen", "nsaid", "belongs_to", {}),
    ("naproxen",  "nsaid", "belongs_to", {}),
    ("aspirin",   "nsaid", "belongs_to", {}),
    ("fluoxetine", "ssri", "belongs_to", {}),
    ("sertraline", "ssri", "belongs_to", {}),

    # interactions (major / moderate)
    ("warfarin", "aspirin",   "interacts_with", {"severity": "major"}),
    ("warfarin", "ibuprofen", "interacts_with", {"severity": "major"}),
    ("warfarin", "naproxen",  "interacts_with", {"severity": "major"}),
    ("fluoxetine", "tramadol", "interacts_with", {"severity": "major"}),
    ("sertraline", "tramadol", "interacts_with", {"severity": "moderate"}),

    # adverse effects caused by
    ("ibuprofen", "gi_bleed", "causes", {}),
    ("naproxen",  "gi_bleed", "causes", {}),
    ("aspirin",   "gi_bleed", "causes", {}),
    ("fluoxetine", "serotonin_syndrome", "causes", {}),

    # contraindications
    ("ibuprofen", "pregnancy", "contraindicated_for", {}),
    ("naproxen",  "pregnancy", "contraindicated_for", {}),
    ("aspirin",   "children",  "contraindicated_for", {}),

    # therapeutic use
    ("ibuprofen", "headache",    "treats", {}),
    ("acetaminophen", "headache", "treats", {}),
    ("fluoxetine", "depression", "treats", {}),
    ("sertraline", "depression", "treats", {}),
]


def build_seed_drug_graph() -> nx.DiGraph:
    """Return a fresh NetworkX DiGraph seeded with the medication-safety fixture."""
    g = nx.DiGraph()
    for cat, items in [
        ("Drug",          _DRUGS),
        ("Ingredient",    _INGREDIENTS),
        ("DrugClass",     _CLASSES),
        ("Condition",     _CONDITIONS),
        ("AdverseEffect", _ADVERSE),
        ("Population",    _POPULATIONS),
    ]:
        for node_id, name in items:
            g.add_node(node_id, category=cat, name=name, confidence=1.0)
    for src, dst, rel, extra in _EDGES:
        g.add_edge(src, dst, relation_type=rel, confidence=1.0, **extra)
    return g


if __name__ == "__main__":
    # Smoke test: build the graph, run a few CogLang queries.
    from coglang import PythonCogLangExecutor, parse, canonicalize

    g = build_seed_drug_graph()
    ex = PythonCogLangExecutor(g)
    for expr_str in [
        'Traverse["warfarin", "interacts_with"]',
        'Query[n_, Equal[Get[n_, "category"], "Drug"]]',
        'IfFound[Traverse["acetaminophen", "interacts_with"], hits_, hits_, "no_known"]',
        'Get["ibuprofen", "name"]',
        'ForEach[List["warfarin", "aspirin", "ibuprofen"], drug_, Get[drug_, "category"]]',
    ]:
        result = ex.execute(parse(expr_str))
        print(f"{expr_str}\n  -> {canonicalize(result) if hasattr(result, 'head') else result!r}\n")

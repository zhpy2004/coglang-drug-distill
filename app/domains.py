# app/domains.py
"""Load app config and domain packs. A domain pack is a folder with
meta.yaml + graph.json + system.txt + fewshot.txt."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@dataclass
class Domain:
    name: str
    description: str
    model: str
    max_tokens: int
    temperature: float
    system: str
    fewshot: str
    graph_data: dict
    dir: Path

    def graph_data_node_ids(self) -> set:
        return {n["id"] for n in self.graph_data.get("nodes", [])}


def load_domain(domains_dir: str | Path, name: str) -> Domain:
    d = Path(domains_dir) / name
    meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
    graph_data = json.loads((d / "graph.json").read_text(encoding="utf-8"))
    return Domain(
        name=meta["name"],
        description=meta.get("description", ""),
        model=meta["model"],
        max_tokens=int(meta.get("max_tokens", 256)),
        temperature=float(meta.get("temperature", 0.0)),
        system=(d / "system.txt").read_text(encoding="utf-8"),
        fewshot=(d / "fewshot.txt").read_text(encoding="utf-8"),
        graph_data=graph_data,
        dir=d,
    )

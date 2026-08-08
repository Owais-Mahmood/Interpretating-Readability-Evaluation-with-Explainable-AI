from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class TaxonomyNode:
    code: str
    name: str
    level: str
    parent: str | None
    definition: str
    examples: tuple[Mapping[str, Any], ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class E2RTaxonomy:
    data: Mapping[str, Any]
    nodes: Mapping[str, TaxonomyNode]
    children_by_parent: Mapping[str, tuple[str, ...]]
    classifier_mapping: Mapping[str, str | None]
    fingerprint: str

    def node(self, code: str) -> TaxonomyNode:
        try:
            return self.nodes[code]
        except KeyError as exc:
            raise KeyError(f"Unknown taxonomy code: {code}") from exc

    def children(self, code: str, recursive: bool = False) -> list[TaxonomyNode]:
        direct = [self.node(child) for child in self.children_by_parent.get(code, ())]
        if not recursive:
            return direct
        result: list[TaxonomyNode] = []
        frontier = direct[:]
        while frontier:
            current = frontier.pop(0)
            result.append(current)
            frontier.extend(self.children(current.code, recursive=False))
        return result

    def ancestors(self, code: str) -> list[TaxonomyNode]:
        result: list[TaxonomyNode] = []
        current = self.node(code)
        seen: set[str] = set()
        while current.parent:
            if current.parent in seen:
                raise ValueError(f"Taxonomy cycle detected at {current.parent}")
            seen.add(current.parent)
            current = self.node(current.parent)
            result.append(current)
        return result

    def macro(self, code: str) -> TaxonomyNode:
        current = self.node(code)
        while current.parent:
            current = self.node(current.parent)
        if current.level != "macro":
            raise ValueError(f"Top-level node is not macro: {current.code}")
        return current

    def classifier_label(self, code: str) -> str | None:
        return self.classifier_mapping.get(self.macro(code).code)

    def macro_for_classifier_label(self, label: str) -> TaxonomyNode:
        for code, mapped in self.classifier_mapping.items():
            if mapped == label:
                return self.node(code)
        raise KeyError(f"No E2R macrostrategy maps to classifier label: {label}")

    def render_macro_card(
        self,
        label_or_code: str,
        *,
        include_descendants: bool = True,
        include_examples: bool = True,
    ) -> str:
        if label_or_code in self.nodes:
            macro = self.macro(label_or_code)
        else:
            macro = self.macro_for_classifier_label(label_or_code)
        macro_text = f"{macro.name} ({macro.code}): {macro.definition}"
        if macro.aliases:
            macro_text += " Aliases: " + ", ".join(macro.aliases)
        chunks = [macro_text]
        descendants = self.children(macro.code, recursive=True) if include_descendants else []
        for child in descendants:
            text = f"- {child.name} ({child.code}): {child.definition}"
            if child.aliases:
                text += " Aliases: " + ", ".join(child.aliases)
            if include_examples and child.examples:
                ex = child.examples[0]
                source = str(ex.get("source", "")).strip()
                simplified = str(ex.get("simplified", "")).strip()
                if source and simplified:
                    text += f" Example: {source} -> {simplified}"
            chunks.append(text)
        return "\n".join(chunks)

    def searchable_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for code, node in self.nodes.items():
            macro = self.macro(code)
            text_parts = [node.name, node.code, node.definition, *node.aliases]
            for example in node.examples:
                text_parts.extend(
                    [str(example.get("source", "")), str(example.get("simplified", ""))]
                )
            documents.append(
                {
                    "id": f"taxonomy:{code}",
                    "kind": "taxonomy_node",
                    "code": code,
                    "name": node.name,
                    "level": node.level,
                    "parent": node.parent,
                    "macro_code": macro.code,
                    "classifier_label": self.classifier_mapping.get(macro.code),
                    "text": " ".join(part for part in text_parts if part).strip(),
                    "properties": {
                        "definition": node.definition,
                        "aliases": list(node.aliases),
                        "examples": [dict(value) for value in node.examples],
                    },
                }
            )
        concepts = self.data.get("simplification_concepts", {})
        for domain, payload in concepts.items():
            documents.append(
                {
                    "id": f"rule:{domain}",
                    "kind": "simplification_rule_set",
                    "domain": domain,
                    "text": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    "properties": payload,
                }
            )
        return documents


def _normalise(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def taxonomy_fingerprint(data: Mapping[str, Any]) -> str:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_taxonomy(data: Mapping[str, Any]) -> None:
    if data.get("schema") != "e2r-diastratic-taxonomy-v1":
        raise ValueError("Unsupported taxonomy schema.")
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("Taxonomy requires a non-empty nodes list.")
    codes: set[str] = set()
    names: set[str] = set()
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise ValueError("Each taxonomy node must be a mapping.")
        code = str(raw.get("code", "")).strip()
        name = str(raw.get("name", "")).strip()
        level = str(raw.get("level", "")).strip()
        if not code or not name or level not in {"macro", "strategy", "micro"}:
            raise ValueError(f"Invalid taxonomy node: {raw}")
        if code in codes:
            raise ValueError(f"Duplicate taxonomy code: {code}")
        codes.add(code)
        key = _normalise(name)
        if key in names:
            raise ValueError(f"Duplicate taxonomy name: {name}")
        names.add(key)
    for raw in raw_nodes:
        parent = raw.get("parent")
        if parent is not None and str(parent) not in codes:
            raise ValueError(f"Unknown parent {parent} for node {raw['code']}")
        if parent is None and raw["level"] != "macro":
            raise ValueError(f"Only macro nodes may be roots: {raw['code']}")
    mapping = data.get("classifier_alignment", {}).get("macro_to_classifier_label", {})
    macro_codes = {str(node["code"]) for node in raw_nodes if node["level"] == "macro"}
    if set(mapping) != macro_codes:
        raise ValueError("classifier_alignment must cover every macrostrategy exactly once.")
    current = data.get("classifier_alignment", {}).get("current_labels", [])
    mapped = [value for value in mapping.values() if value is not None]
    missing = [label for label in current if label not in mapped]
    if missing:
        raise ValueError(f"Classifier labels missing from taxonomy alignment: {missing}")
    continuum = data.get("continuum", {}).get("ordered_macro_codes", [])
    if set(continuum) != macro_codes or len(continuum) != len(macro_codes):
        raise ValueError("Continuum must contain every macrostrategy exactly once.")


def load_taxonomy(path: str | Path) -> E2RTaxonomy:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Taxonomy YAML must contain a mapping.")
    validate_taxonomy(data)
    nodes: dict[str, TaxonomyNode] = {}
    children: dict[str, list[str]] = {}
    for raw in data["nodes"]:
        node = TaxonomyNode(
            code=str(raw["code"]),
            name=str(raw["name"]),
            level=str(raw["level"]),
            parent=str(raw["parent"]) if raw.get("parent") else None,
            definition=str(raw.get("definition", "")),
            examples=tuple(raw.get("examples", []) or []),
            aliases=tuple(str(value) for value in raw.get("aliases", []) or []),
        )
        nodes[node.code] = node
        if node.parent:
            children.setdefault(node.parent, []).append(node.code)
    return E2RTaxonomy(
        data=data,
        nodes=nodes,
        children_by_parent={key: tuple(values) for key, values in children.items()},
        classifier_mapping=data["classifier_alignment"]["macro_to_classifier_label"],
        fingerprint=taxonomy_fingerprint(data),
    )


def export_kag_artifacts(taxonomy: E2RTaxonomy, output_dir: str | Path) -> dict[str, Any]:
    """Export implementation-neutral KAG inputs for E11A/E11B.

    The exports avoid committing to OpenSPG. They can be loaded by NetworkX,
    property-graph databases, vector stores, or OpenSPG in later experiments.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for node in taxonomy.nodes.values():
        macro = taxonomy.macro(node.code)
        node_rows.append(
            {
                "id": node.code,
                "name": node.name,
                "level": node.level,
                "definition": node.definition,
                "parent": node.parent or "",
                "macro_code": macro.code,
                "classifier_label": taxonomy.classifier_mapping.get(macro.code) or "",
                "aliases": json.dumps(list(node.aliases), ensure_ascii=False),
                "examples": json.dumps([dict(value) for value in node.examples], ensure_ascii=False),
            }
        )
        if node.parent:
            edge_rows.append(
                {
                    "source": node.parent,
                    "relation": "HAS_CHILD",
                    "target": node.code,
                    "properties": "{}",
                }
            )
    ordered = list(taxonomy.data["continuum"]["ordered_macro_codes"])
    for left, right in zip(ordered, ordered[1:]):
        edge_rows.append(
            {"source": left, "relation": "PRECEDES", "target": right, "properties": "{}"}
        )
    for code, label in taxonomy.classifier_mapping.items():
        if label is not None:
            edge_rows.append(
                {
                    "source": code,
                    "relation": "MAPS_TO_CLASSIFIER_LABEL",
                    "target": f"label:{label}",
                    "properties": "{}",
                }
            )
    with (output / "taxonomy_nodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(node_rows[0]))
        writer.writeheader()
        writer.writerows(node_rows)
    with (output / "taxonomy_edges.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)
    documents = taxonomy.searchable_documents()
    with (output / "kag_documents.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    metadata = {
        "schema": "e2r-kag-export-v1",
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "node_count": len(node_rows),
        "edge_count": len(edge_rows),
        "document_count": len(documents),
        "source": dict(taxonomy.data.get("source", {})),
        "classifier_alignment": taxonomy.data["classifier_alignment"],
    }
    (output / "kag_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata

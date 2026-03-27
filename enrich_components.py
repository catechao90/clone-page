import argparse
import copy
import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class _OpeningTagParser(HTMLParser):
    """Extract only the first opening tag and its attributes from HTML snippet."""

    def __init__(self) -> None:
        super().__init__()
        self.tag: Optional[str] = None
        self.attrs: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if self.tag is None:
            self.tag = tag
            self.attrs = {k: (v if v is not None else "") for k, v in attrs}


def parse_opening_tag(html: str) -> Tuple[str, Dict[str, str]]:
    parser = _OpeningTagParser()
    parser.feed(html or "")
    return (parser.tag or "div", parser.attrs)


def css_escape_attr_value(value: str) -> str:
    # Keep it simple: use double quotes for attribute selectors.
    return value.replace('"', '\\"')


def is_stable_class_token(token: str) -> bool:
    if not token:
        return False
    # Exclude framework-generated or highly volatile classes.
    volatile_prefixes = ("svelte-", "css-")
    if token.startswith(volatile_prefixes):
        return False
    if re.search(r"__\w+__[0-9a-f]{5,}", token):
        return False
    return True


def build_stable_selectors(tag: str, attrs: Dict[str, str]) -> List[str]:
    selectors: List[str] = []

    elem_id = attrs.get("id", "").strip()
    if elem_id:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*$", elem_id):
            selectors.append(f"#{elem_id}")
        else:
            selectors.append(f'{tag}[id="{css_escape_attr_value(elem_id)}"]')

    for data_key in ("data-testid", "data-test", "data-qa", "data-cy"):
        if attrs.get(data_key):
            selectors.append(
                f'{tag}[{data_key}="{css_escape_attr_value(attrs[data_key])}"]'
            )

    if attrs.get("aria-label"):
        selectors.append(
            f'{tag}[aria-label="{css_escape_attr_value(attrs["aria-label"])}"]'
        )

    if attrs.get("role"):
        selectors.append(f'{tag}[role="{css_escape_attr_value(attrs["role"])}"]')

    for attr_key in ("name", "type", "href", "aria-controls"):
        if attrs.get(attr_key):
            selectors.append(
                f'{tag}[{attr_key}="{css_escape_attr_value(attrs[attr_key])}"]'
            )

    class_tokens = [
        c for c in attrs.get("class", "").split() if is_stable_class_token(c)
    ]
    if class_tokens:
        selectors.append(f"{tag}.{class_tokens[0]}")
    if len(class_tokens) >= 2:
        selectors.append(f"{tag}.{class_tokens[0]}.{class_tokens[1]}")

    # Fallback: always keep tag selector so downstream has at least one candidate.
    selectors.append(tag)

    dedup: List[str] = []
    seen = set()
    for s in selectors:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup


def infer_instance_key(component: Dict[str, Any], attrs: Dict[str, str]) -> Dict[str, Any]:
    repeatable = bool(component.get("Repeatability", False))
    if not repeatable:
        return {"strategy": "singleton", "keyTemplate": None, "source": "Repeatability=false"}

    for attr_name in (
        "data-offer-id",
        "data-id",
        "id",
        "href",
        "aria-label",
        "name",
    ):
        if attrs.get(attr_name):
            return {
                "strategy": "attribute",
                "keyTemplate": f"{attr_name}:{{value}}",
                "source": f"source_html_code:{attr_name}",
            }

    return {
        "strategy": "path-index",
        "keyTemplate": "{componentPath}#{indexWithinParent}",
        "source": "fallback",
    }


def infer_state_markers(component: Dict[str, Any], attrs: Dict[str, str]) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []

    for state_attr in (
        "aria-pressed",
        "aria-expanded",
        "aria-selected",
        "aria-checked",
        "hidden",
    ):
        if state_attr in attrs:
            markers.append(
                {
                    "marker": state_attr,
                    "kind": "attribute",
                    "expectedValues": [attrs.get(state_attr, "")],
                    "source": "source_html_code",
                }
            )

    class_value = attrs.get("class", "")
    if class_value:
        class_tokens = [c for c in class_value.split() if is_stable_class_token(c)]
        if class_tokens:
            markers.append(
                {
                    "marker": "class",
                    "kind": "class-token",
                    "expectedValues": class_tokens[:5],
                    "source": "source_html_code",
                }
            )

    if attrs.get("style"):
        markers.append(
            {
                "marker": "style",
                "kind": "inline-style",
                "expectedValues": [attrs["style"]],
                "source": "source_html_code",
            }
        )

    if not markers:
        markers.append(
            {
                "marker": "dom-mutation",
                "kind": "unknown",
                "expectedValues": [],
                "source": "needs-playwright-capture",
            }
        )

    return markers


def infer_interaction_contracts(
    component: Dict[str, Any],
    tag: str,
    attrs: Dict[str, str],
    selector: str,
) -> List[Dict[str, Any]]:
    name = (component.get("name") or "").lower()
    contracts: List[Dict[str, Any]] = []

    def add_contract(event: str, effect: str, precondition: str = "") -> None:
        contracts.append(
            {
                "event": event,
                "triggerSelector": selector,
                "precondition": precondition,
                "expectedEffects": [effect],
                "confidence": "heuristic",
            }
        )

    if tag == "button" or "button" in name:
        if "favorite" in name or attrs.get("aria-pressed") is not None:
            add_contract("click", "toggle-state", "button is visible and enabled")
        elif "next" in name or "prev" in name or "carousel" in name:
            add_contract("click", "advance-carousel", "carousel is initialized")
        elif "skip" in name:
            add_contract("click", "scroll-into-view", "main target exists")
        else:
            add_contract("click", "set-attribute", "button is visible and enabled")

    if tag == "a":
        add_contract("click", "navigate", "link href is valid")

    if tag in ("input", "textarea", "select"):
        add_contract("input", "update-input", "field is editable")
        add_contract("change", "set-attribute", "value changed")
        add_contract("keydown", "update-input", "keyboard interaction")

    if tag == "form" or "search" in name:
        add_contract("submit", "submit-form", "form validation passes")

    if "carousel" in name:
        add_contract("mouseover", "show-element", "hover behavior enabled")

    if not contracts:
        add_contract("click", "set-attribute", "generic interactive surface")

    # Deduplicate by (event,effect,selector)
    dedup = {}
    for c in contracts:
        key = (c["event"], c["expectedEffects"][0], c["triggerSelector"])
        dedup[key] = c
    return list(dedup.values())


def index_component_tree(components: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}

    def walk(nodes: List[Dict[str, Any]], parent_path: str = "") -> None:
        for node in nodes:
            name = node.get("name", "Unknown")
            path = f"{parent_path}/{name}" if parent_path else name
            idx[name] = node
            idx[path] = node
            walk(node.get("children", []), path)

    walk(components)
    return idx


def merge_from_interactions(
    root_components: List[Dict[str, Any]], interactions_doc: Dict[str, Any]
) -> None:
    comp_index = index_component_tree(root_components)

    inferred_events: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    inferred_markers: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    # Build id(node) lookup for mutation safety.
    node_ids: Dict[int, Dict[str, Any]] = {}

    def register(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            node_ids[id(node)] = node
            register(node.get("children", []))

    register(root_components)

    def update_node(node: Dict[str, Any], key: str, value: Dict[str, Any], bucket: Dict[int, Dict[str, Dict[str, Any]]]) -> None:
        bucket[id(node)][key] = value

    for interaction in interactions_doc.get("interactions", []):
        event = interaction.get("type")
        effects = [e.get("type") for e in interaction.get("effects", []) if e.get("type")]

        changes = interaction.get("domChanges", [])
        markers: List[Dict[str, Any]] = []
        for ch in changes:
            kind = ch.get("kind")
            attr = ch.get("attributeName")
            if kind == "attributes" and attr:
                markers.append(
                    {
                        "marker": attr,
                        "kind": "attribute",
                        "expectedValues": [ch.get("newValue")],
                        "source": "playwright-domChange",
                    }
                )
            elif kind in ("visibility", "layout"):
                markers.append(
                    {
                        "marker": kind,
                        "kind": "mutation-kind",
                        "expectedValues": [ch.get("newValue")],
                        "source": "playwright-domChange",
                    }
                )

        for match in interaction.get("componentMatches", []):
            comp_name = match.get("componentName")
            comp_path = match.get("componentPath")
            node = comp_index.get(comp_path) or comp_index.get(comp_name)
            if not node:
                continue

            selector = (match.get("matchedSelector") or "").strip() or "*"
            rel = match.get("relation", "")
            event_key = f"{event}|{selector}|{rel}"
            contract = {
                "event": event,
                "triggerSelector": selector,
                "precondition": "captured from playwright",
                "expectedEffects": effects or ["set-attribute"],
                "confidence": "captured",
            }
            update_node(node, event_key, contract, inferred_events)

            for m in markers:
                marker_key = f"{m.get('marker')}|{m.get('kind')}|{m.get('source')}"
                update_node(node, marker_key, m, inferred_markers)

    for node_id, contracts in inferred_events.items():
        node = node_ids[node_id]
        existing = node.setdefault("interaction_contracts", [])
        existing.extend(list(contracts.values()))

    for node_id, markers in inferred_markers.items():
        node = node_ids[node_id]
        existing = node.setdefault("state_markers", [])
        existing.extend(list(markers.values()))


def enrich_component_tree(
    nodes: List[Dict[str, Any]], parent_name: Optional[str] = None, parent_path: str = ""
) -> None:
    for node in nodes:
        name = node.get("name", "Unknown")
        path = f"{parent_path}/{name}" if parent_path else name
        source_html = node.get("source_html_code", "")
        tag, attrs = parse_opening_tag(source_html)

        stable_selectors = build_stable_selectors(tag, attrs)

        node["path"] = path
        node["parent"] = parent_name
        node["stable_selectors"] = stable_selectors
        node["instance_key"] = infer_instance_key(node, attrs)
        node["state_markers"] = infer_state_markers(node, attrs)

        trigger_selector = stable_selectors[0] if stable_selectors else tag
        node["interaction_contracts"] = infer_interaction_contracts(
            node, tag, attrs, trigger_selector
        )

        enrich_component_tree(node.get("children", []), parent_name=name, parent_path=path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich component.json with stable_selectors, instance_key, "
            "state_markers, interaction_contracts. Optionally merge Playwright interactions."
        )
    )
    parser.add_argument(
        "--component",
        default="component.json",
        help="Path to input component.json",
    )
    parser.add_argument(
        "--interactions",
        default="",
        help="Optional path to interaction capture json (example.interaction.json style)",
    )
    parser.add_argument(
        "--output",
        default="component.enriched.json",
        help="Path to output enriched json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    component_path = Path(args.component)
    if not component_path.exists():
        raise FileNotFoundError(f"Component file not found: {component_path}")

    with component_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    result = copy.deepcopy(doc)
    components = result.get("result_text", {}).get("components", [])
    if not isinstance(components, list):
        raise ValueError("Invalid component.json format: result_text.components must be a list")

    enrich_component_tree(components)

    if args.interactions:
        interactions_path = Path(args.interactions)
        if interactions_path.exists():
            with interactions_path.open("r", encoding="utf-8") as f:
                interactions_doc = json.load(f)
            merge_from_interactions(components, interactions_doc)
        else:
            raise FileNotFoundError(f"Interactions file not found: {interactions_path}")

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote enriched components to: {args.output}")


if __name__ == "__main__":
    main()

"""Streamlit controls shared by onboarding and live hierarchy navigation."""
from __future__ import annotations

import streamlit as st


def group_paths(nodes: list[dict]) -> dict:
    """Map derived group IDs to their root-to-group registration path."""
    paths = {}
    pending = [(node, []) for node in nodes]
    while pending:
        node, parent = pending.pop()
        if node["children"]:
            path = [*parent, (node["level"], node["name"])]
            paths[node["id"]] = path
            pending.extend((child, path) for child in node["children"])
    return paths


def select_monitoring_node(nodes: list[dict], preferred: str) -> dict:
    """Expand only the selected branch so wide and deep trees stay navigable."""
    paths = {}
    pending = [(node, []) for node in nodes]
    while pending:
        node, parent = pending.pop()
        path = [*parent, node["id"]]
        paths[node["id"]] = path
        pending.extend((child, path) for child in node["children"])
    preferred_path = paths.get(preferred, [])
    current = None
    depth = 0
    while nodes:
        options = {node["id"]: node for node in nodes}
        ids = list(options)
        if current is not None:
            ids.insert(0, None)
        default = preferred_path[depth] if depth < len(preferred_path) and preferred_path[depth] in ids else ids[0]
        labels = {key: f"{node['name']} ({node['level']})" for key, node in options.items()}
        if current is not None:
            labels[None] = f"View {current['name']}"
        selection = st.selectbox(
            "Station hierarchy" if current is None else f"Within {current['name']}",
            ids, index=ids.index(default),
            format_func=labels.__getitem__,
            key=f"monitor_hierarchy_{current['id'] if current else 'root'}",
        )
        if selection is None:
            break
        current = options[selection]
        nodes = current["children"]
        depth += 1
    return current

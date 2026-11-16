"""Hierarchy paths are attached to data-producing stations in the station registry."""
from __future__ import annotations

from hashlib import sha256
import json
import unicodedata
from collections.abc import Iterable

DEFAULT_LEVELS = ("Country", "State", "Station")
MAX_DEPTH = 32


def _clean(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def parse_path(text: str) -> list[dict[str, str]]:
    """Read one parent per line, optionally prefixed by its level (Region: NRW)."""
    path = []
    for line in text.splitlines():
        if not line.strip():
            continue
        level, separator, name = line.partition(":")
        if not separator:
            index = len(path)
            level = DEFAULT_LEVELS[index] if index < len(DEFAULT_LEVELS) else "Group"
            name = line
        node = {"level": _clean(level), "name": _clean(name)}
        if not node["name"] or not node["level"] or len(node["name"]) > 100 or len(node["level"]) > 40:
            raise ValueError("Each parent needs a name (up to 100 characters) and a level (up to 40).")
        path.append(node)
    if len(path) > MAX_DEPTH:
        raise ValueError(f"A station can have up to {MAX_DEPTH} parent levels.")
    return path


def group_key(names: Iterable[str]) -> str:
    parts = [_clean(name).casefold() for name in names]
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return "group-" + sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_tree(stations: dict) -> list[dict]:
    """Return nested groups and source stations without mixing their namespaces."""
    roots: list[dict] = []
    groups: dict[str, dict] = {}
    for station, entry in stations.items():
        children = roots
        prefix: list[str] = []
        for part in entry.get("hierarchy_path") or []:
            prefix.append(part["name"])
            key = group_key(prefix)
            node = groups.get(key)
            if node is None:
                node = {"id": key, "name": part["name"], "level": part["level"],
                        "children": [], "leaf_count": 0}
                groups[key] = node
                children.append(node)
            node["leaf_count"] += 1
            children = node["children"]
        children.append({
            "id": station, "name": entry.get("display_name") or station,
            "level": "Substation" if prefix else "Station", "children": [], "leaf_count": 1,
        })
    for node in groups.values():
        node["children"].sort(key=lambda child: child["name"].casefold())
    return sorted(roots, key=lambda node: node["name"].casefold())


def group_stations(stations: dict, key: str) -> dict:
    """Resolve leaves only; a measurement enters every ancestor at most once."""
    members = {}
    for station, entry in stations.items():
        prefix: list[str] = []
        for part in entry.get("hierarchy_path") or []:
            prefix.append(part["name"])
            if group_key(prefix) == key:
                members[station] = entry
                break
    return members


def validate_path(stations: dict, path: list[dict[str, str]], interval: int,
                  grid_offset_seconds: int | None = None) -> list[dict[str, str]]:
    """Reuse existing group names and enforce comparable time grids within a tree."""
    if not path:
        return path
    root = group_key([path[0]["name"]])
    members = group_stations(stations, root)
    canonical = [part.copy() for part in path]
    for entry in members.values():
        if int(entry.get("interval_minutes", 15)) != interval:
            raise ValueError("Substations under the same root must use the same observation interval.")
        offset = entry.get("hierarchy_grid_offset_seconds")
        if grid_offset_seconds is not None and offset is not None and offset != grid_offset_seconds:
            raise ValueError("Substation timestamps must fall on the same UTC interval grid as the existing substations.")
        for index, (new, old) in enumerate(zip(path, entry["hierarchy_path"])):
            if new["name"].casefold() != old["name"].casefold():
                break
            if new["level"].casefold() != old["level"].casefold():
                raise ValueError(f"{old['name']} is already registered at the {old['level']} level.")
            canonical[index] = old.copy()
    return canonical


def station_label(station: str, entry: dict) -> str:
    parts = [part["name"] for part in entry.get("hierarchy_path") or []]
    return " / ".join([*parts, entry.get("display_name") or station])

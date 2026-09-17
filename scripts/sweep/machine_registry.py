"""Canonical machine names and descriptions for every benchmark report."""
import fnmatch
import sys
import tomllib
from pathlib import Path

# Machine registry — which sweep names are the same machine
# ---------------------------------------------------------------------------

REGISTRY_NAME = "machines.toml"
_REGISTRY_KEYS = {"name", "description", "names", "hosts", "specs"}


def find_registry(results_dir: Path | None, inputs: list[Path]) -> Path | None:
    """Look for machines.toml beside the results, then one level up."""
    base = results_dir or (inputs[0].parent if inputs else None)
    if base is None:
        return None
    for candidate in (base / REGISTRY_NAME, base.parent / REGISTRY_NAME):
        if candidate.is_file():
            return candidate
    return None


def load_registry(path: Path | None) -> list[dict]:
    """Read the registry, refusing anything malformed rather than guessing."""
    if path is None:
        return []
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise SystemExit(f"{path}: {e}")

    entries = data.get("machine", [])
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: expected a list of [[machine]] entries")

    registry: list[dict] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        name = entry.get("name")
        if not name:
            raise SystemExit(f"{path}: machine #{position} has no name")
        if name in seen:
            raise SystemExit(f"{path}: two machines are named {name}")
        seen.add(name)
        unknown = sorted(set(entry) - _REGISTRY_KEYS)
        if unknown:
            print(f"Warning: {path}: {name}: ignoring unknown key(s) {', '.join(unknown)}",
                  file=sys.stderr)
        registry.append({
            "name": name,
            "description": str(entry.get("description", "")),
            "specs": {str(k): str(v) for k, v in (entry.get("specs") or {}).items()},
            "names": [str(x) for x in entry.get("names", [])],
            "hosts": [str(x) for x in entry.get("hosts", [])],
        })
    return registry


def match_registry(registry: list[dict], name: str, host: str) -> dict | None:
    for entry in registry:
        for pattern in [entry["name"], *entry["names"]]:
            if fnmatch.fnmatch(name.lower(), pattern.lower()):
                return entry
        for pattern in entry["hosts"]:
            if host and fnmatch.fnmatch(host.lower(), pattern.lower()):
                return entry
    return None


def machine_id(registry, name, host="", run_id=""):
    """Resolve display identity without changing the recorded host or run."""
    entry = match_registry(registry, name or "", host or "")
    if entry is None and run_id:
        entry = match_registry(registry, run_id, "")
    return entry["name"] if entry else name or host or run_id or "unknown"


def machine_catalog(registry):
    return {"version": 1, "machines": [{key: entry[key] for key in ("name", "description", "specs")}
            for entry in registry]}

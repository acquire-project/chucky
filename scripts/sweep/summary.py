"""Condense sweep result files into the payload the overview page draws.

One results file is one sweep: one machine, one commit. The overview needs every
sweep at once, so it gets a trimmed copy of each run (config plus a few numbers)
and drops the per-stage detail, which only the explorer page uses.
"""

from __future__ import annotations

import fnmatch
import re
import sys
import tomllib
from pathlib import Path

from models import retired_metrics

# <machine>-<commit>-<yyyymmdd>.json, the name sweep.py writes. The machine part
# is the name a person chose, so it survives a cluster handing out a new hostname
# for every allocation.
FILENAME_RE = re.compile(r"^(?P<machine>.+)-(?P<commit>[0-9a-f]{7,40})-(?P<date>\d{8})$")

# Bumped to 2 when report.py started packing the runs into columns.
OVERVIEW_VERSION = 2

CONFIG_KEYS = (
    "scenario", "codec", "fill", "backend", "dtype",
    "chunk_bytes", "chunk_bytes_label", "sink", "status",
)

RUN_METRICS = (
    "throughput_in_gibs", "throughput_out_gibs", "compression_fold",
    "input_gib", "compressed_gib", "elapsed_s", "wall_s", "init_s",
)

STALL_METRICS = (
    "max_append_ms", "peak_pending_mib", "backpressure_ms",
    "flush_stall_ms", "io_fence_ms",
)


def parse_filename(path: Path) -> tuple[str | None, str | None, str | None]:
    m = FILENAME_RE.match(path.stem)
    if not m:
        return None, None, None
    date = m["date"]
    return m["machine"], m["commit"], f"{date[:4]}-{date[4:6]}-{date[6:]}"


def machine_identity(path: Path, machine: dict) -> tuple[str, str]:
    """What this one sweep calls itself, and the commit it ran at.

    The file name wins: renaming a results file is how a sweep gets reassigned
    after the fact. `machine.name` is what sweep.py --machine recorded, and the
    hostname is the last resort.
    """
    name, commit, _ = parse_filename(path)
    return (
        name or machine.get("name") or machine.get("hostname") or path.stem,
        commit or machine.get("commit") or "unknown",
    )


# ---------------------------------------------------------------------------
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
        for pattern in entry["names"]:
            if fnmatch.fnmatch(name.lower(), pattern.lower()):
                return entry
        for pattern in entry["hosts"]:
            if host and fnmatch.fnmatch(host.lower(), pattern.lower()):
                return entry
    return None


def sweep_day(machine: dict, path: Path) -> str:
    date = str(machine.get("date", ""))[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return date
    return parse_filename(path)[2] or ""


def trim_run(run: dict) -> dict:
    out = {"id": run.get("id", "")}
    for key in CONFIG_KEYS:
        if key in run:
            out[key] = run[key]
    for key in RUN_METRICS:
        value = run.get(key)
        if isinstance(value, (int, float)):
            out[key] = value
    stalls = run.get("stalls")
    if isinstance(stalls, dict):
        kept = {k: stalls[k] for k in STALL_METRICS
                if isinstance(stalls.get(k), (int, float))}
        if kept:
            out["stalls"] = kept
    return out


def status_counts(runs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        status = run.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def summarize_sweep(path: Path, data: dict, registry: list[dict]) -> dict:
    machine = data.get("machine", {})
    member, commit = machine_identity(path, machine)
    host = machine.get("hostname", "")
    entry = match_registry(registry, member, host)
    runs = data.get("runs", [])
    return {
        "machine": entry["name"] if entry else member,
        "member": member,
        "description": entry["description"] if entry else "",
        "specs": entry["specs"] if entry else {},
        "registered": entry is not None,
        "commit": commit,
        "host": machine.get("hostname", ""),
        "gpu": machine.get("gpu", ""),
        "date": machine.get("date", ""),
        "day": sweep_day(machine, path),
        "filename": path.name,
        "version": data.get("version"),
        "migrated_from": data.get("migrated_from"),
        "retired": list(retired_metrics(data)),
        "counts": status_counts(runs),
        "runs": [trim_run(r) for r in runs],
    }


def build_summary(files: list[tuple[Path, dict]], registry: list[dict] | None = None) -> dict:
    """Every sweep, ordered oldest first so the overview can read it as history."""
    sweeps = [summarize_sweep(path, data, registry or []) for path, data in files]
    sweeps.sort(key=lambda s: (s["day"], s["date"], s["machine"]))

    machines: dict[str, dict] = {}
    for sweep in sweeps:
        entry = machines.setdefault(sweep["machine"], {
            "name": sweep["machine"],
            "description": sweep["description"],
            "specs": sweep["specs"],
            "registered": sweep["registered"],
            "gpus": [],
            "hosts": [],
            "members": [],
            "sweeps": 0,
            "first_day": sweep["day"],
            "last_day": sweep["day"],
        })
        entry["sweeps"] += 1
        entry["last_day"] = sweep["day"]
        for key, value in (("gpus", sweep["gpu"]), ("hosts", sweep["host"]),
                           ("members", sweep["member"])):
            if value and value not in entry[key]:
                entry[key].append(value)

    return {
        "version": OVERVIEW_VERSION,
        "machines": sorted(machines.values(), key=lambda m: m["name"].lower()),
        "sweeps": sweeps,
    }

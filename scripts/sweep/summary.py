"""Condense sweep result files into the payload the overview page draws.

One results file is one sweep: one machine, one commit. The overview needs every
sweep at once, so it gets a trimmed copy of each run (config plus a few numbers)
and drops the per-stage detail, which only the explorer page uses.
"""

from __future__ import annotations

import re
from pathlib import Path

from models import retired_metrics

# <machine>-<commit>-<yyyymmdd>.json, the name sweep.py writes. The machine part
# is the name a person chose, so it survives a cluster handing out a new hostname
# for every allocation.
FILENAME_RE = re.compile(r"^(?P<machine>.+)-(?P<commit>[0-9a-f]{7,40})-(?P<date>\d{8})$")

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
    """Machine name shown everywhere, and the commit the sweep ran at."""
    name, commit, _ = parse_filename(path)
    return (
        name or machine.get("hostname") or path.stem,
        commit or machine.get("commit") or "unknown",
    )


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


def summarize_sweep(path: Path, data: dict) -> dict:
    machine = data.get("machine", {})
    name, commit = machine_identity(path, machine)
    runs = data.get("runs", [])
    return {
        "machine": name,
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


def build_summary(files: list[tuple[Path, dict]]) -> dict:
    """Every sweep, ordered oldest first so the overview can read it as history."""
    sweeps = [summarize_sweep(path, data) for path, data in files]
    sweeps.sort(key=lambda s: (s["day"], s["date"], s["machine"]))

    machines: dict[str, dict] = {}
    for sweep in sweeps:
        entry = machines.setdefault(sweep["machine"], {
            "name": sweep["machine"],
            "gpu": sweep["gpu"],
            "hosts": [],
            "sweeps": 0,
            "first_day": sweep["day"],
            "last_day": sweep["day"],
        })
        entry["sweeps"] += 1
        entry["last_day"] = sweep["day"]
        entry["gpu"] = entry["gpu"] or sweep["gpu"]
        if sweep["host"] and sweep["host"] not in entry["hosts"]:
            entry["hosts"].append(sweep["host"])

    return {
        "version": 1,
        "machines": sorted(machines.values(), key=lambda m: m["name"].lower()),
        "sweeps": sweeps,
    }

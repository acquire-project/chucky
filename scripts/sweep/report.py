# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic",
# ]
# ///
"""
Generate the benchmark site from sweep result files.

Two pages, both self-contained (data embedded at generation time):
    index.html    every sweep at once — per-machine trend, latest standings, movers
    explore.html  one sweep at a time, down to per-stage timing

Usage:
    uv run scripts/sweep/report.py --results-dir bench/results/ -o _site
    uv run scripts/sweep/report.py bench/results/*.json -o _site
    uv run scripts/sweep/report.py --results-dir bench/results/ -o _site/index.html

Re-run after changing any results file; nothing is read at page load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from models import migrate_results, validate_results
from summary import build_summary, find_registry, load_registry, machine_identity

TEMPLATE_DIR = Path(__file__).parent
EXPLORER_TEMPLATE = TEMPLATE_DIR / "template.html"
OVERVIEW_TEMPLATE = TEMPLATE_DIR / "overview.html"
PLACEHOLDER = "__DATA_PLACEHOLDER__"


def load_files(paths: list[Path], *, warn: bool = True) -> list[tuple[Path, dict]]:
    """Read, migrate, and validate result files, skipping ones that will not parse."""
    loaded: list[tuple[Path, dict]] = []
    for p in paths:
        with open(p) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping corrupt JSON file {p}: {e}", file=sys.stderr)
                continue

        migrate_results(data)

        if warn:
            try:
                validate_results(data)
            except ValidationError as e:
                for err in e.errors():
                    loc = " -> ".join(str(x) for x in err["loc"])
                    print(f"Warning: {p.name}: {loc}: {err['msg']}", file=sys.stderr)

        loaded.append((p, data))
    return loaded


def explorer_payload(loaded: list[tuple[Path, dict]]) -> dict:
    """What the one-sweep-at-a-time page needs: full runs, including stages."""
    files = []
    for path, data in loaded:
        machine = data.get("machine", {})
        name, commit = machine_identity(path, machine)
        day = str(machine.get("date", ""))[:10]
        label = " ".join(x for x in [name, commit, day] if x) or "unknown"
        files.append({
            "label": label,
            "filename": path.name,
            "machine": machine,
            "runs": data.get("runs", []),
        })
    return {"version": 2, "files": files}


def write_page(template: Path, payload: dict, output: Path) -> None:
    text = template.read_text()
    if PLACEHOLDER not in text:
        raise SystemExit(f"{template.name} has no {PLACEHOLDER} to fill")
    embedded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.replace(PLACEHOLDER, embedded))
    print(f"Wrote {output} ({output.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Generate the benchmark site from sweep results")
    ap.add_argument("input", type=Path, nargs="*", help="Result JSON file(s) from sweep.py")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Directory to glob for *.json result files")
    ap.add_argument("-o", "--output", type=Path, default=Path("build/html"),
                    help="Output directory (a path ending in .html names the overview page)")
    ap.add_argument("--machines", type=Path, default=None,
                    help="Machine registry TOML (default: machines.toml beside the results)")
    args = ap.parse_args()

    paths: list[Path] = list(args.input or [])
    if args.results_dir:
        paths.extend(sorted(args.results_dir.glob("*.json")))
    if not paths:
        ap.error("No input files. Provide paths or use --results-dir.")

    loaded = load_files(paths)
    if not loaded:
        raise SystemExit("No readable result files.")
    total_runs = sum(len(data.get("runs", [])) for _, data in loaded)
    print(f"Loaded {len(loaded)} file(s), {total_runs} runs", file=sys.stderr)

    registry_path = args.machines or find_registry(args.results_dir, paths)
    if args.machines and not args.machines.is_file():
        raise SystemExit(f"No machine registry at {args.machines}")
    registry = load_registry(registry_path)
    if registry_path:
        print(f"Machine registry: {registry_path} ({len(registry)} machines)", file=sys.stderr)
    else:
        print("No machine registry found; each sweep name is its own machine", file=sys.stderr)

    if args.output.suffix == ".html":
        out_dir, overview_name = args.output.parent, args.output.name
    else:
        out_dir, overview_name = args.output, "index.html"

    write_page(OVERVIEW_TEMPLATE, build_summary(loaded, registry), out_dir / overview_name)
    write_page(EXPLORER_TEMPLATE, explorer_payload(loaded), out_dir / "explore.html")


if __name__ == "__main__":
    main()

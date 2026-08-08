# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic",
# ]
# ///
"""
Generate the benchmark site from sweep result files.

Two pages:
    index.html    every sweep at once — per-machine trend, latest standings, movers
    explore.html  one sweep at a time, down to per-stage timing

The pages are code only; their data sits beside them and is fetched at load:
    data/overview.json          every sweep, trimmed, for the overview
    data/sweeps.json            the sweep list the explorer offers
    data/sweeps/<result>.json   one sweep in full, fetched when it is opened

That keeps the explorer from downloading every sweep to show one, and lets an
unchanged sweep revalidate instead of being re-sent. It also means the pages
need to be served over http — opening them from a file:// path will not work.

Usage:
    uv run scripts/sweep/report.py --results-dir bench/results/ -o _site
    uv run scripts/sweep/report.py bench/results/*.json -o _site
    uv run scripts/sweep/report.py --results-dir bench/results/ -o _site/index.html
    python -m http.server -d _site      # then open http://localhost:8000/

Re-run after changing any results file.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import ValidationError

from columnar import EXPLORER_OMITS, StringTable, decode_runs, encode_runs, rounded
from models import migrate_results, validate_results
from summary import build_summary, find_registry, load_registry, machine_identity

TEMPLATE_DIR = Path(__file__).parent
EXPLORER_TEMPLATE = TEMPLATE_DIR / "template.html"
OVERVIEW_TEMPLATE = TEMPLATE_DIR / "overview.html"


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


def explorer_index(loaded: list[tuple[Path, dict]]) -> dict:
    """The sweep list the explorer shows before anything is opened."""
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
        })
    return {"version": 3, "files": files}


def check_roundtrip(runs: list[dict], block: dict, strings: list[str],
                    where: str, omit: tuple[str, ...] = ()) -> None:
    """Refuse to publish a packed sweep the pages would unpack differently."""
    unpacked = decode_runs(block, strings)
    if len(unpacked) != len(runs):
        raise SystemExit(f"{where}: packed {len(runs)} runs, unpacked {len(unpacked)}")
    for original, restored in zip(runs, unpacked):
        want = rounded(original, omit)
        if restored != want:
            differing = sorted(set(want) ^ set(restored)) or \
                sorted(k for k in want if want[k] != restored.get(k))
            raise SystemExit(f"{where}: packing changed run "
                             f"{original.get('id', '?')}: {differing}")


def write_json(payload: dict, output: Path) -> int:
    # allow_nan would write Infinity, which JSON.parse rejects — better to stop
    # here than to publish a page that cannot read its own data.
    try:
        text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    except ValueError as e:
        raise SystemExit(f"{output}: {e}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return output.stat().st_size


def write_page(template: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.read_text())
    print(f"Wrote {output} ({output.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)


def write_data(loaded: list[tuple[Path, dict]], registry: list[dict], data_dir: Path) -> None:
    overview = build_summary(loaded, registry)
    strings = StringTable([s["runs"] for s in overview["sweeps"]])
    for sweep in overview["sweeps"]:
        packed = encode_runs(sweep["runs"], strings)
        check_roundtrip(sweep["runs"], packed, strings.items, sweep["filename"])
        sweep["runs"] = packed
    overview["version"] = 2
    overview["strings"] = strings.items
    total = write_json(overview, data_dir / "overview.json")
    print(f"Wrote {data_dir / 'overview.json'} ({total / 1024:.0f} KiB)", file=sys.stderr)

    write_json(explorer_index(loaded), data_dir / "sweeps.json")
    biggest = 0
    for path, data in loaded:
        runs = data.get("runs", [])
        sweep_strings = StringTable([runs], EXPLORER_OMITS)
        packed = encode_runs(runs, sweep_strings, EXPLORER_OMITS)
        check_roundtrip(runs, packed, sweep_strings.items, path.name, EXPLORER_OMITS)
        payload = {"version": 3, "strings": sweep_strings.items, "runs": packed}
        biggest = max(biggest, write_json(payload, data_dir / "sweeps" / path.name))
    print(f"Wrote {len(loaded)} sweep file(s) to {data_dir / 'sweeps'} "
          f"(largest {biggest / 1024:.0f} KiB)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Generate the benchmark site from sweep results")
    ap.add_argument("input", type=Path, nargs="*", help="Result JSON file(s) from sweep.py")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Directory to glob for *.json result files")
    ap.add_argument("-o", "--output", type=Path, default=Path("build/html"),
                    help="Output directory (a path ending in .html names the overview page)")
    ap.add_argument("--machines", type=Path, default=None,
                    help="Machine registry TOML (default: machines.toml beside the results)")
    ap.add_argument("--serve", nargs="?", type=int, const=8000, default=None,
                    metavar="PORT",
                    help="Serve the site after writing it (default port 8000). The pages "
                         "fetch their data, so they need this rather than a file:// path.")
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

    if args.machines and not args.machines.is_file():
        raise SystemExit(f"No machine registry at {args.machines}")
    registry_path = args.machines or find_registry(args.results_dir, paths)
    registry = load_registry(registry_path)
    if registry_path:
        print(f"Machine registry: {registry_path} ({len(registry)} machines)", file=sys.stderr)
    else:
        print("No machine registry found; each sweep name is its own machine", file=sys.stderr)

    if args.output.suffix == ".html":
        out_dir, overview_name = args.output.parent, args.output.name
    else:
        out_dir, overview_name = args.output, "index.html"

    write_page(OVERVIEW_TEMPLATE, out_dir / overview_name)
    write_page(EXPLORER_TEMPLATE, out_dir / "explore.html")
    write_data(loaded, registry, out_dir / "data")

    if args.serve is not None:
        serve(out_dir, overview_name, args.serve)


def serve(out_dir: Path, overview_name: str, port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(out_dir))
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        raise SystemExit(f"Cannot serve on port {port}: {e}")
    print(f"\nServing {out_dir} at http://127.0.0.1:{port}/{overview_name}\n"
          f"Ctrl-C to stop.", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()

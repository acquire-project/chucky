# /// script
# requires-python = ">=3.11"
# ///
"""
Generate a self-contained HTML+D3 report from benchmark sweep results.

Usage:
    uv run scripts/sweep/report.py results.json -o build/html/report.html
    uv run scripts/sweep/report.py bench/results/*.json
    uv run scripts/sweep/report.py --results-dir bench/results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def load_files(paths: list[Path]) -> list[dict]:
    """Load result files, returning a list of {label, machine, runs} dicts."""
    files = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        machine = data.get("machine", {})
        commit = machine.get("commit", "unknown")
        date = machine.get("date", "")[:10]
        label = f"{commit} ({date})" if date else commit
        files.append({
            "label": label,
            "filename": p.name,
            "machine": machine,
            "runs": data.get("runs", []),
        })
    return files


def main():
    ap = argparse.ArgumentParser(description="Generate HTML report from sweep results")
    ap.add_argument("input", type=Path, nargs="*", help="Result JSON file(s) from sweep.py")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Directory to glob for *.json result files")
    ap.add_argument("-o", "--output", type=Path, default=Path("build/html/report.html"),
                    help="Output HTML path")
    args = ap.parse_args()

    paths: list[Path] = list(args.input or [])
    if args.results_dir:
        paths.extend(sorted(args.results_dir.glob("*.json")))
    if not paths:
        ap.error("No input files. Provide paths or use --results-dir.")

    files = load_files(paths)
    total_runs = sum(len(f["runs"]) for f in files)
    print(f"Loaded {len(files)} file(s), {total_runs} runs", file=sys.stderr)

    data = {"version": 2, "files": files}

    template = TEMPLATE_PATH.read_text()
    json_str = json.dumps(data).replace("</", "<\\/")
    html = template.replace("__DATA_PLACEHOLDER__", json_str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KiB)",
          file=sys.stderr)


if __name__ == "__main__":
    main()

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


def load_and_merge(paths: list[Path]) -> dict:
    """Load one or more result files and merge into a single data dict."""
    if not paths:
        print("No input files.", file=sys.stderr)
        sys.exit(1)

    all_runs = []
    machine = None

    for p in paths:
        with open(p) as f:
            data = json.load(f)
        file_machine = data.get("machine", {})
        file_commit = file_machine.get("commit", "unknown")

        if machine is None:
            machine = file_machine

        for run in data.get("runs", []):
            if "commit" not in run:
                run["commit"] = file_commit
            all_runs.append(run)

    return {
        "version": 1,
        "machine": machine,
        "runs": all_runs,
    }


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

    data = load_and_merge(paths)
    print(f"Loaded {len(paths)} file(s), {len(data['runs'])} runs", file=sys.stderr)

    template = TEMPLATE_PATH.read_text()
    # Escape closing tags to prevent XSS via </script> in benchmark output
    json_str = json.dumps(data).replace("</", "<\\/")
    html = template.replace("__DATA_PLACEHOLDER__", json_str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KiB)",
          file=sys.stderr)


if __name__ == "__main__":
    main()

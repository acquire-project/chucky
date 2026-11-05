# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic"]
# ///
"""Write migrated copies of sweep files; report.py also migrates on read."""

import argparse
import json
from pathlib import Path

from image_results import image_sweep
from models import migrate_results, validate_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepared = {}
    # Validate the whole set before creating output files.
    for source in args.inputs:
        target = args.output_dir / source.name
        if target.exists() or target in prepared:
            parser.error(f"Output already exists or has a duplicate name: {target}")
        data = json.loads(source.read_text())
        if data.get("benchmark") == "microscopy-images":
            data = image_sweep(data)
        elif "runs" not in data:
            parser.error(f"Not a sweep file: {source}; retained studies migrate when reported")
        migrate_results(data)
        validate_results(data)
        prepared[target] = json.dumps(data, indent=2, allow_nan=False) + "\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target, content in prepared.items():
        with target.open("x") as output:
            output.write(content)
        print(target)


if __name__ == "__main__":
    main()

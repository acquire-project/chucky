# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ome-zarr-models>=1.6",
# ]
# ///
"""Validate OME-NGFF metadata in zarr stores.

Usage: uv run tests/validate_ome_ngff.py <store_path> [<store_path> ...]

Validates each store against the OME-NGFF v0.5 spec using ome-zarr-models.
Exits 0 if all stores are valid, 1 if any fail.
"""

import sys
from pathlib import Path

from ome_zarr_models import open_ome_zarr


def validate_store(store_path: Path, group_path: str = "") -> bool:
    label = f"{store_path.name}/{group_path}" if group_path else store_path.name
    full = str(store_path / group_path) if group_path else str(store_path)
    try:
        group = open_ome_zarr(full)
        print(f"  PASS {label}: {type(group).__name__}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"  FAIL {label}: {e}", file=sys.stderr)
        return False


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <store_path> [...]", file=sys.stderr)
        sys.exit(1)

    ok = True
    for arg in sys.argv[1:]:
        # arg can be "path" or "path:group" to validate a subgroup
        if ":" in arg:
            path, group = arg.rsplit(":", 1)
        else:
            path, group = arg, ""
        if not validate_store(Path(path), group):
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

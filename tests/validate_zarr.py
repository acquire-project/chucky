# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr>=3",
#   "numcodecs",
# ]
# ///
"""Validate zarr stores written by test_zarr_readback.

Usage: uv run tests/validate_zarr.py <tmpdir> <nt> <ny> <nx>

Each subdirectory of <tmpdir> is a zarr store with array "0".
Validates shape, dtype, and data values for each codec variant.
"""

import sys
from pathlib import Path

import numpy as np
import zarr


def validate_store(store_path: Path, nt: int, ny: int, nx: int) -> bool:
    name = store_path.name
    array_path = store_path / "0"
    if not array_path.exists():
        print(f"  FAIL {name}: array path {array_path} not found", file=sys.stderr)
        return False

    try:
        store = zarr.open_group(str(store_path), mode="r")
        arr = store["0"]
    except Exception as e:
        print(f"  FAIL {name}: could not open: {e}", file=sys.stderr)
        return False

    # Check shape (t is append dim, should be nt)
    expected_shape = (nt, ny, nx)
    if arr.shape != expected_shape:
        print(
            f"  FAIL {name}: shape {arr.shape} != expected {expected_shape}",
            file=sys.stderr,
        )
        return False

    # Check dtype
    if arr.dtype != np.uint16:
        print(f"  FAIL {name}: dtype {arr.dtype} != uint16", file=sys.stderr)
        return False

    # Check values: src[i] = i & 0xFFFF
    data = arr[:]
    expected = np.arange(nt * ny * nx, dtype=np.uint16).reshape(nt, ny, nx)
    if not np.array_equal(data, expected):
        mismatches = np.sum(data != expected)
        print(
            f"  FAIL {name}: {mismatches} mismatched values",
            file=sys.stderr,
        )
        return False

    print(f"  PASS {name}", file=sys.stderr)
    return True


def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <tmpdir> <nt> <ny> <nx>", file=sys.stderr)
        sys.exit(1)

    tmpdir = Path(sys.argv[1])
    nt, ny, nx = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    stores = sorted(p for p in tmpdir.iterdir() if p.is_dir())
    if not stores:
        print(f"No stores found in {tmpdir}", file=sys.stderr)
        sys.exit(1)

    ok = True
    for store in stores:
        if not validate_store(store, nt, ny, nx):
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

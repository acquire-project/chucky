# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr>=3",
#   "numcodecs",
#   "tensorstore>=0.1.76",
#   "ome-zarr-models>=1.6",
# ]
# ///
"""Read every NGFF level and exercise the reader's missing/corrupt-data checks.

The affine reference and metadata checks are shared by the CPU/GPU fixtures
and can also be used by the streaming S3 readback test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

from validate_ome_ngff import validate_store as validate_ome_store
from validate_zarr import validate_tensorstore


def expected_level(level: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Analytic mean of each spatial block of the input affine ramp.

    Mean coordinates of a block of width s are s*i + (s-1)/2. Applying
    coefficients 8 and 4 gives a combined offset of 6*(s-1), exactly integral.
    This reference neither reads L0 nor uses Chucky's downsampling routines.
    """
    nt, ny, nx = shape
    scale = 2**level
    t = np.arange(nt, dtype=np.uint32)[:, None, None]
    y = np.arange(ny // scale, dtype=np.uint32)[None, :, None]
    x = np.arange(nx // scale, dtype=np.uint32)[None, None, :]
    return (1 + 1024 * t + 8 * scale * y + 4 * scale * x + 6 * (scale - 1)).astype(
        np.uint16
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_pyramid(group_path: Path, shape: tuple[int, int, int]) -> None:
    require(validate_ome_store(group_path), f"Invalid NGFF metadata: {group_path}")
    metadata = json.loads((group_path / "zarr.json").read_text())
    ome = metadata["attributes"]["ome"]
    require(ome["version"] == "0.5", "Expected OME-NGFF 0.5")
    require(len(ome["multiscales"]) == 1, "Expected one multiscale image")
    multiscale = ome["multiscales"][0]
    require(
        multiscale["axes"]
        == [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "Incorrect axis names, order, types, or units",
    )
    datasets = multiscale["datasets"]
    require(len(datasets) == 3, "Expected three resolution levels")
    require(len({d["path"] for d in datasets}) == 3, "Duplicate dataset paths")
    group = zarr.open_group(str(group_path), mode="r")
    for level, dataset in enumerate(datasets):
        scale = 2**level
        require(
            dataset["coordinateTransformations"]
            == [{"type": "scale", "scale": [0.25, 0.5 * scale, 0.75 * scale]}],
            f"Incorrect coordinate transform at level {level}",
        )
        # Follow the metadata's paths rather than assuming names like 0/1/2.
        path = dataset["path"]
        expected = expected_level(level, shape)
        arr = group[path]
        require(arr.shape == expected.shape, f"{path}: incorrect zarr-python shape")
        require(arr.dtype == expected.dtype, f"{path}: incorrect zarr-python dtype")
        np.testing.assert_array_equal(arr[:], expected, err_msg=f"zarr-python: {path}")
        validate_tensorstore(group_path / path, expected)
    print(f"  PASS {group_path.parent.name}: all three NGFF levels", file=sys.stderr)


def expect_rejected(label, check):
    try:
        check()
    except (ValueError, AssertionError):
        print(f"  PASS rejected {label}", file=sys.stderr)
        return
    raise AssertionError(f"Validator accepted {label}")


def check_reader_failures(group_path: Path, shape: tuple[int, int, int]) -> None:
    """Prove that every level is read, including the final partial shard."""
    metadata_path = group_path / "zarr.json"
    original_metadata = metadata_path.read_bytes()
    metadata = json.loads(original_metadata)
    datasets = metadata["attributes"]["ome"]["multiscales"][0]["datasets"]
    for level, dataset in enumerate(datasets):
        array_path = group_path / dataset["path"]
        expected = expected_level(level, shape)
        shard = sorted(p for p in (array_path / "c").rglob("*") if p.is_file())[-1]
        original = shard.read_bytes()
        try:
            shard.unlink()
            expect_rejected(
                f"missing level {level} shard",
                lambda: validate_tensorstore(array_path, expected),
            )
            for offset, label in ((0, "payload"), (-1, "index checksum")):
                corrupt = bytearray(original)
                corrupt[offset] ^= 1
                shard.write_bytes(corrupt)
                expect_rejected(
                    f"corrupt level {level} {label}",
                    lambda: validate_tensorstore(array_path, expected),
                )
        finally:
            shard.write_bytes(original)

    # A valid but incorrect physical scale must fail our semantic checks.
    try:
        datasets[2]["coordinateTransformations"][0]["scale"][1] *= 2
        metadata_path.write_text(json.dumps(metadata))
        expect_rejected("incorrect NGFF scale", lambda: validate_pyramid(group_path, shape))
    finally:
        metadata_path.write_bytes(original_metadata)

    # A renamed level still works when its advertised path is updated.
    metadata = json.loads(original_metadata)
    dataset = metadata["attributes"]["ome"]["multiscales"][0]["datasets"][2]
    original_path = group_path / dataset["path"]
    renamed_path = group_path / "coarse"
    original_path.rename(renamed_path)
    try:
        dataset["path"] = "coarse"
        metadata_path.write_text(json.dumps(metadata))
        validate_pyramid(group_path, shape)
    finally:
        renamed_path.rename(original_path)
        metadata_path.write_bytes(original_metadata)
    validate_pyramid(group_path, shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("nt", type=int)
    parser.add_argument("ny", type=int)
    parser.add_argument("nx", type=int)
    parser.add_argument("expected_stores", type=int)
    args = parser.parse_args()
    shape = (args.nt, args.ny, args.nx)
    stores = sorted(p for p in args.directory.iterdir() if p.is_dir())
    require(len(stores) == args.expected_stores > 0, "Missing readback fixtures")
    for store in stores:
        validate_pyramid(store / "pyramid", shape)
    # CODEC_NONE makes payload corruption independent of decoder behavior.
    check_reader_failures(args.directory / "none" / "pyramid", shape)


if __name__ == "__main__":
    main()

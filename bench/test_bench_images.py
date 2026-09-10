"""Image replay shares coverage and reports logical/padded work independently."""

from array import array
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

exe, backend = sys.argv[1:]
height, width, planes = 66, 65, 3
frame = height * width
source = array("H", ((p * 1000 + y * width + x) % 65536
                     for p in range(planes) for y in range(height) for x in range(width)))
if sys.byteorder != "little":
    source.byteswap()

with tempfile.TemporaryDirectory(prefix="chucky-image-window-") as directory:
    root = Path(directory)
    raw = root / "input.raw"
    raw.write_bytes(source.tobytes())
    base = [exe, "--backend", backend, "--input", str(raw),
            "--width", str(width), "--height", str(height),
            "--codec", "none", "--chunk-bytes", "32K", "--geometry-frames", "64",
            "--batch-bytes", "1M", "--max-threads", "2", "--warmup", "0",
            "--duration", "0.01", "--json"]
    geometry = None
    for append in (17, 100000):
        process = subprocess.run([*base, "--frames", "7", "--append-elements", str(append)],
                                 capture_output=True, text=True, timeout=60)
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout)
        window, replay = result["measurement"], result["image_replay"]
        assert window["coverage_status"] == "sufficient"
        assert window["warmup_complete_batches"] >= 2 and window["complete_batches"] >= 4
        assert window["generation_transitions"] >= 2 and window["drain_fraction"] <= 0.1
        assert geometry is None or window["geometry"] == geometry
        geometry = window["geometry"]
        cy, cx = replay["chunk_shape"][1:]
        padded_frame = math.ceil(height / cy) * cy * math.ceil(width / cx) * cx
        measured = replay["shape"][0]
        assert measured >= 7
        assert result["submitted_bytes"] == window["input_bytes"] == measured * padded_frame * 2
        assert result["logical_input_bytes"] == measured * frame * 2
        assert result["logical_input_bytes"] < result["submitted_bytes"]
        assert math.isclose(result["throughput_in_gibs"],
                            result["submitted_bytes"] / 2**30 / window["elapsed_s"], rel_tol=1e-5)
        assert math.isclose(result["throughput_logical_gibs"],
                            result["logical_input_bytes"] / 2**30 / window["elapsed_s"], rel_tol=1e-5)
        assert replay["source_bytes"] == len(source) * 2
        assert window["source_bytes"] == planes * padded_frame * 2
        assert replay["reference_shape"] == [64, height, width]

    # Compression keeps this filesystem metadata/index readback small.
    output = root / "output"
    process = subprocess.run([*base, "--codec", "zstd", "--frames", "7",
                              "-o", str(output), "--no-boundary-timing"],
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    window = result["measurement"]
    metadata = json.loads((output / "images" / "zarr.json").read_text())
    total_frames = (window["warmup_input_bytes"] + window["input_bytes"]) // (padded_frame * 2)
    assert metadata["shape"] == [total_frames, height, width]
    assert window["boundary_timing"] is False
    assert all(group["crossing"]["calls"] == 0 for group in window["boundaries"].values())
    # Every published chunk has a valid index extent, including the last shard.
    config = metadata["codecs"][0]["configuration"]
    entries = math.prod(s // c for s, c in zip(
        metadata["chunk_grid"]["configuration"]["chunk_shape"], config["chunk_shape"]))
    chunks = 0
    for shard in (output / "images" / "c").rglob("*"):
        if not shard.is_file():
            continue
        data = shard.read_bytes()
        for offset, size in struct.iter_unpack("<QQ", data[-(entries * 16 + 4):-4]):
            if offset != 2**64 - 1:
                assert size > 0 and offset + size <= len(data)
                chunks += 1
    assert chunks == math.prod(math.ceil(s / c) for s, c in zip(
        metadata["shape"], config["chunk_shape"]))

print(f"Image source wrapping, accounting, geometry and final metadata passed ({backend})")

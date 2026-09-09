"""Accounting tests use exact work totals; elapsed times only bound windows."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

exe, backend, multiscale = sys.argv[1:]
base = [exe, "--backend", backend, "--codec", "none", "--geometry-frames", "65536",
        "--chunk-bytes", "1M", "--batch-bytes", "1M",
        "--memory-budget", "1G", "--max-threads", "4", "--json"]


def run(*args):
    return subprocess.run([*base, *args], capture_output=True, text=True, timeout=60)


def check(p):
    assert p.returncode == 0, p.stderr
    report = json.loads(p.stdout)
    assert report["status"] == "pass"
    w = report["measurement"]
    assert w["policy"] == "drained-warmup-through-final-close-v1"
    assert w["warmup_s"] >= w["requested_warmup_s"]
    # A synchronous empty drain can finish within one platform clock tick.
    assert w["drain_s"] >= 0
    assert math.isclose(w["elapsed_s"], w["append_s"] + w["drain_s"], rel_tol=1e-5)
    assert math.isclose(report["wall_s"], w["elapsed_s"], rel_tol=1e-5)
    assert math.isclose(w["drain_fraction"], w["drain_s"] / w["elapsed_s"], rel_tol=1e-5)
    assert math.isclose(report["input_gib"], w["input_bytes"] / 2**30, rel_tol=1e-5)
    for direction, key in (("in", "input"), ("out", "output")):
        expected = w[f"{key}_bytes"] / 2**30 / w["elapsed_s"]
        assert math.isclose(w[f"throughput_{direction}_gibs"], expected, rel_tol=1e-5)
        assert math.isclose(report[f"throughput_{direction}_gibs"], expected, rel_tol=1e-5)
    assert w["complete_batches"] == w["input_bytes"] // w["boundaries"]["batch"]["bytes"]
    assert w["batch_reuses_lower_bound"] == max(0, w["complete_batches"] - 2)
    assert f'Coverage: {w["coverage_status"]}' in p.stderr
    section = p.stderr.split("--- Measurement window ---")[1].split("---")[0]
    assert all(len(line) <= 80 for line in section.splitlines()), section
    assert "Whole run" not in p.stderr
    for group in w["boundaries"].values():
        for key in ("crossing", "following"):
            sample = group[key]
            assert 0 <= sample["over_100ms"] <= sample["calls"]
            assert 0 <= sample["max_ms"] <= sample["total_ms"]
    return report, w


geometry = None
for append, warmup, frames in ((256, 0, 16384), (4096, 0.03, 65536),
                               (33554432, 0.03, 131072)):
    r, w = check(run("--frames", str(frames), "--warmup", str(warmup),
                     "--append-elements", str(append)))
    assert w["reference_frames"] == 65536
    assert w["input_bytes"] == frames * 512
    assert w["source_bytes"] == 64 << 20
    assert w["append_bytes"] == append * 2
    assert geometry is None or geometry == w["geometry"]
    geometry = w["geometry"]
    # Whole-run metrics would overcount here when warmup is nonzero.
    assert r["stages"]["scatter"]["in_bytes"] == frames * 512
    if backend == "gpu":
        assert r["memcpy_work"]["bytes"] == frames * 512
        assert r["stages"]["h2d"]["in_bytes"] == frames * 512
    assert bool(w["warmup_input_bytes"]) == bool(warmup)
    assert w["coverage_status"] == "insufficient"

for duration in (0.02, 0.06):
    r, w = check(run("--frames", "0", "--duration", str(duration), "--warmup", "0.03"))
    assert w["append_s"] >= duration
    assert w["geometry"] == geometry
    assert w["input_bytes"] > 0

# Defaults must do more than the old one-append, 32 MiB smallepoch sweep.
r, w = check(run())
assert w["requested_warmup_s"] == 0.25 and w["requested_duration_s"] == 1
assert w["input_bytes"] > 32 << 20
assert w["coverage_status"] == "sufficient"

r, w = check(run("--frames", "1", "--warmup", "0", "--no-boundary-timing"))
assert w["input_bytes"] == 512
assert w["coverage_status"] == "insufficient"
assert w["boundary_timing"] is False
assert all(group["crossing"]["calls"] == group["following"]["calls"] == 0
           for group in w["boundaries"].values())

# Synthetic asynchronous IO's final drain belongs to the same denominator.
r, w = check(run("--frames", "16384", "--warmup", "0.01", "--io-bw-mbps", "100"))
assert w["drain_s"] > 0.01
assert w["drain_fraction"] > 0.1 and w["coverage_status"] == "insufficient"

for args in (("--duration", "0"), ("--duration", "nan"), ("--duration", "inf"),
             ("--duration", "-1"), ("--duration", "1x"),
             ("--duration", "1", "--warmup", "-1"),
             ("--append-elements", "3"), ("--append-elements", "bad"),
             ("--append-elements", "67108864"), ("--frames", "bad"),
             ("--frames", "-1"), ("--frames", "18446744073709551616"),
             ("--geometry-frames", "0"), ("--geometry-frames", "bad"),
             ("--frames", "10", "--duration", "1")):
    p = run(*args)
    assert p.returncode != 0, args

for mode in ("spatial", "append"):
    with tempfile.TemporaryDirectory(prefix="chucky-measurement-") as directory:
        p = subprocess.run([multiscale, mode, "--backend", backend, "--codec", "none",
                            "--fill", "zeros", "--batch-bytes", "256K", "--memory-budget", "1G",
                            "--frames", "33", "--warmup", "0.02", "--max-threads", "4",
                            "--json", "-o", directory], capture_output=True, text=True, timeout=60)
        r, w = check(p)
        assert w["input_bytes"] == 33 * 32 * 32 * 2
        assert w["warmup_input_bytes"] > 0
        assert r["stages"]["sink"]["in_bytes"] == w["output_bytes"]
        assert r["stages"]["sink"]["owner"] == "delivery"
        assert "Sink" in p.stderr
        levels = sorted(Path(directory).glob("multiscale/*/zarr.json"))
        assert len(levels) > 1
        frames = (w["warmup_input_bytes"] + w["input_bytes"]) // (32 * 32 * 2)
        for lv, metadata in enumerate(levels):
            m = json.loads(metadata.read_text())
            expected_frames = math.ceil(frames / 2**lv) if mode == "append" else frames
            assert m["shape"] == [expected_frames, 32 >> lv, 32 >> lv], m
            # For zeros/none every indexed chunk must be readable and zero,
            # across the warmup checkpoint as well as the final partial batch.
            shard_shape = m["chunk_grid"]["configuration"]["chunk_shape"]
            chunk_shape = m["codecs"][0]["configuration"]["chunk_shape"]
            chunks_per_shard = math.prod(s // c for s, c in zip(shard_shape, chunk_shape))
            seen = 0
            for shard in metadata.parent.glob("c/**/*"):
                if not shard.is_file():
                    continue
                data = shard.read_bytes()
                index = data[-(chunks_per_shard * 16 + 4):-4]
                for offset, size in struct.iter_unpack("<QQ", index):
                    if offset == 2**64 - 1:
                        continue
                    assert 0 < size <= len(data) and offset + size <= len(data)
                    assert not any(data[offset:offset + size])
                    seen += 1
            expected_chunks = math.prod(math.ceil(s / c) for s, c in zip(m["shape"], chunk_shape))
            assert seen == expected_chunks, (seen, expected_chunks, metadata)

print(f"Common-window accounting, geometry, multiscale readback and CLI passed ({backend})")

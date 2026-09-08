import json
import math
from pathlib import Path
import subprocess
import sys


exe, backend = sys.argv[1:]
base = [exe, "--backend", backend, "--codec", "none", "--frames", "65536",
        "--chunk-bytes", "1M", "--batch-bytes", "1M",
        "--memory-budget", "1G", "--json"]


def run(*args):
    return subprocess.run([*base, *args], capture_output=True, text=True, timeout=60)


geometry = None
for append, warmup in ((256, 0), (4096, 0.05), (33554432, 0)):
    p = run("--duration", "0.15", "--warmup", str(warmup),
            "--append-elements", str(append))
    assert p.returncode == 0, p.stderr
    report = json.loads(p.stdout)
    assert report["status"] == "pass"
    window = report["sustained"]
    assert window["boundary_timing"] is True
    assert window["reference_frames"] == 65536
    assert window["source_bytes"] == 64 << 20
    assert window["append_bytes"] == append * 2
    assert window["warmup_s"] >= warmup
    assert window["elapsed_s"] >= 0.15
    assert window["drain_s"] > 0
    assert window["input_bytes"] % (append * 2) == 0
    assert window["output_bytes"] > 0
    for direction, key in (("in", "input"), ("out", "output")):
        expected = window[f"{key}_bytes"] / 2**30 / window["elapsed_s"]
        assert math.isclose(window[f"throughput_{direction}_gibs"], expected, rel_tol=1e-5)
    total = round(report["input_gib"] * 2**30)
    if warmup == 0:
        assert math.isclose(total, window["input_bytes"], rel_tol=1e-5)
    else:
        assert total > window["input_bytes"]
    assert report["wall_s"] >= window["warmup_s"] + window["elapsed_s"]
    boundaries = window["boundaries"]
    sizes = {key: value["bytes"] for key, value in boundaries.items()}
    assert geometry is None or geometry == sizes
    geometry = sizes
    for group in boundaries.values():
        assert group["crossing"]["calls"] > 0
        for key in ("crossing", "following"):
            sample = group[key]
            assert 0 <= sample["over_100ms"] <= sample["calls"]
            assert 0 <= sample["max_ms"] <= sample["total_ms"]
            if sample["calls"]:
                assert sample["max_ms"] > 0
        if warmup == 0:
            calls = window["input_bytes"] // window["append_bytes"]
            expected = min(calls, window["input_bytes"] // group["bytes"])
            assert group["crossing"]["calls"] == expected
    section = p.stderr.split("--- Sustained window ---")[1].split("--- Whole run")[0]
    assert all(len(line) <= 80 for line in section.splitlines()), section
    assert "Full API samples" in section and "internal flush" in section

p = run()
assert p.returncode == 0, p.stderr
assert "sustained" not in json.loads(p.stdout)

p = run("--duration", "0.01", "--no-boundary-timing")
assert p.returncode == 0, p.stderr
window = json.loads(p.stdout)["sustained"]
assert window["boundary_timing"] is False
assert "Full API boundary sampling: disabled" in p.stderr
assert all(group["crossing"]["calls"] == group["following"]["calls"] == 0
           for group in window["boundaries"].values())

for args in (("--duration", "0"), ("--duration", "nan"), ("--duration", "inf"),
             ("--duration", "-1"), ("--duration", "1x"), ("--warmup", "1"),
             ("--duration", "1", "--warmup", "-1"),
             ("--duration", "1", "--append-elements", "3"),
             ("--duration", "1", "--append-elements", "bad"),
             ("--duration", "1", "--append-elements", "67108864"),
             ("--duration", "1", "-o", "/unused-sustained-output"),
             ("--duration", "1", "--s3-bucket", "unused"),
             ("--duration", "1", "--io-bw-mbps", "100")):
    p = run(*args)
    assert p.returncode != 0, args

for scenario in ("256cube_multiscale", "256cube_two_streams"):
    other = Path(exe).with_name("bench_stream_" + scenario + Path(exe).suffix)
    if other.exists():
        p = subprocess.run([str(other), "--backend", backend, "--duration", "1"],
                           capture_output=True, text=True, timeout=60)
        assert p.returncode != 0, p.stderr

print(f"Sustained accounting, geometry, CLI and TTY checks passed ({backend})")

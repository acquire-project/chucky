"""Check that sampled observations cannot masquerade as whole-stage totals."""
import json
import math
import subprocess
import sys

for mode in ("sampled", "full", "empty"):
    p = subprocess.run([sys.argv[1], mode], capture_output=True, text=True, check=True)
    report = json.loads(p.stdout)
    stages = report["stages"]
    if mode == "empty":
        assert "memcpy_work" not in report
        assert not {"memcpy", "memcpy_sample"} & stages.keys()
        assert "Memcpy work" not in p.stderr
        continue
    assert report["memcpy_work"] == {
        "calls": 128, "bytes": 65536, "timing_scope": mode
    }
    sampled = mode == "sampled"
    key = "memcpy_sample" if sampled else "memcpy"
    assert ("memcpy" if sampled else "memcpy_sample") not in stages
    row = stages[key]
    assert row["count"] == (2 if sampled else 128)
    assert row["in_bytes"] == row["out_bytes"] == (1024 if sampled else 65536)
    assert row["total_ms"] == 1
    assert math.isclose(row["in_gibs"], row["in_bytes"] / 2**30 * 1000, rel_tol=1e-5)
    assert ("Memcpy[smp]" in p.stderr) == sampled
    assert ("not extrapolated" in p.stderr) == sampled

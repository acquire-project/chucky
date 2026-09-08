"""Regression checks for the plain-text, 80-column benchmark report."""
import subprocess
import sys


def section(lines, title):
    start = next(i for i, line in enumerate(lines) if line.strip() == title)
    end = next((i for i in range(start + 1, len(lines)) if not lines[i]), len(lines))
    return lines[start + 1:end]


stage_names = [
    "Memcpy[smp]", "H2D", "Scatter", "LOD gather", "LOD reduce", "Append fold",
    "LOD to chunks", "Compress", "Aggregate", "D2H", "Sink",
]
stage_header = (
    f"  {'Stage':<14} {'avg GiB/s':>10} {'best GiB/s':>10}"
    f" {'avg ms':>9} {'best ms':>9}"
)

for mode in ("regular", "large", "empty"):
    p = subprocess.run([sys.argv[1], mode], capture_output=True, text=True, check=True)
    assert not p.stdout, "The text report belongs on stderr"
    lines = p.stderr.splitlines()
    for line in lines:
        assert len(line) <= 80, (mode, len(line), line)
        assert "\t" not in line and "\x1b" not in line, repr(line)
    assert "GB/s" not in p.stderr
    assert "backend_internal_name" not in p.stderr
    assert stage_header in lines
    if mode == "empty":
        assert "Host memory:      unavailable" in p.stderr
        assert "Append latency" not in p.stderr
        assert "Host blocking" not in p.stderr
        assert "Memcpy work" not in p.stderr
        continue

    start = lines.index(stage_header) + 1
    rows = lines[start:start + len(stage_names)]
    for name, row in zip(stage_names, rows):
        assert row[2:16].rstrip() == name, row
        assert len(row) == len(stage_header), row
        # Every row uses the same four right-aligned numeric fields.
        cells = [row[17:27], row[28:38], row[39:48], row[49:58]]
        assert all(cell == cell.strip().rjust(len(cell)) for cell in cells), row
    assert rows[0].split()[-2:] == ["4.00e-05", "3.00e-05"]
    assert rows[7].split()[-3:] == ["-", "4.00e-05", "-"]
    assert rows[-1].split()[1:3] == ["1.00", "1.00"]
    assert lines[start + len(stage_names)] == "", "Coverage must not split stage rows"

    coverage = section(lines, "Memcpy work:")
    assert "not extrapolated" in coverage[-1]
    totals = [str(2**64 - 1)] * 2 if mode == "large" else ["128", "65536"]
    assert coverage[1].split() == ["Total", *totals]
    assert coverage[2].split() == ["Timed", "2", "1024"]

    blocking = section(lines, "--- Host blocking ---")
    assert len(blocking[0]) == 79
    assert "  [producer timeline]" in blocking
    assert "  [delivery timeline]" in blocking
    assert "  Aggregate dependency before metadata" in blocking
    assert "  Chunk metadata ready (inclusive)" in blocking
    numeric = [line for line in blocking[1:] if len(line) == 79]
    assert len(numeric) == 10, numeric
    for row in numeric:
        cells = [row[29:39], row[40:50], row[51:60], row[61:70], row[71:79]]
        assert all(cell == cell.strip().rjust(len(cell)) for cell in cells), row
    if mode == "large":
        assert "1.845e+19" in p.stderr, "Large table counts must stay bounded"
    staging = next(line for line in numeric if "Staging-buffer reuse" in line)
    assert staging[29:].split() == ["0", "1", "-", "-", "0.00"]
    for row in section(lines, "--- Delivery latency ---"):
        assert len(row) == 77, row
    latency = section(lines, "--- Append latency ---")
    assert latency[0].split() == ["Appends:", str(2**64 - 1 if mode == "large" else 128)]
    assert len(latency[1]) == len(latency[2]) == 51
    assert all(float(cell) > 0 for cell in latency[2].split())
    assert "-1.00 KiB (observed minus estimated)" in p.stderr

    # All key/value summaries share the same value column.
    for label in ("Input:", "Output:", "Chunks:", "Host memory:", "Device memory:",
                  "Device overhead:", "Estimate:", "Init time:", "Flush time:",
                  "Wall time:", "Throughput:", "Payload:", "Metadata:",
                  "Logical payload:", "Physical payload:", "Peak pending:"):
        row = next(line for line in lines if line.startswith(f"  {label}"))
        assert row[:20] == f"  {label:<17} ", row

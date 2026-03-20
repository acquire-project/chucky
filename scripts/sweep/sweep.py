# /// script
# requires-python = ">=3.11"
# ///
"""
Benchmark sweep runner for chucky streaming zarr write benchmarks.

Usage:
    uv run scripts/sweep/sweep.py --tier compress --build-dir build --dry-run
    uv run scripts/sweep/sweep.py --tier compress --build-dir build
    uv run scripts/sweep/sweep.py --tier compress --tier backend --build-dir build -o results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "orca2_single": {
        "exe": "bench_stream_orca2_single",
        "frames": 200,
    },
    "256cube_single": {
        "exe": "bench_stream_256cube_single",
        "frames": 40,
    },
    "medfmt_single": {
        "exe": "bench_stream_medfmt_single",
        "frames": 10,
    },
    "smallepoch_single": {
        "exe": "bench_stream_smallepoch_single",
        "frames": 65536,
    },
    "orca2_multiscale": {
        "exe": "bench_stream_orca2_multiscale",
        "frames": 200,
    },
    "256cube_multiscale": {
        "exe": "bench_stream_256cube_multiscale",
        "frames": 40,
    },
    "orca2_multiscale_dim0": {
        "exe": "bench_stream_orca2_multiscale_dim0",
        "frames": 200,
    },
    "256cube_multiscale_dim0": {
        "exe": "bench_stream_256cube_multiscale_dim0",
        "frames": 40,
    },
}

# Chunk-byte labels -> values (ordered small to large)
CHUNK_BYTES = {
    "16K": 16 << 10,
    "32K": 32 << 10,
    "64K": 64 << 10,
    "128K": 128 << 10,
    "256K": 256 << 10,
    "512K": 512 << 10,
    "1M": 1 << 20,
    "2M": 2 << 20,
}

SINGLE_SCENARIOS = ["orca2_single", "256cube_single", "medfmt_single", "smallepoch_single"]

# ---------------------------------------------------------------------------
# Run matrix generation
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    scenario: str
    codec: str
    fill: str
    backend: str
    dtype: str
    chunk_label: str

    @property
    def chunk_bytes(self) -> int:
        return CHUNK_BYTES[self.chunk_label]

    @property
    def id(self) -> str:
        return f"{self.scenario}__{self.codec}__{self.fill}__{self.backend}__{self.dtype}__{self.chunk_label}"

    def base_result(self) -> dict:
        """Common fields shared by success, error, and timeout results."""
        return {
            "id": self.id,
            "scenario": self.scenario,
            "codec": self.codec,
            "fill": self.fill,
            "backend": self.backend,
            "dtype": self.dtype,
            "chunk_bytes": self.chunk_bytes,
            "chunk_bytes_label": self.chunk_label,
        }


def compress_runs() -> list[RunSpec]:
    """Core sweep: chunk_size x scenario x codec (GPU-only).

    This is a subset of backend_runs() — all compress runs use backend="gpu".
    Use the 'backend' tier to compare GPU vs CPU.
    """
    runs = []
    for sc in SINGLE_SCENARIOS:
        for codec in ["none", "lz4", "zstd"]:
            for cl in CHUNK_BYTES:
                runs.append(RunSpec(sc, codec, "xor", "gpu", "u16", cl))
    return runs


def backend_runs() -> list[RunSpec]:
    """GPU vs CPU backend comparison (superset of compress tier)."""
    runs = []

    for sc in SINGLE_SCENARIOS:
        for cl in CHUNK_BYTES:
            for codec in ["none", "lz4", "zstd"]:
                for backend in ["gpu", "cpu"]:
                    runs.append(RunSpec(sc, codec, "xor", backend, "u16", cl))

    return runs


def lod_runs() -> list[RunSpec]:
    """Multiscale / LOD sweeps."""
    runs = []
    chunk_labels = ["64K", "256K", "1M"]
    scenarios = ["orca2_multiscale", "256cube_multiscale",
                 "orca2_multiscale_dim0", "256cube_multiscale_dim0"]

    for sc in scenarios:
        for cl in chunk_labels:
            for codec in ["none", "zstd"]:
                for backend in ["gpu", "cpu"]:
                    runs.append(RunSpec(sc, codec, "xor", backend, "u16", cl))

    return runs


TIERS = {
    "compress": compress_runs,
    "backend": backend_runs,
    "lod": lod_runs,
}


def deduplicate(runs: list[RunSpec]) -> list[RunSpec]:
    seen: set[str] = set()
    out = []
    for r in runs:
        if r.id not in seen:
            seen.add(r.id)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# GPU name helper
# ---------------------------------------------------------------------------

def git_commit() -> str:
    """Return short commit hash of HEAD, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip().split("\n")[0]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(spec: RunSpec, build_dir: Path) -> dict:
    """Execute a single benchmark run, return result dict."""
    sc = SCENARIOS[spec.scenario]
    exe = build_dir / "bench" / sc["exe"]
    if not exe.exists():
        return {"id": spec.id, "status": "missing", "error": f"{exe} not found"}

    cmd = [
        str(exe),
        "--codec", spec.codec,
        "--fill", spec.fill,
        "--backend", spec.backend,
        "--dtype", spec.dtype,
        "--chunk-bytes", spec.chunk_label,
        "--frames", str(sc["frames"]),
        "--json",
    ]

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.monotonic() - t0

    # Parse JSON from stdout
    parsed: dict = {}
    stdout = proc.stdout.strip()
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            pass

    if not parsed:
        # No JSON output -- mark as error
        if proc.returncode != 0:
            parsed["status"] = "error"
            parsed["returncode"] = proc.returncode
        else:
            parsed["status"] = "unknown"

    return {
        "id": spec.id,
        "scenario": spec.scenario,
        "codec": spec.codec,
        "fill": spec.fill,
        "backend": spec.backend,
        "dtype": spec.dtype,
        "chunk_bytes": spec.chunk_bytes,
        "chunk_bytes_label": spec.chunk_label,
        "elapsed_s": round(elapsed, 2),
        **parsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tier_names = ", ".join(TIERS.keys())
    ap = argparse.ArgumentParser(description="Benchmark sweep runner")
    ap.add_argument("--tier", type=str, action="append", default=[],
                    help=f"Tier(s) to run ({tier_names}). Repeat for multiple.")
    ap.add_argument("--build-dir", type=Path, default=Path("build"),
                    help="CMake build directory")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output JSON path (default: bench/results/<commit>-<date>.json)")
    ap.add_argument("--skip", type=str, action="append", default=[],
                    help="Scenario(s) to skip (repeat for multiple)")
    ap.add_argument("--retry", action="store_true",
                    help="Re-run previously failed or timed-out benchmarks")
    ap.add_argument("--rerun", type=str, action="append", default=[],
                    help="Re-run benchmarks whose id contains this substring (repeat for multiple)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print run matrix without executing")
    args = ap.parse_args()

    commit = git_commit()
    if args.output is None:
        results_dir = Path("bench/results")
        # Reuse existing file for this commit if one exists
        existing_files = sorted(results_dir.glob(f"{commit}-*.json"))
        if existing_files:
            args.output = existing_files[-1]
        else:
            date_str = time.strftime("%Y%m%d")
            args.output = results_dir / f"{commit}-{date_str}.json"

    tiers = args.tier or ["compress"]

    runs: list[RunSpec] = []
    for t in tiers:
        if t not in TIERS:
            print(f"Unknown tier: {t} (available: {tier_names})", file=sys.stderr)
            return 1
        runs.extend(TIERS[t]())
    runs = deduplicate(runs)
    if args.skip:
        runs = [r for r in runs if r.scenario not in args.skip]

    if args.dry_run:
        for i, r in enumerate(runs, 1):
            print(f"[{i}/{len(runs)}] {r.scenario} {r.codec} {r.fill} "
                  f"{r.backend} {r.dtype} {r.chunk_label}")
        print(f"\nTotal: {len(runs)} runs")
        print(f"Output: {args.output}")
        return

    # Load existing results for resumability
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if args.output.exists():
        with open(args.output) as f:
            data = json.load(f)
        for r in data.get("runs", []):
            rid = r["id"]
            if args.retry and r.get("status") in ("error", "timeout"):
                continue
            if args.rerun and any(pat in rid for pat in args.rerun):
                continue
            existing[rid] = r
    else:
        data = {
            "version": 1,
            "machine": {
                "hostname": platform.node(),
                "gpu": gpu_name(),
                "commit": commit,
                "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "runs": [],
        }

    total = len(runs)
    skipped = 0
    for i, spec in enumerate(runs, 1):
        if spec.id in existing:
            skipped += 1
            continue

        tag = (f"[{i}/{total}] {spec.scenario} {spec.codec} {spec.fill} "
               f"{spec.backend} {spec.dtype} {spec.chunk_label}")
        print(f"{tag} ...", end="", flush=True, file=sys.stderr)

        try:
            result = run_one(spec, args.build_dir)
        except subprocess.TimeoutExpired:
            result = {**spec.base_result(), "status": "timeout"}
        except Exception as e:
            result = {**spec.base_result(), "status": "error", "error": str(e)}

        tp = result.get("throughput_in_gibs")
        st = result.get("status", "?")
        suffix = f" {tp:.2f} GiB/s" if tp else ""
        print(f"\r{tag} ...{suffix} {st.upper()}", file=sys.stderr)

        existing[spec.id] = result

        # Save incrementally
        data["runs"] = list(existing.values())
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)

    if skipped:
        print(f"Skipped {skipped} existing runs", file=sys.stderr)
    print(f"Results: {args.output} ({len(existing)} runs)", file=sys.stderr)


if __name__ == "__main__":
    main()

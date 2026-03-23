# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "rich",
#   "pydantic",
# ]
# ///
"""
Benchmark sweep runner for chucky streaming zarr write benchmarks.

Usage:
    uv run scripts/sweep/sweep.py --tier compress --dry-run
    uv run scripts/sweep/sweep.py --tier compress
    uv run scripts/sweep/sweep.py --all
    uv run scripts/sweep/sweep.py --tier io
    uv run scripts/sweep/sweep.py --tier s3 --s3-bucket my-bucket
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
from pydantic import BaseModel, model_validator
from rich.console import Console
from models import (
    VALID_BACKENDS,
    VALID_CODECS,
    VALID_DTYPES,
    VALID_FILLS,
    VALID_SINKS,
    VALID_STATUSES,
    validate_results,
)
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "orca2_single": 200,
    "256cube_single": 40,
    "medfmt_single": 10,
    "smallepoch_single": 65536,
    "orca2_multiscale": 200,
    "256cube_multiscale": 40,
    "orca2_multiscale_dim0": 200,
    "256cube_multiscale_dim0": 40,
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

SINGLE_SCENARIOS = [k for k in SCENARIOS if k.endswith("_single")]

# ---------------------------------------------------------------------------
# Run spec (pydantic-validated)
# ---------------------------------------------------------------------------


class RunSpec(BaseModel):
    scenario: str
    codec: str
    fill: str
    backend: str
    dtype: str
    chunk_label: str
    sink: str = "discard"
    s3_throughput_gbps: float = 0

    @model_validator(mode="after")
    def _validate_enums(self) -> RunSpec:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        if self.codec not in VALID_CODECS:
            raise ValueError(f"Unknown codec: {self.codec} (expected one of {VALID_CODECS})")
        if self.fill not in VALID_FILLS:
            raise ValueError(f"Unknown fill: {self.fill} (expected one of {VALID_FILLS})")
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"Unknown backend: {self.backend} (expected one of {VALID_BACKENDS})")
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"Unknown dtype: {self.dtype} (expected one of {VALID_DTYPES})")
        if self.chunk_label not in CHUNK_BYTES:
            raise ValueError(f"Unknown chunk_label: {self.chunk_label} (expected one of {set(CHUNK_BYTES)})")
        if self.sink not in VALID_SINKS:
            raise ValueError(f"Unknown sink: {self.sink} (expected one of {VALID_SINKS})")
        return self

    @property
    def chunk_bytes(self) -> int:
        return CHUNK_BYTES[self.chunk_label]

    @property
    def id(self) -> str:
        suffix = f"__{self.sink}" if self.sink != "discard" else ""
        if self.s3_throughput_gbps > 0:
            suffix += f"__{int(self.s3_throughput_gbps)}gbps"
        return f"{self.scenario}__{self.codec}__{self.fill}__{self.backend}__{self.dtype}__{self.chunk_label}{suffix}"

    def base_result(self) -> dict:
        """Common fields shared by success, error, and timeout results."""
        d = {
            "id": self.id,
            "scenario": self.scenario,
            "codec": self.codec,
            "fill": self.fill,
            "backend": self.backend,
            "dtype": self.dtype,
            "chunk_bytes": self.chunk_bytes,
            "chunk_bytes_label": self.chunk_label,
            "sink": self.sink,
        }
        if self.s3_throughput_gbps > 0:
            d["s3_throughput_gbps"] = self.s3_throughput_gbps
        return d


# ---------------------------------------------------------------------------
# Run matrix generation
# ---------------------------------------------------------------------------


def compress_runs() -> list[RunSpec]:
    """Core sweep: chunk_size x scenario x codec (GPU-only)."""
    runs = []
    for sc in SINGLE_SCENARIOS:
        for codec in ["none", "lz4", "zstd"]:
            for cl in CHUNK_BYTES:
                runs.append(RunSpec(scenario=sc, codec=codec, fill="xor", backend="gpu", dtype="u16", chunk_label=cl))
    return runs


def backend_runs() -> list[RunSpec]:
    """GPU vs CPU backend comparison (superset of compress tier)."""
    runs = []
    for sc in SINGLE_SCENARIOS:
        for codec in ["none", "lz4", "zstd"]:
            for cl in CHUNK_BYTES:
                for backend in ["gpu", "cpu"]:
                    runs.append(RunSpec(scenario=sc, codec=codec, fill="xor", backend=backend, dtype="u16", chunk_label=cl))
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
                    runs.append(RunSpec(scenario=sc, codec=codec, fill="xor", backend=backend, dtype="u16", chunk_label=cl))
    return runs


def io_runs() -> list[RunSpec]:
    """I/O tier: measure impact of zarr output vs discard sink."""
    runs = []
    chunk_labels = ["32K", "256K", "2M"]
    scenarios = ["orca2_single", "256cube_single"]
    for sc in scenarios:
        for cl in chunk_labels:
            for codec in ["none", "zstd"]:
                for backend in ["gpu", "cpu"]:
                    runs.append(RunSpec(scenario=sc, codec=codec, fill="xor", backend=backend, dtype="u16", chunk_label=cl, sink="fs"))
    return runs


def s3_runs() -> list[RunSpec]:
    """S3 tier: discard vs fs vs S3 sink comparison, with throughput sweep."""
    runs = []
    chunk_labels = ["32K", "256K", "2M"]
    scenarios = ["orca2_single", "256cube_single"]
    for sc in scenarios:
        for cl in chunk_labels:
            for codec in ["none", "zstd"]:
                for backend in ["gpu", "cpu"]:
                    for throughput in [10, 100]:
                        for sink in ["discard", "fs", "s3"]:
                            runs.append(RunSpec(
                                scenario=sc, codec=codec, fill="xor",
                                backend=backend, dtype="u16", chunk_label=cl,
                                sink=sink,
                                s3_throughput_gbps=throughput if sink == "s3" else 0,
                            ))
    return runs


TIERS = {
    "compress": compress_runs,
    "backend": backend_runs,
    "lod": lod_runs,
    "io": io_runs,
    "s3": s3_runs,
}

ALL_TIER_NAMES = list(TIERS.keys())


def deduplicate(runs: list[RunSpec]) -> list[RunSpec]:
    seen: set[str] = set()
    out = []
    for r in runs:
        if r.id not in seen:
            seen.add(r.id)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def git_commit() -> str:
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

def run_one(spec: RunSpec, build_dir: Path, s3_bucket: str | None = None,
            s3_region: str | None = None, s3_endpoint: str | None = None) -> dict | None:
    """Execute a single benchmark run, return result dict or None if exe missing."""
    exe = build_dir / "bench" / f"bench_stream_{spec.scenario}"
    if sys.platform == "win32":
        exe = exe.with_suffix(".exe")
    if not exe.exists():
        return None

    frames = SCENARIOS[spec.scenario]
    cmd = [
        str(exe),
        "--codec", spec.codec,
        "--fill", spec.fill,
        "--backend", spec.backend,
        "--dtype", spec.dtype,
        "--chunk-bytes", spec.chunk_label,
        "--frames", str(frames),
        "--json",
    ]

    tmpdir = None
    if spec.sink == "fs":
        tmpdir = tempfile.mkdtemp(prefix="chucky_io_")
        cmd.extend(["-o", tmpdir])
    elif spec.sink == "s3":
        if not s3_bucket or not s3_region or not s3_endpoint:
            return {**spec.base_result(), "status": "error",
                    "error": "s3 sink requires --s3-bucket, --s3-region, --s3-endpoint"}
        prefix = f"bench/{spec.id}"
        cmd.extend(["--s3-bucket", s3_bucket,
                     "--s3-prefix", prefix,
                     "--s3-region", s3_region,
                     "--s3-endpoint", s3_endpoint])
        if spec.s3_throughput_gbps > 0:
            cmd.extend(["--s3-throughput-gbps", str(spec.s3_throughput_gbps)])

    try:
        t0 = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.monotonic() - t0

        parsed: dict = {}
        stdout = proc.stdout.strip()
        if stdout:
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                pass

        if not parsed:
            if proc.returncode != 0:
                parsed["status"] = "error"
                parsed["returncode"] = proc.returncode
            else:
                parsed["status"] = "unknown"

        result = {**spec.base_result(), "elapsed_s": round(elapsed, 2), **parsed}
        if spec.sink == "s3":
            if s3_endpoint:
                result.setdefault("s3_endpoint", s3_endpoint)
            if s3_region:
                result.setdefault("s3_region", s3_region)
            if s3_bucket:
                result.setdefault("s3_bucket", s3_bucket)
        return result
    finally:
        if tmpdir is not None:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Status formatting
# ---------------------------------------------------------------------------

_STATUS_STYLE = {
    "pass": "green",
    "error": "red",
    "timeout": "dark_orange",
    "missing": "red",
    "unknown": "yellow",
}


def status_style(status: str) -> str:
    return _STATUS_STYLE.get(status, "white")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--tier", "-t", multiple=True, type=click.Choice(ALL_TIER_NAMES),
              help="Tier(s) to run. Repeat for multiple.")
@click.option("--all", "run_all", is_flag=True, help="Run all tiers.")
@click.option("--build-dir", type=click.Path(exists=False, path_type=Path),
              default=Path("build"),
              show_default=True, help="CMake build directory.")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None,
              help="Output JSON path (default: bench/results/<host>-<commit>-<date>.json).")
@click.option("--skip", multiple=True, help="Scenario(s) to skip.")
@click.option("--retry", is_flag=True, help="Re-run previously failed or timed-out benchmarks.")
@click.option("--rerun", multiple=True, help="Re-run benchmarks whose id contains this substring.")
@click.option("--dry-run", is_flag=True, help="Preview run matrix without executing.")
@click.option("--s3-bucket", default=None, help="S3 bucket (required for s3 tier).")
@click.option("--s3-region", default="us-east-1", show_default=True, help="S3 region.")
@click.option("--s3-endpoint", default="http://localhost:9000", show_default=True,
              help="S3 endpoint URL.")
def main(tier, run_all, build_dir, output, skip, retry, rerun, dry_run,
         s3_bucket, s3_region, s3_endpoint):
    """Benchmark sweep runner for chucky."""
    commit = git_commit()
    hostname = platform.node()

    if output is None:
        results_dir = Path("bench/results")
        existing_files = sorted(results_dir.glob(f"{hostname}-{commit}-*.json"))
        if existing_files:
            output = existing_files[-1]
        else:
            date_str = time.strftime("%Y%m%d")
            output = results_dir / f"{hostname}-{commit}-{date_str}.json"

    # Resolve tiers
    if run_all:
        selected_tiers = ALL_TIER_NAMES
    elif tier:
        selected_tiers = list(tier)
    else:
        selected_tiers = ALL_TIER_NAMES

    runs: list[RunSpec] = []
    for t in selected_tiers:
        runs.extend(TIERS[t]())
    runs = deduplicate(runs)
    if skip:
        runs = [r for r in runs if not any(pat in r.scenario for pat in skip)]

    # Skip S3 runs if --s3-bucket not provided
    if not s3_bucket:
        s3_count = sum(1 for r in runs if r.sink == "s3")
        if s3_count:
            runs = [r for r in runs if r.sink != "s3"]
            console.print(f"Skipping [bold]{s3_count}[/bold] S3 runs (no --s3-bucket provided)")

    # -- dry run: rich table --
    if dry_run:
        table = Table(title="Sweep Matrix", show_lines=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Scenario")
        table.add_column("Codec")
        table.add_column("Fill")
        table.add_column("Backend")
        table.add_column("Dtype")
        table.add_column("Chunk")
        table.add_column("Sink", justify="center")
        for i, r in enumerate(runs, 1):
            table.add_row(
                str(i), r.scenario, r.codec, r.fill,
                r.backend, r.dtype, r.chunk_label,
                r.sink if r.sink != "discard" else "",
            )
        console.print(table)
        console.print(f"\nTotal: [bold]{len(runs)}[/bold] runs across tiers: {', '.join(selected_tiers)}")
        console.print(f"Output: {output}")
        return

    # -- load existing results for resumability --
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if output.exists():
        with open(output) as f:
            raw_data = json.load(f)
        # Validate loaded results
        try:
            validated = validate_results(raw_data)
        except Exception as e:
            console.print(f"[yellow]Warning: results file validation failed: {e}[/yellow]")
            console.print("[yellow]Continuing with raw data.[/yellow]")
        data = raw_data
        for r in data.get("runs", []):
            rid = r["id"]
            if retry and r.get("status") != "pass":
                continue
            if rerun and any(pat in rid for pat in rerun):
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

    # -- count how many actually need to run --
    to_run = [spec for spec in runs if spec.id not in existing]
    skip_count = len(runs) - len(to_run)

    if skip_count:
        console.print(f"Skipping [bold]{skip_count}[/bold] existing runs")

    if not to_run:
        console.print(f"[green]All {len(runs)} runs already complete.[/green]")
        console.print(f"Results: {output}")
        return

    # -- run with progress bar --
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Sweeping", total=len(to_run))

        for spec in to_run:
            tag = f"{spec.scenario} {spec.codec} {spec.backend} {spec.chunk_label}"
            if spec.sink != "discard":
                tag += f" {spec.sink}"
            progress.update(task, description=f"[bold]{tag}")

            try:
                result = run_one(spec, build_dir,
                                 s3_bucket=s3_bucket,
                                 s3_region=s3_region,
                                 s3_endpoint=s3_endpoint)
            except subprocess.TimeoutExpired:
                result = {**spec.base_result(), "status": "timeout"}
            except Exception as e:
                result = {**spec.base_result(), "status": "error", "error": str(e)}

            if result is None:
                progress.console.print(f"  {tag} [dim]SKIP (exe not found)[/dim]")
                progress.advance(task)
                continue

            st = result.get("status", "?")
            tp = result.get("throughput_in_gibs")
            suffix = f" {tp:.2f} GiB/s" if tp else ""
            style = status_style(st)
            progress.console.print(f"  {tag} [{style}]{st.upper()}[/{style}]{suffix}")

            existing[spec.id] = result

            # Save incrementally
            data["runs"] = list(existing.values())
            with open(output, "w") as f:
                json.dump(data, f, indent=2)

            progress.advance(task)

    console.print(f"\n[bold green]Done.[/bold green] Results: {output} ({len(existing)} runs)")


if __name__ == "__main__":
    main()

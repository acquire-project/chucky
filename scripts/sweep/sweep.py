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
    uv run scripts/sweep/sweep.py --tier s3 --backend cpu --s3-bucket my-bucket
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
from pydantic import BaseModel, model_validator
from rich.console import Console
from models import (
    CURRENT_VERSION,
    VALID_BACKENDS,
    VALID_CODECS,
    VALID_DTYPES,
    VALID_FILLS,
    VALID_SINKS,
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
    "smallepoch_4shards": 65536,
    "orca2_multiscale": 200,
    "256cube_multiscale": 40,
    "medfmt_multiscale": 10,
    "orca2_multiscale_dim0": 200,
    "256cube_multiscale_dim0": 40,
    "medfmt_multiscale_dim0": 10,
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

# The compress and backend tiers write to the discard sink, so
# smallepoch_4shards is left out: a second shard file buys nothing there
# but run time. Only the io tier wants it.
SINGLE_SCENARIOS = [
    "orca2_single",
    "256cube_single",
    "medfmt_single",
    "smallepoch_single",
]

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
    # Write scheduling for the filesystem sink. 0 leaves the binary's default,
    # which is what every tier but iodepth wants.
    io_writes_in_flight: int = 0
    io_writes_in_flight_per_file: int = 0

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
    def frames(self) -> int:
        return SCENARIOS[self.scenario]

    @property
    def id(self) -> str:
        suffix = f"__{self.sink}" if self.sink != "discard" else ""
        if self.s3_throughput_gbps > 0:
            suffix += f"__{int(self.s3_throughput_gbps)}gbps"
        if self.io_writes_in_flight > 0:
            suffix += f"__wif{self.io_writes_in_flight}"
        if self.io_writes_in_flight_per_file > 0:
            suffix += f"__perfile{self.io_writes_in_flight_per_file}"
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
            "frames": self.frames,
        }
        if self.s3_throughput_gbps > 0:
            d["s3_throughput_gbps"] = self.s3_throughput_gbps
        # The binary reports what it resolved these to, and that reading wins.
        # These are only here so a run that never got that far still says what
        # it was asked for.
        if self.io_writes_in_flight > 0:
            d["io_writes_in_flight"] = self.io_writes_in_flight
        if self.io_writes_in_flight_per_file > 0:
            d["io_writes_in_flight_per_file"] = self.io_writes_in_flight_per_file
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
        # blosc codecs are CPU-only
        for codec in ["blosc-lz4", "blosc-zstd"]:
            for cl in CHUNK_BYTES:
                runs.append(RunSpec(scenario=sc, codec=codec, fill="xor", backend="cpu", dtype="u16", chunk_label=cl))
    return runs


def lod_runs() -> list[RunSpec]:
    """Multiscale / LOD sweeps."""
    runs = []
    chunk_labels = ["64K", "256K", "1M"]
    scenarios = ["orca2_multiscale", "256cube_multiscale", "medfmt_multiscale",
                 "orca2_multiscale_dim0", "256cube_multiscale_dim0", "medfmt_multiscale_dim0"]
    for sc in scenarios:
        for cl in chunk_labels:
            for codec in ["none", "zstd"]:
                for backend in ["gpu", "cpu"]:
                    runs.append(RunSpec(scenario=sc, codec=codec, fill="xor", backend=backend, dtype="u16", chunk_label=cl))
            # blosc codecs are CPU-only
            for codec in ["blosc-lz4", "blosc-zstd"]:
                runs.append(RunSpec(scenario=sc, codec=codec, fill="xor", backend="cpu", dtype="u16", chunk_label=cl))
    return runs


def io_runs() -> list[RunSpec]:
    """I/O tier: measure impact of zarr output vs discard sink."""
    runs = []
    chunk_labels = ["32K", "256K", "2M"]
    scenarios = ["orca2_single", "256cube_single",
                  "orca2_multiscale_dim0", "256cube_multiscale_dim0",
                  "smallepoch_single", "smallepoch_4shards"]
    for sc in scenarios:
        for cl in chunk_labels:
            for codec in ["none", "zstd"]:
                for backend in ["gpu", "cpu"]:
                    runs.append(RunSpec(scenario=sc, codec=codec, fill="xor", backend=backend, dtype="u16", chunk_label=cl, sink="fs"))
            # blosc codecs are CPU-only
            for codec in ["blosc-lz4", "blosc-zstd"]:
                runs.append(RunSpec(scenario=sc, codec=codec, fill="xor", backend="cpu", dtype="u16", chunk_label=cl, sink="fs"))
    return runs


# How deep the write queue is allowed to get. Measured on xfs, more is worth
# about 1.4x and flat past 16; 32 is here to show the flat part.
WRITES_IN_FLIGHT = [1, 2, 4, 8, 16, 32]


def iodepth_runs() -> list[RunSpec]:
    """Write-queue depth sweep, over shard files and within one."""
    runs = []
    scenarios = ["smallepoch_single", "smallepoch_4shards", "orca2_single"]
    for sc in scenarios:
        for codec in ["none", "zstd"]:
            for wif in WRITES_IN_FLIGHT:
                # One per file isolates the gain from writing to several shard
                # files at once; four adds the gain from pre-sizing a file so
                # its own writes can run together.
                for per_file in [1, 4]:
                    runs.append(RunSpec(
                        scenario=sc, codec=codec, fill="xor", backend="gpu",
                        dtype="u16", chunk_label="256K", sink="fs",
                        io_writes_in_flight=wif,
                        io_writes_in_flight_per_file=per_file,
                    ))
    return runs


def fill_runs() -> list[RunSpec]:
    """Fill-pattern sweep: xor vs zeros vs rand across codecs and chunk sizes."""
    runs = []
    chunk_labels = ["32K", "256K", "2M"]
    scenarios = ["orca2_single", "256cube_single"]
    for sc in scenarios:
        for fill in ["xor", "zeros", "rand"]:
            for codec in ["none", "lz4", "zstd"]:
                for cl in chunk_labels:
                    runs.append(RunSpec(scenario=sc, codec=codec, fill=fill, backend="gpu", dtype="u16", chunk_label=cl))
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
            # blosc codecs are CPU-only
            for codec in ["blosc-lz4", "blosc-zstd"]:
                for throughput in [10, 100]:
                    for sink in ["discard", "fs", "s3"]:
                        runs.append(RunSpec(
                            scenario=sc, codec=codec, fill="xor",
                            backend="cpu", dtype="u16", chunk_label=cl,
                            sink=sink,
                            s3_throughput_gbps=throughput if sink == "s3" else 0,
                        ))
    return runs


TIERS = {
    "compress": compress_runs,
    "backend": backend_runs,
    "lod": lod_runs,
    "fill": fill_runs,
    "io": io_runs,
    "iodepth": iodepth_runs,
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


def gpu_and_driver() -> tuple[str, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        name, _, driver = out.stdout.strip().split("\n")[0].partition(",")
        return name.strip() or "unknown", driver.strip() or "unknown"
    except Exception:
        return "unknown", "unknown"


def cpu_count() -> int:
    """Cores this process may run on, which on a cluster is below the machine's."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 0


BUILD_CACHE_KEYS = {
    "CMAKE_BUILD_TYPE": "build_type",
    "CMAKE_CUDA_ARCHITECTURES": "cuda_architectures",
    "CMAKE_CXX_COMPILER": "cxx_compiler",
    "CHUCKY_ENABLE_GPU": "gpu_enabled",
    "NVCOMP_LIBRARY": "nvcomp",
}


def build_info(build_dir: Path) -> dict:
    """The build directory's last configuration is returned.

    Read from the build directory rather than the environment the sweep runs
    in, so it describes the binaries rather than the machine. It describes the
    last configure, not the last build, so it goes stale if someone
    reconfigures without rebuilding. Missing keys are left out: a reader should
    see nothing rather than a default that was never true.
    """
    info: dict = {}
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            name, _, value = line.partition("=")
            key = BUILD_CACHE_KEYS.get(name.split(":")[0])
            if key and value:
                info[key] = value

    # Not a cache entry. CMake writes one directory per version it configured
    # with, so the newest file is the one that describes this build.
    newest = max(build_dir.glob("CMakeFiles/*/CMakeCUDACompiler.cmake"),
                 key=lambda p: p.stat().st_mtime, default=None)
    if newest:
        found = re.search(r'set\(CMAKE_CUDA_COMPILER_VERSION "([^"]+)"\)',
                          newest.read_text(errors="replace"))
        if found:
            info["cuda_compiler_version"] = found.group(1)
    return info


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(spec: RunSpec, build_dir: Path, s3_bucket: str | None = None,
            s3_region: str | None = None, s3_endpoint: str | None = None,
            tmpdir_root: Path | None = None) -> dict | None:
    """Execute a single benchmark run, return result dict or None if exe missing."""
    exe = build_dir / "bench" / f"bench_stream_{spec.scenario}"
    if sys.platform == "win32":
        exe = exe.with_suffix(".exe")
    if not exe.exists():
        return None

    cmd = [
        str(exe),
        "--codec", spec.codec,
        "--fill", spec.fill,
        "--backend", spec.backend,
        "--dtype", spec.dtype,
        "--chunk-bytes", spec.chunk_label,
        "--frames", str(spec.frames),
        "--json",
    ]

    # The worker count is what caps how many writes a blocking backend can have
    # running, so it tracks the ceiling rather than being swept on its own.
    if spec.io_writes_in_flight > 0:
        cmd.extend(["--io-writes-in-flight", str(spec.io_writes_in_flight),
                    "--io-workers", str(spec.io_writes_in_flight)])
    if spec.io_writes_in_flight_per_file > 0:
        cmd.extend(["--io-writes-in-flight-per-file",
                    str(spec.io_writes_in_flight_per_file)])

    tmpdir = None
    if spec.sink == "fs":
        tmpdir = tempfile.mkdtemp(prefix="chucky_io_",
                                  dir=str(tmpdir_root) if tmpdir_root else None)
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
@click.option("--backend", "backend_filter", type=click.Choice(sorted(VALID_BACKENDS)),
              help="Only run benchmarks for this backend.")
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
@click.option("--tmpdir", "tmpdir_root", type=click.Path(path_type=Path), default=None,
              help="Parent directory for fs-sink scratch dirs (default: system temp).")
@click.option("--machine", "machine_name", default=None, envvar="CHUCKY_MACHINE",
              help="Name this machine goes by in the report (default: hostname). "
                   "Give a stable name where the hostname changes between runs, as "
                   "it does on a cluster; group names live in bench/machines.toml.")
def main(tier, run_all, backend_filter, build_dir, output, skip, retry, rerun, dry_run,
         s3_bucket, s3_region, s3_endpoint, tmpdir_root, machine_name):
    """Benchmark sweep runner for chucky."""
    commit = git_commit()
    hostname = platform.node()
    machine_name = machine_name or hostname

    # The name becomes a file name and a glob pattern, so keep it to characters
    # that mean the same thing in both.
    if not re.fullmatch(r"[A-Za-z0-9._-]+", machine_name):
        raise click.BadParameter(
            f"machine name {machine_name!r} may only contain letters, digits, dot, "
            "underscore, and dash",
            param_hint="--machine",
        )

    if output is None:
        results_dir = Path("bench/results")
        existing_files = sorted(results_dir.glob(f"{machine_name}-{commit}-*.json"))
        if existing_files:
            output = existing_files[-1]
        else:
            date_str = time.strftime("%Y%m%d")
            output = results_dir / f"{machine_name}-{commit}-{date_str}.json"

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
    if backend_filter:
        runs = [r for r in runs if r.backend == backend_filter]
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
        table.add_column("In flight all/file", justify="center")
        for i, r in enumerate(runs, 1):
            in_flight = ""
            if r.io_writes_in_flight > 0:
                in_flight = f"{r.io_writes_in_flight}/{r.io_writes_in_flight_per_file}"
            table.add_row(
                str(i), r.scenario, r.codec, r.fill,
                r.backend, r.dtype, r.chunk_label,
                r.sink if r.sink != "discard" else "",
                in_flight,
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
        try:
            validate_results(raw_data)
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
        gpu, driver = gpu_and_driver()
        data = {
            "version": CURRENT_VERSION,
            "machine": {
                "name": machine_name,
                "hostname": platform.node(),
                "gpu": gpu,
                "driver_version": driver,
                "cpu_count": cpu_count(),
                "build": build_info(build_dir),
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
                                 s3_endpoint=s3_endpoint,
                                 tmpdir_root=tmpdir_root)
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

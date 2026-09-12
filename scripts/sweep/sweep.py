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
    uv run scripts/sweep/sweep.py --tier backend --scenario microscopy
    uv run scripts/sweep/sweep.py --tier backend --scenario orca2_single --scenario microscopy
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from image_results import input_label
from measurements import (
    DEFAULT_DURATION_S, DEFAULT_WARMUP_S, aggregate_repetitions, execute,
    measurement_policy, validate_measurement,
)
from models import (
    CURRENT_VERSION,
    VALID_BACKENDS,
    VALID_CODECS,
    VALID_DTYPES,
    VALID_FILLS,
    VALID_SINKS,
    VALID_SHUFFLES,
    default_level,
    migrate_scenario,
    run_id,
    validate_results,
)

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

MICROSCOPY_SCENARIO = "microscopy"
SCENARIOS: dict[str, int | None] = {
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
    # Image frame counts depend on each registered asset and --min-gib.
    MICROSCOPY_SCENARIO: None,
}

DEFAULT_DATA_REGISTRY = Path(__file__).resolve().parents[2] / "bench/data.json"
DEFAULT_IMAGE_MIN_GIB = 32.0
DEFAULT_IMAGE_REPEATS = 5
IMAGE_DTYPES = {"u8": ("u8", 1), "u16": ("u16le", 2), "f32": ("f32le", 4)}
IMAGE_SUPPORTED_TIERS = {"compress", "backend"}
IMAGE_CODEC_SETTINGS = {
    "none": {"level": 0, "blosc_shuffle": "none"},
    "lz4": {"level": 1, "blosc_shuffle": "none"},
    "zstd": {"level": 3, "blosc_shuffle": "none"},
    "blosc-lz4": {"level": 3, "blosc_shuffle": "bit"},
    "blosc-zstd": {"level": 3, "blosc_shuffle": "bit"},
}
IMAGE_LAYOUT_KEYS = (
    "reference_shape",
    "chunk_shape",
    "chunks_per_shard",
    "epochs_per_batch",
    "target_batch_bytes",
    "actual_batch_bytes",
    "append_elements",
    "dtype",
)

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
    blosc_block_bytes: int | None = Field(default=None, ge=128, le=715827542, strict=True)
    blosc_shuffle: str = "none"
    level: int | None = Field(default=None, ge=0, le=255)
    input_id: str | None = None
    image_asset_id: str | None = None
    image_split: str | None = None

    @model_validator(mode="after")
    def _validate_enums(self) -> RunSpec:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        if self.codec not in VALID_CODECS:
            raise ValueError(f"Unknown codec: {self.codec} (expected one of {VALID_CODECS})")
        if self.scenario == MICROSCOPY_SCENARIO:
            if self.fill != "images":
                raise ValueError(f"{MICROSCOPY_SCENARIO} scenario requires fill=images")
            if not self.input_id or not self.image_asset_id or not self.image_split:
                raise ValueError(
                    f"{MICROSCOPY_SCENARIO} scenario requires input_id, image_asset_id, "
                    "and image_split"
                )
            if self.dtype not in IMAGE_DTYPES or self.sink != "discard":
                raise ValueError(
                    f"{MICROSCOPY_SCENARIO} scenario requires u8, u16, or f32 and the discard sink"
                )
        elif self.fill not in VALID_FILLS:
            raise ValueError(f"Unknown fill: {self.fill} (expected one of {VALID_FILLS})")
        elif any(
            value is not None
            for value in (self.input_id, self.image_asset_id, self.image_split)
        ):
            raise ValueError("Image input fields are only valid for the microscopy scenario")
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"Unknown backend: {self.backend} (expected one of {VALID_BACKENDS})")
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"Unknown dtype: {self.dtype} (expected one of {VALID_DTYPES})")
        if self.chunk_label not in CHUNK_BYTES:
            raise ValueError(f"Unknown chunk_label: {self.chunk_label} (expected one of {set(CHUNK_BYTES)})")
        if self.sink not in VALID_SINKS:
            raise ValueError(f"Unknown sink: {self.sink} (expected one of {VALID_SINKS})")
        if self.codec.startswith("blosc-"):
            if self.blosc_block_bytes is None:
                raise ValueError("Blosc runs require explicit blosc_block_bytes")
        elif self.blosc_block_bytes is not None:
            raise ValueError("blosc_block_bytes is only valid for Blosc runs")
        if self.blosc_shuffle not in VALID_SHUFFLES:
            raise ValueError(f"Unknown shuffle: {self.blosc_shuffle}")
        if self.blosc_shuffle != "none" and not self.codec.startswith("blosc-"):
            raise ValueError("Byte/bit shuffle requires a Blosc codec")
        if self.level is None:
            self.level = default_level(self.codec)
        if self.codec.startswith("blosc-") and self.level > 9:
            raise ValueError("Blosc level must be 0..9")
        if self.codec == "lz4" and self.level == 0:
            raise ValueError("LZ4 level must be at least 1")
        return self

    @property
    def chunk_bytes(self) -> int:
        return CHUNK_BYTES[self.chunk_label]

    @property
    def frames(self) -> int:
        frames = SCENARIOS[self.scenario]
        if frames is None:
            raise ValueError("Image frame count depends on the selected corpus")
        return frames

    @property
    def id(self) -> str:
        return self.base_result()["id"]

    def base_result(self) -> dict:
        """Common fields shared by success, error, and timeout results."""
        d: dict = {
            "scenario": self.scenario,
            "codec": self.codec,
            "fill": self.fill,
            "backend": self.backend,
            "dtype": self.dtype,
            "chunk_bytes": self.chunk_bytes,
            "chunk_bytes_label": self.chunk_label,
            "sink": self.sink,
        }
        if self.scenario == MICROSCOPY_SCENARIO:
            d["input_id"] = self.input_id
            d["image_asset_id"] = self.image_asset_id
            d["image_split"] = self.image_split
        else:
            d["geometry_frames"] = self.frames
        if self.s3_throughput_gbps > 0:
            d["s3_throughput_gbps"] = self.s3_throughput_gbps
        if self.codec.startswith("blosc-"):
            d["blosc_block_bytes"] = self.blosc_block_bytes
            d["blosc_shuffle"] = self.blosc_shuffle
            d["blosc_level"] = self.level
        else:
            d["level"] = self.level
        identity = None
        if self.scenario == MICROSCOPY_SCENARIO:
            identity = "__".join(
                (
                    self.scenario,
                    self.codec,
                    str(self.input_id),
                    str(self.image_asset_id),
                    str(self.image_split),
                    self.backend,
                    self.dtype,
                    self.chunk_label,
                )
            )
        d["id"] = run_id({**d, **({"id": identity} if identity else {})})
        return d


# ---------------------------------------------------------------------------
# Run matrix generation
# ---------------------------------------------------------------------------


def compress_runs() -> list[RunSpec]:
    """Core sweep: chunk_size x scenario x codec (GPU-only)."""
    runs = []
    for sc in SINGLE_SCENARIOS:
        for codec in ["none", "lz4", "zstd", "blosc-lz4", "blosc-zstd"]:
            for cl in CHUNK_BYTES:
                runs.append(RunSpec(
                    scenario=sc, codec=codec, fill="xor",
                    backend="gpu", dtype="u16", chunk_label=cl,
                    blosc_block_bytes=16 * 1024 if codec.startswith("blosc-") else None,
                ))
    return runs


def backend_runs() -> list[RunSpec]:
    """GPU vs CPU backend comparison (superset of compress tier)."""
    runs = []
    for sc in SINGLE_SCENARIOS:
        for codec in ["none", "lz4", "zstd", "blosc-lz4", "blosc-zstd"]:
            for cl in CHUNK_BYTES:
                for backend in ["gpu", "cpu"]:
                    runs.append(RunSpec(
                        scenario=sc, codec=codec, fill="xor",
                        backend=backend, dtype="u16", chunk_label=cl,
                        blosc_block_bytes=16 * 1024 if codec.startswith("blosc-") else None,
                    ))
    return runs


def image_runs(tier: str, members: list[tuple[str, str, str]]) -> list[RunSpec]:
    """Full chunk x codec x backend image matrix for a sweep tier."""
    if tier not in IMAGE_SUPPORTED_TIERS:
        return []
    backends = ("gpu", "cpu")
    runs = []
    for asset_id, input_id, dtype in members:
        for codec, settings in IMAGE_CODEC_SETTINGS.items():
            for cl in CHUNK_BYTES:
                for backend in backends:
                    runs.append(
                        RunSpec(
                            scenario=MICROSCOPY_SCENARIO,
                            codec=codec,
                            fill="images",
                            input_id=input_id,
                            image_asset_id=asset_id,
                            image_split="core",
                            backend=backend,
                            dtype=dtype,
                            chunk_label=cl,
                            level=settings["level"],
                            blosc_shuffle=settings["blosc_shuffle"],
                            blosc_block_bytes=(
                                16 * 1024 if codec.startswith("blosc-") else None
                            ),
                        )
                    )
    return runs


def lod_runs() -> list[RunSpec]:
    """Multiscale / LOD sweeps."""
    runs = []
    chunk_labels = ["64K", "256K", "1M"]
    scenarios = ["orca2_multiscale", "256cube_multiscale", "medfmt_multiscale",
                 "orca2_multiscale_dim0", "256cube_multiscale_dim0", "medfmt_multiscale_dim0"]
    for sc in scenarios:
        for cl in chunk_labels:
            for codec in ["none", "zstd", "blosc-lz4", "blosc-zstd"]:
                for backend in ["gpu", "cpu"]:
                    runs.append(RunSpec(
                        scenario=sc, codec=codec, fill="xor",
                        backend=backend, dtype="u16", chunk_label=cl,
                        blosc_block_bytes=16 * 1024 if codec.startswith("blosc-") else None,
                    ))
    return runs


def io_runs() -> list[RunSpec]:
    """I/O tier: measure impact of zarr output vs discard sink."""
    runs = []
    default_chunk_labels = ["32K", "256K", "2M"]
    scenarios = ["orca2_single", "256cube_single",
                  "orca2_multiscale_dim0", "256cube_multiscale_dim0",
                  "smallepoch_single", "smallepoch_4shards"]
    for sc in scenarios:
        chunk_labels = (
            ["16K", *default_chunk_labels]
            if sc == "orca2_single"
            else default_chunk_labels
        )
        for cl in chunk_labels:
            for codec in ["none", "zstd", "blosc-lz4", "blosc-zstd"]:
                for backend in ["gpu", "cpu"]:
                    runs.append(RunSpec(
                        scenario=sc, codec=codec, fill="xor",
                        backend=backend, dtype="u16", chunk_label=cl,
                        sink="fs", blosc_block_bytes=16 * 1024 if codec.startswith("blosc-") else None,
                    ))
    return runs


def fill_runs() -> list[RunSpec]:
    """Fill-pattern sweep: xor vs zeros vs rand across codecs and chunk sizes."""
    runs = []
    chunk_labels = ["16K", "32K", "256K", "2M"]
    scenarios = ["orca2_single", "256cube_single"]
    for sc in scenarios:
        for fill in ["xor", "zeros", "rand"]:
            for codec in ["none", "lz4", "zstd", "blosc-lz4", "blosc-zstd"]:
                for cl in chunk_labels:
                    runs.append(RunSpec(
                        scenario=sc, codec=codec, fill=fill,
                        backend="gpu", dtype="u16", chunk_label=cl,
                        blosc_block_bytes=16 * 1024 if codec.startswith("blosc-") else None,
                    ))
    return runs


def s3_runs() -> list[RunSpec]:
    """S3 tier: discard vs fs vs S3 sink comparison, with throughput sweep."""
    runs = []
    chunk_labels = ["32K", "256K", "2M"]
    scenarios = ["orca2_single", "256cube_single"]
    for sc in scenarios:
        for cl in chunk_labels:
            for codec in ["none", "zstd", "blosc-lz4", "blosc-zstd"]:
                for backend in ["gpu", "cpu"]:
                    for throughput in [10, 100]:
                        for sink in ["discard", "fs", "s3"]:
                            runs.append(RunSpec(
                                scenario=sc, codec=codec, fill="xor",
                                backend=backend, dtype="u16", chunk_label=cl,
                                blosc_block_bytes=16 * 1024 if codec.startswith("blosc-") else None,
                                sink=sink,
                                s3_throughput_gbps=throughput if sink == "s3" else 0,
                            ))
    return runs


def blosc_runs() -> list[RunSpec]:
    runs = []
    for cl in ["16K", "256K", "1M"]:
        for backend in ["gpu", "cpu"]:
            for codec in ["lz4", "zstd"]:
                runs.append(RunSpec(
                    scenario="orca2_single", codec=codec, fill="xor",
                    backend=backend, dtype="u16", chunk_label=cl,
                ))
            for codec in ["blosc-lz4", "blosc-zstd"]:
                for shuffle in ["none", "byte", "bit"]:
                    runs.append(RunSpec(
                        scenario="orca2_single", codec=codec, fill="xor",
                        backend=backend, dtype="u16", chunk_label=cl,
                        blosc_shuffle=shuffle, blosc_block_bytes=16 * 1024,
                    ))
    return runs


TIERS = {
    "compress": compress_runs,
    "backend": backend_runs,
    "lod": lod_runs,
    "fill": fill_runs,
    "io": io_runs,
    "s3": s3_runs,
    "blosc": blosc_runs,
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
# Registered image inputs
# ---------------------------------------------------------------------------

def _image_modules():
    """Load the dataset tooling only when the microscopy scenario is selected."""
    datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
    path = str(datasets_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    import data_sources
    import run as image_runner

    return data_sources, image_runner


def load_image_members(
    registry_path: Path, dataset_id: str | None, corpus_path: Path | None = None
) -> list[tuple[str, str, str]]:
    data_sources, _ = _image_modules()
    return data_sources.load_members(registry_path, dataset_id, corpus_path)


def load_image_corpus(
    registry_path: Path, dataset_id: str | None, corpus_path: Path | None
):
    data_sources, _ = _image_modules()
    return data_sources.load_corpus(
        registry_path, dataset_id, corpus_path, None
    )


def check_image_result(
    result: dict, pack: dict, frames: int, spec: RunSpec, process_wall_s: float,
) -> dict:
    validate_measurement(result)
    replay = result["image_replay"]
    window = result["measurement"]
    dtype, bytes_per_element = IMAGE_DTYPES[spec.dtype]
    expected = {
        "backend": spec.backend, "dtype": dtype, "codec": spec.codec,
        "codec_level": spec.level, "shuffle": spec.blosc_shuffle,
        "source_bytes": pack["bytes"], "order": "cyclic",
        "target_batch_bytes": 64 * 1024**2,
    }
    for key, value in expected.items():
        if replay.get(key) != value:
            raise ValueError(f"Image replay changed {key}: expected {value}")
    chunk = replay["chunk_shape"]
    if (len(chunk) != 3 or any(type(n) is not int or n <= 0 for n in chunk)
            or math.prod(chunk) * bytes_per_element != CHUNK_BYTES[spec.chunk_label]):
        raise ValueError("Image chunk geometry disagrees with requested target")
    height, width = pack["height"], pack["width"]
    padded_frame = math.ceil(height / chunk[1]) * chunk[1] * math.ceil(width / chunk[2]) * chunk[2] * bytes_per_element
    measured_frames, partial = divmod(window["input_bytes"], padded_frame)
    if partial or measured_frames < frames:
        raise ValueError("Image measurement has partial or insufficient measured frames")
    if (replay["shape"] != [measured_frames, height, width]
            or result["submitted_bytes"] != measured_frames * padded_frame
            or result["logical_input_bytes"] != measured_frames * height * width * bytes_per_element):
        raise ValueError("Image logical/submitted byte accounting disagrees")
    source_bytes = len(pack["planes"]) * padded_frame
    if replay["source_padded_bytes"] != source_bytes or window["source_bytes"] != source_bytes:
        raise ValueError("Image padded source byte accounting disagrees")
    if result["worker_threads"] != 4:
        raise ValueError("Image benchmark did not use four workers")
    if spec.codec.startswith("blosc-") and result["blosc_block_bytes"] != spec.blosc_block_bytes:
        raise ValueError("Image Blosc block size changed")
    return {key: replay[key] for key in IMAGE_LAYOUT_KEYS}


def image_build_record(executable: Path) -> dict:
    _, image_runner = _image_modules()
    return image_runner.build_record(executable, None)


def image_executable(build_dir: Path) -> Path:
    executable = build_dir / "bench" / "bench_stream_images"
    return executable.with_suffix(".exe") if sys.platform == "win32" else executable


def image_protocol(min_gib: float, repeats: int, smoke: bool) -> dict:
    return {
        "minimum_bytes": math.ceil(min_gib * 1024**3),
        "warmups": 0,
        "repeats": repeats,
        "smoke": smoke,
        "batch_bytes": 64 * 1024**2,
        "workers": 4,
        "sink": "discard",
        "chunk_ratios": [1, 4, 4],
        "min_full_shard_bytes": 512 * 1024**2,
        "max_full_shard_bytes": 1024**3,
        "order": "cyclic",
        "scales": 1,
        "repeatability_range_percent": 5,
    }


def image_corpus_record(corpus) -> dict:
    record = corpus.record()
    record["selected_assets"] = [
        {
            "asset": pack["id"],
            "input": pack.get("input_id", pack["source_group"]),
            "sha256": pack["sha256"],
            "dtype": pack["dtype"],
            "width": pack["width"],
            "height": pack["height"],
            "split": pack["split"],
        }
        for pack in corpus.manifest["packs"]
    ]
    return record


def image_execution_count(runs: list[RunSpec], repeats: int, smoke: bool) -> int:
    return sum(repeats if repeats is not None else
               DEFAULT_IMAGE_REPEATS if run.scenario == MICROSCOPY_SCENARIO else 1
               for run in runs)


def existing_image_layouts(runs: list[dict]) -> dict:
    layouts = {}
    for run in runs:
        if run.get("scenario") != MICROSCOPY_SCENARIO or run.get("status") != "pass":
            continue
        replay = run.get("image_replay")
        asset_id = run.get("image_asset_id")
        chunk_label = run.get("chunk_bytes_label")
        if not isinstance(replay, dict) or not asset_id or not chunk_label:
            continue
        try:
            layout = {key: replay[key] for key in IMAGE_LAYOUT_KEYS}
        except KeyError:
            continue
        identity = (asset_id, chunk_label)
        if identity in layouts and layouts[identity] != layout:
            raise ValueError(
                f"Existing image layouts disagree for {asset_id} at {chunk_label}"
            )
        layouts[identity] = layout
    return layouts


def run_image_one(
    spec: RunSpec,
    build_dir: Path,
    corpus,
    min_gib: float,
    repeats: int,
    smoke: bool,
    layouts: dict | None = None,
    *, warmup: float = DEFAULT_WARMUP_S, duration: float = DEFAULT_DURATION_S,
    geometry_frames: int | None = None,
) -> dict | None:
    """Run image repetitions with the common invocation policy."""
    executable = image_executable(build_dir)
    if not executable.exists():
        return None

    packs = {pack["id"]: pack for pack in corpus.manifest["packs"]}
    try:
        pack = packs[spec.image_asset_id]
        input_path = corpus.pack_files[spec.image_asset_id]
    except KeyError as error:
        raise ValueError(
            f"Image asset {spec.image_asset_id!r} is absent from the selected dataset"
        ) from error
    input_id = pack.get("input_id", pack["source_group"])
    if input_id != spec.input_id:
        raise ValueError(
            f"Image asset {spec.image_asset_id!r} maps to {input_id!r}, not {spec.input_id!r}"
        )
    if pack["split"] != spec.image_split:
        raise ValueError(
            f"Image asset {spec.image_asset_id!r} is in split {pack['split']!r}, "
            f"not {spec.image_split!r}"
        )

    dtype, bytes_per_element = IMAGE_DTYPES[spec.dtype]
    if pack["dtype"] != dtype:
        raise ValueError(
            f"Image asset {spec.image_asset_id!r} has dtype {pack['dtype']!r}, "
            f"not {dtype!r}"
        )
    minimum_bytes = math.ceil(min_gib * 1024**3)
    frame_bytes = pack["width"] * pack["height"] * bytes_per_element
    frames = (minimum_bytes + frame_bytes - 1) // frame_bytes
    executions = repeats
    measured = []
    case_layout = None

    for iteration in range(executions):
        command = [
            str(executable),
            "--input", str(input_path),
            "--width", str(pack["width"]),
            "--height", str(pack["height"]),
            "--dtype", spec.dtype,
            "--frames", str(frames),
            "--warmup", str(warmup),
            "--duration", str(duration),
            "--backend", spec.backend,
            "--chunk-bytes", spec.chunk_label,
            "--batch-bytes", "64M",
            "--max-threads", "4",
            "--json",
            "--codec", spec.codec,
            "--codec-level", str(spec.level),
        ]
        if geometry_frames is not None:
            command.extend(["--geometry-frames", str(geometry_frames)])
        if spec.codec.startswith("blosc-"):
            command.extend(
                [
                    "--shuffle", spec.blosc_shuffle,
                    "--blosc-block-bytes", str(spec.blosc_block_bytes),
                ]
            )

        measurement = execute(command)
        if measurement["status"] != "pass":
            return {**spec.base_result(), **measurement}
        layout = check_image_result(
            measurement, pack, frames, spec, measurement["process_wall_s"]
        )
        if case_layout is not None and layout != case_layout:
            raise ValueError("Image layout changed between repetitions")
        case_layout = layout
        measured.append(measurement)

    if len(measured) != repeats:
        raise ValueError("Image benchmark produced an incomplete repetition set")
    if layouts is not None:
        key = (spec.image_asset_id, spec.chunk_label)
        if key in layouts and layouts[key] != case_layout:
            raise ValueError(
                f"Image layout changed across codecs/backends for {spec.image_asset_id} "
                f"at {spec.chunk_label}"
            )
        layouts[key] = case_layout

    result = aggregate_repetitions(measured)
    result.update(spec.base_result())
    result.update(
        {
            "status": "pass",
            "frames": result["image_replay"]["shape"][0],
            "geometry_frames": result["measurement"]["reference_frames"],
            "input_label": input_label(str(spec.input_id))
            + (f" ({corpus.manifest['kind']})" if corpus.manifest["kind"] != "raw" else "")
            + (" (smoke)" if smoke else ""),
            "image_input": {
                "release": corpus.manifest.get("release"),
                "kind": corpus.manifest["kind"],
                "manifest_sha256": corpus.sha256,
                "pack_id": pack["id"],
                "pack_sha256": pack["sha256"],
                "dtype": pack["dtype"],
                "modality": pack["modality"],
                "split": pack["split"],
                "input_id": spec.input_id,
                "source_group": pack["source_group"],
                "plane_order": [plane["id"] for plane in pack["planes"]],
            },
        }
    )
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(spec: RunSpec, build_dir: Path, s3_bucket: str | None = None,
            s3_region: str | None = None, s3_endpoint: str | None = None,
            tmpdir_root: Path | None = None, *, image_corpus=None,
            image_min_gib: float = DEFAULT_IMAGE_MIN_GIB,
            image_repeats: int = DEFAULT_IMAGE_REPEATS,
            image_smoke: bool = False,
            image_layouts: dict | None = None,
            warmup: float = DEFAULT_WARMUP_S,
            duration: float = DEFAULT_DURATION_S,
            repeats: int = 1,
            geometry_frames: int | None = None) -> dict | None:
    """Execute a single benchmark run, return result dict or None if exe missing."""
    if spec.scenario == MICROSCOPY_SCENARIO:
        if image_corpus is None:
            raise ValueError("The microscopy scenario requires a verified corpus")
        return run_image_one(
            spec,
            build_dir,
            image_corpus,
            image_min_gib,
            image_repeats,
            image_smoke,
            image_layouts,
            warmup=warmup, duration=duration, geometry_frames=geometry_frames,
        )

    exe = build_dir / "bench" / f"bench_stream_{spec.scenario}"
    if sys.platform == "win32":
        exe = exe.with_suffix(".exe")
    if not exe.exists():
        return None

    cmd = [
        str(exe),
        "--codec", spec.codec,
        "--level", str(spec.level),
        "--fill", spec.fill,
        "--backend", spec.backend,
        "--dtype", spec.dtype,
        "--chunk-bytes", spec.chunk_label,
        "--geometry-frames", str(geometry_frames or spec.frames),
        "--warmup", str(warmup),
        "--duration", str(duration),
        "--json",
    ]
    if spec.codec.startswith("blosc-"):
        cmd.extend(["--blosc-block-bytes", str(spec.blosc_block_bytes),
                    "--blosc-shuffle", spec.blosc_shuffle])

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
        executions = []
        for _ in range(repeats):
            parsed = execute(cmd)
            if parsed["status"] != "pass":
                return {**spec.base_result(), **parsed}
            executions.append(parsed)
        result = {**spec.base_result(), **aggregate_repetitions(executions)}
        result["geometry_frames"] = result["measurement"]["reference_frames"]
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
@click.option("--scenario", "scenario_filter", multiple=True,
              type=click.Choice(sorted(SCENARIOS)),
              help="Only run this scenario. Repeat to combine scenarios; microscopy is opt-in.")
@click.option("--backend", "backend_filter", type=click.Choice(sorted(VALID_BACKENDS)),
              help="Only run benchmarks for this backend.")
@click.option("--blosc-shuffle", type=click.Choice(sorted(VALID_SHUFFLES)), default=None,
              help="Use this shuffle for all selected Blosc runs (deduplicated).")
@click.option("--level", type=click.IntRange(0, 9), default=None,
              help="Use this level for all selected Blosc runs (0 = store only).")
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
@click.option("--data-registry", type=click.Path(path_type=Path),
              default=DEFAULT_DATA_REGISTRY, show_default=True,
              help="Registered data sources used by the microscopy scenario.")
@click.option("--dataset", "image_dataset", default=None,
              help="Registered image dataset id (default: registry default).")
@click.option("--corpus", "image_corpus_path", type=click.Path(path_type=Path),
              default=None, help="Override the registered image corpus checkout.")
@click.option("--min-gib", "image_min_gib", type=float,
              default=DEFAULT_IMAGE_MIN_GIB, show_default=True,
              help="Minimum native image GiB per image process execution.")
@click.option("--repeats", type=click.IntRange(min=1), default=None,
              help="Measured executions per configuration (default: microscopy 5, generated 1).")
@click.option("--warmup", type=click.FloatRange(min=0), default=DEFAULT_WARMUP_S,
              show_default=True, help="Minimum warmup seconds for every execution.")
@click.option("--duration", type=click.FloatRange(min=0, min_open=True),
              default=DEFAULT_DURATION_S, show_default=True,
              help="Minimum measured append seconds, extended for coverage.")
@click.option("--geometry-frames", type=click.IntRange(min=1), default=None,
              help="Override the fixed geometry reference for selected scenarios.")
@click.option("--codec", "codec_filter", multiple=True,
              type=click.Choice(sorted(VALID_CODECS)), help="Select codecs.")
@click.option("--chunk-bytes", "chunk_filter", multiple=True,
              type=click.Choice(list(CHUNK_BYTES)), help="Select chunk targets.")
@click.option("--input", "input_filter", multiple=True,
              help="Select semantic input ids, such as opencell-dna.")
@click.option("--smoke", "image_smoke", is_flag=True,
              help="Exclude this sweep from performance trends; permit short image replays. Coverage still applies.")
@click.option("--machine", "machine_name", default=None, envvar="CHUCKY_MACHINE",
              help="Name this machine goes by in the report (default: hostname). "
                   "Give a stable name where the hostname changes between runs, as "
                   "it does on a cluster; group names live in bench/machines.toml.")
def main(tier, run_all, scenario_filter, backend_filter, blosc_shuffle, level,
         build_dir, output, skip, retry, rerun, dry_run, s3_bucket, s3_region,
         s3_endpoint, tmpdir_root, data_registry, image_dataset,
         image_corpus_path, image_min_gib, repeats, image_smoke,
         machine_name, warmup, duration, geometry_frames, codec_filter,
         chunk_filter, input_filter):
    """Benchmark sweep runner for chucky."""
    if not math.isfinite(warmup) or not math.isfinite(duration):
        raise click.BadParameter("warmup and duration must be finite")
    policy = measurement_policy(warmup, duration)
    policy["geometry_frames"] = geometry_frames
    repetition_policy = {"generated": repeats or 1, "images": repeats or DEFAULT_IMAGE_REPEATS}
    image_repeats = repetition_policy["images"]
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
    selected_scenarios = set(scenario_filter)
    if selected_scenarios:
        runs = [run for run in runs if run.scenario in selected_scenarios]
    if MICROSCOPY_SCENARIO in selected_scenarios:
        supported = [tier for tier in selected_tiers if tier in IMAGE_SUPPORTED_TIERS]
        if not supported:
            raise click.UsageError(
                "The microscopy scenario is available in the compress and backend tiers"
            )
        try:
            members = load_image_members(data_registry, image_dataset, image_corpus_path)
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        for name in supported:
            runs.extend(image_runs(name, members))
    if blosc_shuffle is not None or level is not None:
        overrides = {}
        if blosc_shuffle is not None:
            overrides["blosc_shuffle"] = blosc_shuffle
        if level is not None:
            overrides["level"] = level
        runs = [RunSpec(**{**r.model_dump(), **overrides})
                if r.codec.startswith("blosc-") else r for r in runs]
    runs = deduplicate(runs)
    if backend_filter:
        runs = [r for r in runs if r.backend == backend_filter]
    if skip:
        runs = [r for r in runs if not any(pat in r.scenario for pat in skip)]
    if codec_filter:
        runs = [r for r in runs if r.codec in codec_filter]
    if chunk_filter:
        runs = [r for r in runs if r.chunk_label in chunk_filter]
    if input_filter:
        runs = [r for r in runs if (r.input_id or r.fill) in input_filter]

    image_specs = [run for run in runs if run.scenario == MICROSCOPY_SCENARIO]
    image_was_skipped = any(pattern in MICROSCOPY_SCENARIO for pattern in skip)
    if (
        MICROSCOPY_SCENARIO in selected_scenarios
        and not image_specs
        and not image_was_skipped
    ):
        raise click.UsageError(
            "No image configurations remain after filtering."
        )
    if image_specs:
        if image_repeats < 1 or (not image_smoke and image_repeats < 3):
            raise click.BadParameter(
                "a non-smoke image sweep needs at least three measured runs",
                param_hint="--repeats",
            )
        if not math.isfinite(image_min_gib) or image_min_gib <= 0:
            raise click.BadParameter("must be positive", param_hint="--min-gib")
        if not image_smoke and image_min_gib < DEFAULT_IMAGE_MIN_GIB:
            raise click.BadParameter(
                "a throughput image sweep needs at least 32 GiB; use --smoke for a quick check",
                param_hint="--min-gib",
            )

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
        table.add_column("Input")
        table.add_column("Codec")
        table.add_column("Shuffle")
        table.add_column("Level", justify="right")
        table.add_column("Fill")
        table.add_column("Backend")
        table.add_column("Dtype")
        table.add_column("Chunk")
        table.add_column("Blosc block (bytes)", justify="right")
        table.add_column("Sink", justify="center")
        for i, r in enumerate(runs, 1):
            table.add_row(
                str(i), r.scenario, r.input_id or r.fill, r.codec,
                r.blosc_shuffle, str(r.level), r.fill,
                r.backend, r.dtype, r.chunk_label,
                str(r.blosc_block_bytes) if r.blosc_block_bytes is not None else "",
                r.sink if r.sink != "discard" else "",
            )
        console.print(table)
        executions = image_execution_count(runs, repeats, image_smoke)
        console.print(
            f"\nTotal: [bold]{len(runs)}[/bold] configurations, "
            f"[bold]{executions}[/bold] process executions across tiers: "
            f"{', '.join(selected_tiers)}"
        )
        console.print(f"Minimum warmup: {warmup:g} s; measurement: {duration:g} s plus drain")
        console.print(f"Output: {output}")
        return

    verified_image_corpus = None
    current_image_corpus = None
    current_image_protocol = None
    if image_specs:
        try:
            verified_image_corpus = load_image_corpus(
                data_registry, image_dataset, image_corpus_path
            )
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        current_image_corpus = image_corpus_record(verified_image_corpus)
        current_image_protocol = image_protocol(
            image_min_gib, image_repeats, image_smoke
        )

    # -- load existing results for resumability --
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if output.exists():
        with open(output) as f:
            raw_data = json.load(f)
        if (raw_data.get("version") != CURRENT_VERSION
                or raw_data.get("measurement_policy") != policy
                or raw_data.get("repetition_policy") != repetition_policy
                or raw_data.get("smoke", False) != image_smoke):
            raise click.ClickException(
                "Existing results use a different or unknown measurement/repetition policy; "
                "choose a new --output file.")
        try:
            validate_results(raw_data)
        except Exception as e:
            console.print(f"[yellow]Warning: results file validation failed: {e}[/yellow]")
            console.print("[yellow]Continuing with raw data.[/yellow]")
        data = raw_data
        for r in data.get("runs", []):
            migrate_scenario(r)
            rid = run_id(r)
            if retry and r.get("status") != "pass":
                continue
            if rerun and any(pat in rid for pat in rerun):
                continue
            existing[rid] = r
    else:
        gpu, driver = gpu_and_driver()
        data = {
            "version": CURRENT_VERSION,
            "measurement_policy": policy,
            "repetition_policy": repetition_policy,
            "smoke": image_smoke,
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

    previous_image_runs = any(
        run.get("scenario") == MICROSCOPY_SCENARIO for run in data.get("runs", [])
    )
    if image_specs:
        if previous_image_runs:
            previous_corpus = data.get("corpus", {})
            previous_identity = {
                key: previous_corpus.get(key)
                for key in ("dataset", "selected_assets")
            }
            current_identity = {
                key: current_image_corpus.get(key)
                for key in ("dataset", "selected_assets")
            }
            if previous_identity != current_identity:
                raise click.ClickException(
                    "Existing image runs use a different registered dataset or asset content"
                )
            if data.get("image_protocol") != current_image_protocol:
                raise click.ClickException(
                    "Existing image runs use a different repetition or replay protocol"
                )
        data["corpus"] = current_image_corpus
        data["image_protocol"] = current_image_protocol
        executable = image_executable(build_dir)
        if not previous_image_runs and executable.exists():
            data["image_benchmark"] = image_build_record(executable)

    # -- count how many actually need to run --
    to_run = [spec for spec in runs if spec.id not in existing]
    skip_count = len(runs) - len(to_run)

    if skip_count:
        console.print(f"Skipping [bold]{skip_count}[/bold] existing runs")

    if not to_run:
        console.print(f"[green]All {len(runs)} runs already complete.[/green]")
        console.print(f"Results: {output}")
        return

    try:
        image_layouts = existing_image_layouts(data.get("runs", []))
    except ValueError as error:
        raise click.ClickException(str(error)) from error

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
            if spec.input_id:
                tag = (
                    f"{spec.scenario}/{spec.input_id} {spec.codec} "
                    f"{spec.backend} {spec.chunk_label}"
                )
            if spec.codec.startswith("blosc-"):
                tag += f" {spec.blosc_shuffle} level={spec.level}"
            if spec.sink != "discard":
                tag += f" {spec.sink}"
            progress.update(task, description=f"[bold]{tag}")

            try:
                result = run_one(spec, build_dir,
                                 s3_bucket=s3_bucket,
                                 s3_region=s3_region,
                                 s3_endpoint=s3_endpoint,
                                 tmpdir_root=tmpdir_root,
                                 image_corpus=verified_image_corpus,
                                 image_min_gib=image_min_gib,
                                 image_repeats=image_repeats,
                                 image_smoke=image_smoke,
                                 image_layouts=image_layouts,
                                 warmup=warmup, duration=duration,
                                 repeats=repetition_policy["generated"],
                                 geometry_frames=geometry_frames)
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

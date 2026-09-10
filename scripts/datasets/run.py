from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from data_sources import DEFAULT_REGISTRY, load_corpus
from environment import collect_environment, digest_source, source_tree_digest
from manifest import digest_file, git_output, read_json

BLOSC_BLOCK_BYTES = 16 * 1024
MIN_FULL_SHARD_BYTES = 512 * 1024**2
MAX_FULL_SHARD_BYTES = 1024**3
DEFAULT_MIN_GIB = 32
DEFAULT_REPEATS = 5
DEFAULT_TIER = "codec"
DEFAULT_CHUNK_BYTES = ("32K",)
DEFAULT_PROFILES = ("none", "zstd", "blosc-lz4", "blosc-zstd")
SCENARIO = "images"

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

PROFILES = {
    "none": ["--codec", "none", "--codec-level", "0"],
    "lz4": ["--codec", "lz4", "--codec-level", "1"],
    "zstd": ["--codec", "zstd", "--codec-level", "3"],
    "blosc-lz4": [
        "--codec",
        "blosc-lz4",
        "--codec-level",
        "3",
        "--shuffle",
        "bit",
        "--blosc-block-bytes",
        str(BLOSC_BLOCK_BYTES),
    ],
    "blosc-zstd": [
        "--codec",
        "blosc-zstd",
        "--codec-level",
        "3",
        "--shuffle",
        "bit",
        "--blosc-block-bytes",
        str(BLOSC_BLOCK_BYTES),
    ],
}
PROFILE_LEVELS = {
    "none": 0,
    "lz4": 1,
    "zstd": 3,
    "blosc-lz4": 3,
    "blosc-zstd": 3,
}

# The named presets match the codec/chunk/backend axes in sweep.py. Explicit
# --backends, --profiles, and --chunk-bytes values replace a preset axis, which
# lets a CPU-only machine run the compress matrix without inventing a new tier.
IMAGE_TIERS = {
    "codec": {
        "backends": ("gpu",),
        "profiles": DEFAULT_PROFILES,
        "chunk_bytes": DEFAULT_CHUNK_BYTES,
    },
    "compress": {
        "backends": ("gpu", "cpu"),
        "profiles": tuple(PROFILES),
        "chunk_bytes": tuple(CHUNK_BYTES),
    },
    "backend": {
        "backends": ("gpu", "cpu"),
        "profiles": tuple(PROFILES),
        "chunk_bytes": tuple(CHUNK_BYTES),
    },
}
LAYOUT_KEYS = (
    "shape",
    "chunk_shape",
    "chunks_per_shard",
    "epochs_per_batch",
    "target_batch_bytes",
    "actual_batch_bytes",
    "append_elements",
    "dtype",
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def machine_record(name: str, gpu: bool) -> dict:
    result = {
        "name": name,
        "hostname": platform.node(),
        "system": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "environment": {
            key: os.environ[key]
            for key in (
                "SLURM_JOB_ID",
                "SLURM_CPUS_PER_TASK",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
            if key in os.environ
        },
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        result["cpu_models"] = sorted(
            {
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            }
        )
    if gpu:
        result["nvidia_smi"] = command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
    return result


def build_record(executable: Path, toolchain: Path | None) -> dict:
    root = Path(__file__).resolve().parents[2]
    result = {
        "revision": git_output(root, "rev-parse", "HEAD"),
        "worktree_status": git_output(root, "status", "--porcelain"),
        "executable_sha256": digest_file(executable),
        "executable": str(executable),
    }
    sources = [
        root / "bench/bench_util.c",
        root / "bench/bench_input.c",
        root / "bench/bench_stream_images.c",
        root / "bench/bench_report.c",
    ]
    result["benchmark_source_sha256"] = {
        path.relative_to(root).as_posix(): digest_source(path) for path in sources
    }
    result["source_tree_sha256"] = source_tree_digest(root)
    for parent in executable.parents:
        cache = parent / "CMakeCache.txt"
        if cache.is_file():
            result["cmake_cache_sha256"] = digest_file(cache)
            result["toolchain"] = collect_environment(parent)
            result["build_settings"] = {
                line.split(":", 1)[0]: line.split("=", 1)[1]
                for line in cache.read_text().splitlines()
                if "=" in line
                and ":" in line
                and not line.startswith(("#", "//"))
                and any(
                    token in line.split(":", 1)[0].lower()
                    for token in (
                        "compiler",
                        "cuda",
                        "nvcomp",
                        "blosc",
                        "zstd",
                        "build_type",
                    )
                )
            }
            break
    if toolchain is not None:
        result["toolchain"] = read_json(toolchain)
        built_source = result["toolchain"].get("source_tree_sha256")
        if built_source and built_source != result["source_tree_sha256"]:
            raise ValueError(
                "Sources changed after the recorded build; rebuild before benchmarking"
            )
    return result


def check_result(
    result: dict,
    pack: dict,
    frames: int,
    backend: str,
    profile: str,
    chunk_bytes: int,
    process_wall_s: float | None = None,
    *,
    expected_level: int | None = None,
    expected_shuffle: str | None = None,
    expected_blosc_block_bytes: int | None = None,
) -> dict:
    if result.get("status") != "pass":
        raise ValueError("Benchmark did not report success")
    replay = result["image_replay"]
    expected = {
        "backend": backend,
        "codec": profile,
        "dtype": "u16le",
        "shape": [frames, pack["height"], pack["width"]],
    }
    for key, value in expected.items():
        if replay.get(key) != value:
            raise ValueError(
                f"Benchmark changed {key}: expected {value}, got {replay.get(key)}"
            )
    chunk_shape = replay.get("chunk_shape")
    if (
        not isinstance(chunk_shape, list)
        or len(chunk_shape) != 3
        or any(not isinstance(value, int) or value <= 0 for value in chunk_shape)
        or math.prod(chunk_shape) * 2 != chunk_bytes
    ):
        raise ValueError(
            f"Benchmark changed chunk_shape: expected {chunk_bytes} decoded bytes, "
            f"got {chunk_shape}"
        )
    codec_level = (
        PROFILE_LEVELS[profile] if expected_level is None else expected_level
    )
    shuffle = (
        ("bit" if profile.startswith("blosc-") else "none")
        if expected_shuffle is None
        else expected_shuffle
    )
    expected.update(
        target_batch_bytes=64 * 1024**2,
        source_bytes=pack["bytes"],
        order="cyclic",
        codec_level=codec_level,
        shuffle=shuffle,
    )
    for key, value in expected.items():
        if replay.get(key) != value:
            raise ValueError(
                f"Benchmark changed {key}: expected {value}, got {replay.get(key)}"
            )
    if result["worker_threads"] != 4:
        raise ValueError("Benchmark did not use four workers")
    if result["input_bytes"] != frames * pack["width"] * pack["height"] * 2:
        raise ValueError("Benchmark input byte count is wrong")
    block_bytes = (
        BLOSC_BLOCK_BYTES
        if expected_blosc_block_bytes is None
        else expected_blosc_block_bytes
    )
    if profile.startswith("blosc-") and result.get("blosc_block_bytes") != block_bytes:
        raise ValueError("Blosc block size changed")
    for key in ("wall_s", "throughput_in_gibs", "logical_compression_fold"):
        value = result[key]
        if (
            not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid measurement: {key}")
    if (
        process_wall_s is not None
        and result["wall_s"] > process_wall_s * 1.05 + 0.1
    ):
        raise ValueError(
            "Benchmark clock disagrees with process clock; the machine may have "
            "slept during the measurement"
        )
    padded_height = (pack["height"] + chunk_shape[1] - 1) // chunk_shape[1]
    padded_width = (pack["width"] + chunk_shape[2] - 1) // chunk_shape[2]
    padded_frame_bytes = (
        padded_height
        * padded_width
        * chunk_shape[1]
        * chunk_shape[2]
        * 2
    )
    padded_frames = (
        (frames + chunk_shape[0] - 1) // chunk_shape[0] * chunk_shape[0]
    )
    if result["padded_input_bytes"] != padded_frames * padded_frame_bytes:
        raise ValueError("Padded input byte count is wrong")
    source_frames = pack["bytes"] // (pack["height"] * pack["width"] * 2)
    if replay["source_padded_bytes"] != source_frames * padded_frame_bytes:
        raise ValueError("Preloaded padded source byte count is wrong")
    full_shard_bytes = (
        math.prod(replay["chunk_shape"])
        * math.prod(replay["chunks_per_shard"])
        * 2
    )
    if full_shard_bytes > MAX_FULL_SHARD_BYTES:
        raise ValueError("Full decoded shard geometry exceeds 1 GiB")
    if (
        frames * padded_frame_bytes >= 4 * MIN_FULL_SHARD_BYTES
        and full_shard_bytes < MIN_FULL_SHARD_BYTES
    ):
        raise ValueError("Full decoded shard geometry is smaller than 0.5 GiB")
    return {key: replay[key] for key in LAYOUT_KEYS}


def summaries(runs: list[dict]) -> list[dict]:
    grouped = {}
    for run in runs:
        if run.get("status") != "pass" or run["warmup"]:
            continue
        grouped.setdefault(
            (
                run["pack_id"],
                run["backend"],
                run["profile"],
                run["chunk_bytes_label"],
            ),
            [],
        ).append(run)
    rows = []
    for key, values in sorted(grouped.items()):
        first = values[0]
        measurements = [v["measurement"] for v in values]
        total_input = sum(v["input_bytes"] for v in measurements)
        total_output = sum(v["output_bytes"] for v in measurements)
        rates = [v["throughput_in_gibs"] for v in measurements]
        median = statistics.median(rates)
        spread_percent = 100 * (max(rates) - min(rates)) / median
        rows.append(
            {
                "pack_id": key[0],
                "backend": key[1],
                "profile": key[2],
                "chunk_bytes_label": key[3],
                "chunk_bytes": first["chunk_bytes"],
                "modality": first["modality"],
                "split": first["split"],
                "input_id": first.get("input_id", first["source_group"]),
                "source_group": first["source_group"],
                "width": first["width"],
                "height": first["height"],
                "pack_sha256": first["pack_sha256"],
                "repeats": len(values),
                "median_throughput_gibs": median,
                "min_throughput_gibs": min(rates),
                "max_throughput_gibs": max(rates),
                "throughput_spread_percent": spread_percent,
                "repeatability": (
                    "inconclusive"
                    if len(values) < 3
                    else "within_target"
                    if spread_percent <= 5
                    else "above_target"
                ),
                "logical_compression_fold": total_input / total_output,
                "edge_padding_fraction": sum(
                    v["padded_input_bytes"] for v in measurements
                )
                / total_input
                - 1,
                "median_load_s": statistics.median(
                    v["image_replay"]["load_s"] for v in measurements
                ),
                "input_bytes": total_input,
                "output_bytes": total_output,
                "layout": first["layout"],
            }
        )
    return rows


def assess(rows: list[dict], smoke: bool) -> list[dict]:
    groups = {}
    for row in rows:
        key = (
            row["modality"],
            row.get("input_id", row["source_group"]),
            row["width"],
            row["height"],
            row["backend"],
            row["profile"],
            row.get("chunk_bytes_label", "unknown"),
        )
        groups.setdefault(key, {}).setdefault(row["split"], []).append(row)
    results = []
    for key, splits in sorted(groups.items()):
        item = dict(
            zip(
                (
                    "modality",
                    "input_id",
                    "width",
                    "height",
                    "backend",
                    "profile",
                    "chunk_bytes_label",
                ),
                key,
            )
        )
        incomplete = any(
            row["repeats"] < 3 for values in splits.values() for row in values
        )
        split_names = set(splits)
        if smoke or incomplete or split_names != {"core", "heldout"}:
            if smoke or incomplete:
                reason = "At least three measured repetitions are required"
            elif split_names == {"core"}:
                reason = "Corpus has no heldout sample"
            else:
                reason = "A matched core and heldout pilot is required"
            results.append(
                item
                | {
                    "status": "inconclusive",
                    "reason": reason,
                }
            )
            continue
        layouts = {
            json.dumps(row["layout"], sort_keys=True)
            for values in splits.values()
            for row in values
        }
        if len(layouts) != 1:
            results.append(
                item
                | {
                    "status": "inconclusive",
                    "reason": "Core and heldout layouts differ",
                }
            )
            continue

        def metrics(values):
            return (
                sum(v["input_bytes"] for v in values)
                / sum(v["output_bytes"] for v in values),
                statistics.median(v["median_throughput_gibs"] for v in values),
            )

        core, heldout = metrics(splits["core"]), metrics(splits["heldout"])
        ratio_delta = abs(core[0] / heldout[0] - 1)
        speed_delta = abs(core[1] / heldout[1] - 1)
        results.append(
            item
            | {
                "status": "within_target"
                if max(ratio_delta, speed_delta) <= 0.10
                else "revise_selection",
                "compression_relative_difference": ratio_delta,
                "throughput_relative_difference": speed_delta,
            }
        )
    return results


def save_results(output: Path, document: dict) -> None:
    rows = summaries(document["runs"])
    document["summary"] = rows
    document["representativeness"] = assess(
        rows, document["protocol"]["smoke"] or document["corpus"]["kind"] != "raw"
    )
    write_json(output / "results.json", document)
    if rows:
        columns = [key for key in rows[0] if key != "layout"]
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def resolve_axes(args) -> None:
    defaults = IMAGE_TIERS[args.tier]
    args.backends = list(args.backends or defaults["backends"])
    args.profiles = list(args.profiles or defaults["profiles"])
    args.chunk_bytes = list(args.chunk_bytes or defaults["chunk_bytes"])


def run(args, corpus) -> None:
    resolve_axes(args)
    for name in ("backends", "profiles", "chunk_bytes"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            raise ValueError(f"{name.replace('_', ' ').capitalize()} must not repeat")
    if args.repeats < 1 or (not args.smoke and args.repeats < 3):
        raise ValueError("A pilot needs at least three measured runs")
    if not args.smoke and args.min_gib < 32:
        raise ValueError(
            "A throughput run needs at least 32 GiB; use --smoke for a quick check"
        )
    if not math.isfinite(args.min_gib) or args.min_gib <= 0:
        raise ValueError("--min-gib must be positive")
    packs = corpus.manifest["packs"]
    if not packs:
        raise ValueError("The selected dataset has no image assets")
    executable = args.executable.expanduser().resolve(strict=True)
    executions_per_case = args.repeats + (0 if args.smoke else 1)
    cases = len(packs) * len(args.chunk_bytes) * len(args.backends) * len(args.profiles)
    print(
        f"{args.tier} tier: {len(packs)} pack(s) x {len(args.chunk_bytes)} chunk(s) "
        f"x {len(args.backends)} backend(s) x {len(args.profiles)} codec(s) = "
        f"{cases} cases, {cases * executions_per_case} executions",
        flush=True,
    )
    if args.dry_run:
        return
    if args.output is None:
        raise ValueError("--output is required unless --dry-run is used")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    minimum_bytes = math.ceil(args.min_gib * 1024**3)
    document = {
        "schema_version": 2,
        "benchmark": "microscopy-images",
        "status": "running",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "corpus": corpus.record(),
        "machine": machine_record(args.machine, "gpu" in args.backends),
        "chucky": build_record(executable, args.toolchain),
        "protocol": {
            "tier": args.tier,
            "minimum_bytes": minimum_bytes,
            "warmups": 0 if args.smoke else 1,
            "repeats": args.repeats,
            "smoke": args.smoke,
            "batch_bytes": 64 * 1024**2,
            "workers": 4,
            "sink": "discard",
            "chunk_targets": [
                {"label": label, "bytes": CHUNK_BYTES[label]}
                for label in args.chunk_bytes
            ],
            "chunk_ratios": [1, 4, 4],
            "min_full_shard_bytes": MIN_FULL_SHARD_BYTES,
            "max_full_shard_bytes": MAX_FULL_SHARD_BYTES,
            "order": "cyclic",
            "scales": 1,
            "codec_profiles": {name: PROFILES[name] for name in args.profiles},
            "repeatability_range_percent": 5,
        },
        "runs": [],
    }
    layouts = {}
    save_results(output, document)
    try:
        for pack in packs:
            frame_bytes = pack["width"] * pack["height"] * 2
            frames = (minimum_bytes + frame_bytes - 1) // frame_bytes
            for chunk_label in args.chunk_bytes:
                chunk_bytes = CHUNK_BYTES[chunk_label]
                for backend in args.backends:
                    for profile in args.profiles:
                        for iteration in range(
                            args.repeats + (0 if args.smoke else 1)
                        ):
                            warmup = not args.smoke and iteration == 0
                            label = (
                                f"{pack['id']}-{chunk_label}-{backend}-{profile}-"
                                f"{iteration}"
                            )
                            command = [
                                str(executable),
                                "--input",
                                str(corpus.pack_files[pack["id"]]),
                                "--width",
                                str(pack["width"]),
                                "--height",
                                str(pack["height"]),
                                "--dtype",
                                "u16",
                                "--frames",
                                str(frames),
                                "--backend",
                                backend,
                                "--chunk-bytes",
                                chunk_label,
                                "--batch-bytes",
                                "64M",
                                "--max-threads",
                                "4",
                                "--append-elements",
                                str(pack["width"] * pack["height"]),
                                "--json",
                                *PROFILES[profile],
                            ]
                            print(
                                f"{label}: {'warmup' if warmup else 'measured'}",
                                flush=True,
                            )
                            started = time.perf_counter()
                            result = subprocess.run(
                                command, capture_output=True, text=True, check=False
                            )
                            process_s = time.perf_counter() - started
                            (output / f"{label}.log").write_text(result.stderr)
                            (output / f"{label}.stdout").write_text(result.stdout)
                            record = {
                                "scenario": SCENARIO,
                                "pack_id": pack["id"],
                                "pack_sha256": pack["sha256"],
                                "pack_path": pack["path"],
                                "input_path": str(corpus.pack_files[pack["id"]]),
                                "plane_order": [p["id"] for p in pack["planes"]],
                                "modality": pack["modality"],
                                "split": pack["split"],
                                "input_id": pack.get(
                                    "input_id", pack["source_group"]
                                ),
                                "source_group": pack["source_group"],
                                "width": pack["width"],
                                "height": pack["height"],
                                "backend": backend,
                                "profile": profile,
                                "chunk_bytes": chunk_bytes,
                                "chunk_bytes_label": chunk_label,
                                "iteration": iteration,
                                "warmup": warmup,
                                "frames": frames,
                                "process_wall_s": process_s,
                                "command": command,
                                "returncode": result.returncode,
                                "status": "error",
                            }
                            document["runs"].append(record)
                            if result.returncode != 0:
                                raise ValueError(
                                    f"Benchmark failed: see {output / (label + '.log')}"
                                )
                            measurement = json.loads(result.stdout)
                            layout = check_result(
                                measurement,
                                pack,
                                frames,
                                backend,
                                profile,
                                chunk_bytes,
                                process_s,
                            )
                            layout_key = (pack["id"], chunk_label)
                            if layout_key in layouts and layouts[layout_key] != layout:
                                raise ValueError(
                                    "Layout changed across profiles/backends for "
                                    f"{pack['id']} at {chunk_label}"
                                )
                            layouts[layout_key] = layout
                            record.update(
                                status="pass", measurement=measurement, layout=layout
                            )
                            save_results(output, document)
        document["status"] = "complete"
    except Exception:
        document["status"] = "error"
        raise
    finally:
        save_results(output, document)
    print(output / "summary.csv")


def compare(args) -> None:
    left, right = read_json(args.left), read_json(args.right)
    if left.get("status") != "complete" or right.get("status") != "complete":
        raise ValueError("Only complete benchmark runs can be compared")
    if left["protocol"] != right["protocol"]:
        raise ValueError("Benchmark protocols differ")
    left_dataset = left["corpus"].get("dataset")
    right_dataset = right["corpus"].get("dataset")
    if left_dataset != right_dataset:
        raise ValueError("Dataset contracts differ")
    for key in ("revision", "benchmark_source_sha256", "source_tree_sha256"):
        if left["chucky"][key] != right["chucky"][key]:
            raise ValueError(f"Chucky sources differ: {key}")

    def index(document):
        return {
            (
                r["pack_id"],
                r["backend"],
                r["profile"],
                r.get("chunk_bytes_label", "32K"),
            ): r
            for r in document["summary"]
        }

    before, after = index(left), index(right)
    if before.keys() != after.keys():
        raise ValueError("Measured packs, profiles, backends, or chunks differ")
    rows = []
    for key in sorted(before):
        a, b = before[key], after[key]
        if a["layout"] != b["layout"] or a["pack_sha256"] != b["pack_sha256"]:
            raise ValueError(f"Input or layout differs: {key}")
        rows.append(
            {
                "pack_id": key[0],
                "backend": key[1],
                "profile": key[2],
                "chunk_bytes_label": key[3],
                "throughput_right_over_left": b["median_throughput_gibs"]
                / a["median_throughput_gibs"],
                "compression_right_over_left": b["logical_compression_fold"]
                / a["logical_compression_fold"],
            }
        )
    document = {
        "left_machine": left["machine"],
        "right_machine": right["machine"],
        "left_toolchain": left["chucky"].get("toolchain"),
        "right_toolchain": right["chucky"].get("toolchain"),
        "rows": rows,
    }
    if args.output:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        write_json(args.output, document)
    else:
        print(json.dumps(document, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and replay a versioned microscopy image dataset"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "run"):
        p = sub.add_parser(name)
        p.add_argument(
            "--data-registry",
            type=Path,
            default=DEFAULT_REGISTRY,
            help="Chucky data registry (default: bench/data.json)",
        )
        p.add_argument(
            "--dataset",
            help="Registered dataset id (default: registry default)",
        )
        source = p.add_mutually_exclusive_group()
        source.add_argument(
            "--corpus",
            type=Path,
            help="Override the registered dataset's source path",
        )
        source.add_argument(
            "--direct-corpus",
            type=Path,
            help="Load an unregistered legacy corpus directly",
        )
        p.add_argument("--allow-test-data", action="store_true")
        p.add_argument("--allow-provisional", action="store_true")
        if name == "run":
            p.add_argument("--executable", type=Path, required=True)
            p.add_argument("--output", type=Path)
            p.add_argument("--machine", default=platform.node())
            p.add_argument(
                "--tier",
                choices=tuple(IMAGE_TIERS),
                default=DEFAULT_TIER,
                help="Image matrix preset; explicit matrix axes replace preset values",
            )
            p.add_argument(
                "--backends", nargs="+", choices=("cpu", "gpu")
            )
            p.add_argument(
                "--profiles", nargs="+", choices=tuple(PROFILES)
            )
            p.add_argument(
                "--chunk-bytes",
                nargs="+",
                choices=tuple(CHUNK_BYTES),
                help="Decoded chunk-size targets; replaces the tier's chunk axis",
            )
            p.add_argument("--min-gib", type=float, default=DEFAULT_MIN_GIB)
            p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
            p.add_argument("--smoke", action="store_true")
            p.add_argument(
                "--dry-run", action="store_true", help="Verify and print the matrix only"
            )
            p.add_argument("--toolchain", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            compare(args)
        else:
            corpus = load_corpus(
                args.data_registry,
                args.dataset,
                args.corpus,
                args.direct_corpus,
                allow_test_data=args.allow_test_data,
                allow_provisional=args.allow_provisional,
            )
            if args.command == "verify":
                print(json.dumps(corpus.record(), indent=2))
            else:
                run(args, corpus)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

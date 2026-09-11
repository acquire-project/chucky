# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "zarr>=3,<4"]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr

from data_sources import DEFAULT_REGISTRY, load_corpus
from run import BLOSC_BLOCK_BYTES, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sweep"))
from measurements import validate_measurement


# The raw nvCOMP LZ4 format has no compatible Zarr Python reader.
PROFILES = {
    "none": ["--codec", "none", "--codec-level", "0"],
    "zstd": ["--codec", "zstd", "--codec-level", "3"],
    **{
        name: ["--codec", name, "--codec-level", "3", "--shuffle", "bit",
               "--blosc-block-bytes", str(BLOSC_BLOCK_BYTES)]
        for name in ("blosc-lz4", "blosc-zstd")
    },
}


def check_replay(report, images, minimum_frames, backend, profile):
    validate_measurement(report)
    window, replay = report["measurement"], report["image_replay"]
    height, width = images.shape[1:]
    chunk = replay["chunk_shape"]
    if len(chunk) != 3 or any(type(n) is not int or n <= 0 for n in chunk):
        raise ValueError("Invalid image chunk shape")
    if math.prod(chunk) * 2 != 32 * 1024:
        raise ValueError("Image chunk target changed")
    padded_frame = math.prod(
        (size + step - 1) // step * step
        for size, step in zip((height, width), chunk[1:])
    ) * 2
    measured, partial = divmod(window["input_bytes"], padded_frame)
    warmup, warmup_partial = divmod(window["warmup_input_bytes"], padded_frame)
    if partial or warmup_partial or measured < minimum_frames:
        raise ValueError("Image replay has partial or insufficient frames")
    expected = {
        "backend": backend, "codec": profile, "dtype": "u16le",
        "shape": [measured, height, width], "source_bytes": images.nbytes,
        "source_padded_bytes": len(images) * padded_frame, "order": "cyclic",
        "codec_level": 0 if profile == "none" else 3,
        "shuffle": "bit" if profile.startswith("blosc-") else "none",
    }
    for key, value in expected.items():
        if replay.get(key) != value:
            raise ValueError(f"Image replay changed {key}: expected {value}")
    if (report["submitted_bytes"] != measured * padded_frame
            or report["logical_input_bytes"] != measured * height * width * 2):
        raise ValueError("Image logical/submitted byte accounting disagrees")
    return warmup + measured


def check_pixels(destination, images, total_frames):
    actual = zarr.open_array(str(destination / "images"), mode="r")
    expected_shape = (total_frames, *images.shape[1:])
    if actual.dtype != np.dtype("<u2") or actual.shape != expected_shape:
        raise ValueError(
            f"Independent Zarr readback differs: shape={actual.shape}, "
            f"dtype={actual.dtype}, expected={expected_shape} uint16"
        )
    # Coverage can extend a short frame request by thousands of frames.
    # Compare the entire warmup + measured stream without materializing it.
    block_frames = max(1, (16 * 1024**2) // images[0].nbytes)
    digest = hashlib.sha256()
    for start in range(0, total_frames, block_frames):
        stop = min(start + block_frames, total_frames)
        expected = images[np.arange(start, stop) % len(images)]
        block = actual[start:stop]
        if not np.array_equal(block, expected):
            raise ValueError(f"Independent Zarr readback differs at frames {start}:{stop}")
        digest.update(block.astype("<u2", copy=False).tobytes())
    return digest.hexdigest(), list(actual.shape)


def invoke(command, log):
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=300
    )
    Path(str(log) + ".log").write_text(result.stderr)
    Path(str(log) + ".stdout").write_text(result.stdout)
    return result


def blosc_settings(value):
    if isinstance(value, dict):
        if value.get("name") == "blosc":
            yield value["configuration"]
        for child in value.values():
            yield from blosc_settings(child)
    elif isinstance(value, list):
        for child in value:
            yield from blosc_settings(child)


def replay(executable, raw, images, frames, backends, appends, output, prefix, checks):
    height, width = images.shape[1:]
    base = [
        executable,
        "--input",
        str(raw),
        "--width",
        str(width),
        "--height",
        str(height),
        "--frames",
        str(frames),
        "--geometry-frames",
        "64",
        "--chunk-bytes",
        "32K",
        "--batch-bytes",
        "1M",
        "--warmup",
        "0",
        "--duration",
        "0.01",
        "--max-threads",
        "4",
        "--json",
    ]
    for backend in backends:
        for profile, flags in PROFILES.items():
            for append in appends:
                name = f"{prefix}-{backend}-{profile}-{append}"
                destination = output / name
                result = invoke(
                    base
                    + [
                        "--backend",
                        backend,
                        "--append-elements",
                        str(append),
                        "-o",
                        str(destination),
                        *flags,
                    ],
                    output / name,
                )
                if result.returncode:
                    raise RuntimeError(f"Benchmark failed: {output / (name + '.log')}")
                report = json.loads(result.stdout)
                total_frames = check_replay(
                    report, images, frames, backend, profile
                )
                checksum, shape = check_pixels(destination, images, total_frames)
                settings = list(
                    blosc_settings(
                        json.loads((destination / "images" / "zarr.json").read_text())
                    )
                )
                if profile.startswith("blosc-") and (
                    len(settings) != 1
                    or settings[0].get("blocksize") != BLOSC_BLOCK_BYTES
                    or settings[0].get("shuffle") != "bitshuffle"
                ):
                    raise RuntimeError(
                        f"Saved Blosc settings differ: {name}: {settings}"
                    )
                checks.append(
                    {
                        "name": name,
                        "status": "pass",
                        "sha256": checksum,
                        "shape": shape,
                        "measured_frames": report["image_replay"]["shape"][0],
                        "input_bytes": report["input_bytes"],
                        "blosc_settings": settings,
                    }
                )
                write_json(output / "checks.json", checks)
                print(f"Passed {name}", flush=True)
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Independent Zarr readback of cyclic image replay"
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument(
        "--backends", nargs="+", choices=("cpu", "gpu"), default=["cpu"]
    )
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--corpus", type=Path, help="Override the registered dataset's source path"
    )
    source.add_argument(
        "--direct-corpus", type=Path, help="Load an unregistered legacy corpus"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--data-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--allow-test-data", action="store_true")
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args()
    corpus = (
        load_corpus(
            args.data_registry,
            args.dataset,
            args.corpus,
            args.direct_corpus,
            allow_test_data=args.allow_test_data,
            allow_provisional=args.allow_provisional,
        )
        if args.corpus or args.direct_corpus or args.dataset
        else None
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = str(args.executable.resolve(strict=True))
    y, x = np.indices((257, 259), dtype=np.uint32)
    images = np.stack(
        [
            ((y * 937 + x * 269 + frame * 10003) ^ ((y // 32) * 7)).astype("<u2")
            for frame in range(3)
        ]
    )
    raw = output / "fixture.raw"
    raw.write_bytes(images.tobytes())
    checks = []
    base = replay(
        executable,
        raw,
        images,
        8,
        args.backends,
        (257 * 259, 257 * 259 // 3 + 7),
        output,
        "fixture",
        checks,
    )
    for index, flags in enumerate(
        (
            ["--width", "0"],
            ["--height", "-1"],
            ["--frames", "2garbage"],
            ["--frames", "18446744073709551615"],
            ["--append-elements", "-1"],
            ["--dtype", "f32"],
            ["--chunk-bytes", "3"],
            ["--shuffle", "bogus"],
            ["--codec", "zstd", "--shuffle", "bit"],
            ["--codec-level", "256"],
        )
    ):
        result = invoke(
            base + ["--backend", "cpu", *flags], output / f"invalid-{index}"
        )
        if result.returncode == 0:
            raise RuntimeError(f"Invalid arguments were accepted: {flags}")
        checks.append({"name": f"invalid-{index}", "status": "pass", "flags": flags})
    raw.write_bytes(b"")
    if invoke(base + ["--backend", "cpu"], output / "empty").returncode == 0:
        raise RuntimeError("Empty pack was accepted")
    raw.write_bytes(images.tobytes()[:-1])
    if invoke(base + ["--backend", "cpu"], output / "truncated").returncode == 0:
        raise RuntimeError("Truncated pack was accepted")
    raw.unlink()
    if invoke(base + ["--backend", "cpu"], output / "missing").returncode == 0:
        raise RuntimeError("Missing pack was accepted")
    checks.extend(
        {"name": name, "status": "pass"} for name in ("empty", "truncated", "missing")
    )
    if corpus:
        write_json(output / "corpus.json", corpus.record())
        for pack in corpus.manifest["packs"]:
            raw = corpus.pack_files[pack["id"]]
            images = np.fromfile(raw, dtype="<u2").reshape(
                -1, pack["height"], pack["width"]
            )
            replay(
                executable,
                raw,
                images,
                2 * len(images) + 1,
                args.backends,
                (pack["height"] * pack["width"],),
                output,
                pack["id"],
                checks,
            )
    write_json(output / "checks.json", checks)
    print(f"{len(checks)} checks passed")


if __name__ == "__main__":
    main()

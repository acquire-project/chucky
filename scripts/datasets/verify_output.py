from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import zarr

from data_sources import DEFAULT_REGISTRY, load_corpus
from run import BLOSC_BLOCK_BYTES, PROFILES, check_result, write_json


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
    expected = images[np.arange(frames) % len(images)]
    expected_sha = hashlib.sha256(expected.tobytes()).hexdigest()
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
        "--batch-bytes",
        "64M",
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
                check_result(
                    report,
                    {"width": width, "height": height, "bytes": images.nbytes},
                    frames,
                    backend,
                    profile,
                )
                actual = zarr.open_array(str(destination / "images"), mode="r")[:]
                if actual.dtype != np.dtype("<u2") or not np.array_equal(
                    actual, expected
                ):
                    raise RuntimeError(
                        f"Independent Zarr readback differs: {name}, "
                        f"shape={actual.shape}, expected={expected.shape}"
                    )
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
                        "sha256": expected_sha,
                        "shape": list(actual.shape),
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
            ["--append-elements", "0"],
            ["--dtype", "f32"],
            ["--chunk-bytes", "64K"],
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

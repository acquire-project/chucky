from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path


def command_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def digest_source(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_tree_digest(root: Path) -> str:
    files = {root / "CMakeLists.txt"}
    for directory in ("src", "bench", "cmake"):
        for path in (root / directory).rglob("*"):
            if path.name == "CMakeLists.txt" or path.suffix in {
                ".c",
                ".h",
                ".cc",
                ".cpp",
                ".hpp",
                ".cu",
                ".cuh",
                ".cmake",
            }:
                files.add(path)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda path: path.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(digest_source(path).encode() + b"\n")
    return digest.hexdigest()


def collect_environment(build: Path) -> dict:
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        key, value = line.split("=", 1)
        cache[key.split(":", 1)[0]] = value
    compiler = cache.get("CMAKE_C_COMPILER")
    cuda = cache.get("CMAKE_CUDA_COMPILER")
    result = {
        "cmake": command_output(["cmake", "--version"]),
        "c_compiler": command_output([compiler, "--version"]) if compiler else None,
        "cuda_compiler": command_output([cuda, "--version"]) if cuda else None,
        "compiler_configuration": {},
        "packages": {},
        "codec_headers": {},
    }
    for path in sorted((build / "CMakeFiles").glob("*/CMake*Compiler.cmake")):
        result["compiler_configuration"][path.name] = {
            name: value
            for name, value in re.findall(
                r'set\((CMAKE_\w+_COMPILER_(?:ID|VERSION)) "([^"]+)"\)',
                path.read_text(errors="replace"),
            )
        }
    for package in ("numpy", "numcodecs", "tifffile", "zarr"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    roots = {
        Path(value)
        for key, value in cache.items()
        if key.endswith("INCLUDE_DIR") and Path(value).is_dir()
    }
    roots.update(
        Path(value) / "include"
        for value in cache.get("CMAKE_PREFIX_PATH", "").split(";")
        if value
    )
    for root in sorted(roots):
        for name in ("nvcomp.h", "nvcomp/version.h", "blosc.h", "zstd.h", "lz4.h"):
            path = root / name
            if path.is_file():
                lines = [
                    line
                    for line in path.read_text(errors="replace").splitlines()
                    if re.search(r"#define\s+\w*(VERSION|VER_)\w*", line)
                ]
                if lines:
                    result["codec_headers"][str(path)] = lines
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Record benchmark build tools and codec headers"
    )
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect_environment(args.build)
    result["source_tree_sha256"] = source_tree_digest(
        Path(__file__).resolve().parents[2]
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

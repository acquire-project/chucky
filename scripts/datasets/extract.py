from __future__ import annotations

import argparse
import hashlib
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numcodecs
import numpy as np
import tifffile

from manifest import (
    IDENTIFIER,
    MAX_BYTES,
    check_kind,
    digest_file,
    read_json,
    relative_file,
    verify_corpus,
    verify_source,
)
from run import BLOSC_BLOCK_BYTES, PROFILES, write_json


def read_zarr2(candidate: dict) -> tuple[np.ndarray, dict]:
    path = Path(candidate["path"])
    metadata_path = path / ".zarray"
    metadata = read_json(metadata_path)
    shape, chunks = metadata["shape"], metadata["chunks"]
    dtype = np.dtype(metadata["dtype"])
    if len(shape) != 5 or len(chunks) != 5 or candidate.get("axes") != "TCZYX":
        raise ValueError(f"Expected explicit TCZYX source axes: {path}")
    if dtype.kind != "u" or dtype.itemsize != 2 or metadata.get("filters"):
        raise ValueError(f"Expected unfiltered uint16 camera data: {path}")
    if any(type(v) is not int or v <= 0 for v in shape + chunks):
        raise ValueError(f"Invalid source dimensions: {path}")
    coordinates = [candidate["coordinates"][axis] for axis in ("t", "c", "z")]
    if any(
        type(v) is not int or v < 0 or v >= shape[i] for i, v in enumerate(coordinates)
    ):
        raise ValueError(f"Source coordinates are outside the array: {path}")
    compressor = metadata.get("compressor")
    if compressor and compressor["id"] not in {"blosc", "zstd", "gzip", "zlib", "lz4"}:
        raise ValueError(f"Unsupported source compression: {path}")
    codec = numcodecs.get_codec(compressor) if compressor else None
    order = metadata.get("order", "C")
    if order not in {"C", "F"}:
        raise ValueError(f"Unsupported source order: {path}")
    separator = metadata.get("dimension_separator", ".")
    if separator not in {".", "/"}:
        raise ValueError(f"Unsupported chunk key encoding: {path}")
    image = np.empty(shape[-2:], dtype="<u2")
    used = []
    for y, x in itertools.product(
        range(0, shape[3], chunks[3]), range(0, shape[4], chunks[4])
    ):
        indices = [v // chunks[i] for i, v in enumerate(coordinates)]
        indices.extend((y // chunks[3], x // chunks[4]))
        file = path / separator.join(str(v) for v in indices)
        if not file.is_file():
            raise ValueError(f"Missing source chunk; refusing fill values: {file}")
        payload = file.read_bytes()
        decoded = codec.decode(payload) if codec else payload
        block = np.frombuffer(decoded, dtype=dtype)
        if block.size != int(np.prod(chunks, dtype=np.int64)):
            raise ValueError(f"Truncated or malformed source chunk: {file}")
        block = block.reshape(chunks, order=order)
        local = tuple(v % chunks[i] for i, v in enumerate(coordinates))
        height, width = min(chunks[3], shape[3] - y), min(chunks[4], shape[4] - x)
        image[y : y + height, x : x + width] = block[local][0:height, 0:width]
        used.append(
            {
                "key": str(file.relative_to(path)),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return image, {"metadata_sha256": digest_file(metadata_path), "chunks": used}


def read_tiff(candidate: dict) -> tuple[np.ndarray, dict]:
    path = Path(candidate["path"])
    index = candidate["coordinates"]["page"]
    if type(index) is not int or index < 0:
        raise ValueError(f"Invalid TIFF page: {path}")
    with tifffile.TiffFile(path) as file:
        page = file.pages[index]
        if page.compression.name not in {
            "NONE",
            "LZW",
            "ADOBE_DEFLATE",
            "DEFLATE",
            "PACKBITS",
            "LZMA",
            "ZSTD",
        }:
            raise ValueError(
                f"TIFF compression is not approved as lossless: {page.compression}"
            )
        if len(page.shape) != 2 or page.dtype.kind != "u" or page.dtype.itemsize != 2:
            raise ValueError(f"Expected a monochrome uint16 TIFF page: {path}")
        image = np.ascontiguousarray(page.asarray(), dtype="<u2")
        details = {
            "file_bytes": path.stat().st_size,
            "page": index,
            "page_offset": page.offset,
            "compression": page.compression.name,
            "photometric": page.photometric.name,
        }
    return image, details


def read_plane(candidate: dict) -> tuple[np.ndarray, dict]:
    readers = {"zarr2": read_zarr2, "tiff": read_tiff}
    try:
        reader = readers[candidate["reader"]]
    except KeyError as error:
        raise ValueError(
            "Reader must be zarr2 or tiff; reconstructed float arrays are excluded"
        ) from error
    image, details = reader(candidate)
    if image.ndim != 2 or image.dtype != np.dtype("<u2"):
        raise ValueError("Source is not an unchanged uint16 plane")
    return image, details


def statistics(image: np.ndarray) -> dict:
    counts = np.bincount(image.ravel(), minlength=65536)
    probabilities = counts[counts > 0] / image.size
    low = np.bincount((image.ravel() & 15), minlength=16)
    low_probabilities = low[low > 0] / image.size
    quantiles = np.percentile(image, [1, 50, 99])
    compressors = {
        "zstd": numcodecs.Zstd(level=3),
        "blosc_lz4": numcodecs.Blosc(
            cname="lz4", clevel=3, shuffle=2, blocksize=BLOSC_BLOCK_BYTES
        ),
        "blosc_zstd": numcodecs.Blosc(
            cname="zstd", clevel=3, shuffle=2, blocksize=BLOSC_BLOCK_BYTES
        ),
    }
    sizes = {name: 0 for name in compressors}
    padded_bytes = 0
    for y, x in itertools.product(
        range(0, image.shape[0], 256), range(0, image.shape[1], 256)
    ):
        chunk = np.zeros((256, 256), dtype="<u2")
        block = image[y : y + 256, x : x + 256]
        chunk[: block.shape[0], : block.shape[1]] = block
        padded_bytes += chunk.nbytes
        for name, codec in compressors.items():
            sizes[name] += len(codec.encode(chunk))
    return {
        "minimum": int(image.min()),
        "maximum": int(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "p01": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "zero_fraction": float(counts[0] / image.size),
        "u16_max_fraction": float(counts[-1] / image.size),
        "entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
        "low_four_bit_entropy": float(
            -np.sum(low_probabilities * np.log2(low_probabilities))
        ),
        "neighbor_difference_median": float(
            np.median(np.abs(np.diff(image.astype(np.int32), axis=1)))
        ),
        "padded_bytes": padded_bytes,
        "compressed_chunk_bytes": sizes,
    }


def survey(args):
    plan = read_json(args.plan)
    kind = plan.get("kind", "raw")
    check_kind(kind, args.allow_test_data, args.allow_provisional)
    if kind == "provisional" and not plan.get("selection_reason"):
        raise ValueError("Provisional selection needs a reason in the source plan")
    candidates = plan["candidates"]
    counts = Counter(c["modality"] for c in candidates)
    if counts != {
        "fluorescence": args.candidates_per_modality,
        "brightfield": args.candidates_per_modality,
    }:
        raise ValueError(
            f"Expected {args.candidates_per_modality} candidates per modality, got {counts}"
        )
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output / "planes"
    cache.mkdir()
    fields = sorted(
        {c["field_id"] for c in candidates},
        key=lambda value: hashlib.sha256(("chucky-v1:" + value).encode()).hexdigest(),
    )
    heldout_fields = set() if kind == "provisional" else set(fields[::4])
    document = {
        "schema_version": 1,
        "kind": kind,
        "release": plan.get("release"),
        "selection_reason": plan.get("selection_reason"),
        "codec_profiles": PROFILES,
        "status": "running",
        "source_plan_sha256": digest_file(args.plan),
        "sources": plan["sources"],
        "field_split": (
            "All planned fields are core; provisional proof of concept has no holdout"
            if kind == "provisional"
            else "Sort fields by SHA256(chucky-v1: + field_id); every fourth field is held out"
        ),
        "candidates": [],
    }
    for index, candidate in enumerate(candidates):
        print(f"{index + 1}/{len(candidates)}: {candidate['id']}", flush=True)
        image, details = read_plane(candidate)
        data = image.tobytes(order="C")
        image_sha = hashlib.sha256(data).hexdigest()
        original = candidate.get("original")
        if original:
            reference, original_details = read_plane(original)
            if reference.shape != image.shape or not np.array_equal(reference, image):
                raise ValueError(
                    f"Camera values differ from the original: {candidate['id']}"
                )
            details["original_comparison"] = {
                "path": original["path"],
                "coordinates": original["coordinates"],
                "plane_sha256": image_sha,
                "details": original_details,
            }
        file = cache / f"{hashlib.sha256(candidate['id'].encode()).hexdigest()}.raw"
        file.write_bytes(data)
        document["candidates"].append(
            candidate
            | {
                "split": "heldout"
                if candidate["field_id"] in heldout_fields
                else "core",
                "width": image.shape[1],
                "height": image.shape[0],
                "bytes": len(data),
                "sha256": image_sha,
                "statistics": statistics(image),
                "source_details": details,
                "cache": file.relative_to(args.output).as_posix(),
            }
        )
        write_json(args.output / "survey.json", document)
    document["status"] = "complete"
    write_json(args.output / "survey.json", document)


def representatives(candidates, count):
    candidates = sorted(candidates, key=lambda c: c["id"])
    if count >= len(candidates):
        return candidates
    vectors = []
    for candidate in candidates:
        stats = candidate["statistics"]
        vectors.append(
            [
                np.log1p(stats[key])
                for key in ("p01", "p50", "p99", "std", "neighbor_difference_median")
            ]
            + [
                stats["entropy_bits"],
                stats["low_four_bit_entropy"],
                np.log(candidate["bytes"] / stats["compressed_chunk_bytes"]["zstd"]),
                np.log(
                    candidate["bytes"] / stats["compressed_chunk_bytes"]["blosc_lz4"]
                ),
            ]
        )
    features = np.array(vectors, dtype=np.float64)
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    features = (features - features.mean(axis=0)) / scale
    distances = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
    selected = [int(np.argmin(distances.sum(axis=1)))]
    while len(selected) < count:
        costs = [
            float(np.min(distances[:, selected + [i]], axis=1).sum())
            if i not in selected
            else np.inf
            for i in range(len(candidates))
        ]
        selected.append(int(np.argmin(costs)))
    while True:
        cost = float(np.min(distances[:, selected], axis=1).sum())
        best = None
        for position in range(count):
            for candidate in range(len(candidates)):
                if candidate in selected:
                    continue
                proposal = selected.copy()
                proposal[position] = candidate
                proposal_cost = float(np.min(distances[:, proposal], axis=1).sum())
                if proposal_cost < cost - 1e-9:
                    cost, best = proposal_cost, proposal
        if best is None:
            return [candidates[i] for i in sorted(selected)]
        selected = best


def allocate(groups, target):
    if target < len(groups):
        raise ValueError("Too few selected images to cover every source and shape")
    counts = {key: 1 for key in groups}
    for _ in range(min(target, sum(len(v) for v in groups.values())) - len(groups)):
        available = [key for key in groups if counts[key] < len(groups[key])]
        key = max(available, key=lambda key: len(groups[key]) / (counts[key] + 1))
        counts[key] += 1
    return counts


def build(args):
    survey_path = args.survey / "survey.json"
    document = read_json(survey_path)
    root = args.corpus
    if document.get("status") != "complete":
        raise ValueError("A completed survey is required")
    kind = document.get("kind", "raw")
    check_kind(kind, args.allow_test_data, args.allow_provisional)
    release = (
        document.get("release")
        if kind == "provisional"
        else ("test" if kind == "synthetic-test" else "v1")
    )
    if not isinstance(release, str) or not IDENTIFIER.fullmatch(release):
        raise ValueError("A valid release name is required")
    if kind == "provisional" and release in {"v1", "test"}:
        raise ValueError("Provisional release must have a separate name")
    if any(
        (root / name).exists()
        for name in ("manifest.json", "INCOMPLETE", f"data/{release}")
    ):
        raise ValueError("Refusing to replace an existing or incomplete corpus")
    sources = document["sources"]
    for name, source in sources.items():
        verify_source(root, name, source, kind)
    groups = defaultdict(list)
    for candidate in document["candidates"]:
        if candidate["source_id"] not in sources:
            raise ValueError(f"Unknown source: {candidate['id']}")
        key = (
            candidate["modality"],
            candidate["split"],
            candidate["source_group"],
            candidate["height"],
            candidate["width"],
        )
        groups[key].append(candidate)
    selected_counts = {}
    for modality in () if kind == "provisional" else ("fluorescence", "brightfield"):
        for split, target in (
            ("core", args.core_per_modality),
            ("heldout", args.heldout_per_modality),
        ):
            subset = {
                k: v for k, v in sorted(groups.items()) if k[:2] == (modality, split)
            }
            if not subset:
                raise ValueError(f"No independent fields for {modality}/{split}")
            selected_counts.update(allocate(subset, target))

    if kind == "provisional":
        if not document.get("selection_reason") or any(
            key[1] != "core" for key in groups
        ):
            raise ValueError(
                "Provisional extraction must keep all planned images as core"
            )
        selected_counts = {key: len(values) for key, values in groups.items()}

    def total_bytes():
        return sum(
            count * key[-1] * key[-2] * 2 for key, count in selected_counts.items()
        )

    if kind == "provisional" and total_bytes() > MAX_BYTES:
        raise ValueError(
            "Provisional images exceed the corpus limit; revise the source plan"
        )
    while total_bytes() > MAX_BYTES:
        eligible = [key for key, count in selected_counts.items() if count > 1]
        if not eligible:
            raise ValueError("Native planes cannot fit within the corpus limit")
        key = max(eligible, key=lambda key: key[-1] * key[-2] * selected_counts[key])
        selected_counts[key] -= 1
    manifest = {
        "schema_version": 1,
        "kind": kind,
        "release": release,
        "sources": sources,
        "selection": {
            "survey_path": f"provenance/survey-{release}.json",
            "survey_sha256": digest_file(survey_path),
            "candidates_per_modality": dict(
                Counter(c["modality"] for c in document["candidates"])
            ),
            "field_split": document["field_split"],
            "method": (
                "All candidates from the explicit provisional source plan"
                if kind == "provisional"
                else "Deterministic medoids of brightness, adjacent differences, entropy, and chunk compression"
            ),
            "reason": document.get("selection_reason"),
            "core_target_per_modality": None
            if kind == "provisional"
            else args.core_per_modality,
            "heldout_target_per_modality": 0
            if kind == "provisional"
            else args.heldout_per_modality,
            "decoded_limit_bytes": MAX_BYTES,
        },
        "packs": [],
    }
    destination = root / "data" / release
    destination.mkdir(parents=True)
    try:
        saved_survey = relative_file(root, manifest["selection"]["survey_path"])
        saved_survey.parent.mkdir(parents=True, exist_ok=True)
        with saved_survey.open("xb") as output:
            output.write(survey_path.read_bytes())
        for index, (key, count) in enumerate(sorted(selected_counts.items())):
            modality, split, source_group, height, width = key
            chosen = representatives(groups[key], count)
            name = f"{modality}-{split}-{index:02d}-{width}x{height}"
            file = destination / f"{name}.raw"
            checksum = hashlib.sha256()
            planes = []
            with file.open("xb") as output:
                for candidate in sorted(chosen, key=lambda c: c["id"]):
                    data = relative_file(args.survey, candidate["cache"]).read_bytes()
                    if (
                        len(data) != height * width * 2
                        or hashlib.sha256(data).hexdigest() != candidate["sha256"]
                    ):
                        raise ValueError(f"Survey cache changed: {candidate['id']}")
                    output.write(data)
                    checksum.update(data)
                    planes.append(
                        {
                            k: v
                            for k, v in candidate.items()
                            if k
                            not in {
                                "cache",
                                "split",
                                "source_group",
                                "modality",
                                "width",
                                "height",
                                "bytes",
                            }
                        }
                    )
            manifest["packs"].append(
                {
                    "id": name,
                    "path": file.relative_to(root).as_posix(),
                    "modality": modality,
                    "split": split,
                    "source_group": source_group,
                    "dtype": "u16le",
                    "height": height,
                    "width": width,
                    "bytes": file.stat().st_size,
                    "sha256": checksum.hexdigest(),
                    "planes": planes,
                }
            )
        write_json(root / "manifest.json", manifest)
        verify_corpus(
            root,
            allow_test_data=args.allow_test_data,
            allow_provisional=args.allow_provisional,
        )
    except Exception:
        (root / "INCOMPLETE").write_text(
            "Extraction failed; do not tag or benchmark this corpus.\n"
        )
        raise
    print(root / "manifest.json")


def main():
    parser = argparse.ArgumentParser(
        description="Survey and extract unchanged native uint16 image planes"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("survey")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidates-per-modality", type=int, default=64)
    p.add_argument("--allow-test-data", action="store_true")
    p.add_argument("--allow-provisional", action="store_true")
    p = sub.add_parser("build")
    p.add_argument("--survey", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--core-per-modality", type=int, default=12)
    p.add_argument("--heldout-per-modality", type=int, default=4)
    p.add_argument("--allow-test-data", action="store_true")
    p.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args()
    if args.command == "survey":
        survey(args)
    else:
        build(args)


if __name__ == "__main__":
    main()

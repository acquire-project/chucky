"""Versioned microscopy configurations and their execution order."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

DEFAULT_DEFINITION = Path(__file__).resolve().parents[2] / "bench/studies/microscopy/discovery.json"
CHUNKS = {f"{size}K": size << 10 for size in (16, 32, 64, 128, 256, 512)} | {
    "1M": 1 << 20, "2M": 2 << 20,
}
CODECS = {"none", "lz4", "zstd", "blosc-lz4", "blosc-zstd"}


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict:
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


def positive(value, name, *, integer=False, minimum=0):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value <= minimum):
        raise ValueError(f"Invalid {name}: {value!r}")


def validate_profile(profile, *, repeats=False):
    for key in ("warmup_s", "duration_s", "min_gib"):
        positive(profile[key], key)
    if min(profile["warmup_s"], profile["duration_s"]) < 0.25:
        raise ValueError("Warmup and duration must be at least 0.25 s")
    if repeats:
        positive(profile["repeats"], "repeats", integer=True)


def validate_definition(definition):
    expected = {"version", "id", "label", "dataset", "inputs", "backends", "chunk_labels",
                "blosc_block_bytes", "codecs", "chunk_depth", "geometry_frames", "sink",
                "profiles", "reference", "reference_every", "tolerance_percent", "seed",
                "environment", "pilot"}
    if set(definition) != expected or type(definition["version"]) is not int or definition["version"] != 1:
        raise ValueError("Unsupported microscopy definition or fields")
    for key in ("id", "label", "dataset"):
        if not isinstance(definition[key], str) or not definition[key]:
            raise ValueError(f"Missing study {key}")
    for key in ("inputs", "backends", "chunk_labels"):
        values = definition[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"Study {key} must be nonempty and unique")
    if set(definition["backends"]) - {"cpu", "gpu"}:
        raise ValueError("Unknown study backend")
    if set(definition["chunk_labels"]) - CHUNKS.keys():
        raise ValueError("Unknown chunk label")
    if definition["sink"] not in {"discard", "fs", "s3"}:
        raise ValueError("Unknown study sink")
    if not definition["codecs"] or definition["codecs"].keys() - CODECS:
        raise ValueError("Unknown study codec")
    for codec, settings in definition["codecs"].items():
        if set(settings) != {"level", "shuffle"} or type(settings["level"]) is not int:
            raise ValueError("Codec requires an integer level and shuffle")
        if not 0 <= settings["level"] <= (9 if codec.startswith("blosc-") else 255):
            raise ValueError("Codec level out of range")
        if settings["shuffle"] != ("bit" if codec.startswith("blosc-") else "none"):
            raise ValueError("The discovery study fixes Blosc bitshuffle and uses raw controls")
    positive(definition["chunk_depth"], "chunk_depth", integer=True)
    if definition["geometry_frames"] is not None:
        positive(definition["geometry_frames"], "geometry_frames", integer=True)
    positive(definition["reference_every"], "reference_every", integer=True)
    positive(definition["tolerance_percent"], "tolerance_percent")
    if type(definition["seed"]) is not int:
        raise ValueError("Study seed must be an integer")
    if set(definition["profiles"]) != set(definition["backends"]):
        raise ValueError("Each backend needs an explicit profile")
    for profile in definition["profiles"].values():
        if set(profile) != {"warmup_s", "duration_s", "min_gib", "repeats"}:
            raise ValueError("Unknown measurement profile fields")
        validate_profile(profile, repeats=True)
    reference = definition["reference"]
    if set(reference) != {"warmup_s", "duration_s", "min_gib", "chunk_label", "blosc_block_bytes", "codecs"}:
        raise ValueError("Unknown reference fields")
    validate_profile(reference)
    if reference["chunk_label"] not in CHUNKS:
        raise ValueError("Unknown reference chunk")
    if set(reference["codecs"]) != set(definition["inputs"]):
        raise ValueError("Each input needs a reference codec")
    if set(reference["codecs"].values()) - definition["codecs"].keys():
        raise ValueError("Unknown reference codec")
    if set(definition["environment"]) != {"gpu", "cpu_count"}:
        raise ValueError("Record the intended GPU and CPU allocation")
    positive(definition["environment"]["cpu_count"], "cpu_count", integer=True)
    blocks = definition["blosc_block_bytes"]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("No Blosc block requests")
    if type(reference["blosc_block_bytes"]) is not int:
        raise ValueError("Reference block size must be an explicit byte count")
    for value in [*blocks, reference["blosc_block_bytes"]]:
        if value not in ("quarter-chunk", "chunk") and (
                type(value) is not int or not 128 <= value <= 715827542):
            raise ValueError(f"Invalid Blosc block request: {value}")
    if not isinstance(definition["pilot"], list) or not definition["pilot"]:
        raise ValueError("No pilot cases")
    for case in definition["pilot"]:
        expected = {"codec", "chunk_label"}
        if case.get("codec", "").startswith("blosc-"):
            expected.add("blosc_block_bytes")
        if (set(case) != expected or case["codec"] not in definition["codecs"]
                or case["chunk_label"] not in definition["chunk_labels"]):
            raise ValueError("Invalid pilot configuration")
        if (case["codec"].startswith("blosc-")
                and case["blosc_block_bytes"] not in block_sizes(definition, CHUNKS[case["chunk_label"]])):
            raise ValueError("Pilot block size is outside the discovery grid")
    return definition


def block_sizes(definition, chunk):
    return sorted({chunk if value == "chunk" else chunk // 4 if value == "quarter-chunk"
                   else value for value in definition["blosc_block_bytes"]})


def make_plan(definition: dict, members: list[tuple[str, str, str]], phase="discovery") -> dict:
    validate_definition(definition)
    if phase not in {"pilot", "discovery"}:
        raise ValueError(f"Unknown study phase: {phase}")
    by_input = {}
    for asset, input_id, dtype in members:
        if input_id in by_input:
            raise ValueError(f"Ambiguous study input: {input_id}")
        by_input[input_id] = (asset, dtype)
    if set(definition["inputs"]) - by_input.keys():
        raise ValueError("Study input is absent from the registered dataset")
    configs, batches, schedule = {}, [], []
    rng = random.Random(definition["seed"])

    def case(input_id, backend, codec, chunk_label, block=None):
        asset, dtype = by_input[input_id]
        settings = definition["codecs"][codec]
        spec = {"scenario": "microscopy", "fill": "images", "input_id": input_id,
                "image_asset_id": asset, "image_split": "core", "dtype": dtype,
                "backend": backend, "codec": codec, "chunk_label": chunk_label,
                "chunk_depth": definition["chunk_depth"], "sink": definition["sink"],
                "blosc_block_bytes": block if codec.startswith("blosc-") else None,
                "blosc_shuffle": settings["shuffle"], "level": settings["level"]}
        key = fingerprint(spec)[:16]
        configs[key] = spec
        return key

    groups = [(input_id, backend) for input_id in definition["inputs"] for backend in definition["backends"]]
    rng.shuffle(groups)
    for input_id, backend in groups:
        selected = []
        for chunk_label in definition["chunk_labels"]:
            for codec in definition["codecs"]:
                blocks = block_sizes(definition, CHUNKS[chunk_label]) if codec.startswith("blosc-") else [None]
                for block in blocks:
                    options = {"codec": codec, "chunk_label": chunk_label}
                    if block is not None:
                        options["blosc_block_bytes"] = block
                    if phase == "pilot" and options not in definition["pilot"]:
                        continue
                    selected.append(case(input_id, backend, codec, chunk_label, block))
        if not selected:
            raise ValueError("Pilot selects no configurations")
        reference = definition["reference"]
        reference_id = case(input_id, backend, reference["codecs"][input_id],
                            reference["chunk_label"], reference["blosc_block_bytes"])
        rng.shuffle(selected)
        size = definition["reference_every"]
        for start in range(0, len(selected), size):
            batch_id = f"group-{len(batches) + 1:03d}"
            ids = selected[start:start + size]
            batch = {"id": batch_id, "case_ids": ids, "reference_case_id": reference_id}
            batches.append(batch)
            ref_profile = {key: reference[key] for key in ("warmup_s", "duration_s", "min_gib")}
            profile = {key: definition["profiles"][backend][key] for key in ref_profile}
            count = 1 if phase == "pilot" else definition["profiles"][backend]["repeats"]
            tasks = [{"case_id": reference_id, "role": "reference-before", "repeat": 1, "profile": ref_profile}]
            observations = [{"case_id": key, "role": "sample", "repeat": repeat, "profile": profile}
                            for repeat in range(1, count + 1) for key in ids]
            rng.shuffle(observations)
            tasks.extend(observations)
            tasks.append({"case_id": reference_id, "role": "reference-after", "repeat": 1, "profile": ref_profile})
            for task in tasks:
                schedule.append({"id": f"execution-{len(schedule) + 1:04d}", "batch_id": batch_id, **task})
    measured = {task["case_id"] for task in schedule if task["role"] == "sample"}
    return {"version": 1, "definition": definition, "phase": phase, "cases": configs,
            "batches": batches, "schedule": schedule,
            "counts": {"configurations": len(measured),
                       "samples": sum(task["role"] == "sample" for task in schedule),
                       "references": sum(task["role"] != "sample" for task in schedule),
                       "executions": len(schedule)},
            "requested_seconds": sum(task["profile"]["warmup_s"] + task["profile"]["duration_s"]
                                     for task in schedule)}


def validate_plan(plan):
    members = sorted({(spec["image_asset_id"], spec["input_id"], spec["dtype"])
                      for spec in plan["cases"].values()})
    expected = make_plan(plan["definition"], members, plan["phase"])
    if expected != plan:
        raise ValueError("Study plan differs from its definition and deterministic schedule")
    return plan

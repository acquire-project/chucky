from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics

from models import run_id


def content_hash(value: dict) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def image_sweep(document: dict) -> dict:
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported image benchmark schema")
    if document.get("status") != "complete":
        raise ValueError("Image benchmark is incomplete")
    corpus = document["corpus"]
    protocol = document["protocol"]
    groups = {}
    for record in document["runs"]:
        if record["status"] != "pass":
            raise ValueError("Image benchmark contains a failed execution")
        if not record["warmup"]:
            key = (record["pack_id"], record["backend"], record["profile"])
            groups.setdefault(key, []).append(record)
    if not groups:
        raise ValueError("Image benchmark has no measured executions")

    runs = []
    for (pack_id, backend, profile), records in sorted(groups.items()):
        if len(records) != protocol["repeats"]:
            raise ValueError(f"Incomplete repetitions: {pack_id}/{backend}/{profile}")
        iterations = [r["iteration"] for r in records]
        if len(set(iterations)) != len(iterations):
            raise ValueError(f"Duplicate repetitions: {pack_id}/{backend}/{profile}")
        first = records[0]
        for record in records:
            for key in (
                "pack_sha256",
                "plane_order",
                "layout",
                "frames",
                "width",
                "height",
            ):
                if record[key] != first[key]:
                    raise ValueError(
                        f"Image input or layout changed between repetitions: {key}"
                    )
        measurements = [r["measurement"] for r in records]
        rates = [m["throughput_in_gibs"] for m in measurements]
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("Image throughput must be finite and positive")
        median = statistics.median(rates)
        selected = min(
            records, key=lambda r: abs(r["measurement"]["throughput_in_gibs"] - median)
        )
        result = copy.deepcopy(selected["measurement"])
        replay = result["image_replay"]
        layout = selected["layout"]
        chunk_bytes = math.prod(layout["chunk_shape"]) * 2
        input_id = "images-" + content_hash(
            {
                "corpus": corpus["manifest_sha256"],
                "pack": selected["pack_sha256"],
                "pack_id": pack_id,
                "plane_order": selected["plane_order"],
                "protocol": protocol,
                "layout": layout,
            }
        )
        label = f"{corpus['release']} / {pack_id}"
        if corpus["kind"] != "raw":
            label += f" ({corpus['kind']})"
        if protocol["smoke"]:
            label += " (smoke)"
        total_input = sum(m["input_bytes"] for m in measurements)
        total_output = sum(m["output_bytes"] for m in measurements)
        if total_input <= 0 or total_output <= 0:
            raise ValueError("Image byte counts must be positive")
        result.update(
            {
                "scenario": "images",
                "codec": profile,
                "fill": "images",
                "input_id": input_id,
                "input_label": label,
                "backend": backend,
                "dtype": "u16",
                "chunk_bytes": chunk_bytes,
                "chunk_bytes_label": f"{chunk_bytes // 1024}K"
                if chunk_bytes % 1024 == 0
                else str(chunk_bytes),
                "sink": protocol["sink"],
                "frames": selected["frames"],
                "throughput_in_gibs": median,
                "throughput_out_gibs": statistics.median(
                    m["throughput_out_gibs"] for m in measurements
                ),
                "compression_fold": total_input / total_output,
                "logical_compression_fold": total_input / total_output,
                "image_input": {
                    "release": corpus["release"],
                    "kind": corpus["kind"],
                    "manifest_sha256": corpus["manifest_sha256"],
                    "pack_id": pack_id,
                    "pack_sha256": selected["pack_sha256"],
                    "modality": selected["modality"],
                    "split": selected["split"],
                    "source_group": selected["source_group"],
                    "plane_order": selected["plane_order"],
                },
                "repetitions": {
                    "count": len(records),
                    "warmups": protocol["warmups"],
                    "throughput_min_gibs": min(rates),
                    "throughput_max_gibs": max(rates),
                    "throughput_spread_percent": 100
                    * (max(rates) - min(rates))
                    / median,
                    "detail_iteration": selected["iteration"],
                    "detail_repeat": records.index(selected) + 1,
                },
            }
        )
        if profile.startswith("blosc-"):
            result["blosc_shuffle"] = replay["shuffle"]
            result["blosc_level"] = replay["codec_level"]
        else:
            result["level"] = replay["codec_level"]
        result["id"] = run_id(
            {
                **result,
                "id": f"images__{profile}__{input_id}__{backend}__u16__{result['chunk_bytes_label']}"
                f"__settings-{content_hash({key: replay[key] for key in ('codec_level', 'shuffle')})}",
            }
        )
        runs.append(result)

    machine = copy.deepcopy(document["machine"])
    source = document["chucky"]
    machine["commit"] = (source.get("revision") or "unknown")[:7]
    machine["date"] = document["created_utc"]
    affinity = machine.get("cpu_affinity")
    if affinity:
        machine["cpu_count"] = len(affinity)
    devices = (machine.get("nvidia_smi") or "").splitlines()
    if devices:
        fields = [value.strip() for value in devices[0].split(",")]
        if len(fields) >= 4:
            machine["gpu"], machine["driver_version"] = fields[1], fields[3]
    settings = source.get("build_settings", {})
    compilers = source.get("toolchain", {}).get("compiler_configuration", {})
    machine["build"] = {
        "build_type": settings.get("CMAKE_BUILD_TYPE"),
        "cuda_architectures": settings.get("CMAKE_CUDA_ARCHITECTURES"),
        "cuda_compiler_version": compilers.get("CMakeCUDACompiler.cmake", {}).get(
            "CMAKE_CUDA_COMPILER_VERSION"
        ),
    }
    return {
        "version": 10,
        "machine": machine,
        "runs": runs,
        "corpus": copy.deepcopy(corpus),
        "protocol": copy.deepcopy(protocol),
        "chucky": copy.deepcopy(source),
    }

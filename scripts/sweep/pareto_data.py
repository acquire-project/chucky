"""Strict adapters for retained Blosc archives. No benchmark or frontier code.

Only display formatting may round values. Legacy summary fields survive in
source_metrics; unavailable measurements are null, including summary-only
memory ranges. Hashes of unavailable historical artifacts are provenance, not
claims that we can verify them. See docs/benchmarks/README.md for the contract.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

VERSION = 1
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "docs/benchmarks/datasets.json"
GIB = 2**30
KEYS = ("fill", "chunk_kib", "codec", "shuffle", "block_kib")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)


def invalid_constant(value):
    raise ValueError(f"Nonfinite JSON: {value}")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows and all(None not in row and None not in row.values() for row in rows),
            f"Malformed or empty CSV: {path}")
    return rows


def number(value, *, optional=False):
    if optional and value in (None, ""):
        return None
    require(not isinstance(value, bool), f"Invalid numeric value: {value!r}")
    result = float(value)
    require(math.isfinite(result), f"Nonfinite metric: {value!r}")
    return result


def integer(value):
    result = number(value)
    require(result.is_integer(), f"Expected integer: {value!r}")
    return int(result)


def identity(row):
    return tuple(integer(row[k]) if k in ("chunk_kib", "block_kib") else row[k] for k in KEYS)


def configuration_id(row, level):
    return "__".join(map(str, (*identity(row), level)))


def safe_file(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), f"Archive path escapes root: {name}")
    return path


def check_hash(path, expected, *, uncompressed=False):
    raw = path.read_bytes()
    if uncompressed:
        raw = gzip.decompress(raw)
    if hashlib.sha256(raw).hexdigest() == expected:
        return
    # Git's Windows checkout can expand LF text to CRLF. Compare canonical
    # repository text as well; compressed/raw JSONL bytes must match exactly.
    if not uncompressed and path.suffix != ".gz":
        raw.decode("utf-8")
        if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected:
            return
    raise ValueError(f"SHA256 mismatch: {path}")


def expected_configs(matrix):
    return {(fill, chunk, codec, shuffle, block)
            for fill in matrix["fills"] for chunk in matrix["chunks_kib"]
            for codec, spec in matrix["codecs"].items()
            for shuffle in spec["shuffles"] for block in spec["blocks_kib"]
            if block <= chunk}


def stats(values):
    # Partial observations cannot establish a complete repetition summary.
    if not values or any(v is None for v in values):
        return {"median": None, "min": None, "max": None}
    values = [number(v) for v in values]
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def close(actual, expected, label):
    require(actual is None and expected is None or actual is not None and expected is not None
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"Summary differs from repetitions: {label} ({actual} != {expected})")


def summary_row(source, format_name):
    node = format_name == "node-jsonl-v1"
    median, lo, hi, fold, estimate = (("speed", "lo", "hi", "fold", "estimate_gib") if node else
        ("throughput_median_gibs", "throughput_min_gibs", "throughput_max_gibs", "compression_fold",
         "estimated_device_gib" if format_name == "python-jsonl-v1" else "estimated_total_gib"))
    throughput = dict(zip(("median", "min", "max"), (number(source[k]) for k in (median, lo, hi))))
    require(0 < throughput["min"] <= throughput["median"] <= throughput["max"], "Invalid throughput range")
    compression = number(source[fold])
    require(compression > 0, "Compression fold must be positive")
    memory = {"median": number(source.get("device_gib"), optional=True),
              "min": number(source.get("device_min_gib"), optional=True),
              "max": number(source.get("device_max_gib"), optional=True)}
    allocation = number(source.get(estimate), optional=True)
    require(all(v is None or v >= 0 for v in (*memory.values(), allocation)), "Negative memory")
    return {"throughput_gibs": throughput, "compression_fold": compression,
            "repetitions": integer(source["repeats"]), "measured_device_gib": memory,
            "estimated_device_gib": allocation,
            "estimated_pinned_gib": number(source.get("pinned_gib"), optional=True)}


def validate_result(record, experiment, workload, *, node):
    config, result = record["config"], record["result"]
    require(result["status"] == "pass" and not record.get("validation_error")
            and not record.get("timed_out") and record.get("code", 0) == 0, "Failed raw execution")
    chunk = integer(config["chunk_kib"])
    geometry = workload["chunks"][str(chunk)]
    shape, count = geometry["shape"], geometry["chunks_per_epoch"]
    require(result["chunks_per_epoch"] == count and
            result["total_chunks"] == math.ceil(workload["frames"] / shape[0]) * count,
            "Changed measured chunk count")
    require(result["worker_threads"] == workload["worker_threads"] and
            math.isclose(result["input_gib"], workload["input_bytes"] / GIB, rel_tol=1e-5),
            "Changed worker count or input size")
    require(result["stages"]["memcpy"]["in_bytes"] == workload["input_bytes"] and
            result["stages"]["compress"]["in_bytes"] == result["total_chunks"] * chunk * 1024,
            "Changed input or padded bytes")
    level = experiment["matrix"]["codecs"][config["codec"]]["level"]
    if config["block_kib"]:
        require(result["blosc_block_bytes"] == config["block_kib"] * 1024, "Returned block size mismatch")
        require(result["blosc_shuffle" if node else "shuffle"] == config["shuffle"] and
                result["blosc_level" if node else "level"] == level, "Returned codec settings mismatch")
    if not node:
        actual = record["actual"]
        require(actual["chunk_shape"] == shape and actual["chunks_per_epoch"] == count and
                actual["epochs_per_batch"] == 1 and
                actual["padded_batch_bytes"] == geometry["padded_batch_bytes"], "Changed batch geometry")
        require(config["level"] == level and result["level"] == level and
                result["shuffle"] == config["shuffle"], "Returned codec settings mismatch")
    else:
        args = record["args"]
        shapes = [int(v) for v in re.findall(r"^\s+[0-3]\s+[tcyx]\s+\d+\s+(\d+)", record["stderr"], re.MULTILINE)]
        require(shapes == shape and f"auto-fit: {chunk * 1024} bytes/chunk (batch=1)" in record["stderr"],
                "Returned chunk shape or batch differs")
        for flag, value in (("--fill", config["fill"]), ("--codec", config["codec"]),
                            ("--frames", str(workload["frames"])), ("--dtype", workload["dtype"]),
                            ("--backend", workload["backend"]),
                            ("--chunk-bytes", f"{chunk}K"),
                            ("--max-threads", str(workload["worker_threads"]))):
            require(flag in args and args[args.index(flag) + 1] == value, f"Command differs: {flag}")
    number(result["throughput_in_gibs"])
    number(result["compression_fold"])
    used, estimated, residual = (result.get(k) for k in
        ("memory_device_used_bytes", "memory_estimate_total_bytes", "memory_device_overhead_bytes"))
    if all(v is not None for v in (used, estimated, residual)):
        require(used - estimated == residual, "Memory residual differs from observed minus estimated bytes")


def validate_repetitions(records, sources, experiment, workload, *, node):
    """Return measured records grouped by configuration; check warmups separately."""
    expected = expected_configs(experiment["matrix"])
    require({identity(s) for s in sources} == expected, "Incomplete summary configuration matrix")
    groups = defaultdict(list)
    all_groups = defaultdict(list)
    repeat_key = "pass" if node else "repeat"
    warmups, repeats = experiment["warmups"], experiment["repeats"]
    require(len(records) == len(expected) * (warmups + repeats), "Wrong raw execution count")
    for record in records:
        key = identity(record["config"])
        require(key in expected, "Unknown raw configuration")
        repeat = integer(record[repeat_key])
        require(type(record["warmup"]) is bool and record["warmup"] == (repeat < warmups),
                "Warmup flag differs from repetition identity")
        validate_result(record, experiment, workload, node=node)
        all_groups[key].append(repeat)
        if not record["warmup"]:
            groups[key].append(record)
    require(set(all_groups) == expected, "Incomplete raw configuration matrix")
    for key, passes in all_groups.items():
        require(sorted(passes) == list(range(warmups + repeats)), f"Missing or duplicate repetitions: {key}")
    return groups


def validate_compact(records, samples):
    require(len(records) == len(samples), "Raw records and compact measurements disagree")
    for record, sample in zip(records, samples):
        require(identity(record["config"]) == identity(sample) and
                record["repeat"] == integer(sample["repeat"]) and
                str(record["warmup"]) == sample["warmup"], "Compact execution identities differ")
        require(sample["utc"] == record["utc"] and sample["id"] == record["config"]["id"] and
                integer(sample["level"]) == record["config"]["level"], "Compact metadata differs")
        for field, value in sample.items():
            raw = (record["elapsed_s"] if field == "elapsed_s" else
                   record["result"]["stages"]["compress"]["total_ms"] if field == "compress_total_ms" else
                   record["result"].get(field))
            if field not in (*KEYS, "level", "id", "repeat", "warmup", "utc"):
                close(number(value, optional=True), raw, f"compact {field}")


def normalize_samples(row, source, records, *, node):
    results = [r["result"] for r in records]
    throughput = stats([r["throughput_in_gibs"] for r in results])
    for key in throughput:
        close(row["throughput_gibs"][key], throughput[key], key)
    for field in ("compression_fold", "memory_estimate_total_bytes", "memory_estimate_pinned_bytes"):
        require(len({r.get(field) for r in results}) == 1, f"Metric varied across repetitions: {field}")
    close(row["compression_fold"], results[0]["compression_fold"], "compression_fold")
    close(number(source["span_pct" if node else "throughput_span_pct"]),
          100 * (throughput["max"] - throughput["min"]) / throughput["median"], "throughput span")
    measured = stats([r.get("memory_device_used_bytes") for r in results])
    measured = {k: v / GIB if v is not None else None for k, v in measured.items()}
    close(row["measured_device_gib"]["median"], measured["median"], "measured memory")
    if node:
        for k in ("min", "max"):
            close(row["measured_device_gib"][k], measured[k], f"measured memory {k}")
    row["measured_device_gib"] = measured
    estimated = results[0].get("memory_estimate_total_bytes")
    close(row["estimated_device_gib"], estimated / GIB if estimated is not None else None, "allocation")
    pinned = results[0].get("memory_estimate_pinned_bytes")
    if node:
        close(row["estimated_pinned_gib"], pinned / GIB if pinned is not None else None, "pinned allocation")
    row["estimated_pinned_gib"] = pinned / GIB if pinned is not None else None
    metric = lambda path: statistics.median(_field(r, path) for r in results)
    if node:
        checks = {"overhead_mib": metric("memory_device_overhead_bytes") / 2**20
                  if all(r.get("memory_device_overhead_bytes") is not None for r in results) else None,
                  "input_bytes": results[0]["stages"]["memcpy"]["in_bytes"],
                  "compressed_payload_bytes": results[0]["stages"]["compress"]["out_bytes"],
                  "padded_bytes": results[0]["stages"]["compress"]["in_bytes"],
                  "compress_ms": metric("stages.compress.total_ms"),
                  "compress_gibs": metric("stages.compress.in_gibs"),
                  "h2d_ms": metric("stages.h2d.total_ms"), "d2h_ms": metric("stages.d2h.total_ms"),
                  "wall_s": metric("wall_s"), "init_s": metric("init_s"),
                  "chunks_per_epoch": results[0]["chunks_per_epoch"], "total_chunks": results[0]["total_chunks"]}
        checks.update(sink_bytes_approx=checks["padded_bytes"] / row["compression_fold"],
                      input_per_encoded=row["compression_fold"] * checks["input_bytes"] / checks["padded_bytes"],
                      padding_factor=checks["padded_bytes"] / checks["input_bytes"])
    else:
        checks = {"compress_total_ms": metric("stages.compress.total_ms"),
                  "estimated_device_bytes": estimated,
                  "device_overhead_bytes": metric("memory_device_overhead_bytes")
                  if all(r.get("memory_device_overhead_bytes") is not None for r in results) else None}
    for key, value in checks.items():
        close(number(source.get(key), optional=True), value, key)
    # Keep every additional source metric available without assigning new meaning.
    # Large per-run diagnostics remain in the unchanged, linked raw archive.
    row["samples"] = [{"repeat": r.get("pass", r.get("repeat")),
                       "throughput_gibs": r["result"]["throughput_in_gibs"],
                       "measured_device_bytes": r["result"].get("memory_device_used_bytes"),
                       "raw_line": r["_line"]} for r in records]


def _field(row, path):
    for part in path.split("."):
        row = row[part]
    return row


def summary_adapter(root, spec, provenance, workload):
    require(provenance["complete"], "Incomplete experiment")
    require(provenance["options"]["repeats"] == spec["repeats"] and
            provenance["options"]["warmups"] == spec["warmups"], "Repetition metadata mismatch")
    check_hash(root / spec["summary"], provenance["archived_summary_csv_sha256"])
    return None


def node_adapter(root, spec, provenance, workload):
    require(provenance["complete"] and not provenance["failures"], "Incomplete experiment")
    require(provenance["options"]["repeats"] == spec["repeats"] and
            provenance["options"]["warmups"] == spec["warmups"], "Repetition metadata mismatch")
    check_hash(root / spec["raw"], provenance["raw_results_sha256"], uncompressed=True)
    records = raw_records(root / spec["raw"])
    require(len(records) == provenance["runs"] and
            sum(not r["warmup"] for r in records) == provenance["measured_runs"], "Provenance count mismatch")
    for chunk, geometry in provenance["geometry"].items():
        require(geometry == workload["chunks"][chunk], "Provenance geometry mismatch")
    return records


def python_adapter(root, spec, provenance, workload):
    build, manifest, validation = (read_json(root / spec[k]) for k in ("build", "collection_manifest", "validation"))
    require(provenance["build"] == build and provenance["manifest"] == manifest, "Provenance differs from collection inputs")
    for field, name in (("manifest_sha256", spec["collection_manifest"]),
                        ("harness_sha256", "sweep.py"), ("patch_sha256", "benchmark-controls.patch")):
        check_hash(root / name, build[field])
    require(provenance["complete"] and manifest["warmups"] == spec["warmups"] and
            manifest["repeats"] == spec["repeats"], "Incomplete experiment or wrong repetition metadata")
    require({identity(r) for r in manifest["configurations"]} == expected_configs(spec["matrix"])
            and len(manifest["configurations"]) == len(expected_configs(spec["matrix"])), "Collection matrix mismatch")
    for key in ("scenario", "frames", "dtype", "sink", "target_batch_bytes"):
        require(manifest[key] == workload[key], f"Collection workload mismatch: {key}")
    check_hash(root / spec["raw"], validation["results_jsonl_sha256"], uncompressed=True)
    records = raw_records(root / spec["raw"])
    require(len(records) == provenance["completed"] == validation["benchmark_executions_passed"], "Provenance count mismatch")
    validate_compact(records, read_csv(root / spec["runs"]))
    return records


ADAPTERS = {"summary-v1": summary_adapter, "node-jsonl-v1": node_adapter, "python-jsonl-v1": python_adapter}


def raw_records(path):
    data = gzip.decompress(path.read_bytes())
    records = []
    for i, line in enumerate(data.splitlines(), 1):
        record = json.loads(line, parse_constant=invalid_constant)
        record["_line"] = i
        records.append(record)
    return records


def normalize_experiment(base, spec, workload):
    root = safe_file(base, spec["directory"])
    retained_names = [f["path"] for f in spec["retained_files"]]
    require(len(retained_names) == len(set(retained_names)), "Duplicate retained file path")
    for key in ("summary", "provenance", "raw", "build", "collection_manifest", "validation", "runs"):
        require(key not in spec or spec[key] in retained_names, f"Unretained adapter input: {key}")
    for retained in spec["retained_files"]:
        path = safe_file(root, retained["path"])
        require(path.is_file(), f"Missing retained file: {path}")
        if "sha256" in retained:
            check_hash(path, retained["sha256"])
    if "checksums.sha256" in retained_names:
        inventoried = set()
        for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            require(name in retained_names and name not in inventoried, f"Invalid current checksum entry: {name}")
            check_hash(safe_file(root, name), digest)
            inventoried.add(name)
        require(inventoried == set(retained_names) - {"checksums.sha256"}, "Incomplete current checksum inventory")
    for reference in spec.get("related_hashes", []):
        check_hash(safe_file(base, reference["path"]), reference["sha256"])
    provenance = read_json(root / spec["provenance"])
    if "options" in provenance:
        for source_key, target_key in (("frames", "frames"), ("dtype", "dtype"), ("max_threads", "worker_threads")):
            require(provenance["options"][source_key] == workload[target_key], f"Provenance workload differs: {target_key}")
        require(provenance["scenario"] == workload["scenario"], "Provenance scenario differs")
    if "padded_batch_bytes" in provenance:
        require(all(provenance["padded_batch_bytes"][k] == v["padded_batch_bytes"] for k, v in workload["chunks"].items()),
                "Provenance padded batch differs")
    sources = read_csv(root / spec["summary"])
    expected = expected_configs(spec["matrix"])
    require(len(sources) == len(expected) and {identity(s) for s in sources} == expected,
            "Duplicate identities or incomplete summary matrix")
    require(spec["format"] in ADAPTERS, f"Unsupported format: {spec['format']}")
    records = ADAPTERS[spec["format"]](root, spec, provenance, workload)
    groups = validate_repetitions(records, sources, spec, workload, node=spec["format"] == "node-jsonl-v1") if records else None
    rows, workloads = [], {}
    for index, source in enumerate(sources, 2):
        row = summary_row(source, spec["format"])
        require(row["repetitions"] == spec["repeats"], "Summary repetition count mismatch")
        level = spec["matrix"]["codecs"][source["codec"]]["level"]
        if "level" in source:
            require(integer(source["level"]) == level, "Summary codec level mismatch")
        chunk = integer(source["chunk_kib"])
        geometry = {k: v for k, v in workload.items() if k != "chunks"}
        geometry.update(fill=source["fill"], chunk_kib=chunk, **workload["chunks"][str(chunk)])
        # Source locations describe the workload but do not change its measured
        # geometry or identity.
        identity_geometry = {k: v for k, v in geometry.items() if k != "source_url"}
        group_id = hashlib.sha256(json.dumps(identity_geometry, sort_keys=True).encode()).hexdigest()[:16]
        workloads[group_id] = {"id": group_id, **geometry}
        config_id = configuration_id(source, level)
        row.update(id=f"{spec['id']}:{config_id}", experiment_id=spec["id"], configuration_id=config_id,
                   workload_id=group_id, fill=source["fill"], chunk_kib=chunk,
                   codec=source["codec"], shuffle=source["shuffle"], block_kib=integer(source["block_kib"]),
                   level=level, control=not source["codec"].startswith("blosc-"),
                   source_metrics=source,
                   provenance={"summary": f"archives/{spec['id']}/{spec['summary']}", "summary_line": index,
                               "metadata": f"archives/{spec['id']}/{spec['provenance']}"})
        if groups:
            normalize_samples(row, source, groups[identity(source)], node=spec["format"] == "node-jsonl-v1")
            row["provenance"]["raw"] = f"archives/{spec['id']}/{spec['raw']}"
        else:
            row["samples"] = None
        rows.append(row)
    start, finish = provenance["start_utc"], provenance["finish_utc"]
    require(datetime.fromisoformat(start) <= datetime.fromisoformat(finish), "Invalid experiment dates")
    experiment = {"id": spec["id"], "label": spec["label"], "format": spec["format"],
                  "start_utc": start, "finish_utc": finish, "hardware": spec["hardware"],
                  "build": provenance.get("build"), "source_commit": provenance.get("source_commit", provenance.get("build", {}).get("source_commit")),
                  "summary_only": records is None, "repetitions": spec["repeats"], "warmups": spec["warmups"],
                  "configuration_count": len(rows), "validated_executions": len(records) if records else None,
                  "notes": spec["notes"], "methodology": spec["methodology"],
                  "files": [{"label": f["path"], "href": f"archives/{spec['id']}/{f['path']}"} for f in spec["retained_files"]],
                  "data": f"data/pareto/{spec['id']}.json"}
    return {"version": VERSION, "experiment": experiment, "workloads": list(workloads.values()), "measurements": sorted(rows, key=lambda r: r["id"])}


def load_datasets(manifest_path=DEFAULT_MANIFEST):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    require(manifest["version"] == VERSION, "Unsupported dataset manifest version")
    specs = manifest["experiments"]
    require(len({e["id"] for e in specs}) == len(specs), "Duplicate experiment identity")
    datasets = []
    for entry in specs:
        require(re.fullmatch(r"[a-z0-9][a-z0-9-]*", entry["id"]), "Invalid experiment identity")
        spec = {**entry, "matrix": manifest["matrices"][entry["matrix"]]}
        try:
            datasets.append(normalize_experiment(manifest_path.parent, spec, manifest["workloads"][entry["workload"]]))
        except (ValueError, KeyError, TypeError, OSError, EOFError) as error:
            raise ValueError(f"{entry['id']}: {error}") from error
    return manifest, datasets


def write_datasets(output, manifest_path=DEFAULT_MANIFEST):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest, datasets = load_datasets(manifest_path)
    data_dir = output / "data/pareto"
    data_dir.mkdir(parents=True, exist_ok=True)
    dump = lambda path, payload: path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    for spec, data in zip(manifest["experiments"], datasets):
        dump(data_dir / f"{spec['id']}.json", data)
        for retained in spec["retained_files"]:
            target = output / "archives" / spec["id"] / retained["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest_path.parent / spec["directory"] / retained["path"], target)
    dump(data_dir / "index.json", {"version": VERSION, "definitions": manifest["definitions"],
                                  "experiments": [d["experiment"] for d in datasets]})
    shutil.copyfile(manifest_path, data_dir / "manifest.json")
    return datasets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    _, datasets = load_datasets(args.manifest)
    for data in datasets:
        experiment = data["experiment"]
        print(f"{experiment['label']}: {experiment['configuration_count']} configurations; "
              f"{experiment['validated_executions'] or 'summary-only'} validated executions")

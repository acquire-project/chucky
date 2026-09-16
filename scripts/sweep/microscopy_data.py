"""Validate retained microscopy observations and derive logical-byte summaries."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from image_results import input_label
from measurements import validate_measurement
from microscopy_entropy import ITEM_BYTES, ROWS_PER_PLANE, sample_rows
from microscopy_plan import CHUNKS, DEFAULT_DEFINITION, decode_json, fingerprint, read_json, validate_plan

DEFAULT_INDEX = Path(__file__).resolve().parents[2] / "bench/studies/microscopy/index.json"
DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "bench/data/microscopy"
DEFAULT_ENTROPY = DEFAULT_INDEX.parent / "entropy.json"
LAYOUT_KEYS = ("reference_shape", "chunk_shape", "chunks_per_shard", "epochs_per_batch",
               "target_batch_bytes", "actual_batch_bytes", "append_elements", "dtype")


def span(values):
    middle = statistics.median(values)
    return {"median": middle, "min": min(values), "max": max(values),
            "spread_percent": 100 * (max(values) - min(values)) / middle}


def check_observation(record, task, config, definition):
    if {key: record.get(key) for key in task} != task:
        raise ValueError("Observation disagrees with the planned execution")
    result = record["result"]
    validate_measurement(result)
    replay = result["image_replay"]
    for key in ("scenario", "input_id", "image_asset_id", "image_split", "codec", "backend", "dtype", "sink"):
        if result.get(key) != config[key]:
            raise ValueError(f"Observation changed {key}")
    expected = {"codec": config["codec"], "backend": config["backend"],
                "dtype": {"u8": "u8", "u16": "u16le", "f32": "f32le"}[config["dtype"]],
                "codec_level": config["level"], "shuffle": config["blosc_shuffle"],
                "order": "cyclic", "target_batch_bytes": 64 << 20}
    if any(replay.get(key) != value for key, value in expected.items()):
        raise ValueError("Observation changed codec, input, or replay policy")
    expected_workers = config.get("max_threads", 4)
    if config["backend"] == "gpu":
        expected_workers = min(expected_workers, 4)
    if "max_threads" in config and (type(result.get("max_threads")) is not int
                                    or result["max_threads"] != config["max_threads"]):
        raise ValueError("Observation changed the requested thread limit")
    chunk = replay["chunk_shape"]
    element_size = {"u8": 1, "u16": 2, "f32": 4}[config["dtype"]]
    if (len(chunk) != 3 or any(type(n) is not int or n <= 0 for n in chunk)
            or chunk[0] != definition["chunk_depth"]
            or math.prod(chunk) * element_size != CHUNKS[config["chunk_label"]]
            or type(result.get("worker_threads")) is not int
            or result["worker_threads"] != expected_workers):
        raise ValueError("Observation changed chunk geometry or worker count")
    if config["codec"].startswith("blosc-") and result.get("blosc_block_bytes") != config["blosc_block_bytes"]:
        raise ValueError("Observation changed the Blosc block request")
    window = result["measurement"]
    for recorded, requested in (("requested_warmup_s", "warmup_s"), ("requested_duration_s", "duration_s")):
        if not math.isclose(window[recorded], task["profile"][requested], rel_tol=1e-6):
            raise ValueError("Observation changed the requested timing policy")
    if (window["warmup_s"] + 1e-6 < task["profile"]["warmup_s"]
            or window["append_s"] + 1e-6 < task["profile"]["duration_s"]):
        raise ValueError("Observation did not cover the requested timing window")
    logical = result["logical_input_bytes"]
    if (type(logical) is not int or logical < math.ceil(task["profile"]["min_gib"] * 2**30)
            or logical > window["input_bytes"] or logical <= 0
            or logical != math.prod(replay["shape"]) * element_size
            or not math.isclose(result["throughput_logical_gibs"], logical / 2**30 / window["elapsed_s"], rel_tol=1e-5)):
        raise ValueError("Invalid logical byte accounting")
    frames, height, width = replay["shape"]
    frame_bytes = math.ceil(height / chunk[1]) * chunk[1] * math.ceil(width / chunk[2]) * chunk[2] * element_size
    if (window["input_bytes"] != frames * frame_bytes
            or result["submitted_bytes"] != window["input_bytes"]
            or replay["source_padded_bytes"] != window["source_bytes"]
            or replay["append_elements"] * element_size != window["append_bytes"]):
        raise ValueError("Invalid padded byte accounting or replay buffer")
    if definition["geometry_frames"] is not None and window["reference_frames"] != definition["geometry_frames"]:
        raise ValueError("Observation changed reference geometry")
    if not math.isfinite(result["process_wall_s"]) or result["process_wall_s"] <= 0:
        raise ValueError("Invalid process duration")
    if not isinstance(result.get("command"), list) or not result["command"]:
        raise ValueError("Observation must retain the executed command")



def check_machine(machine, definition):
    environment = definition["environment"]
    if type(machine.get("cpu_count")) is not int or machine["cpu_count"] <= 0:
        raise ValueError("Recorded machine has no CPU allocation")
    if environment["cpu_count"] is not None and machine["cpu_count"] != environment["cpu_count"]:
        raise ValueError("CPU allocation differs from the study definition")
    if definition["version"] == 3 and "cpu" in definition["backends"]:
        if max(definition["cpu_workers"]) > machine["cpu_count"]:
            raise ValueError("CPU compression workers exceed the allowed CPU count")
    if "gpu" in definition["backends"]:
        if not machine.get("gpu") or machine["gpu"] == "unknown":
            raise ValueError("The study requires an available GPU")
        if environment["gpu"] is not None and machine["gpu"] != environment["gpu"]:
            raise ValueError("GPU differs from the study definition")


def validate_study(document, *, complete=True):
    if type(document.get("version")) is not int or document["version"] != 1 or document.get("benchmark") != "microscopy-study":
        raise ValueError("Unsupported microscopy study")
    if complete and document.get("status") != "complete":
        raise ValueError("Microscopy study is incomplete")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", document.get("id", "")):
        raise ValueError("Invalid retained study id")
    plan = validate_plan(document["plan"])
    if document.get("plan_sha256") != fingerprint(plan):
        raise ValueError("Study plan checksum disagrees")
    check_machine(document["machine"], plan["definition"])
    if (not re.fullmatch(r"[0-9a-f]{40}", document["build"].get("revision", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", document["build"].get("executable_sha256", ""))):
        raise ValueError("Missing source revision or executable checksum")
    records = document["records"]
    schedule = plan["schedule"]
    if len(records) > len(schedule) or complete and len(records) != len(schedule):
        raise ValueError("Incomplete or extra study observations")
    assets = {asset["asset"]: asset for asset in document["corpus"]["selected_assets"]}
    layouts, sources = {}, {}
    for record, task in zip(records, schedule):
        config = plan["cases"][task["case_id"]]
        check_observation(record, task, config, plan["definition"])
        result = record["result"]
        asset = config["image_asset_id"]
        if result["image_input"]["pack_sha256"] != assets[asset]["sha256"]:
            raise ValueError("Observation input checksum disagrees with the corpus")
        expected_asset = assets[asset]
        if (result["image_replay"]["shape"][1:] != [expected_asset["height"], expected_asset["width"]]
                or result["image_input"]["pack_id"] != asset
                or result["image_input"]["input_id"] != config["input_id"]):
            raise ValueError("Observation changed image dimensions or identity")
        source = result["image_input"]
        if asset in sources and sources[asset] != source:
            raise ValueError("Input changed during the study")
        sources[asset] = source
        replay = result["image_replay"]
        layout = {key: replay[key] for key in LAYOUT_KEYS}
        layout["measurement"] = {key: result["measurement"][key] for key in
                                 ("geometry", "source_bytes", "append_bytes", "boundary_timing")}
        key = (asset, config["chunk_label"])
        if key in layouts and layouts[key] != layout:
            raise ValueError("Replay geometry changed across codecs or backends")
        layouts[key] = layout
    return document


def summarize(document):
    validate_study(document)
    plan = document["plan"]
    limit = plan["definition"]["tolerance_percent"]
    records = document["records"]
    by_batch = {batch["id"]: [] for batch in plan["batches"]}
    for record in records:
        by_batch[record["batch_id"]].append(record)
    condition_checks = {}
    for record in records:
        if record["role"] != "sample":
            condition_checks.setdefault(record["case_id"], []).append(record)
    references = {}
    for key, group in by_batch.items():
        checks = [record for record in group if record["role"] != "sample"]
        values = [record["result"]["throughput_logical_gibs"] for record in checks]
        all_checks = condition_checks[checks[0]["case_id"]]
        condition_range = span([record["result"]["throughput_logical_gibs"] for record in all_checks])
        references[key] = {**span(values), "drift": span(values)["spread_percent"] > limit
                           or condition_range["spread_percent"] > limit,
                           "condition_spread_percent": condition_range["spread_percent"],
                           "execution_ids": [record["id"] for record in checks],
                           "condition_execution_ids": [record["id"] for record in all_checks]}
    samples = {}
    for record in records:
        if record["role"] == "sample":
            samples.setdefault(record["case_id"], []).append(record)
    rows = []
    for case_id, observations in samples.items():
        config = plan["cases"][case_id]
        rates = [record["result"]["throughput_logical_gibs"] for record in observations]
        throughput = span(rates)
        detail = min(observations, key=lambda record: abs(record["result"]["throughput_logical_gibs"] - throughput["median"]))
        output = sum(record["result"]["measurement"]["output_bytes"] for record in observations)
        logical = sum(record["result"]["logical_input_bytes"] for record in observations)
        physical = sum(record["result"]["measurement"]["input_bytes"] for record in observations)
        reference = references[detail["batch_id"]]
        result = detail["result"]
        condition = [document["id"], config["input_id"], config["image_asset_id"], config["image_split"],
                     config["backend"], config["sink"], result["image_input"]["pack_sha256"]]
        if "max_threads" in config:
            condition.append(config["max_threads"])
        rows.append({"id": document["id"] + ":" + case_id, "case_id": case_id,
                     "study_id": document["id"], "condition": fingerprint(condition)[:16],
                     "config": config, "input_label": input_label(
                         config["input_id"], result["image_input"].get("dataset_version")),
                     "throughput": throughput, "compression_fold": logical / output,
                     "padding_percent": 100 * (physical / logical - 1), "count": len(rates),
                     "reference": reference, "needs_confirmation": len(rates) < 3
                     or throughput["spread_percent"] > limit or reference["drift"],
                     "detail_execution": detail["id"], "detail": copy.deepcopy(result),
                     "samples": [{"id": record["id"], "started": record["started"],
                                  "throughput_gibs": record["result"]["throughput_logical_gibs"],
                                  "compression_fold": record["result"]["logical_input_bytes"] / record["result"]["measurement"]["output_bytes"],
                                  "process_wall_s": record["result"]["process_wall_s"]}
                                 for record in observations]})
    if plan["phase"] == "comparison":
        from microscopy_uncertainty import summarize_rounds
        for row in rows:
            row["phase"] = "comparison"
            for sample, record in zip(row["samples"], samples[row["case_id"]]):
                sample.update(round=record["round"], batch_id=record["batch_id"],
                              logical_input_bytes=record["result"]["logical_input_bytes"],
                              output_bytes=record["result"]["measurement"]["output_bytes"])
            row["compression_range"] = {"min": min(sample["compression_fold"] for sample in row["samples"]),
                                        "max": max(sample["compression_fold"] for sample in row["samples"])}
        summarize_rounds(rows, seed=plan["definition"]["seed"])
    rows.sort(key=lambda row: (plan["definition"]["inputs"].index(row["config"]["input_id"]),
                              row["config"]["backend"], row["case_id"]))
    data = {"version": 1, "study": {key: document[key] for key in
            ("id", "created", "machine", "build", "corpus", "plan_sha256")},
            "phase": plan["phase"], "definition": plan["definition"], "counts": plan["counts"],
            "measurements": rows}
    if plan["phase"] == "comparison":
        data["study"]["sink_options"] = copy.deepcopy(document.get("sink_options", {}))
        data["uncertainty"] = {
            "method": "Observed ranges; with at least six matched rounds, paired resampling of the throughput median and ratio of summed logical/output bytes",
            "interpretation": "Observed ranges are not confidence intervals. When available, resampled intervals and frontier frequencies are approximate and conditional on those rounds; not posterior probabilities or simultaneous confidence bounds",
            "scope": "Run variation in one recorded machine session on fixed image inputs",
        }
    return data


def estimate_seconds(pilot, plan):
    validate_study(pilot)
    if pilot["plan"]["phase"] != "pilot" or pilot["plan"]["definition"] != plan["definition"]:
        raise ValueError("Cost estimate requires a complete pilot of the same definition")
    costs = {}
    for record in pilot["records"]:
        spec = pilot["plan"]["cases"][record["case_id"]]
        key = (spec["input_id"], spec["backend"], record["role"] != "sample")
        costs.setdefault(key, []).append(record["result"]["process_wall_s"])
    total = 0
    for task in plan["schedule"]:
        spec = plan["cases"][task["case_id"]]
        key = (spec["input_id"], spec["backend"], task["role"] != "sample")
        total += max(costs[key])
    return {"process_seconds": total, "basis": "slowest pilot process per input/backend/role",
            "includes_setup": False, "is_upper_bound": False}


def validate_report(report, datasets):
    if not isinstance(report, list):
        raise ValueError("Microscopy report must list its image inputs")
    studies = {data["study"]["id"]: data for data in datasets}
    inputs, conditions = set(), set()
    for item in report:
        if (not isinstance(item, dict) or set(item) != {"input", "label", "sources"}
                or not isinstance(item["input"], str) or not item["input"]
                or not isinstance(item["label"], str) or not item["label"].strip()
                or not isinstance(item["sources"], list) or not item["sources"]):
            raise ValueError("Invalid microscopy report input")
        if item["input"] in inputs:
            raise ValueError("Duplicate microscopy report input")
        inputs.add(item["input"])
        identities, layouts = set(), {}
        for source in item["sources"]:
            if (not isinstance(source, dict) or set(source) != {"study", "backends"}
                    or not isinstance(source["study"], str) or source["study"] not in studies
                    or not isinstance(source["backends"], list) or not source["backends"]
                    or any(backend not in ("cpu", "gpu") for backend in source["backends"])
                    or len(set(source["backends"])) != len(source["backends"])):
                raise ValueError("Invalid microscopy report source")
            data = studies[source["study"]]
            selected = [row for row in data["measurements"]
                        if row["config"]["input_id"] == item["input"]
                        and row["config"]["backend"] in source["backends"]]
            if {row["config"]["backend"] for row in selected} != set(source["backends"]):
                raise ValueError("Microscopy report source has no matching measurements")
            identities.update((row["config"]["image_asset_id"], row["config"]["image_split"],
                               row["config"]["dtype"], row["detail"]["image_input"]["pack_sha256"]) for row in selected)
            if len(identities) != 1:
                raise ValueError("Microscopy report combines different input content or versions")
            for row in selected:
                replay = row["detail"]["image_replay"]
                if replay["chunk_shape"][0] > len(row["detail"]["image_input"]["plane_order"]):
                    raise ValueError("Microscopy report chunk depth exceeds available planes")
                layout = {key: replay[key] for key in LAYOUT_KEYS}
                chunk = row["config"]["chunk_label"]
                if chunk in layouts and layouts[chunk] != layout:
                    raise ValueError("Microscopy report combines different replay geometry")
                layouts[chunk] = layout
            selected_conditions = {(data["study"]["machine"]["name"], item["input"],
                                    row["config"]["backend"], row["config"]["sink"],
                                    row["config"].get("max_threads", 4)) for row in selected}
            if conditions & selected_conditions:
                raise ValueError("Microscopy report sources overlap for a machine/input/backend/sink/worker count")
            conditions.update(selected_conditions)
    return report


def write_previews(output: Path, datasets, corpus=DEFAULT_CORPUS):
    manifest = corpus / "manifest.json"
    if not manifest.is_file():
        return []
    identities = {(row["config"]["image_asset_id"], row["detail"]["image_input"]["pack_sha256"])
                  for data in datasets for row in data["measurements"]}
    previews = []
    for dataset in read_json(manifest)["datasets"]:
        for asset in dataset["assets"]:
            if (asset["id"], asset["sha256"]) not in identities or not asset.get("thumbnail"):
                continue
            thumbnail = Path(asset["thumbnail"])
            source = (corpus / thumbnail).resolve()
            if thumbnail.is_absolute() or not source.is_relative_to((corpus / "thumbnails").resolve()):
                raise ValueError("Microscopy thumbnail escapes its directory")
            relative = f"data/microscopy/thumbnails/{asset['sha256'][:12]}-{source.name}"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = source.read_bytes()
            target.write_bytes(raw)
            previews.append({"asset": asset["id"], "pack_sha256": asset["sha256"],
                             "file": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                             "name": asset["name"], "dtype": asset["dtype"], "shape": asset["shape"],
                             "modality": dataset["modality"], "source": {key: dataset["source"][key]
                             for key in ("collection", "url", "attribution", "license", "license_url")}})
    return previews


def load_entropy(datasets, path=DEFAULT_ENTROPY):
    document = read_json(path)
    if document.get("version") != 2 or document.get("rows_per_plane") != ROWS_PER_PLANE:
        raise ValueError("Unsupported microscopy entropy sample")
    identities = {(row["config"]["image_asset_id"], row["detail"]["image_input"]["pack_sha256"],
                   {"u8": "uint8", "u16": "uint16", "f32": "float32"}[row["config"]["dtype"]])
                  for data in datasets for row in data["measurements"]}
    selected, seen = [], set()
    for record in document["inputs"]:
        identity = record["asset"], record["pack_sha256"]
        shape, entropy = record["shape"], record["pixel_entropy_bits"]
        pixels, unique_values = record["sample_pixels"], record["unique_values"]
        if (identity in seen or record["dtype"] not in ITEM_BYTES
                or len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
                or not all(re.fullmatch(r"[0-9a-f]{64}", record[key]) for key in ("pack_sha256", "sample_sha256"))
                or record["sample_rows"] != sample_rows(shape[1])
                or type(pixels) is not int or pixels != len(record["sample_rows"]) * shape[0] * shape[2]
                or type(unique_values) is not int
                or not 1 <= unique_values <= min(pixels, 1 << (8 * ITEM_BYTES[record["dtype"]]))
                or type(entropy) not in (int, float) or not math.isfinite(entropy)
                or not 0 <= entropy <= math.log2(unique_values) + 1e-12):
            raise ValueError("Invalid microscopy entropy sample")
        seen.add(identity)
        if (*identity, record["dtype"]) in identities:
            selected.append(record)
    return {**document, "inputs": selected}


def write_datasets(output: Path, index_path=DEFAULT_INDEX, extra=()):
    index = read_json(index_path)
    if (index.get("version") != 1 or not {"version", "studies"} <= set(index)
            or set(index) - {"version", "studies", "report"}):
        raise ValueError("Unsupported microscopy study index")
    paths = []
    for entry in index["studies"]:
        path = (index_path.parent / entry["path"]).resolve()
        if not path.is_relative_to(index_path.parent.resolve()):
            raise ValueError("Retained study path escapes its directory")
        paths.append((path, entry["sha256"]))
    paths.extend((path, None) for path in extra)
    studies, datasets, seen = [], [], set()
    data_dir = output / "data/microscopy"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "discovery.json").write_bytes(DEFAULT_DEFINITION.read_bytes())
    for path, expected_sha256 in paths:
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and checksum != expected_sha256:
            raise ValueError("Retained microscopy checksum disagrees")
        document = decode_json(raw)
        data = summarize(document)
        study_id = document["id"]
        if study_id in seen:
            raise ValueError(f"Duplicate microscopy study id: {study_id}")
        seen.add(study_id)
        archive = f"archives/microscopy/{study_id}/study.json"
        target = output / archive
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        data["study"]["archive"] = archive
        data["study"]["sha256"] = checksum
        (data_dir / f"{study_id}.json").write_text(json.dumps(data, allow_nan=False, separators=(",", ":")))
        datasets.append(data)
        studies.append({"id": study_id, "label": document["plan"]["definition"]["label"],
                        "phase": data["phase"], "machine": document["machine"]["name"],
                        "created": document["created"], "file": f"data/microscopy/{study_id}.json"})
    result = {"version": 1, "studies": studies, "previews": write_previews(output, datasets),
              "entropy": load_entropy(datasets)}
    if "report" in index:
        report = copy.deepcopy(index["report"])
        for data in datasets[len(index["studies"]):]:
            for input_id in data["definition"]["inputs"]:
                selected = [row for row in data["measurements"] if row["config"]["input_id"] == input_id]
                if not selected:
                    continue
                item = next((item for item in report if item["input"] == input_id), None)
                if item is None:
                    item = {"input": input_id, "label": selected[0]["input_label"], "sources": []}
                    report.append(item)
                item["sources"].append({"study": data["study"]["id"],
                                        "backends": sorted({row["config"]["backend"] for row in selected})})
        result["report"] = validate_report(report, datasets)
    (data_dir / "index.json").write_text(json.dumps(result, allow_nan=False))
    return studies

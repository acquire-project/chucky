"""Selected microscopy settings measured in matched randomized rounds."""
from __future__ import annotations

import random

from microscopy_plan import CHUNKS, CODECS, fingerprint, positive, validate_profile


def validate_settings(settings, codecs, *, allow_backend=False):
    fields = {"codec", "chunk_label"}
    if settings.get("codec", "").startswith("blosc-"):
        fields.add("blosc_block_bytes")
    if allow_backend and "backend" in settings:
        fields.add("backend")
        if settings["backend"] not in ("cpu", "gpu"):
            raise ValueError("Unknown selected backend")
    if (set(settings) != fields or settings["codec"] not in codecs
            or settings["chunk_label"] not in CHUNKS):
        raise ValueError("Invalid selected compression setting")
    if "blosc_block_bytes" in settings:
        block = settings["blosc_block_bytes"]
        if type(block) is not int or not 128 <= block <= CHUNKS[settings["chunk_label"]]:
            raise ValueError("Selected Blosc block must fit inside its chunk")


def selected_settings(definition, input_id, backend):
    return [setting for setting in definition["configurations"][input_id]
            if setting.get("backend", backend) == backend]


def validate_definition(definition):
    fields = {"version", "id", "label", "dataset", "inputs", "backends", "sinks",
              "configurations", "codecs", "chunk_depth", "geometry_frames", "profile",
              "rounds", "rounds_per_group", "reference", "tolerance_percent", "seed", "environment"}
    if set(definition) != fields or type(definition["version"]) is not int or definition["version"] != 2:
        raise ValueError("Unsupported microscopy comparison definition")
    for key in ("id", "label", "dataset"):
        if not isinstance(definition[key], str) or not definition[key]:
            raise ValueError(f"Missing study {key}")
    for key in ("inputs", "backends", "sinks"):
        values = definition[key]
        if (not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Study {key} must be nonempty and unique")
    if set(definition["backends"]) - {"cpu", "gpu"}:
        raise ValueError("Unknown study backend")
    if set(definition["sinks"]) - {"discard", "fs"}:
        raise ValueError("Comparisons support discard and filesystem sinks")
    codecs = definition["codecs"]
    if not codecs or codecs.keys() - CODECS:
        raise ValueError("Unknown study codec")
    for codec, settings in codecs.items():
        if (set(settings) != {"level", "shuffle"} or type(settings["level"]) is not int
                or not 0 <= settings["level"] <= (9 if codec.startswith("blosc-") else 255)
                or settings["shuffle"] != ("bit" if codec.startswith("blosc-") else "none")):
            raise ValueError("Comparison requires fixed codec levels, Blosc bitshuffle, and raw controls")
    positive(definition["chunk_depth"], "chunk_depth", integer=True)
    if definition["geometry_frames"] is not None:
        positive(definition["geometry_frames"], "geometry_frames", integer=True)
    positive(definition["rounds"], "rounds", integer=True, minimum=1)
    positive(definition["rounds_per_group"], "rounds_per_group", integer=True)
    if definition["rounds"] % definition["rounds_per_group"]:
        raise ValueError("Rounds must divide into complete reference groups")
    positive(definition["tolerance_percent"], "tolerance_percent")
    if type(definition["seed"]) is not int:
        raise ValueError("Study seed must be an integer")
    if set(definition["profile"]) != {"warmup_s", "duration_s", "min_gib"}:
        raise ValueError("Unknown comparison profile fields")
    validate_profile(definition["profile"])
    reference = definition["reference"]
    if set(reference) != {"warmup_s", "duration_s", "min_gib", "configurations"}:
        raise ValueError("Unknown comparison reference fields")
    validate_profile(reference)
    for configurations in (definition["configurations"], reference["configurations"]):
        if set(configurations) != set(definition["inputs"]):
            raise ValueError("Each input needs selected settings and a reference")
    for settings in reference["configurations"].values():
        validate_settings(settings, codecs)
    for input_id, selected in definition["configurations"].items():
        if (not isinstance(selected, list) or not selected
                or len({fingerprint(settings) for settings in selected}) != len(selected)):
            raise ValueError("Selected compression settings must be nonempty and unique")
        for settings in selected:
            validate_settings(settings, codecs, allow_backend=True)
        for backend in ("cpu", "gpu"):
            settings = selected_settings(definition, input_id, backend)
            keys = [fingerprint({key: value for key, value in setting.items() if key != "backend"})
                    for setting in settings]
            if len(keys) != len(set(keys)):
                raise ValueError("Selected compression settings overlap for a backend")
        if not any(selected_settings(definition, input_id, backend) for backend in definition["backends"]):
            raise ValueError(f"No selected settings for the requested backends: {input_id}")
    environment = definition["environment"]
    if set(environment) != {"gpu", "cpu_count"}:
        raise ValueError("Unknown comparison environment fields")
    if environment["cpu_count"] is not None:
        positive(environment["cpu_count"], "cpu_count", integer=True)
    if environment["gpu"] is not None and (not isinstance(environment["gpu"], str) or not environment["gpu"]):
        raise ValueError("Expected GPU must be a name or null")
    return definition


def make_plan(definition, members):
    validate_definition(definition)
    by_input = {}
    for asset, input_id, dtype in members:
        if input_id in by_input:
            raise ValueError(f"Ambiguous study input: {input_id}")
        by_input[input_id] = (asset, dtype)
    if set(definition["inputs"]) - by_input.keys():
        raise ValueError("Study input is absent from the registered dataset")
    configs, batches, schedule = {}, [], []
    rng = random.Random(definition["seed"])

    def case(input_id, backend, sink, selected):
        asset, dtype = by_input[input_id]
        codec = selected["codec"]
        settings = definition["codecs"][codec]
        spec = {"scenario": "microscopy", "fill": "images", "input_id": input_id,
                "image_asset_id": asset, "image_split": "core", "dtype": dtype,
                "backend": backend, "codec": codec, "chunk_label": selected["chunk_label"],
                "chunk_depth": definition["chunk_depth"], "sink": sink,
                "blosc_block_bytes": selected.get("blosc_block_bytes"),
                "blosc_shuffle": settings["shuffle"], "level": settings["level"]}
        key = fingerprint(spec)[:16]
        configs[key] = spec
        return key

    settings = {(input_id, backend): selected_settings(definition, input_id, backend)
                for input_id in definition["inputs"] for backend in definition["backends"]}
    conditions = [(input_id, backend, sink) for (input_id, backend), values in settings.items()
                  if values for sink in definition["sinks"]]
    selected = {condition: [case(*condition, setting) for setting in settings[condition[:2]]]
                for condition in conditions}
    references = {condition: case(*condition, definition["reference"]["configurations"][condition[0]])
                  for condition in conditions}
    reference_profile = {key: definition["reference"][key] for key in ("warmup_s", "duration_s", "min_gib")}

    def append(condition, case_id, role, round_number, profile, group_ids):
        schedule.append({"id": f"execution-{len(schedule) + 1:04d}", "batch_id": group_ids[condition],
                         "case_id": case_id, "role": role, "repeat": round_number,
                         "round": round_number, "profile": profile})

    size = definition["rounds_per_group"]
    for start in range(1, definition["rounds"] + 1, size):
        group_ids = {}
        for condition in conditions:
            batch_id = f"group-{len(batches) + 1:03d}"
            group_ids[condition] = batch_id
            batches.append({"id": batch_id, "case_ids": selected[condition],
                            "reference_case_id": references[condition],
                            "rounds": list(range(start, start + size))})
        ordered = list(conditions)
        rng.shuffle(ordered)
        for condition in ordered:
            append(condition, references[condition], "reference-before", start, reference_profile, group_ids)
        for round_number in range(start, start + size):
            pairs = [(input_id, backend, index) for (input_id, backend), values in settings.items()
                     for index in range(len(values))]
            rng.shuffle(pairs)
            for input_id, backend, index in pairs:
                sinks = list(definition["sinks"])
                rng.shuffle(sinks)
                for sink in sinks:
                    condition = (input_id, backend, sink)
                    append(condition, selected[condition][index], "sample", round_number,
                           definition["profile"], group_ids)
        rng.shuffle(ordered)
        for condition in ordered:
            append(condition, references[condition], "reference-after", start + size - 1, reference_profile, group_ids)
    measured = {task["case_id"] for task in schedule if task["role"] == "sample"}
    return {"version": 2, "definition": definition, "phase": "comparison", "cases": configs,
            "batches": batches, "schedule": schedule,
            "counts": {"configurations": len(measured),
                       "samples": sum(task["role"] == "sample" for task in schedule),
                       "references": sum(task["role"] != "sample" for task in schedule),
                       "executions": len(schedule)},
            "requested_seconds": sum(task["profile"]["warmup_s"] + task["profile"]["duration_s"]
                                     for task in schedule)}

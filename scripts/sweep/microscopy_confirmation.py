# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "rich", "pydantic"]
# ///
"""Select a fixed confirmation study from matching discard and filesystem screens."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path

from microscopy_data import LAYOUT_KEYS, summarize, validate_study
from microscopy_plan import CHUNKS, fingerprint, make_plan, read_json


def setting(config):
    keys = ("backend", "codec", "chunk_label", "blosc_block_bytes")
    return {key: config[key] for key in keys if config.get(key) is not None}


def without_sink(config):
    return {key: value for key, value in config.items() if key != "sink"}


def candidates(rows, tolerance_percent=5):
    if not math.isfinite(tolerance_percent) or tolerance_percent <= 0:
        raise ValueError("The selection tolerance must be finite and positive")
    margin = 1 + tolerance_percent / 100
    selected = {}
    for row in rows:
        rate, fold = row["throughput"]["median"], row["compression_fold"]
        frontier = not any(other["throughput"]["median"] >= rate and other["compression_fold"] >= fold
                           and (other["throughput"]["median"] > rate or other["compression_fold"] > fold)
                           for other in rows)
        high_rate, high_fold = row["throughput"]["max"], row["compression_range"]["max"]
        contender = not any(other["throughput"]["min"] >= high_rate
                            and other["compression_range"]["min"] >= high_fold
                            and (other["throughput"]["min"] >= margin * high_rate
                                 or other["compression_range"]["min"] >= margin * high_fold)
                            for other in rows if other is not row)
        if frontier or contender:
            selected[row["case_id"]] = "observed-frontier" if frontier else "overlapping-observed-ranges"
    return selected


def representatives(rows, tolerance_percent=5):
    groups = defaultdict(list)
    for row in rows:
        groups[row["config"]["codec"]].append(row)
    selected = []
    for codec, group in groups.items():
        if codec in ("zstd", "blosc-zstd"):
            maximum = max(row["compression_fold"] for row in group)
            group = [row for row in group if row["compression_fold"] >= maximum / (1 + tolerance_percent / 100)]
        selected.append(max(group, key=lambda row: (row["throughput"]["median"], row["compression_fold"],
                                                    fingerprint(setting(row["config"])))))
    return selected


def check_screens(filesystem, discard):
    for document, sink in ((filesystem, "fs"), (discard, "discard")):
        validate_study(document)
        if document["plan"]["version"] != 2 or document["plan"]["definition"]["sinks"] != [sink]:
            raise ValueError("Confirmation requires one filesystem screen and one discard screen")
    first, second = filesystem["plan"], discard["plan"]
    for key in ("dataset", "inputs", "backends", "codecs", "chunk_depth", "geometry_frames", "profile", "reference"):
        if first["definition"][key] != second["definition"][key]:
            raise ValueError(f"The screens changed the measurement protocol: {key}")
    for key in ("executable_sha256", "cmake_cache_sha256", "source_tree_sha256"):
        if not filesystem["build"].get(key) or filesystem["build"][key] != discard["build"].get(key):
            raise ValueError(f"The screens must use the same native build: {key}")
    for key in ("name", "cpu_count", "gpu"):
        if filesystem["machine"].get(key) != discard["machine"].get(key):
            raise ValueError(f"The screens must use the same machine class: {key}")
    sources = [{asset["asset"]: asset for asset in document["corpus"]["selected_assets"]}
               for document in (filesystem, discard)]
    layouts = []
    for document in (filesystem, discard):
        measured = {}
        for record in document["records"]:
            if record["role"] != "sample":
                continue
            config = document["plan"]["cases"][record["case_id"]]
            result = record["result"]
            measured[fingerprint(without_sink(config))] = {
                "replay": {key: result["image_replay"][key] for key in LAYOUT_KEYS},
                "measurement": {key: result["measurement"][key]
                                for key in ("geometry", "source_bytes", "append_bytes", "boundary_timing")},
            }
            asset = config["image_asset_id"]
            if sources[0].get(asset) != sources[1].get(asset):
                raise ValueError(f"The screens changed the input: {asset}")
        layouts.append(measured)
    if layouts[0].keys() != layouts[1].keys():
        raise ValueError("The screens must measure the same compression grid on both backends")
    if layouts[0] != layouts[1]:
        raise ValueError("The screens changed replay geometry")


def select_confirmation(filesystem, discard, *, max_settings=64):
    if type(max_settings) is not int or max_settings < 1:
        raise ValueError("The setting budget must be a positive integer")
    check_screens(filesystem, discard)
    definition = copy.deepcopy(filesystem["plan"]["definition"])
    selected, evidence = {}, defaultdict(list)
    summaries = [summarize(document) for document in (filesystem, discard)]

    def add(row, reason):
        config = row["config"]
        key = (config["input_id"], fingerprint(setting(config)))
        selected[key] = setting(config)
        evidence[key].append({"study": row["study_id"], "case_id": row["case_id"], "reason": reason})

    for data in summaries:
        groups = defaultdict(list)
        for row in data["measurements"]:
            groups[row["condition"]].append(row)
        for rows in groups.values():
            reasons = candidates(rows, definition["tolerance_percent"])
            for row in rows:
                if row["case_id"] in reasons:
                    add(row, reasons[row["case_id"]])
    rows = summaries[0]["measurements"]
    backend = "gpu" if "gpu" in definition["backends"] else "cpu"
    for input_id in definition["inputs"]:
        controls = representatives([row for row in rows if row["config"]["input_id"] == input_id
                                    and row["config"]["backend"] == backend], definition["tolerance_percent"])
        for control in controls:
            compression = {key: value for key, value in setting(control["config"]).items() if key != "backend"}
            for row in rows:
                if (row["config"]["input_id"] == input_id
                        and {key: value for key, value in setting(row["config"]).items() if key != "backend"} == compression):
                    add(row, "matched-codec-control")
    if len(selected) > max_settings:
        raise ValueError(f"Selected {len(selected)} input/backend settings, exceeding the budget of {max_settings}; "
                         "review the budget instead of dropping candidates")

    def order(key):
        input_id, _ = key
        config = selected[key]
        return (definition["inputs"].index(input_id), definition["backends"].index(config["backend"]),
                list(definition["codecs"]).index(config["codec"]), CHUNKS[config["chunk_label"]],
                config.get("blosc_block_bytes", 0))

    definition.update(id="microscopy-final-comparison-v1", label="Microscopy final sink comparison",
                      sinks=["discard", "fs"], rounds=3, rounds_per_group=3, seed=2026091603,
                      environment={"gpu": None, "cpu_count": None},
                      configurations={input_id: [] for input_id in definition["inputs"]})
    selection = []
    for key in sorted(selected, key=order):
        definition["configurations"][key[0]].append(selected[key])
        selection.append({"input_id": key[0], **selected[key], "evidence": evidence[key]})
    members = {(config["image_asset_id"], config["input_id"], config["dtype"])
               for config in filesystem["plan"]["cases"].values()}
    plan = make_plan(definition, sorted(members))
    return definition, {
        "version": 1, "settings": selection, "setting_budget": max_settings, "counts": plan["counts"],
        "tolerance_percent": definition["tolerance_percent"],
        "rule": "Union of both sink frontiers and contenders not clearly dominated across observed ranges; "
                "one representative per codec is also measured on both backends",
        "controls": "Fastest filesystem LZ4 and uncompressed settings; fastest Zstd settings within the selection "
                    "tolerance of that codec's best logical compression fold, selected on GPU when available",
        "interpretation": "Selection uses observed ranges, not confidence bounds. Screen observations are not "
                          "pooled into the three new matched rounds. Report observed ranges for those rounds, "
                          "without confidence intervals or frontier probability estimates.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filesystem", type=Path, required=True)
    parser.add_argument("--discard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-settings", type=int, default=64,
                        help="Maximum input/backend settings before review; never truncates candidates")
    args = parser.parse_args()
    try:
        definition, selection = select_confirmation(read_json(args.filesystem), read_json(args.discard),
                                                     max_settings=args.max_settings)
        selection["sources"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                for path in (args.filesystem, args.discard)]
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in (("definition.json", definition), ("selection.json", selection)):
            (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"settings": len(selection["settings"]), **selection["counts"]}, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Microscopy confirmation: {error}\n")


if __name__ == "__main__":
    main()

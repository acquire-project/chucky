# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic"]
# ///
"""Compare CPU worker counts and machines using retained microscopy observations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from microscopy_data import span, summarize
from microscopy_plan import fingerprint, read_json


def matched_ratios(row, baseline):
    current = {sample["round"]: sample["throughput_gibs"] for sample in row["samples"]}
    previous = {sample["round"]: sample["throughput_gibs"] for sample in baseline["samples"]}
    if current.keys() != previous.keys() or len(current) < 2:
        return None
    return {**span([current[number] / previous[number] for number in sorted(current)]),
            "rounds": sorted(current), "baseline": baseline["id"]}


def compare(documents):
    studies = [summarize(document) for document in documents]
    ids = [data["study"]["id"] for data in studies]
    if len(set(ids)) != len(ids):
        raise ValueError("A study was supplied more than once")
    policies = {fingerprint({key: data["definition"].get(key) for key in
                             ("profile", "chunk_depth", "geometry_frames")}) for data in studies}
    if len(policies) > 1:
        raise ValueError("Measurement profiles or geometry policy differ between studies")
    identities, compact, grouped, settings, rows = {}, {}, defaultdict(list), defaultdict(dict), []
    for data in studies:
        if data["phase"] != "comparison":
            raise ValueError("Worker comparisons require matched rounds")
        for row in data["measurements"]:
            config = row["config"]
            input_id = config["input_id"]
            identity = (config["image_asset_id"], config["image_split"], config["dtype"],
                        row["detail"]["image_input"]["pack_sha256"])
            if input_id in identities and identities[input_id] != identity:
                raise ValueError(f"Input content differs between studies: {input_id}")
            identities[input_id] = identity
            workers = row["detail"]["worker_threads"]
            group = (row["study_id"], input_id, config["backend"], config["sink"], workers)
            grouped[group].append(row)
            budget = (input_id, config["sink"])
            compact[budget] = max(compact.get(budget, 0), row["compression_fold"])
            key = (row["study_id"], fingerprint({key: value for key, value in config.items() if key != "max_threads"}))
            settings[key][workers] = row
            rows.append((row, key, group))
    measurements, extension = [], []
    for row, key, group in rows:
        config = row["config"]
        item = {"id": row["id"], "study": group[0], "input": group[1], "backend": group[2],
                "sink": group[3], "workers": group[4], "codec": config["codec"],
                "chunk": config["chunk_label"], "block_bytes": config["blosc_block_bytes"],
                "throughput": row["throughput"], "compression_fold": row["compression_fold"],
                "observations": row["count"], "reference": row["reference"],
                "detail_execution": row["detail_execution"],
                "stages": row["detail"].get("stages", {})}
        if group[2] == "cpu" and 4 in settings[key]:
            item["speedup_over_four_workers"] = matched_ratios(row, settings[key][4])
        if group[2] == "cpu" and group[4] == 32 and 16 in settings[key]:
            item["speedup_over_sixteen_workers"] = matched_ratios(row, settings[key][16])
            ratios = item["speedup_over_sixteen_workers"]
            if ratios and ratios["median"] > 1.10:
                extension.append({"id": row["id"], "speedup_32_over_16": ratios})
        measurements.append(item)
    choices = []
    for group, candidates in grouped.items():
        best_fold = compact[(group[1], group[3])]
        eligible = [row for row in candidates if row["compression_fold"] >= best_fold / 1.10]
        chosen = max(eligible, key=lambda row: (row["throughput"]["median"], row["compression_fold"])) if eligible else None
        choices.append({"study": group[0], "input": group[1], "backend": group[2], "sink": group[3],
                        "workers": group[4], "smallest_output_fold": best_fold,
                        "configuration": chosen["id"] if chosen else None})
    return {"version": 1, "studies": [{"id": data["study"]["id"],
            "machine": data["study"]["machine"]["name"], "cpu_models": data["study"]["machine"].get("cpu_models"),
            "allowed_cpus": data["study"]["machine"]["cpu_count"],
            "allowed_physical_cores": data["study"]["machine"].get("cpu_topology", {}).get("allowed_physical_cores"),
            "revision": data["study"]["build"]["revision"], "native_source_sha256": data["study"]["build"].get("source_tree_sha256"),
            "plan_sha256": data["study"]["plan_sha256"]}
            for data in studies], "measurements": measurements, "choices": choices,
            "possible_64_worker_checks": extension,
            "interpretation": {
                "ranges": "Observed minimum and maximum, not confidence intervals",
                "speedups": "Ratios pair the same configuration and round within one machine session",
                "choices": "Fastest median within 10% of the smallest output observed across supplied studies for this input and sink",
                "extension": "Consider 64 workers where the median within-round 32/16 speedup exceeds 1.10; no automatic extra measurements",
                "stages": "Stage values come from the recorded representative execution; overlapping stage times must not be added",
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare([read_json(path) for path in args.study])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "measurements.csv").open("w", newline="") as output:
        fields = ["id", "study", "input", "backend", "sink", "workers", "codec", "chunk", "block_bytes",
                  "compression_fold", "observations", "throughput_median", "throughput_min", "throughput_max",
                  "speedup_median", "speedup_min", "speedup_max"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in report["measurements"]:
            flat = {field: row[field] for field in fields if field in row}
            for label, value in (("throughput", row["throughput"]), ("speedup", row.get("speedup_over_four_workers"))):
                flat.update({f"{label}_{key}": value[key] if value else None for key in ("median", "min", "max")})
            writer.writerow(flat)
    print(json.dumps({"measurements": len(report["measurements"]), "choices": len(report["choices"]),
                      "possible_64_worker_checks": len(report["possible_64_worker_checks"])}, indent=2))


if __name__ == "__main__":
    main()

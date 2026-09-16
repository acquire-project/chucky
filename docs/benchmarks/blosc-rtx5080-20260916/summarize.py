"""Summarize the retained RTX 5080 Blosc run, including failed executions."""
import csv
import gzip
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
GIB = 2**30


def write_csv(name, rows, fields):
    with (ROOT / name).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    archive = ROOT / "raw-results.jsonl.gz"
    if not archive.is_file():
        raise SystemExit("The collector has not completed raw-results.jsonl.gz")
    with gzip.open(archive, "rt") as stream:
        records = [json.loads(line) for line in stream]
    groups = {}
    failures = []
    for record in records:
        config = record["config"]
        key = tuple(config[field] for field in ("fill", "chunk_kib", "block_kib", "codec", "shuffle"))
        groups.setdefault(key, []).append(record)
        if record.get("validation_error"):
            failures.append({**config, "pass": record["pass"], "warmup": record["warmup"],
                             "code": record["code"], "timed_out": record["timed_out"],
                             "error": record["validation_error"],
                             "stderr": record.get("stderr", "")[-2000:]})
    rows = []
    for key, group in sorted(groups.items()):
        measured = [record["result"] for record in group
                    if not record["warmup"] and not record.get("validation_error")]
        speeds = [result["throughput_in_gibs"] for result in measured]
        folds = {result["compression_fold"] for result in measured}
        status = "complete" if (len(group) == 4 and len(measured) == 3
                                and not any(record.get("validation_error") for record in group)
                                and len(folds) == 1) else "incomplete"
        rows.append(dict(zip(("fill", "chunk_kib", "block_kib", "codec", "shuffle"), key)) | {
            "status": status, "measured_repeats": len(measured),
            "failed_executions": sum(bool(record.get("validation_error")) for record in group),
            "throughput_median_gibs": statistics.median(speeds) if speeds else "",
            "throughput_min_gibs": min(speeds) if speeds else "",
            "throughput_max_gibs": max(speeds) if speeds else "",
            "compression_fold": next(iter(folds)) if len(folds) == 1 else "",
            "measured_device_gib": statistics.median(result["memory_device_used_bytes"] for result in measured) / GIB if speeds else "",
            "estimated_device_gib": statistics.median(result["memory_estimate_total_bytes"] for result in measured) / GIB if speeds else "",
            "estimated_pinned_gib": statistics.median(result["memory_estimate_pinned_bytes"] for result in measured) / GIB if speeds else "",
        })

    fields = list(rows[0])
    write_csv("summary.csv", rows, fields)
    write_csv("failures.csv", failures,
              ["fill", "chunk_kib", "block_kib", "codec", "shuffle", "pass", "warmup",
               "code", "timed_out", "error", "stderr"])

    frontier = []
    eligible = [row for row in rows if row["status"] == "complete" and row["codec"].startswith("blosc-")]
    for row in eligible:
        peers = [other for other in eligible if (other["fill"], other["chunk_kib"]) ==
                 (row["fill"], row["chunk_kib"])]
        dominated = any(
            other["throughput_median_gibs"] >= row["throughput_median_gibs"]
            and other["compression_fold"] >= row["compression_fold"]
            and (other["throughput_median_gibs"] > row["throughput_median_gibs"]
                 or other["compression_fold"] > row["compression_fold"])
            for other in peers)
        if not dominated:
            frontier.append(row)
    write_csv("pareto-frontier.csv", frontier, fields)
    print(f"{len(records)} executions, {len(rows)} configurations, "
          f"{len(failures)} failures, {len(frontier)} frontier points")


if __name__ == "__main__":
    main()

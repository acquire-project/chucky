"""Run all archived Blosc commands, retaining failures and continuing."""
import csv
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

import sweep as s


def succeeded(record):
    return not record.get("error") and record["result"].get("status") == "pass"


def summarize(records):
    outcomes = []
    for config in s.matrix():
        attempts = [r for r in records if r["config"] == config]
        measured = [r for r in attempts if not r["warmup"]]
        passed = [r for r in measured if succeeded(r)]
        rates = [r["result"]["throughput_in_gibs"] for r in passed]
        folds = [r["result"]["compression_fold"] for r in passed]
        status = "complete" if len(attempts) == 6 and all(succeeded(r) for r in attempts) else "partial" if passed else "failed"
        outcomes.append({**config, "status": status,
                         "measured_passed": len(passed), "measured_failed": len(measured) - len(passed),
                         "warmup_failed": sum(r["warmup"] and not succeeded(r) for r in attempts),
                         "throughput_median_gibs": statistics.median(rates) if rates else None,
                         "throughput_min_gibs": min(rates) if rates else None,
                         "throughput_max_gibs": max(rates) if rates else None,
                         "compression_fold": statistics.median(folds) if folds else None})
    with (s.ROOT / "outcomes.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(outcomes[0]))
        writer.writeheader(); writer.writerows(outcomes)
    failures = [{"config": r["config"], "repeat": r["repeat"], "warmup": r["warmup"],
                 "utc": r["utc"], "error": r["error"], "error_kind": r["error_kind"],
                 "returncode": r["code"], "command": r["command"], "log": r["log"]}
                for r in records if not succeeded(r)]
    (s.ROOT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if not failures:
        s.summarize()
    else:
        with (s.ROOT / "summary.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(outcomes[0]))
            writer.writeheader(); writer.writerows(outcomes)


def run():
    if s.command("hostname") != "auk":
        raise RuntimeError("This collection is authorized for Auk")
    s.check_build()
    raw = s.ROOT / "raw"
    raw.mkdir(exist_ok=False)
    provenance = dict(start_utc=s.now(), node=s.command("hostname"),
                      cpu_affinity=sorted(os.sched_getaffinity(0)),
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      gpu_before=s.gpu_info(), build=json.loads((s.ROOT / "build.json").read_text()),
                      manifest=json.loads((s.ROOT / "manifest.json").read_text()), complete=False,
                      collection_driver_sha256=s.sha(Path(__file__)),
                      failure_policy="Record and continue; preserve every scheduled attempt")
    path = s.ROOT / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    records = []
    rng = random.Random(169)
    with (s.ROOT / "results.jsonl").open("x") as output:
        for repeat in range(s.REPEATS + 1):
            order = s.matrix()
            rng.shuffle(order)
            for row in order:
                label = f"{repeat}__{row['id']}"
                command = s.bench_command(row)
                record = dict(config=row, repeat=repeat, warmup=repeat == 0, utc=s.now(),
                              command=command, log=f"raw/{label}.log")
                started = time.monotonic()
                stdout, stderr, code = "", "", None
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                    stdout, stderr, code = result.stdout, result.stderr, result.returncode
                    record["result"] = json.loads(stdout) if stdout.strip() else {"status": "error"}
                    if code:
                        raise RuntimeError(f"Benchmark exit {code}: {stderr[-4000:]}")
                    record["actual"] = s.verify(row, record["result"], stderr)
                except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
                    if isinstance(error, subprocess.TimeoutExpired):
                        stdout = error.stdout or ""; stderr = error.stderr or ""
                        if isinstance(stdout, bytes): stdout = stdout.decode(errors="replace")
                        if isinstance(stderr, bytes): stderr = stderr.decode(errors="replace")
                    record.setdefault("result", {"status": "error"})
                    record["error"] = str(error)
                    record["error_kind"] = "out-of-memory" if re.search(r"out.of.memory|CUDA_ERROR_OUT_OF_MEMORY", str(error) + stderr, re.I) else "timeout" if isinstance(error, subprocess.TimeoutExpired) else "execution-or-validation"
                record.update(code=code, elapsed_s=time.monotonic() - started)
                (raw / f"{label}.json").write_text(stdout)
                (raw / f"{label}.log").write_text(stderr)
                output.write(json.dumps(record) + "\n"); output.flush()
                records.append(record)
                message = f"{record['result']['throughput_in_gibs']:.3f} GiB/s" if succeeded(record) else f"FAILED ({record['error_kind']})"
                print(f"{len(records)}/1200 {label}: {message}", flush=True)
                provenance.update(completed=len(records),
                                  passed=sum(succeeded(r) for r in records),
                                  failed=sum(not succeeded(r) for r in records))
                path.write_text(json.dumps(provenance, indent=2) + "\n")
    provenance.update(complete=True, finish_utc=s.now(), gpu_after=s.gpu_info())
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    summarize(records)


if __name__ == "__main__":
    run()

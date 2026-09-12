"""One execution policy and one repetition summary for every input source."""

from __future__ import annotations

import copy
import json
import math
import statistics
import subprocess
import time

MEASUREMENT_POLICY = "coverage-qualified-through-final-close-v2"
DEFAULT_WARMUP_S = 0.25
DEFAULT_DURATION_S = 1.0


def measurement_policy(warmup=DEFAULT_WARMUP_S, duration=DEFAULT_DURATION_S):
    return {"policy": MEASUREMENT_POLICY, "warmup_s": warmup, "duration_s": duration}


def validate_measurement(result: dict) -> None:
    window = result.get("measurement")
    if (result.get("status") != "pass" or not isinstance(window, dict)
            or window.get("policy") != MEASUREMENT_POLICY
            or window.get("coverage_status") != "sufficient"):
        raise ValueError("benchmark did not satisfy required measurement policy")
    for key, minimum in (("warmup_complete_batches", 2), ("complete_batches", 4),
                         ("generation_transitions", 2)):
        value = window.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"insufficient measurement coverage: {key}")
    for key in ("warmup_s", "append_s", "elapsed_s", "drain_s"):
        value = window.get(key)
        if (not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0):
            raise ValueError(f"invalid measurement time: {key}")
    if (window["warmup_s"] < 0.25 or window["append_s"] < 0.25
            or window["drain_s"] > 0.1 * window["elapsed_s"] + 1e-6
            or not math.isclose(window["elapsed_s"],
                                window["append_s"] + window["drain_s"], rel_tol=1e-5)):
        raise ValueError("insufficient measurement time or excessive final drain")
    for direction, key in (("in", "input_bytes"), ("out", "output_bytes")):
        count = window.get(key)
        if type(count) is not int or count <= 0:
            raise ValueError(f"invalid measured bytes: {key}")
        expected = count / 2**30 / window["elapsed_s"]
        rate = result.get(f"throughput_{direction}_gibs")
        if (not isinstance(rate, (int, float)) or not math.isfinite(rate)
                or not math.isclose(rate, expected, rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError(f"measurement denominator disagrees: {direction}")
    if not math.isclose(result.get("wall_s", -1), window["elapsed_s"], rel_tol=1e-5):
        raise ValueError("measurement wall time disagrees")


def execute(command: list[str]) -> dict:
    started = time.monotonic()
    process = subprocess.run(command, capture_output=True, text=True, timeout=600)
    elapsed = time.monotonic() - started
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        result = {}
    if not isinstance(result, dict):
        result = {}
    result["process_wall_s"] = elapsed
    result["elapsed_s"] = elapsed  # legacy runner field, distinct from the window
    if process.returncode:
        result.update(status="error", returncode=process.returncode)
        result.setdefault("error", process.stderr.strip()[-2000:] or "benchmark failed")
    else:
        try:
            validate_measurement(result)
            if result["wall_s"] > elapsed * 1.05 + 0.1:
                raise ValueError("benchmark clock disagrees with process clock")
        except (ValueError, TypeError, KeyError) as error:
            result.update(status="error", error=str(error))
    return result


def aggregate_repetitions(executions: list[dict]) -> dict:
    if not executions:
        raise ValueError("No measured executions")
    first = executions[0]
    for result in executions:
        validate_measurement(result)
        for key in ("geometry", "source_bytes", "append_bytes", "boundary_timing"):
            if result["measurement"].get(key) != first["measurement"].get(key):
                raise ValueError(f"Measurement configuration changed between repeats: {key}")
    rates = [result["throughput_in_gibs"] for result in executions]
    median = statistics.median(rates)
    detail = min(range(len(rates)), key=lambda i: abs(rates[i] - median))
    result = copy.deepcopy(executions[detail])
    for key in ("throughput_in_gibs", "throughput_out_gibs", "throughput_logical_gibs"):
        if all(key in execution for execution in executions):
            result[key] = statistics.median(execution[key] for execution in executions)
    output = sum(execution["measurement"]["output_bytes"] for execution in executions)
    if all("padded_input_bytes" in execution for execution in executions):
        result["compression_fold"] = sum(
            execution["padded_input_bytes"] for execution in executions
        ) / output
    if all("logical_input_bytes" in execution for execution in executions):
        result["logical_compression_fold"] = sum(
            execution["logical_input_bytes"] for execution in executions
        ) / output
    result["repetitions"] = {
        "count": len(executions),
        "warmups": 0,  # process pre-runs; each execution has its own measured warmup
        "throughput_min_gibs": min(rates),
        "throughput_max_gibs": max(rates),
        "throughput_spread_percent": 100 * (max(rates) - min(rates)) / median,
        "throughput_gibs": rates,
        "detail_repeat": detail + 1,
        "detail_iteration": detail,
        "executions": copy.deepcopy(executions),
    }
    return result

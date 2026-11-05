"""Normalize compression names without treating pool slots as emitted chunks."""

from __future__ import annotations

import math


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def logical_bytes(run, *, generated=False):
    logical = run.get("logical_input_bytes")
    if logical is None and generated:
        logical = run.get("input_bytes")
    output = run.get("measurement", {}).get("output_bytes", run.get("output_bytes"))
    if positive(logical) and positive(output):
        return logical, output
    return None


def migrate_compression(run, *, generated=False):
    """Migrate a report copy in place; raw archives need never be rewritten.

    Existing logical summaries may pool repetitions. Never replace one using
    the representative execution's byte counts. Old padded/total-chunk values
    cannot establish emitted chunks, particularly with append downsampling.
    """
    generated = generated or run.get("fill") in {"xor", "rand", "zeros"}
    if "compression_fold" in run:
        run.setdefault("legacy_compression_fold", run.pop("compression_fold"))
    if "decoded_chunk_bytes" not in run:
        for key in ("padded_input_bytes", "total_chunks"):
            if key in run:
                run.setdefault("legacy_" + key, run.pop(key))
    repetitions = run.get("repetitions", {})
    executions = repetitions.get("executions", [])
    for execution in executions:
        migrate_compression(execution, generated=generated)
    pairs = [logical_bytes(execution, generated=generated) for execution in executions]
    if pairs and all(pair is not None for pair in pairs):
        run["logical_compression_fold"] = sum(pair[0] for pair in pairs) / sum(pair[1] for pair in pairs)
        run["logical_compression_source"] = "pooled_bytes"
    elif not positive(run.get("logical_compression_fold")):
        # A detail repeat cannot stand in for a missing pooled summary.
        if executions or repetitions.get("count", 1) > 1:
            return run
        pair = logical_bytes(run, generated=generated)
        if pair:
            run["logical_compression_fold"] = pair[0] / pair[1]
            run["logical_compression_source"] = "recorded_bytes"
        elif generated and positive(run.get("input_gib")) and positive(run.get("compressed_gib")):
            run["logical_compression_fold"] = run["input_gib"] / run["compressed_gib"]
            run["logical_compression_source"] = "recorded_gib"
    return run

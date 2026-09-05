"""Shared data models for sweep runner and report generator."""

from __future__ import annotations

import re

from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Schema enums (single source of truth)
# ---------------------------------------------------------------------------

VALID_CODECS = {"none", "lz4", "zstd", "blosc-lz4", "blosc-zstd"}
VALID_FILLS = {"xor", "zeros", "rand"}
VALID_BACKENDS = {"gpu", "cpu"}
VALID_SINKS = {"discard", "fs", "s3"}
VALID_DTYPES = {"u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16", "f32", "f64"}
VALID_STATUSES = {"pass", "error", "timeout", "missing", "unknown"}


def run_id(run: dict) -> str:
    """Configuration identity; unrecorded Blosc block sizes are unknown."""
    identity = run.get("id")
    if identity is None:
        parts = [str(run[key]) for key in
                 ("scenario", "codec", "fill", "backend", "dtype", "chunk_bytes_label")]
        sink = run.get("sink", "discard")
        if sink != "discard":
            parts.append(sink)
        throughput = run.get("s3_throughput_gbps") or 0
        if throughput > 0:
            parts.append(f"{int(throughput)}gbps")
        identity = "__".join(parts)
    if run["codec"].startswith("blosc-"):
        identity = re.sub(r"__blosc-block-(?:[0-9]+|unknown)$", "", identity)
        block_bytes = run.get("blosc_block_bytes")
        identity += f"__blosc-block-{block_bytes if block_bytes is not None else 'unknown'}"
    return identity


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class D2HTransfer(BaseModel, extra="allow"):
    payload_bytes_transferred: int
    metadata_bytes_transferred: int
    payload_copy_count: int


class ShardPadding(BaseModel, extra="allow"):
    logical_payload_bytes: int
    internal_padding_bytes: int
    physical_shard_update_count: int
    padded_update_count: int


class RunResult(BaseModel, extra="allow"):
    id: str
    scenario: str
    codec: str
    fill: str
    backend: str
    dtype: str
    chunk_bytes: int
    chunk_bytes_label: str
    sink: str = "discard"
    # Absent in files written before the runner recorded them.
    frames: int | None = None
    worker_threads: int | None = None
    blosc_block_bytes: int | None = None
    memory_estimate_total_bytes: int | None = None
    memory_estimate_pinned_bytes: int | None = None
    memory_host_baseline_bytes: int | None = None
    memory_host_peak_bytes: int | None = None
    memory_host_reading_failed: bool | None = None
    memory_device_used_bytes: int | None = None
    memory_device_overhead_bytes: int | None = None
    memory_measured_bytes: int | None = None
    # Absent in archived results; missing means unknown, not zero.
    d2h_transfer: D2HTransfer | None = None
    shard_padding: ShardPadding | None = None
    s3_endpoint: str | None = None
    s3_region: str | None = None
    s3_bucket: str | None = None
    s3_throughput_gbps: float | None = None
    status: str

    @model_validator(mode="after")
    def _validate_enums(self) -> RunResult:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unknown status: {self.status}")
        return self


class ResultsFile(BaseModel, extra="allow"):
    version: int
    machine: dict
    runs: list[RunResult]


def validate_results(data: dict) -> ResultsFile:
    return ResultsFile.model_validate(data)


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


# One version per sweep, since every run in a file comes from one binary. Bump
# when a metric is renamed, removed, or changes meaning; adding one does not
# need a bump. Version 1 predates the rule and is not a single shape, so
# migrating from it cannot assume which keys are present. A bump also needs a
# line in README.md, the only record of what a stored version number means.
CURRENT_VERSION = 10

# Renames of an unchanged quantity, safe to carry forward.
_RENAMED_STAGES_1_TO_2 = {"lod_dim0_fold": "lod_append_fold"}

# Keys whose value cannot be recovered, by the version that retired them.
RETIRED_AT = {
    2: ("kick_sync_ms", "kick_sync_count"),
    3: ("tail_gate_ms", "tail_gate_count"),
    5: ("peak_pending_mib", "backpressure_ms", "backpressure_count"),
}


def _migrate_1_to_2(data: dict) -> None:
    for run in data.get("runs", []):
        stages = run.get("stages")
        if not isinstance(stages, dict):
            continue
        for old_name, new_name in _RENAMED_STAGES_1_TO_2.items():
            if old_name in stages and new_name not in stages:
                stages[new_name] = stages.pop(old_name)


def _migrate_2_to_3(data: dict) -> None:
    # `tail_gate` now measures the compression-to-aggregation delay while the
    # host coordinator waits for prior tail readiness. The value itself cannot
    # be converted, so the migration only advances the file version; RETIRED_AT
    # prevents older values from being compared with the new measurement.
    pass


def _migrate_3_to_4(data: dict) -> None:
    # The discard sink now reports a shard alignment, so a default sweep
    # measures the page-aligned pipeline instead of the contiguous one. Every
    # timing changes meaning; none of them can be converted, so this only
    # advances the file version.
    pass


def _migrate_4_to_5(data: dict) -> None:
    # The writer stopped starting a transfer per append (#173), so the
    # pending-bytes high-water mark and the backpressure wait are sampled once
    # per staging buffer instead of once per epoch. The counts are far apart for
    # a reason that has nothing to do with the hardware, and neither can be
    # converted, so this only advances the file version.
    pass


def _migrate_5_to_6(data: dict) -> None:
    # The CPU pipeline pool now sizes itself from the cores the process is
    # allowed rather than the cores the machine has, so a sweep run under a
    # batch scheduler no longer oversubscribes them. Every CPU-backend timing
    # moves, and by how much depends on the allocation, so this only advances
    # the file version. GPU runs are unaffected: that pool was already capped.
    pass


def _migrate_6_to_7(data: dict) -> None:
    # The filesystem sink runs several writes at once instead of one, and
    # pre-sizes a shard file when more than one of its writes may run
    # together. Every timing for a run with filesystem output moves, and the
    # write-path counters change meaning with it, so this only advances the
    # file version. Discard and S3 runs are unaffected.
    pass


def _migrate_7_to_8(data: dict) -> None:
    # GPU aggregation became compact and page-aligned tail assembly moved to
    # ordered host copying. D2H stage bytes now mean actual payload
    # bytes transferred. The new d2h_transfer block is deliberately not
    # backfilled: archived transfer totals are unknown, not zero.
    pass


def _migrate_8_to_9(data: dict) -> None:
    renamed = {
        "indexed_aggregate_ready": "indexed_aggregate_wait",
        "chunk_metadata_ready": "chunk_metadata_wait",
    }
    for run in data.get("runs", []):
        diagnostics = run.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        for old_name, new_name in renamed.items():
            if old_name in diagnostics:
                diagnostics.setdefault(new_name, diagnostics.pop(old_name))


def _migrate_9_to_10(data: dict) -> None:
    removed = {
        "host_output_budget_bytes",
        "host_output_buffers_in_use",
        "host_output_bytes_in_use",
        "host_output_buffers_in_use_peak",
        "host_output_bytes_in_use_peak",
        "io_backend",
        "io_workers",
        "io_writes_in_flight",
        "io_writes_in_flight_per_file",
        "io_writes_in_flight_mean",
        "io_writes_in_flight_peak",
        "io_files_waiting_mean",
        "io_files_waiting_peak",
        "io_files_opened",
        "io_files_open_peak",
        "io_writes",
        "io_bytes_copied",
        "io_bytes_borrowed",
        "io_queued_bytes_peak",
        "io_queued_jobs_peak",
        "io_wait_ms_mean",
        "io_wait_ms_max",
        "io_run_ms_mean",
        "io_run_ms_max",
        "io_write_sizes",
    }
    for run in data.get("runs", []):
        for key in removed:
            run.pop(key, None)
        diagnostics = run.get("diagnostics")
        if isinstance(diagnostics, dict):
            for key in ("output_slot_io", "prior_tail_state",
                        "host_output_wait", "host_output_lifetime"):
                diagnostics.pop(key, None)
        delivery = run.get("delivery_timing")
        if isinstance(delivery, dict):
            delivery.pop("writes_posted_to_completion", None)
        stalls = run.get("stalls")
        if isinstance(stalls, dict):
            for key in ("io_fence_ms", "io_fence_count",
                        "tail_gate_ms", "tail_gate_count"):
                stalls.pop(key, None)
            owners = stalls.get("owners")
            if isinstance(owners, dict):
                owners.pop("io_fence", None)
                owners.pop("tail_gate", None)


_MIGRATIONS = {
    1: _migrate_1_to_2,
    2: _migrate_2_to_3,
    3: _migrate_3_to_4,
    4: _migrate_4_to_5,
    5: _migrate_5_to_6,
    6: _migrate_6_to_7,
    7: _migrate_7_to_8,
    8: _migrate_8_to_9,
    9: _migrate_9_to_10,
}


# Canonical diagnostic IDs are deliberately independent of both terminal
# labels and the legacy `stalls` keys. Keep this table in step with
# bench/bench_report.c; it lets report.py give archived runs the same shape as
# runs emitted by a current benchmark binary.
_LEGACY_DIAGNOSTICS = {
    "batch_drain": {
        "label": "Batch delivery (wait/work)", "kind": "host_block",
        "ms": "flush_stall_ms", "count": "flush_stall_count",
        "owner": "flush_stall",
    },
    "d2h_dispatch": {
        "label": "D2H dispatch work", "kind": "host_overhead",
        "ms": "drain_dispatch_ms", "count": "drain_dispatch_count",
        "owner": "drain_dispatch",
    },
    "footer_buffer_io": {
        "label": "Footer-buffer write", "kind": "host_wait",
        "ms": "footer_buffer_ms", "count": "footer_buffer_count",
        "owner": "footer_buffer",
    },
    "append_extent_io": {
        "label": "Closed-shard writes", "kind": "host_wait",
        "ms": "append_extent_ms", "count": "append_extent_count",
        "owner": "append_extent",
    },
    "final_io": {
        "label": "Final queued writes", "kind": "host_wait",
        "ms": "flush_writes_ms", "count": "flush_writes_count",
        "owner": "flush_writes",
    },
    "sink_backpressure": {
        "label": "Sink queue below limit", "kind": "host_wait",
        "ms": "backpressure_ms", "count": "backpressure_count",
        "owner": "backpressure",
    },
}

_LEGACY_EDGE_DIAGNOSTICS = {
    "staging_reuse": ("StagingFree", "Staging-buffer reuse"),
    "chunk_metadata_d2h": ("ChunkIndex", "Chunk metadata ready (inclusive)"),
    "payload_d2h": ("D2HDone", "Payload D2H"),
}


def _canonical_diagnostic(total_ms: object, samples: object, *, label: str,
                          kind: str, owner: object, wall_s: object) -> dict | None:
    if not isinstance(total_ms, (int, float)):
        return None
    n = samples if isinstance(samples, int) and samples >= 0 else 0
    if total_ms <= 0 and n == 0:
        return None
    out = {
        "label": label,
        "kind": kind,
        "owner": owner if isinstance(owner, str) else "unknown",
        "total_ms": total_ms,
        "samples": n,
    }
    if n > 0:
        out["avg_ms"] = total_ms / n
    if isinstance(wall_s, (int, float)) and wall_s > 0:
        out["wall_pct"] = total_ms / (wall_s * 10.0)
    return out


def _backfill_diagnostics(run: dict, retired: tuple[str, ...]) -> None:
    if isinstance(run.get("diagnostics"), dict):
        return
    stalls = run.get("stalls")
    if not isinstance(stalls, dict):
        return

    owners = stalls.get("owners")
    owners = owners if isinstance(owners, dict) else {}
    wall_s = run.get("wall_s")
    diagnostics = {}
    for diagnostic_id, spec in _LEGACY_DIAGNOSTICS.items():
        if spec["ms"] in retired:
            continue
        value = _canonical_diagnostic(
            stalls.get(spec["ms"]), stalls.get(spec["count"]),
            label=spec["label"], kind=spec["kind"],
            owner=owners.get(spec["owner"]), wall_s=wall_s,
        )
        if value:
            diagnostics[diagnostic_id] = value

    edge_stalls = stalls.get("edge_stalls")
    if isinstance(edge_stalls, dict):
        for diagnostic_id, (legacy_name, label) in _LEGACY_EDGE_DIAGNOSTICS.items():
            edge = edge_stalls.get(legacy_name)
            if not isinstance(edge, dict):
                continue
            value = _canonical_diagnostic(
                edge.get("total_ms"), edge.get("count"), label=label,
                kind="host_wait", owner=edge.get("owner"), wall_s=wall_s,
            )
            if value:
                diagnostics[diagnostic_id] = value

    if diagnostics:
        run["diagnostics"] = diagnostics


def migrate_run(run: dict, retired: tuple[str, ...] = ()) -> dict:
    """Fill defaults for fields added after the initial schema."""
    run.setdefault("sink", "discard")
    _backfill_diagnostics(run, retired)
    return run


def retired_metrics(data: dict) -> tuple[str, ...]:
    """Keys this file carries that must not be compared against later sweeps."""
    origin = data.get("migrated_from", data.get("version", 1))
    out: list[str] = []
    for version, keys in sorted(RETIRED_AT.items()):
        if origin < version:
            out.extend(keys)
    return tuple(out)


def migrate_results(data: dict) -> dict:
    """Bring one results file to the current schema, in place.

    Keys whose meaning changed are left as they are; converting them would
    invent data. Use retired_metrics to find them.
    """
    version = data.get("version", 1)
    data.setdefault("migrated_from", version)
    while version < CURRENT_VERSION:
        migrate = _MIGRATIONS.get(version)
        if migrate:
            migrate(data)
        version += 1
        data["version"] = version
    retired = retired_metrics(data)
    for run in data.get("runs", []):
        migrate_run(run, retired)
    return data

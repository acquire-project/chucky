"""Shared data models for sweep runner and report generator."""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


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
    io_backend: str | None = None
    io_workers: int | None = None
    io_writes_in_flight: int | None = None
    io_writes_in_flight_per_file: int | None = None
    io_files_waiting_mean: float | None = None
    io_files_waiting_peak: int | None = None
    io_files_opened: int | None = None
    io_files_open_peak: int | None = None
    io_writes: int | None = None
    io_bytes_copied: int | None = None
    io_bytes_borrowed: int | None = None
    io_queued_bytes_peak: int | None = None
    io_queued_jobs_peak: int | None = None
    io_wait_ms_mean: float | None = None
    io_wait_ms_max: float | None = None
    io_run_ms_mean: float | None = None
    io_run_ms_max: float | None = None
    memory_estimate_total_bytes: int | None = None
    memory_estimate_pinned_bytes: int | None = None
    memory_host_baseline_bytes: int | None = None
    memory_host_peak_bytes: int | None = None
    memory_host_reading_failed: bool | None = None
    memory_device_used_bytes: int | None = None
    memory_measured_bytes: int | None = None
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
CURRENT_VERSION = 7

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


_MIGRATIONS = {
    1: _migrate_1_to_2,
    2: _migrate_2_to_3,
    3: _migrate_3_to_4,
    4: _migrate_4_to_5,
    5: _migrate_5_to_6,
    6: _migrate_6_to_7,
}


def migrate_run(run: dict) -> dict:
    """Fill defaults for fields added after the initial schema."""
    run.setdefault("sink", "discard")
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
    for run in data.get("runs", []):
        migrate_run(run)
    return data

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
# migrating from it cannot assume which keys are present.
CURRENT_VERSION = 3

# Renames of an unchanged quantity, safe to carry forward.
_RENAMED_STAGES_1_TO_2 = {"lod_dim0_fold": "lod_append_fold"}

# Keys whose value cannot be recovered, by the version that retired them.
# Version 3: the writer stopped starting a transfer per append (#173), so the
# pending-bytes high-water mark and the backpressure wait are sampled once per
# staging buffer instead of once per epoch. The numbers are far apart for reasons
# that have nothing to do with the hardware.
RETIRED_AT = {
    2: ("kick_sync_ms", "kick_sync_count"),
    3: ("peak_pending_mib", "backpressure_ms", "backpressure_count"),
}


def _migrate_1_to_2(data: dict) -> None:
    for run in data.get("runs", []):
        stages = run.get("stages")
        if not isinstance(stages, dict):
            continue
        for old_name, new_name in _RENAMED_STAGES_1_TO_2.items():
            if old_name in stages and new_name not in stages:
                stages[new_name] = stages.pop(old_name)


_MIGRATIONS = {1: _migrate_1_to_2}


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
    while version < CURRENT_VERSION and version in _MIGRATIONS:
        _MIGRATIONS[version](data)
        version += 1
        data["version"] = version
    for run in data.get("runs", []):
        migrate_run(run)
    return data

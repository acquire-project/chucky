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


def migrate_run(run: dict) -> dict:
    """Fill defaults for fields added after the initial schema."""
    run.setdefault("sink", "discard")
    return run


def migrate_results(data: dict) -> dict:
    """In-place migration of a results dict to the current schema."""
    for run in data.get("runs", []):
        migrate_run(run)
    return data

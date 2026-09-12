"""Parse and enforce the scenario/input compatibility registry."""

from __future__ import annotations

import tomllib
from pathlib import Path


CURRENT_VERSION = 1
DEFAULT_WORKLOADS = Path(__file__).resolve().parents[2] / "bench/workloads.toml"

_TOP_LEVEL_KEYS = {"version", "input", "scenario"}
_ENTRY_KEYS = {
    "input": {"id", "scenarios", "default_scenario"},
    "scenario": {"id", "inputs", "default_input"},
}


def _identifier(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{where} must be a non-empty, trimmed string")
    if any(character in value for character in "*?["):
        raise ValueError(f"{where} must not contain wildcard characters")
    return value


def _references(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty array")
    references = [_identifier(item, f"{where} item") for item in value]
    duplicate = next(
        (item for index, item in enumerate(references) if item in references[:index]),
        None,
    )
    if duplicate is not None:
        raise ValueError(f"{where} repeats {duplicate}")
    return references


def _entries(data: dict, kind: str, path: Path) -> list[dict]:
    raw = data.get(kind)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected one or more [[{kind}]] entries")

    plural = "scenarios" if kind == "input" else "inputs"
    default_key = "default_scenario" if kind == "input" else "default_input"
    entries: list[dict] = []
    seen: set[str] = set()
    for position, item in enumerate(raw, start=1):
        where = f"{path}: {kind} #{position}"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be a table")
        missing = sorted(_ENTRY_KEYS[kind] - set(item))
        unknown = sorted(set(item) - _ENTRY_KEYS[kind])
        if missing:
            raise ValueError(f"{where} is missing {', '.join(missing)}")
        if unknown:
            raise ValueError(f"{where} has unknown key(s): {', '.join(unknown)}")

        identifier = _identifier(item["id"], f"{where} id")
        if identifier in seen:
            raise ValueError(f"{path}: duplicate {kind} id {identifier}")
        seen.add(identifier)
        references = _references(item[plural], f"{where} {plural}")
        default = _identifier(item[default_key], f"{where} {default_key}")
        if default not in references:
            raise ValueError(
                f"{path}: {kind} {identifier} default {default} is not in {plural}"
            )
        entries.append(
            {"id": identifier, plural: references, default_key: default}
        )
    return entries


def load_workloads(path: Path = DEFAULT_WORKLOADS) -> dict:
    """Return a canonical JSON-ready registry, rejecting ambiguous input."""
    if not path.is_file():
        raise ValueError(f"No workload registry at {path}")
    try:
        with open(path, "rb") as stream:
            data = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path}: {error}") from error

    unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
    missing = sorted(_TOP_LEVEL_KEYS - set(data))
    if missing:
        raise ValueError(f"{path}: missing top-level key(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s): {', '.join(unknown)}")
    if type(data["version"]) is not int or data["version"] != CURRENT_VERSION:
        raise ValueError(
            f"{path}: version must be the integer {CURRENT_VERSION}"
        )

    inputs = _entries(data, "input", path)
    scenarios = _entries(data, "scenario", path)
    input_by_id = {entry["id"]: entry for entry in inputs}
    scenario_by_id = {entry["id"]: entry for entry in scenarios}

    for entry in inputs:
        for scenario_id in entry["scenarios"]:
            scenario = scenario_by_id.get(scenario_id)
            if scenario is None:
                raise ValueError(
                    f"{path}: input {entry['id']} references unknown scenario {scenario_id}"
                )
            if entry["id"] not in scenario["inputs"]:
                raise ValueError(
                    f"{path}: input {entry['id']} lists scenario {scenario_id}, "
                    "but the scenario does not list the input"
                )
    for entry in scenarios:
        for input_id in entry["inputs"]:
            input_entry = input_by_id.get(input_id)
            if input_entry is None:
                raise ValueError(
                    f"{path}: scenario {entry['id']} references unknown input {input_id}"
                )
            if entry["id"] not in input_entry["scenarios"]:
                raise ValueError(
                    f"{path}: scenario {entry['id']} lists input {input_id}, "
                    "but the input does not list the scenario"
                )

    return {"version": CURRENT_VERSION, "inputs": inputs, "scenarios": scenarios}


def validate_run_pairs(files: list[tuple[Path, dict]], workloads: dict) -> None:
    """Reject every result pair not explicitly present in the registry."""
    registered = {
        (scenario["id"], input_id)
        for scenario in workloads["scenarios"]
        for input_id in scenario["inputs"]
    }
    unknown: dict[tuple[str, str], set[str]] = {}
    for path, data in files:
        for run in data.get("runs", []):
            scenario = run.get("scenario")
            input_id = run.get("input_id") or run.get("fill")
            pair = (str(scenario or "<missing>"), str(input_id or "<missing>"))
            if pair not in registered:
                unknown.setdefault(pair, set()).add(path.name)
    if unknown:
        detail = "; ".join(
            f"({scenario}, {input_id}) in {', '.join(sorted(names))}"
            for (scenario, input_id), names in sorted(unknown.items())
        )
        raise ValueError(f"Unregistered scenario/input pair(s): {detail}")

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_BYTES = 256 * 1024**2
SHA256 = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    value = json.loads(path.read_text(), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def positive_int(value, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def relative_file(root: Path, value: str) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("File paths must use forward slashes")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {".", "..", ".git"} or ":" in part for part in path.parts)
    ):
        raise ValueError(f"Invalid corpus path: {value}")
    return root.joinpath(*path.parts)


def git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass
class Corpus:
    root: Path
    manifest: dict
    sha256: str
    revision: str | None
    verify_s: float
    pack_files: dict[str, Path]

    def record(self) -> dict:
        return {
            "manifest_sha256": self.sha256,
            "revision": self.revision,
            "release": self.manifest.get("release"),
            "kind": self.manifest["kind"],
            "verify_s": self.verify_s,
            "resolved_pack_paths": {
                name: str(path) for name, path in self.pack_files.items()
            },
            "decoded_bytes": sum(p["bytes"] for p in self.manifest["packs"]),
        }


def check_kind(kind: str, allow_test_data=False, allow_provisional=False) -> None:
    if kind == "raw" or (kind == "synthetic-test" and allow_test_data):
        return
    if kind == "provisional" and allow_provisional:
        return
    raise ValueError(
        "Only verified raw data are allowed by default; use --allow-test-data for "
        "synthetic fixtures or --allow-provisional for unverified microscopy"
    )


def verify_source(root: Path, name: str, source: dict, kind: str) -> dict:
    expected_status = {
        "raw": "verified",
        "synthetic-test": "synthetic-test",
        "provisional": "unverified",
    }[kind]
    if source.get("raw_status") != expected_status:
        raise ValueError(f"Source is not verified raw: {name}")
    path = relative_file(root, source["evidence"])
    if not SHA256.fullmatch(source.get("evidence_sha256", "")):
        raise ValueError(f"Invalid evidence checksum: {name}")
    if digest_file(path) != source["evidence_sha256"]:
        raise ValueError(f"Provenance checksum mismatch: {name}")
    if kind == "synthetic-test":
        return {}
    evidence = read_json(path)
    if kind == "provisional":
        if (
            evidence.get("raw_status") != "unverified"
            or evidence.get("allowed_for_raw_release") is not False
            or not evidence.get("missing_evidence")
        ):
            raise ValueError(
                f"Provisional source must document its missing raw evidence: {name}"
            )
        return {}
    verification = evidence.get("verification", {})
    if (
        evidence.get("raw_status") != "verified"
        or evidence.get("allowed_for_raw_release") is not True
        or verification.get("method")
        not in {
            "original-acquisition",
            "conversion-code-review",
            "pixel-comparison",
        }
        or not verification.get("references")
    ):
        raise ValueError(f"Evidence does not establish raw camera values: {name}")
    return verification


def verify_corpus(
    root: Path,
    lock: Path | None = None,
    allow_test_data: bool = False,
    allow_provisional: bool = False,
) -> Corpus:
    start = time.perf_counter()
    root = root.expanduser().resolve()
    if (root / "INCOMPLETE").exists():
        raise ValueError("Corpus extraction is incomplete")
    path = root / "manifest.json"
    document = read_json(path)
    selection = document.get("selection")
    if selection is not None:
        survey = relative_file(root, selection["survey_path"])
        if digest_file(survey) != selection["survey_sha256"]:
            raise ValueError("Selection survey checksum mismatch")
    manifest_sha = digest_file(path)
    revision = (
        git_output(root, "rev-parse", "HEAD") if (root / ".git").exists() else None
    )
    if lock is not None:
        pin = read_json(lock)
        if manifest_sha != pin["manifest_sha256"]:
            raise ValueError("Manifest differs from the pinned corpus")
        if revision is not None and revision != pin["revision"]:
            branch = (
                git_output(root, "symbolic-ref", "--quiet", "--short", "HEAD") or ""
            )
            if not branch.startswith("adjusted/"):
                raise ValueError("Check out the pinned corpus revision before running")
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported corpus schema")
    kind = document.get("kind")
    check_kind(kind, allow_test_data, allow_provisional)
    packs = document.get("packs")
    if not isinstance(packs, list) or not packs:
        raise ValueError("Manifest has no image packs")
    sources = document.get("sources", {})
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Manifest has no provenance")
    verifications = {
        name: verify_source(root, name, source, kind)
        for name, source in sources.items()
    }

    identifiers, paths, plane_ids = set(), set(), set()
    fields, modalities = {}, set()
    total_bytes = 0
    pack_files = {}
    for pack in packs:
        name = pack.get("id", "")
        if not IDENTIFIER.fullmatch(name) or name in identifiers:
            raise ValueError(f"Invalid or duplicate pack id: {name}")
        identifiers.add(name)
        if pack.get("modality") not in {"fluorescence", "brightfield"}:
            raise ValueError(f"Unknown modality: {name}")
        modalities.add(pack["modality"])
        if pack.get("split") not in {"core", "heldout"}:
            raise ValueError(f"Unknown split: {name}")
        if pack.get("dtype") != "u16le" or not pack.get("source_group"):
            raise ValueError(f"Expected u16le and a source group: {name}")
        width = positive_int(pack.get("width"), f"{name} width")
        height = positive_int(pack.get("height"), f"{name} height")
        planes = pack.get("planes")
        if not isinstance(planes, list) or not planes:
            raise ValueError(f"No planes: {name}")
        size = positive_int(pack.get("bytes"), f"{name} bytes")
        if size != width * height * 2 * len(planes):
            raise ValueError(f"Pack size does not match its frames: {name}")
        total_bytes += size
        if total_bytes > MAX_BYTES:
            raise ValueError("Corpus exceeds the 256 MiB decoded limit")
        if not SHA256.fullmatch(pack.get("sha256", "")):
            raise ValueError(f"Invalid pack checksum: {name}")
        file = relative_file(root, pack["path"])
        if pack["path"] in paths:
            raise ValueError(f"Duplicate pack path: {pack['path']}")
        paths.add(pack["path"])
        for plane in planes:
            plane_id = plane.get("id")
            if not isinstance(plane_id, str) or not plane_id or plane_id in plane_ids:
                raise ValueError(f"Invalid or duplicate plane id: {plane_id}")
            plane_ids.add(plane_id)
            if plane.get("source_id") not in sources:
                raise ValueError(f"Missing source provenance: {plane_id}")
            verification = verifications[plane["source_id"]]
            if verification.get("method") == "pixel-comparison":
                comparison = plane.get("source_details", {}).get(
                    "original_comparison", {}
                )
                if comparison.get("plane_sha256") != plane.get("sha256"):
                    raise ValueError(f"Missing original-pixel comparison: {plane_id}")
            if not SHA256.fullmatch(plane.get("sha256", "")):
                raise ValueError(f"Invalid plane checksum: {plane_id}")
            field = plane.get("field_id")
            if not isinstance(field, str) or not field:
                raise ValueError(f"Missing field identity: {plane_id}")
            if field in fields and fields[field] != pack["split"]:
                raise ValueError(f"Field occurs in core and heldout sets: {field}")
            fields[field] = pack["split"]
            coordinates = plane.get("coordinates")
            if (
                not isinstance(coordinates, dict)
                or not coordinates
                or any(type(v) is not int or v < 0 for v in coordinates.values())
            ):
                raise ValueError(f"Invalid source coordinates: {plane_id}")
        try:
            resolved = file.resolve(strict=True)
            if resolved.stat().st_size != size:
                raise ValueError(f"Wrong pack length: {file}; run git annex get data/")
            checksum = hashlib.sha256()
            with resolved.open("rb") as stream:
                for plane in planes:
                    data = stream.read(width * height * 2)
                    if len(data) != width * height * 2:
                        raise ValueError(f"Truncated pack: {file}")
                    if hashlib.sha256(data).hexdigest() != plane["sha256"]:
                        raise ValueError(f"Plane checksum mismatch: {plane['id']}")
                    checksum.update(data)
                if stream.read(1):
                    raise ValueError(f"Unexpected bytes at the end of {file}")
            if checksum.hexdigest() != pack["sha256"]:
                raise ValueError(f"Pack checksum mismatch: {file}")
            pack_files[name] = resolved
        except FileNotFoundError as error:
            raise ValueError(
                f"Missing image pack: {file}; run git annex get data/"
            ) from error
    if modalities != {"fluorescence", "brightfield"}:
        raise ValueError("Corpus must contain fluorescence and brightfield")
    return Corpus(
        root, document, manifest_sha, revision, time.perf_counter() - start, pack_files
    )

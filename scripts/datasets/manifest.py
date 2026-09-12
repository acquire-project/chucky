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
CORPUS_FORMAT = "chucky-image-corpus"
CORPUS_FORMAT_VERSION = 2
ASSET_DTYPES = {
    "uint8": ("u8", 1),
    "uint16": ("u16le", 2),
    "float32": ("f32le", 4),
}


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
    source_manifest: str = "manifest.json"
    dataset: dict | None = None

    def record(self) -> dict:
        result = {
            "manifest_sha256": self.sha256,
            "manifest": self.source_manifest,
            "revision": self.revision,
            "release": self.manifest.get("release"),
            "kind": self.manifest["kind"],
            "verify_s": self.verify_s,
            "resolved_pack_paths": {
                name: str(path) for name, path in self.pack_files.items()
            },
            "decoded_bytes": sum(p["bytes"] for p in self.manifest["packs"]),
        }
        for key in ("name", "version", "format", "datasets"):
            if key in self.manifest:
                result[key] = self.manifest[key]
        if self.dataset is not None:
            result["dataset"] = self.dataset
        return result


def nonempty_string(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def resolved_content_file(root: Path, file: Path) -> Path:
    """Return the concrete bytes used for replay, including annex object paths."""
    try:
        relative = file.relative_to(root).as_posix()
    except ValueError:
        return file.resolve(strict=True)
    key = git_output(root, "annex", "lookupkey", "--", relative)
    if key:
        location = git_output(root, "annex", "contentlocation", key)
        if location:
            path = Path(location)
            if path.is_absolute():
                candidate = path
            elif path.parts and path.parts[0] == ".git":
                git_dir = git_output(
                    root, "rev-parse", "--path-format=absolute", "--git-dir"
                )
                candidate = (
                    Path(git_dir).joinpath(*path.parts[1:])
                    if git_dir
                    else file
                )
            else:
                candidate = root.joinpath(*path.parts)
            try:
                return candidate.resolve(strict=True)
            except (FileNotFoundError, NotADirectoryError):
                pass
    return file.resolve(strict=True)


def parse_compact_manifest(
    root: Path, document: dict, selected_ids: set[str] | None = None
) -> dict:
    format_spec = document.get("format")
    expected_format = {
        "name": CORPUS_FORMAT,
        "encoding": "raw",
        "byte_order": "little",
        "axes": ["plane", "y", "x"],
        "order": "C",
    }
    if (
        not isinstance(format_spec, dict)
        or type(format_spec.get("version")) is not int
        or format_spec["version"] not in (1, CORPUS_FORMAT_VERSION)
        or any(format_spec.get(key) != value for key, value in expected_format.items())
        or (format_spec["version"] == 1 and format_spec.get("dtype") != "uint16")
    ):
        raise ValueError(f"Expected {CORPUS_FORMAT} format version 1 or 2")

    corpus_id = nonempty_string(document.get("id"), "Corpus id")
    if not IDENTIFIER.fullmatch(corpus_id):
        raise ValueError(f"Invalid corpus id: {corpus_id}")
    corpus_version = positive_int(document.get("version"), "Corpus version")
    corpus_name = nonempty_string(document.get("name"), "Corpus name")
    datasets = [document] if format_spec["version"] == 1 else document.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Manifest has no datasets")

    identifiers, paths, dataset_ids = set(), set(), set()
    packs, selected_datasets = [], []
    total_bytes = 0
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("Each dataset must be an object")
        dataset_id = nonempty_string(dataset.get("id"), "Dataset id")
        if not IDENTIFIER.fullmatch(dataset_id) or dataset_id in dataset_ids:
            raise ValueError(f"Invalid or duplicate dataset id: {dataset_id}")
        dataset_ids.add(dataset_id)
        dataset_version = positive_int(dataset.get("version"), "Dataset version")
        dataset_name = nonempty_string(dataset.get("name"), "Dataset name")
        modality = dataset.get("modality")
        if not isinstance(modality, str) or modality not in {
            "fluorescence", "brightfield", "quantitative-phase", "electron-microscopy"
        }:
            raise ValueError(f"Unsupported dataset modality: {modality}")
        source = dataset.get("source")
        if not isinstance(source, dict):
            raise ValueError("Dataset source must be an object")
        assets = dataset.get("assets")
        if not isinstance(assets, list) or not assets:
            raise ValueError("Dataset has no image assets")
        selected = False
        for asset in assets:
            if not isinstance(asset, dict):
                raise ValueError("Each image asset must be an object")
            asset_id = asset.get("id", "")
            if (
                not isinstance(asset_id, str)
                or not IDENTIFIER.fullmatch(asset_id)
                or asset_id in identifiers
            ):
                raise ValueError(f"Invalid or duplicate asset id: {asset_id}")
            identifiers.add(asset_id)
            if selected_ids is not None and asset_id not in selected_ids:
                continue
            nonempty_string(asset.get("name"), f"{asset_id} name")
            relative_file(root, asset.get("path"))
            if asset["path"] in paths:
                raise ValueError(f"Duplicate asset path: {asset['path']}")
            paths.add(asset["path"])
            native_dtype = format_spec["dtype"] if format_spec["version"] == 1 else asset.get("dtype")
            if not isinstance(native_dtype, str) or native_dtype not in ASSET_DTYPES:
                raise ValueError(f"Unsupported asset dtype: {asset_id}: {native_dtype}")
            dtype, bytes_per_element = ASSET_DTYPES[native_dtype]
            shape = asset.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or any(type(value) is not int or value <= 0 for value in shape)
            ):
                raise ValueError(f"{asset_id} shape must be [plane, y, x]")
            plane_count, height, width = shape
            size = plane_count * height * width * bytes_per_element
            checksum = asset.get("sha256", "")
            if not isinstance(checksum, str) or not SHA256.fullmatch(checksum):
                raise ValueError(f"Invalid asset checksum: {asset_id}")
            total_bytes += size
            if total_bytes > MAX_BYTES:
                raise ValueError("Selected data exceed the 256 MiB decoded limit")
            packs.append(
                {
                    "id": asset_id,
                    "name": asset["name"],
                    "path": asset["path"],
                    "dataset_id": dataset_id,
                    "dataset_version": dataset_version,
                    "modality": modality,
                    "split": "core",
                    "source_group": asset_id,
                    "dtype": dtype,
                    "height": height,
                    "width": width,
                    "bytes": size,
                    "sha256": checksum,
                    "planes": [
                        {"id": f"{asset_id}-{index}"} for index in range(plane_count)
                    ],
                }
            )
            selected = True
        if selected:
            selected_datasets.append({
                "id": dataset_id,
                "version": dataset_version,
                "name": dataset_name,
                "modality": modality,
                "source": source,
            })

    if selected_ids is not None:
        missing = sorted(selected_ids - identifiers)
        if missing:
            raise ValueError(f"Selected image asset(s) are absent: {', '.join(missing)}")
    if not packs:
        raise ValueError("No image assets were selected")

    normalized = {
        "schema_version": 2,
        "kind": "raw",
        "release": corpus_id,
        "version": corpus_version,
        "name": corpus_name,
        "format": format_spec,
        "modalities": list(dict.fromkeys(pack["modality"] for pack in packs)),
        "datasets": selected_datasets,
        "sources": {},
        "packs": packs,
    }
    if format_spec["version"] == 1:
        normalized["source"] = document["source"]
    return normalized


def verify_compact_corpus(
    root: Path,
    document: dict,
    manifest_sha: str,
    revision: str | None,
    started: float,
    selected_ids: set[str] | None,
    source_manifest: str,
) -> Corpus:
    normalized = parse_compact_manifest(root, document, selected_ids)
    pack_files = {}
    for pack in normalized["packs"]:
        path = relative_file(root, pack["path"])
        try:
            resolved = resolved_content_file(root, path)
            if resolved.stat().st_size != pack["bytes"]:
                raise ValueError(f"Wrong asset length: {path}; run git annex get data/")
            if digest_file(resolved) != pack["sha256"]:
                raise ValueError(f"Asset checksum mismatch: {path}")
        except (FileNotFoundError, NotADirectoryError) as error:
            raise ValueError(f"Missing image asset: {path}; run git annex get data/") from error
        pack_files[pack["id"]] = resolved
    return Corpus(
        root,
        normalized,
        manifest_sha,
        revision,
        time.perf_counter() - started,
        pack_files,
        source_manifest,
    )


def check_kind(kind: str, allow_test_data=False, allow_provisional=False) -> None:
    if kind == "raw" or (kind == "synthetic-test" and allow_test_data):
        return
    if kind == "provisional" and allow_provisional:
        return
    raise ValueError(
        "Only verified raw data are allowed by default; use --allow-test-data for "
        "synthetic fixtures or --allow-provisional for unverified microscopy"
    )


def corpus_modalities(document: dict) -> list[str]:
    modalities = document.get("modalities", ["fluorescence", "brightfield"])
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(
            not isinstance(m, str) or m not in {"fluorescence", "brightfield"}
            for m in modalities
        )
        or len(modalities) != len(set(modalities))
    ):
        raise ValueError("Declare distinct fluorescence or brightfield modalities")
    return modalities


def corpus_notices(root: Path, document: dict) -> list[str]:
    notices = document.get("notices", [])
    if not isinstance(notices, list):
        raise ValueError("Corpus notices must be a list")
    paths = []
    for notice in notices:
        if not isinstance(notice, dict):
            raise ValueError("Each corpus notice needs a path and SHA256")
        path = relative_file(root, notice.get("path"))
        checksum = notice.get("sha256", "")
        if not isinstance(checksum, str) or not SHA256.fullmatch(checksum):
            raise ValueError("Invalid notice checksum")
        if digest_file(path) != checksum:
            raise ValueError(f"Notice checksum mismatch: {path}")
        paths.append(notice["path"])
    return paths


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
    *,
    manifest: str = "manifest.json",
    allow_test_data: bool = False,
    allow_provisional: bool = False,
    selected_ids: set[str] | None = None,
) -> Corpus:
    start = time.perf_counter()
    root = root.expanduser().resolve()
    if (root / "INCOMPLETE").exists():
        raise ValueError("Corpus extraction is incomplete")
    path = relative_file(root, manifest)
    document = read_json(path)
    manifest_sha = digest_file(path)
    revision = (
        git_output(root, "rev-parse", "HEAD") if (root / ".git").exists() else None
    )
    if "format" in document:
        return verify_compact_corpus(
            root,
            document,
            manifest_sha,
            revision,
            start,
            selected_ids,
            manifest,
        )
    selection = document.get("selection")
    if selection is not None:
        survey = relative_file(root, selection["survey_path"])
        if digest_file(survey) != selection["survey_sha256"]:
            raise ValueError("Selection survey checksum mismatch")
    expected_modalities = set(corpus_modalities(document))
    corpus_notices(root, document)
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
    selected_packs = []
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
        if selected_ids is not None and name not in selected_ids:
            continue
        total_bytes += size
        if total_bytes > MAX_BYTES:
            raise ValueError("Selected data exceed the 256 MiB decoded limit")
        try:
            resolved = resolved_content_file(root, file)
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
        except (FileNotFoundError, NotADirectoryError) as error:
            raise ValueError(
                f"Missing image pack: {file}; run git annex get data/"
            ) from error
        selected_packs.append(pack)
    if selected_ids is not None:
        missing = sorted(selected_ids - identifiers)
        if missing:
            raise ValueError(f"Selected image pack(s) are absent: {', '.join(missing)}")
    if not selected_packs:
        raise ValueError("No image packs were selected")
    if modalities != expected_modalities:
        raise ValueError("Image packs differ from the declared modalities")
    document = dict(document)
    document["packs"] = selected_packs
    return Corpus(
        root,
        document,
        manifest_sha,
        revision,
        time.perf_counter() - start,
        pack_files,
        manifest,
    )

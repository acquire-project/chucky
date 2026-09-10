"""Load Chucky-owned image dataset selections.

The data repository owns physical assets and provenance.  This registry owns
which of those assets Chucky benchmarks and the logical input ids used by the
reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from manifest import IDENTIFIER, read_json, verify_corpus


CURRENT_VERSION = 1
DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "bench/data.json"

_TOP_LEVEL_KEYS = {"version", "default_dataset", "source", "dataset"}
_SOURCE_KEYS = {"id", "path", "manifest", "format", "format_version"}
_DATASET_KEYS = {
    "id",
    "source",
    "manifest_id",
    "manifest_version",
    "member",
}
_MEMBER_KEYS = {"asset", "input"}


def identifier(value: object, where: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{where} must be a lowercase identifier")
    return value


def relative_path(value: object, where: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{where} must be a relative path using forward slashes")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {".", "..", ".git"} or ":" in part for part in path.parts)
    ):
        raise ValueError(f"Invalid {where}: {value}")
    return path


def exact_keys(value: object, expected: set[str], where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{where} is missing {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{where} has unknown key(s): {', '.join(unknown)}")
    return value


def positive_version(value: object, where: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{where} must be a positive integer")
    return value


@dataclass(frozen=True)
class Dataset:
    id: str
    source_id: str
    root: Path
    manifest: str
    manifest_format: str
    manifest_format_version: int
    manifest_id: str
    manifest_version: int
    members: dict[str, str]

    def record(self) -> dict:
        return {
            "id": self.id,
            "source": self.source_id,
            "format": {
                "name": self.manifest_format,
                "version": self.manifest_format_version,
            },
            "manifest_id": self.manifest_id,
            "manifest_version": self.manifest_version,
            "members": [
                {"asset": asset, "input": input_id}
                for asset, input_id in self.members.items()
            ],
        }


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"No data registry at {path}; initialize the benchmark data submodule"
        )
    document = read_json(path)

    missing = sorted(_TOP_LEVEL_KEYS - set(document))
    unknown = sorted(set(document) - _TOP_LEVEL_KEYS)
    if missing:
        raise ValueError(f"{path}: missing top-level key(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s): {', '.join(unknown)}")
    if document["version"] != CURRENT_VERSION or type(document["version"]) is not int:
        raise ValueError(f"{path}: version must be the integer {CURRENT_VERSION}")

    sources = document["source"]
    datasets = document["dataset"]
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{path}: expected one or more source entries")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{path}: expected one or more dataset entries")

    source_by_id = {}
    for position, source in enumerate(sources, start=1):
        where = f"{path}: source #{position}"
        source = exact_keys(source, _SOURCE_KEYS, where)
        source_id = identifier(source["id"], f"{where} id")
        if source_id in source_by_id:
            raise ValueError(f"{path}: duplicate source id {source_id}")
        source_path = relative_path(source["path"], f"{where} path")
        manifest = relative_path(source["manifest"], f"{where} manifest")
        source_by_id[source_id] = {
            "id": source_id,
            "root": path.parent.joinpath(*source_path.parts).resolve(),
            "manifest": manifest.as_posix(),
            "format": identifier(source["format"], f"{where} format"),
            "format_version": positive_version(
                source["format_version"], f"{where} format_version"
            ),
        }

    dataset_by_id = {}
    for position, item in enumerate(datasets, start=1):
        where = f"{path}: dataset #{position}"
        item = exact_keys(item, _DATASET_KEYS, where)
        dataset_id = identifier(item["id"], f"{where} id")
        if dataset_id in dataset_by_id:
            raise ValueError(f"{path}: duplicate dataset id {dataset_id}")
        source_id = identifier(item["source"], f"{where} source")
        if source_id not in source_by_id:
            raise ValueError(f"{where} references unknown source {source_id}")
        members = item["member"]
        if not isinstance(members, list) or not members:
            raise ValueError(f"{where} needs one or more member entries")
        mapping = {}
        for member_position, member in enumerate(members, start=1):
            member_where = f"{where} member #{member_position}"
            member = exact_keys(member, _MEMBER_KEYS, member_where)
            asset = identifier(member["asset"], f"{member_where} asset")
            input_id = identifier(member["input"], f"{member_where} input")
            if asset in mapping:
                raise ValueError(f"{where} repeats asset {asset}")
            mapping[asset] = input_id
        source = source_by_id[source_id]
        dataset_by_id[dataset_id] = Dataset(
            id=dataset_id,
            source_id=source_id,
            root=source["root"],
            manifest=source["manifest"],
            manifest_format=source["format"],
            manifest_format_version=source["format_version"],
            manifest_id=identifier(item["manifest_id"], f"{where} manifest_id"),
            manifest_version=positive_version(
                item["manifest_version"], f"{where} manifest_version"
            ),
            members=mapping,
        )

    default = identifier(document["default_dataset"], f"{path}: default_dataset")
    if default not in dataset_by_id:
        raise ValueError(f"{path}: unknown default dataset {default}")
    return {
        "version": CURRENT_VERSION,
        "default_dataset": default,
        "sources": source_by_id,
        "datasets": dataset_by_id,
    }


def verify_dataset(
    registry_path: Path = DEFAULT_REGISTRY,
    dataset_id: str | None = None,
    source_override: Path | None = None,
):
    registry = load_registry(registry_path)
    selected_id = dataset_id or registry["default_dataset"]
    try:
        dataset = registry["datasets"][selected_id]
    except KeyError as error:
        raise ValueError(f"Unknown dataset: {selected_id}") from error
    root = (
        source_override.expanduser().resolve()
        if source_override is not None
        else dataset.root
    )
    manifest_path = root.joinpath(*PurePosixPath(dataset.manifest).parts)
    document = read_json(manifest_path)
    format_spec = document.get("format")
    if not isinstance(format_spec, dict):
        raise ValueError(f"{selected_id} requires a compact versioned manifest")
    if (
        format_spec.get("name") != dataset.manifest_format
        or format_spec.get("version") != dataset.manifest_format_version
    ):
        raise ValueError(
            f"{selected_id} requires {dataset.manifest_format} format version "
            f"{dataset.manifest_format_version}"
        )
    if (
        document.get("id") != dataset.manifest_id
        or document.get("version") != dataset.manifest_version
    ):
        raise ValueError(
            f"{selected_id} requires data {dataset.manifest_id} version "
            f"{dataset.manifest_version}"
        )

    corpus = verify_corpus(
        root, manifest=dataset.manifest, selected_ids=set(dataset.members)
    )
    by_id = {pack["id"]: pack for pack in corpus.manifest["packs"]}
    ordered_packs = []
    ordered_files = {}
    for asset, input_id in dataset.members.items():
        pack = by_id[asset]
        pack["input_id"] = input_id
        ordered_packs.append(pack)
        ordered_files[asset] = corpus.pack_files[asset]
    corpus.manifest["packs"] = ordered_packs
    corpus.pack_files = ordered_files
    corpus.dataset = dataset.record()
    return corpus


def load_corpus(
    registry_path: Path,
    dataset_id: str | None,
    source_override: Path | None,
    direct_root: Path | None,
    *,
    allow_test_data: bool = False,
    allow_provisional: bool = False,
):
    """Load a registered dataset, or explicitly bypass it for an older corpus."""
    if direct_root is not None:
        if dataset_id is not None or source_override is not None:
            raise ValueError(
                "--direct-corpus cannot be combined with --dataset or --corpus"
            )
        return verify_corpus(
            direct_root,
            allow_test_data=allow_test_data,
            allow_provisional=allow_provisional,
        )
    return verify_dataset(registry_path, dataset_id, source_override)

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from manifest import IDENTIFIER, digest_file, read_json, relative_file
from run import write_json

BUCKET = "https://czb-opencell.s3.us-west-2.amazonaws.com/"
NAMESPACE = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
STACK_NAME = re.compile(r"OC-FOV_(.+)_(ENSG\d+)_(CID\d+)_(FID\d+)_stack\.tif")


def stack_ids(key: str) -> dict:
    name = Path(key).name
    match = STACK_NAME.fullmatch(name)
    if not key.startswith("microscopy/raw/") or not match:
        raise ValueError(f"Expected a public OpenCell raw stack: {key}")
    gene, ensg, cell_line, field = match.groups()
    return {"gene": gene, "ensg": ensg, "cell_line": cell_line, "field": field}


def rank(release: str, value: str) -> str:
    return hashlib.sha256(f"{release}:{value}".encode()).hexdigest()


def catalog(args):
    if args.output.exists() or args.fields < 4:
        raise ValueError("Use a new output path and at least four fields")
    if not IDENTIFIER.fullmatch(args.release):
        raise ValueError("Invalid release name")
    params = {"list-type": "2", "prefix": "microscopy/raw/", "max-keys": "1000"}
    targets = {}
    stack_count = 0
    while True:
        with urlopen(BUCKET + "?" + urlencode(params), timeout=60) as response:
            page = ElementTree.fromstring(response.read())
        for item in page.findall("s:Contents", NAMESPACE):
            key = item.findtext("s:Key", namespaces=NAMESPACE)
            if not key.endswith("_stack.tif"):
                continue
            identity = stack_ids(key)
            record = identity | {
                "key": key,
                "bytes": int(item.findtext("s:Size", namespaces=NAMESPACE)),
                "etag": item.findtext("s:ETag", namespaces=NAMESPACE),
                "last_modified": item.findtext("s:LastModified", namespaces=NAMESPACE),
            }
            stack_count += 1
            target = identity["ensg"]
            if target not in targets or rank(args.release, key) < rank(
                args.release, targets[target]["key"]
            ):
                targets[target] = record
        if page.findtext("s:IsTruncated", namespaces=NAMESPACE) != "true":
            break
        params["continuation-token"] = page.findtext(
            "s:NextContinuationToken", namespaces=NAMESPACE
        )
    if len(targets) < args.fields:
        raise ValueError("The catalog has fewer targets than requested")
    chosen = sorted(targets, key=lambda value: rank(args.release, value))[: args.fields]
    document = {
        "release": args.release,
        "catalog_stack_count": stack_count,
        "catalog_target_count": len(targets),
        "selection": (
            "Sort targets by SHA256(release + ':' + ENSG ID); select the first "
            f"{args.fields}. Within each target, select the stack with the lowest "
            "SHA256(release + ':' + object key)."
        ),
        "files": [targets[target] for target in chosen],
    }
    write_json(args.output, document)
    print(
        json.dumps(
            {"fields": len(chosen), "bytes": sum(f["bytes"] for f in document["files"])}
        )
    )


def download(record: dict, root: Path, previous: dict) -> dict:
    identity = stack_ids(record["key"])
    path = root / "images" / Path(record["key"]).name
    url = BUCKET + quote(record["key"], safe="/")
    expected = record.get("sha256") or previous.get(record["key"], {}).get("sha256")
    if path.exists():
        if (
            not expected
            or path.stat().st_size != record["bytes"]
            or digest_file(path) != expected
        ):
            raise ValueError(
                f"Existing source does not match its recorded checksum: {path}"
            )
        checksum = expected
    else:
        with tempfile.NamedTemporaryFile(
            prefix=path.name + ".", suffix=".part", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
        digest = hashlib.sha256()
        request = Request(url, headers={"If-Match": record["etag"]})
        try:
            with (
                urlopen(request, timeout=60) as response,
                temporary.open("wb") as output,
            ):
                if response.headers.get("ETag") != record["etag"]:
                    raise ValueError(f"Source object changed: {url}")
                for block in iter(lambda: response.read(1024**2), b""):
                    output.write(block)
                    digest.update(block)
            checksum = digest.hexdigest()
            if temporary.stat().st_size != record["bytes"] or (
                expected and checksum != expected
            ):
                raise ValueError(f"Downloaded source differs from its record: {url}")
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return (
        record
        | identity
        | {
            "url": url,
            "path": path.relative_to(root).as_posix(),
            "sha256": checksum,
        }
    )


def fetch(args):
    document = read_json(args.plan)
    records = document["files"]
    if not records or sum(record["bytes"] for record in records) > 6 * 1024**3:
        raise ValueError("Source downloads must fit within 6 GiB")
    keys = [record["key"] for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source object")
    for record in records:
        stack_ids(record["key"])
    root = args.output.resolve()
    (root / "images").mkdir(parents=True, exist_ok=True)
    ledger = root / "downloads.json"
    previous = {}
    if ledger.exists():
        saved = read_json(ledger)
        if saved["plan_sha256"] != digest_file(args.plan):
            raise ValueError("Download directory belongs to another source plan")
        previous = {record["key"]: record for record in saved["files"]}
    results = dict(previous)
    completed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {
            pool.submit(download, record, root, previous): record for record in records
        }
        for job in as_completed(jobs):
            record = job.result()
            results[record["key"]] = record
            completed += 1
            saved = document | {
                "status": "complete" if completed == len(records) else "running",
                "plan_sha256": digest_file(args.plan),
                "files": [results[key] for key in keys if key in results],
            }
            write_json(ledger, saved)
            print(
                f"{completed}/{len(records)}: {record['gene']} ({record['bytes']} bytes)",
                flush=True,
            )
    print(ledger)


def make_plan(args):
    import tifffile

    root = args.corpus.resolve()
    output = root / "source-plan.json"
    if output.exists() or (root / "manifest.json").exists():
        raise ValueError("Refusing to replace an existing corpus source plan")
    downloads = read_json(args.downloads)
    if downloads.get("status") != "complete":
        raise ValueError("All source downloads must be complete")
    release = downloads["release"]
    if not IDENTIFIER.fullmatch(release):
        raise ValueError("Invalid release name")
    evidence_path = root / "provenance" / f"{release}.json"
    evidence = read_json(evidence_path)
    if (
        evidence.get("raw_status") != "verified"
        or evidence.get("allowed_for_raw_release") is not True
    ):
        raise ValueError("Record the source's original-acquisition evidence first")
    targets = [record["ensg"] for record in downloads["files"]]
    if len(targets) != len(set(targets)):
        raise ValueError("Select only one field per target protein")
    records, candidates = [], []
    for index, record in enumerate(downloads["files"]):
        file = relative_file(args.downloads.parent, record["path"])
        if digest_file(file) != record["sha256"]:
            raise ValueError(f"Source checksum changed: {file}")
        with tifffile.TiffFile(file) as tiff:
            series = tiff.series[0]
            shape = tuple(series.shape)
            axes = series.axes
            if (
                axes not in {"CZYX", "ZCYX"}
                or len(shape) != 4
                or shape[axes.index("C")] != 2
                or series.dtype.kind != "u"
                or series.dtype.itemsize != 2
            ):
                raise ValueError(
                    f"Expected a two-channel uint16 CZYX or ZCYX stack: {file}: {axes} {shape}"
                )
            if len(tiff.pages) != math.prod(shape[:-2]):
                raise ValueError(
                    f"TIFF page count differs from its channel/Z dimensions: {file}"
                )
            depth = shape[axes.index("Z")]
            z = min(depth - 1, depth * (2 * (index % 4) + 1) // 8)
            record = record | {
                "tiff_axes": series.axes,
                "axes": axes,
                "shape": list(shape),
                "selected_z": z,
                "z_fraction": (2 * (index % 4) + 1) / 8,
            }
            for channel, label, group in [
                (0, "Hoechst", "opencell-dna"),
                (1, "mNeonGreen", "opencell-protein"),
            ]:
                page_index = channel * depth + z if axes == "CZYX" else z * 2 + channel
                page = tiff.pages[page_index]
                if page.shape != shape[-2:] or page.compression.name != "NONE":
                    raise ValueError(
                        f"Unexpected OpenCell page layout or compression: {file}"
                    )
                candidates.append(
                    {
                        "id": f"opencell-{record['field'].lower()}-c{channel}-z{z}",
                        "source_id": "opencell",
                        "source_group": group,
                        "field_id": f"opencell/{record['ensg']}/{record['field']}",
                        "modality": "fluorescence",
                        "reader": "tiff",
                        "path": record["path"],
                        "source_url": record["url"],
                        "source_sha256": record["sha256"],
                        "gene": record["gene"],
                        "ensg": record["ensg"],
                        "cell_line": record["cell_line"],
                        "channel": label,
                        "axes": axes,
                        "coordinates": {"c": channel, "z": z, "page": page_index},
                    }
                )
        records.append(record)
    evidence["source_files"] = records
    evidence["candidate_selection"] = {
        key: downloads[key]
        for key in ["catalog_stack_count", "catalog_target_count", "selection"]
    }
    evidence["candidate_selection"]["z_selection"] = (
        "In catalog selection order, cycle through 1/8, 3/8, 5/8, 7/8 of each stack's Z depth, rounding down; retain both channels."
    )
    write_json(evidence_path, evidence)
    notice = root / "provenance" / f"{release}-NOTICE.md"
    plan = {
        "schema_version": 1,
        "kind": "raw",
        "release": release,
        "modalities": ["fluorescence"],
        "selection_reason": f"Full fields from {len(records)} distinct OpenCell proteins, both fluorescence channels, and four relative Z depths; benchmark and validation fields are separated before selection.",
        "sources": {
            "opencell": {
                "raw_status": "verified",
                "evidence": evidence_path.relative_to(root).as_posix(),
                "evidence_sha256": digest_file(evidence_path),
            }
        },
        "notices": [
            {"path": notice.relative_to(root).as_posix(), "sha256": digest_file(notice)}
        ],
        "candidates": candidates,
    }
    write_json(output, plan)
    print(output)


def main():
    parser = argparse.ArgumentParser(
        description="Select and fetch original OpenCell fluorescence stacks"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("catalog")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release", default="opencell-v1")
    p.add_argument("--fields", type=int, default=32)
    p = commands.add_parser("fetch")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("plan")
    p.add_argument("--downloads", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    {"catalog": catalog, "fetch": fetch, "plan": make_plan}[args.command](args)


if __name__ == "__main__":
    main()

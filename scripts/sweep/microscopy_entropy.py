# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ITEM_BYTES = {"uint8": 1, "uint16": 2, "float32": 4}
ROWS_PER_PLANE = 8
DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "bench/data/microscopy"


def sample_rows(height):
    count = min(ROWS_PER_PLANE, height)
    return [(2 * index + 1) * height // (2 * count) for index in range(count)]


def sample_asset(stream, asset):
    planes, height, width = asset["shape"]
    item_bytes = ITEM_BYTES[asset["dtype"]]
    rows = sample_rows(height)
    blocks = []
    for plane in range(planes):
        for row in rows:
            stream.seek((plane * height + row) * width * item_bytes)
            block = stream.read(width * item_bytes)
            if len(block) != width * item_bytes:
                raise ValueError(f"Incomplete sample: {asset['id']}")
            blocks.append(block)
    raw = b"".join(blocks)
    counts = Counter(raw[index:index + item_bytes] for index in range(0, len(raw), item_bytes))
    pixels = len(raw) // item_bytes
    entropy = math.fsum((count / pixels) * math.log2(pixels / count) for count in counts.values())
    return {"asset": asset["id"], "dtype": asset["dtype"], "shape": asset["shape"],
            "pack_sha256": asset["sha256"], "sample_sha256": hashlib.sha256(raw).hexdigest(),
            "sample_rows": rows, "sample_pixels": pixels, "unique_values": len(counts),
            "pixel_entropy_bits": entropy}


def profile_corpus(corpus):
    manifest_raw = (corpus / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    records = []
    for dataset in manifest["datasets"]:
        for asset in dataset["assets"]:
            relative = Path(asset["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Image pack escapes the corpus")
            path = corpus / relative
            expected_size = math.prod(asset["shape"]) * ITEM_BYTES[asset["dtype"]]
            if path.stat().st_size != expected_size:
                raise ValueError(f"Incorrect pack size: {asset['id']}")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != asset["sha256"]:
                    raise ValueError(f"Pack checksum disagrees: {asset['id']}")
                records.append(sample_asset(stream, asset))
    return {"version": 2, "rows_per_plane": ROWS_PER_PLANE,
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "inputs": records}


def main():
    parser = argparse.ArgumentParser(description="Sample pixel entropy in bits/pixel from verified microscopy packs")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = profile_corpus(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

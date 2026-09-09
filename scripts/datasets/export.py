from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from manifest import relative_file, verify_corpus


def main():
    parser = argparse.ArgumentParser(
        description="Export verified image bytes for machines without annex"
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "bench/datasets/corpus.lock.json",
    )
    parser.add_argument("--allow-test-data", action="store_true")
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args()
    corpus = verify_corpus(
        args.corpus, args.lock, args.allow_test_data, args.allow_provisional
    )
    pack_files = {
        p["path"]: corpus.pack_files[p["id"]] for p in corpus.manifest["packs"]
    }
    files = {"manifest.json"}
    files.update(n["path"] for n in corpus.manifest.get("notices", []))
    selection = corpus.manifest.get("selection")
    if selection is not None:
        files.add(selection["survey_path"])
    files.update(p["path"] for p in corpus.manifest["packs"])
    files.update(s["evidence"] for s in corpus.manifest.get("sources", {}).values())
    with zipfile.ZipFile(
        args.output, "x", compression=zipfile.ZIP_STORED, strict_timestamps=False
    ) as archive:
        for name in sorted(files):
            archive.write(pack_files.get(name, relative_file(corpus.root, name)), name)
    print(args.output)


if __name__ == "__main__":
    main()

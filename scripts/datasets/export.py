from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from data_sources import DEFAULT_REGISTRY, load_corpus
from manifest import relative_file


def main():
    parser = argparse.ArgumentParser(
        description="Export verified image bytes for machines without annex"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--corpus", type=Path, help="Override the registered dataset's source path"
    )
    source.add_argument(
        "--direct-corpus", type=Path, help="Load an unregistered legacy corpus"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--data-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-test-data", action="store_true")
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args()
    corpus = load_corpus(
        args.data_registry,
        args.dataset,
        args.corpus,
        args.direct_corpus,
        allow_test_data=args.allow_test_data,
        allow_provisional=args.allow_provisional,
    )
    pack_files = {
        p["path"]: corpus.pack_files[p["id"]] for p in corpus.manifest["packs"]
    }
    files = {"manifest.json": relative_file(corpus.root, corpus.source_manifest)}
    files.update(
        (n["path"], relative_file(corpus.root, n["path"]))
        for n in corpus.manifest.get("notices", [])
    )
    selection = corpus.manifest.get("selection")
    if selection is not None:
        files[selection["survey_path"]] = relative_file(
            corpus.root, selection["survey_path"]
        )
    files.update((p["path"], pack_files[p["path"]]) for p in corpus.manifest["packs"])
    files.update(
        (s["evidence"], relative_file(corpus.root, s["evidence"]))
        for s in corpus.manifest.get("sources", {}).values()
    )
    with zipfile.ZipFile(
        args.output, "x", compression=zipfile.ZIP_STORED, strict_timestamps=False
    ) as archive:
        for name, file in sorted(files.items()):
            archive.write(file, name)
    print(args.output)


if __name__ == "__main__":
    main()

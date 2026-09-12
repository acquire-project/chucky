import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data_sources import load_corpus, load_members, load_registry, verify_dataset
from test_manifest import collection_fixture


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(root: Path, *, extra_missing: bool = False) -> None:
    selected = bytes(range(12))
    (root / "selected.raw").write_bytes(selected)
    assets = [
        {
            "id": "physical-selected",
            "name": "External name is not authoritative",
            "path": "selected.raw",
            "shape": [1, 2, 3],
            "sha256": sha(selected),
        }
    ]
    if extra_missing:
        assets.append(
            {
                "id": "not-in-sweep",
                "metadata": "The selected view does not depend on this schema",
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": {
                    "name": "chucky-image-corpus",
                    "version": 1,
                    "encoding": "raw",
                    "dtype": "uint16",
                    "byte_order": "little",
                    "axes": ["plane", "y", "x"],
                    "order": "C",
                },
                "id": "external-images",
                "version": 3,
                "name": "External images",
                "modality": "fluorescence",
                "source": {
                    "collection": "Test",
                    "url": "https://example.invalid",
                    "attribution": "Test",
                    "license": "CC0-1.0",
                    "license_url": "https://example.invalid/license",
                    "changes": "Selected test pixels.",
                },
                "assets": assets,
            }
        )
    )


def registry(path: Path, source: Path) -> None:
    relative = source.relative_to(path.parent).as_posix()
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "default_dataset": "test-view",
                "source": [
                    {
                        "id": "test-source",
                        "path": relative,
                        "manifest": "manifest.json",
                        "format": "chucky-image-corpus",
                        "format_version": 1,
                    }
                ],
                "dataset": [
                    {
                        "id": "test-view",
                        "source": "test-source",
                        "manifest_id": "external-images",
                        "manifest_version": 3,
                        "member": [
                            {"asset": "physical-selected", "input": "logical-input"}
                        ],
                    }
                ],
            }
        )
    )


class DataSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="chucky-data-sources-")
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.path = self.root / "data.json"

    def tearDown(self):
        self.directory.cleanup()

    def test_selection_owns_logical_input_and_ignores_unselected_content(self):
        manifest(self.source, extra_missing=True)
        registry(self.path, self.source)
        corpus = verify_dataset(self.path)
        self.assertEqual(
            [p["id"] for p in corpus.manifest["packs"]], ["physical-selected"]
        )
        self.assertEqual(corpus.manifest["packs"][0]["input_id"], "logical-input")
        self.assertEqual(corpus.record()["dataset"]["id"], "test-view")
        self.assertEqual(
            corpus.record()["dataset"]["format"],
            {"name": "chucky-image-corpus", "version": 1},
        )
        self.assertEqual(corpus.record()["decoded_bytes"], 12)
        self.assertEqual(
            corpus.pack_files["physical-selected"],
            (self.source / "selected.raw").resolve(),
        )

    def test_manifest_identity_and_version_are_contracts(self):
        manifest(self.source)
        registry(self.path, self.source)
        document = json.loads((self.source / "manifest.json").read_text())
        document["version"] = 4
        (self.source / "manifest.json").write_text(json.dumps(document))
        with self.assertRaisesRegex(
            ValueError, "requires data external-images version 3"
        ):
            verify_dataset(self.path)

    def test_manifest_format_version_is_a_contract(self):
        manifest(self.source)
        registry(self.path, self.source)
        document = json.loads((self.source / "manifest.json").read_text())
        document["format"]["version"] = 2
        (self.source / "manifest.json").write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "format version 1"):
            verify_dataset(self.path)

    def test_source_override_keeps_selection_and_direct_mode_is_explicit(self):
        manifest(self.source)
        registry(self.path, self.source)
        selected = load_corpus(self.path, None, self.source, None)
        self.assertEqual(selected.manifest["packs"][0]["input_id"], "logical-input")

        direct = load_corpus(self.path, None, None, self.source)
        self.assertIsNone(direct.dataset)
        self.assertNotIn("input_id", direct.manifest["packs"][0])
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            load_corpus(self.path, "test-view", None, self.source)

    def test_registry_member_order_controls_replay_order(self):
        manifest(self.source)
        other = b"abcdefghijkl"
        (self.source / "other.raw").write_bytes(other)
        document = json.loads((self.source / "manifest.json").read_text())
        document["assets"].insert(
            0,
            {
                "id": "physical-other",
                "name": "Other",
                "path": "other.raw",
                "shape": [1, 2, 3],
                "sha256": sha(other),
            },
        )
        (self.source / "manifest.json").write_text(json.dumps(document))
        registry(self.path, self.source)
        document = json.loads(self.path.read_text())
        document["dataset"][0]["member"].append(
            {"asset": "physical-other", "input": "other-input"}
        )
        self.path.write_text(json.dumps(document))

        corpus = verify_dataset(self.path)
        self.assertEqual(
            [pack["id"] for pack in corpus.manifest["packs"]],
            ["physical-selected", "physical-other"],
        )

    def test_collection_selection_reads_native_types_without_image_content(self):
        document = collection_fixture(self.source)
        registry(self.path, self.source)
        selection = json.loads(self.path.read_text())
        selection["source"][0]["format_version"] = 2
        selection["dataset"][0].update(
            manifest_id=document["id"], manifest_version=1,
            member=[
                {"asset": "test-float32", "input": "phase"},
                {"asset": "test-uint8", "input": "em"},
                {"asset": "test-uint16", "input": "fluorescence"},
            ],
        )
        self.path.write_text(json.dumps(selection))
        for path in self.source.glob("*.raw"):
            path.unlink()
        self.assertEqual(load_members(self.path), [
            ("test-float32", "phase", "f32"),
            ("test-uint8", "em", "u8"),
            ("test-uint16", "fluorescence", "u16"),
        ])
        override = self.root / "override"
        override.mkdir()
        document["datasets"][0]["assets"][0]["dtype"] = "float32"
        (override / "manifest.json").write_text(json.dumps(document))
        self.assertEqual(load_members(self.path, source_override=override)[1],
                         ("test-uint8", "em", "f32"))
        with self.assertRaisesRegex(ValueError, "Missing image asset"):
            verify_dataset(self.path)

    def test_duplicate_members_and_unknown_sources_are_rejected(self):
        manifest(self.source)
        registry(self.path, self.source)
        document = json.loads(self.path.read_text())
        document["dataset"][0]["member"].append(
            {"asset": "physical-selected", "input": "other"}
        )
        self.path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "repeats asset"):
            load_registry(self.path)

        registry(self.path, self.source)
        document = json.loads(self.path.read_text())
        document["dataset"][0]["source"] = "missing"
        self.path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "unknown source"):
            load_registry(self.path)

    def test_duplicate_source_dataset_and_unknown_default_are_rejected(self):
        manifest(self.source)
        registry(self.path, self.source)
        document = json.loads(self.path.read_text())
        document["source"].append(document["source"][0])
        self.path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            load_registry(self.path)

        registry(self.path, self.source)
        document = json.loads(self.path.read_text())
        document["dataset"].append(document["dataset"][0])
        self.path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "duplicate dataset"):
            load_registry(self.path)

        registry(self.path, self.source)
        document = json.loads(self.path.read_text())
        document["default_dataset"] = "missing"
        self.path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "unknown default"):
            load_registry(self.path)


if __name__ == "__main__":
    unittest.main()

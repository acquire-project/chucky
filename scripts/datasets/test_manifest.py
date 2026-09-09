import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from manifest import verify_corpus
from run import assess, check_result


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(root):
    evidence = b"Synthetic fixture, not microscope data.\n"
    (root / "evidence.txt").write_bytes(evidence)
    document = {
        "schema_version": 1,
        "kind": "synthetic-test",
        "release": "test",
        "sources": {
            "test": {
                "raw_status": "synthetic-test",
                "evidence": "evidence.txt",
                "evidence_sha256": sha(evidence),
            },
        },
        "packs": [],
    }
    for index, modality in enumerate(("fluorescence", "brightfield")):
        data = bytes(range(index * 12, index * 12 + 12))
        name = f"pack-{index}"
        (root / f"{name}.raw").write_bytes(data)
        document["packs"].append(
            {
                "id": name,
                "modality": modality,
                "split": "core",
                "source_group": "test",
                "path": f"{name}.raw",
                "dtype": "u16le",
                "width": 3,
                "height": 2,
                "bytes": 12,
                "sha256": sha(data),
                "planes": [
                    {
                        "id": f"plane-{index}",
                        "source_id": "test",
                        "field_id": f"field-{index}",
                        "coordinates": {"t": 0, "c": 0, "z": 0},
                        "sha256": sha(data),
                    }
                ],
            }
        )
    (root / "manifest.json").write_text(json.dumps(document))
    return document


def compact_fixture(root):
    data = bytes(range(12))
    (root / "images.raw").write_bytes(data)
    document = {
        "format": {
            "name": "chucky-image-corpus",
            "version": 1,
            "encoding": "raw",
            "dtype": "uint16",
            "byte_order": "little",
            "axes": ["plane", "y", "x"],
            "order": "C",
        },
        "id": "test-images",
        "version": 1,
        "name": "Test images",
        "modality": "fluorescence",
        "source": {
            "collection": "Test collection",
            "url": "https://example.invalid/images",
            "attribution": "Test creator",
            "license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "changes": "Selected and repacked test planes.",
        },
        "assets": [
            {
                "id": "test-input",
                "name": "Test input",
                "path": "images.raw",
                "shape": [1, 2, 3],
                "sha256": sha(data),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(document))
    return document


class CompactManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="chucky-compact-manifest-")
        self.root = Path(self.directory.name)
        self.document = compact_fixture(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def save(self):
        (self.root / "manifest.json").write_text(json.dumps(self.document))

    def test_minimal_manifest_verifies_and_normalizes_for_replay(self):
        corpus = verify_corpus(self.root)
        self.assertEqual(corpus.record()["release"], "test-images")
        self.assertEqual(corpus.record()["version"], 1)
        self.assertEqual(corpus.record()["decoded_bytes"], 12)
        self.assertEqual(corpus.manifest["kind"], "raw")
        self.assertEqual(corpus.manifest["packs"][0]["source_group"], "test-input")
        self.assertEqual(
            corpus.manifest["packs"][0]["planes"], [{"id": "test-input-0"}]
        )

    def test_format_and_attribution_are_required(self):
        self.document["format"]["version"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "format version 1"):
            verify_corpus(self.root)
        self.document["format"]["version"] = 1
        self.document["source"]["attribution"] = ""
        self.save()
        with self.assertRaisesRegex(ValueError, "Source attribution"):
            verify_corpus(self.root)

    def test_asset_shape_path_and_checksum_are_checked(self):
        asset = self.document["assets"][0]
        for key, value, message in (
            ("shape", [1, 2, 4], "Wrong asset length"),
            ("path", "../images.raw", "Invalid corpus path"),
            ("sha256", "0" * 64, "Asset checksum mismatch"),
        ):
            with self.subTest(key=key):
                original = asset[key]
                asset[key] = value
                self.save()
                with self.assertRaisesRegex(ValueError, message):
                    verify_corpus(self.root)
                asset[key] = original

    def test_content_lock_does_not_require_a_git_revision(self):
        lock = self.root / "lock.json"
        lock.write_text(
            json.dumps(
                {
                    "manifest_sha256": sha(
                        (self.root / "manifest.json").read_bytes()
                    )
                }
            )
        )
        self.assertEqual(verify_corpus(self.root, lock).record()["decoded_bytes"], 12)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="chucky-manifest-")
        self.root = Path(self.directory.name)
        self.document = fixture(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def save(self):
        (self.root / "manifest.json").write_text(json.dumps(self.document))

    def verify(self):
        return verify_corpus(self.root, allow_test_data=True)

    def test_fixture_needs_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "Only verified raw"):
            verify_corpus(self.root)
        self.assertEqual(self.verify().record()["decoded_bytes"], 24)

    def provisional(self):
        evidence = {
            "raw_status": "unverified",
            "allowed_for_raw_release": False,
            "missing_evidence": "Test fixture emulating missing original acquisitions",
        }
        (self.root / "evidence.txt").write_text(json.dumps(evidence))
        self.document["kind"] = "provisional"
        self.document["sources"]["test"].update(
            raw_status="unverified",
            evidence_sha256=sha((self.root / "evidence.txt").read_bytes()),
        )
        self.save()

    def test_provisional_needs_its_own_opt_in_and_keeps_pixel_checks(self):
        self.provisional()
        with self.assertRaisesRegex(ValueError, "allow-provisional"):
            self.verify()
        result = verify_corpus(self.root, allow_provisional=True)
        self.assertEqual(result.manifest["kind"], "provisional")
        (self.root / "pack-0.raw").write_bytes(b"\xff" + bytes(range(1, 12)))
        with self.assertRaisesRegex(ValueError, "Plane checksum mismatch"):
            verify_corpus(self.root, allow_provisional=True)

    def test_provisional_opt_in_cannot_qualify_raw_data(self):
        self.provisional()
        self.document["kind"] = "raw"
        self.save()
        with self.assertRaisesRegex(ValueError, "not verified raw"):
            verify_corpus(self.root, allow_provisional=True)

    def test_provisional_requires_recorded_missing_evidence(self):
        self.provisional()
        evidence = {"raw_status": "unverified", "allowed_for_raw_release": False}
        (self.root / "evidence.txt").write_text(json.dumps(evidence))
        self.document["sources"]["test"]["evidence_sha256"] = sha(
            (self.root / "evidence.txt").read_bytes()
        )
        self.save()
        with self.assertRaisesRegex(ValueError, "missing raw evidence"):
            verify_corpus(self.root, allow_provisional=True)

    def test_single_modality_must_be_declared_and_match_the_packs(self):
        self.document["packs"] = self.document["packs"][:1]
        self.save()
        with self.assertRaisesRegex(ValueError, "declared modalities"):
            self.verify()
        self.document["modalities"] = ["fluorescence"]
        self.save()
        self.assertEqual(self.verify().record()["decoded_bytes"], 12)
        for modalities in [
            [],
            ["fluorescence", "fluorescence"],
            ["phase"],
            ["brightfield"],
        ]:
            with self.subTest(modalities=modalities):
                self.document["modalities"] = modalities
                self.save()
                with self.assertRaises(ValueError):
                    self.verify()

    def test_changed_or_missing_attribution_fails_verification(self):
        notice = self.root / "NOTICE.md"
        content = b"Test attribution and license notice.\n"
        notice.write_bytes(content)
        self.document["notices"] = [{"path": "NOTICE.md", "sha256": sha(content)}]
        self.save()
        self.verify()
        notice.write_bytes(b"Changed attribution")
        with self.assertRaisesRegex(ValueError, "Notice checksum"):
            self.verify()
        notice.unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_changed_pixel_is_rejected(self):
        (self.root / "pack-0.raw").write_bytes(b"\xff" + bytes(range(1, 12)))
        with self.assertRaisesRegex(ValueError, "Plane checksum mismatch"):
            self.verify()

    def test_truncated_or_empty_pack_is_rejected(self):
        for data in (bytes(range(10)), b""):
            with self.subTest(length=len(data)):
                (self.root / "pack-0.raw").write_bytes(data)
                with self.assertRaisesRegex(ValueError, "Wrong pack length"):
                    self.verify()

    def test_missing_annex_content_is_actionable(self):
        (self.root / "pack-0.raw").unlink()
        with self.assertRaisesRegex(ValueError, "git annex get"):
            self.verify()

    def test_annex_pointer_is_not_image_data(self):
        (self.root / "pack-0.raw").write_text("/annex/objects/SHA256-key\n")
        with self.assertRaisesRegex(ValueError, "git annex get"):
            self.verify()

    def test_resolved_paths_can_differ_under_the_same_lock(self):
        lock = self.root / "corpus.lock.json"
        lock.write_text(
            json.dumps(
                {
                    "revision": "a" * 40,
                    "manifest_sha256": sha((self.root / "manifest.json").read_bytes()),
                }
            )
        )
        file = self.root / "pack-0.raw"
        pixels = file.read_bytes()
        objects = self.root / "objects"
        objects.mkdir()
        first = objects / "first-content"
        first.write_bytes(pixels)
        file.unlink()
        try:
            file.symlink_to(first.relative_to(self.root))
        except OSError as error:
            self.skipTest(f"Symlinks are unavailable: {error}")
        corpus = verify_corpus(self.root, lock, allow_test_data=True)
        self.assertEqual(corpus.pack_files["pack-0"], first.resolve())
        before = corpus.sha256
        file.unlink()
        second = objects / "different-name"
        second.write_bytes(pixels)
        file.symlink_to(second)
        changed = verify_corpus(self.root, lock, allow_test_data=True)
        self.assertEqual(changed.sha256, before)
        self.assertEqual(changed.pack_files["pack-0"], second.resolve())
        self.assertEqual(corpus.pack_files["pack-0"], first.resolve())
        file.unlink()
        file.write_bytes(pixels)
        plain = verify_corpus(self.root, lock, allow_test_data=True)
        self.assertEqual(plain.sha256, before)
        self.assertEqual(plain.pack_files["pack-0"], file.resolve())
        file.write_bytes(b"\xff" + pixels[1:])
        with self.assertRaisesRegex(ValueError, "Plane checksum mismatch"):
            verify_corpus(self.root, lock, allow_test_data=True)

    def test_missing_or_changed_provenance_is_rejected(self):
        (self.root / "evidence.txt").write_text("Changed evidence")
        with self.assertRaisesRegex(ValueError, "Provenance checksum mismatch"):
            self.verify()

    def test_processed_source_is_rejected(self):
        self.document["kind"] = "raw"
        self.document["sources"]["test"]["raw_status"] = "processed"
        self.save()
        with self.assertRaisesRegex(ValueError, "not verified raw"):
            self.verify()

    def test_shape_dtype_and_paths_are_checked(self):
        cases = (
            ("width", 0),
            ("height", 3),
            ("dtype", "f32"),
            ("path", "../outside.raw"),
            ("path", "C:/outside.raw"),
        )
        for key, value in cases:
            with self.subTest(key=key):
                original = self.document["packs"][0][key]
                self.document["packs"][0][key] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.verify()
                self.document["packs"][0][key] = original

    def test_cap_is_checked_before_reading_large_pack(self):
        pack = self.document["packs"][0]
        pack.update(width=16384, height=16384, bytes=536870912)
        self.save()
        with self.assertRaisesRegex(ValueError, "256 MiB"):
            self.verify()

    def test_field_cannot_leak_into_holdout(self):
        pack = self.document["packs"][1]
        pack["split"] = "heldout"
        pack["planes"][0]["field_id"] = "field-0"
        self.save()
        with self.assertRaisesRegex(ValueError, "core and heldout"):
            self.verify()

    def test_changed_manifest_does_not_match_pin(self):
        lock = self.root / "lock.json"
        lock.write_text(json.dumps({"revision": "f" * 40, "manifest_sha256": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "pinned corpus"):
            verify_corpus(self.root, lock, True)

    def test_adjusted_annex_branch_keeps_the_same_content_pin(self):
        (self.root / ".git").mkdir()
        lock = self.root / "lock.json"
        lock.write_text(
            json.dumps(
                {
                    "revision": "a" * 40,
                    "manifest_sha256": sha((self.root / "manifest.json").read_bytes()),
                }
            )
        )
        with patch(
            "manifest.git_output", side_effect=["b" * 40, "adjusted/main(unlocked)"]
        ):
            self.assertEqual(verify_corpus(self.root, lock, True).revision, "b" * 40)

    def test_copied_files_verify_without_git_or_annex(self):
        with tempfile.TemporaryDirectory(prefix="chucky-copy-") as directory:
            copy = Path(directory) / "copy"
            shutil.copytree(self.root, copy)
            with patch("manifest.git_output", return_value=None):
                original = self.verify()
                copied = verify_corpus(copy, allow_test_data=True)
            self.assertEqual(original.sha256, copied.sha256)
            self.assertEqual(original.manifest["packs"], copied.manifest["packs"])


class ResultTests(unittest.TestCase):
    def test_incomplete_validation_is_inconclusive(self):
        row = {
            "modality": "brightfield",
            "source_group": "test",
            "width": 2048,
            "height": 2048,
            "backend": "cpu",
            "profile": "none",
            "split": "core",
            "repeats": 3,
        }
        result = assess([row], False)[0]
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["reason"], "Corpus has no heldout sample")

    def test_changed_geometry_is_rejected(self):
        pack = {"width": 256, "height": 256, "bytes": 131072}
        result = {
            "status": "pass",
            "image_replay": {
                "backend": "cpu",
                "codec": "none",
                "dtype": "u16le",
                "shape": [8, 256, 256],
                "chunk_shape": [2, 128, 256],
            },
        }
        with self.assertRaisesRegex(ValueError, "chunk_shape"):
            check_result(result, pack, 8, "cpu", "none")

    def test_clock_disagreement_is_rejected(self):
        pack = {"width": 64, "height": 64, "bytes": 32768}
        result = {
            "status": "pass",
            "image_replay": {
                "backend": "cpu",
                "codec": "none",
                "dtype": "u16le",
                "shape": [4, 64, 64],
                "chunk_shape": [4, 64, 64],
                "target_batch_bytes": 64 * 1024**2,
                "source_bytes": 32768,
                "order": "cyclic",
                "codec_level": 0,
                "shuffle": "none",
            },
            "worker_threads": 4,
            "input_bytes": 32768,
            "wall_s": 883.0,
            "throughput_in_gibs": 0.001,
            "logical_compression_fold": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "clock disagrees"):
            check_result(result, pack, 4, "cpu", "none", process_wall_s=19.0)


if __name__ == "__main__":
    unittest.main()

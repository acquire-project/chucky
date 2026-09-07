import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numcodecs
    import numpy as np
    from extract import build, read_plane, survey
    from manifest import verify_corpus
except ImportError as error:
    raise unittest.SkipTest(
        "Extraction tests require requirements-extract.txt"
    ) from error


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="chucky-extract-")
        self.root = Path(self.directory.name)
        self.image = np.arange(30, dtype="<u2").reshape(1, 1, 2, 3, 5)
        self.codec = numcodecs.Blosc(cname="zstd", clevel=1, shuffle=2)
        (self.root / ".zarray").write_text(
            json.dumps(
                {
                    "zarr_format": 2,
                    "shape": [1, 1, 2, 3, 5],
                    "chunks": [1, 1, 2, 3, 5],
                    "dtype": "<u2",
                    "compressor": self.codec.get_config(),
                    "filters": None,
                    "fill_value": 0,
                    "order": "C",
                    "dimension_separator": ".",
                }
            )
        )
        self.chunk = self.root / "0.0.0.0.0"
        self.chunk.write_bytes(self.codec.encode(self.image))
        self.candidate = {
            "reader": "zarr2",
            "path": str(self.root),
            "axes": "TCZYX",
            "coordinates": {"t": 0, "c": 0, "z": 1},
        }

    def tearDown(self):
        self.directory.cleanup()

    def test_selects_exact_plane_without_rescaling(self):
        actual, details = read_plane(self.candidate)
        np.testing.assert_array_equal(actual, self.image[0, 0, 1])
        self.assertEqual(actual.dtype, np.dtype("<u2"))
        self.assertEqual(len(details["chunks"]), 1)

    def test_missing_chunk_is_not_returned_as_zeros(self):
        self.chunk.unlink()
        with self.assertRaisesRegex(ValueError, "refusing fill values"):
            read_plane(self.candidate)

    def test_truncated_decoded_chunk_is_rejected(self):
        self.chunk.write_bytes(self.codec.encode(self.image.ravel()[:-1]))
        with self.assertRaisesRegex(ValueError, "Truncated"):
            read_plane(self.candidate)

    def test_float_source_is_not_cast_to_uint16(self):
        path = self.root / ".zarray"
        metadata = json.loads(path.read_text())
        metadata["dtype"] = "<f4"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "uint16 camera"):
            read_plane(self.candidate)

    def test_out_of_range_coordinates_fail(self):
        self.candidate["coordinates"]["z"] = 2
        with self.assertRaisesRegex(ValueError, "outside"):
            read_plane(self.candidate)

    def make_survey(self, kind="synthetic-test"):
        corpus = self.root / "corpus"
        corpus.mkdir()
        evidence = b"Synthetic extraction fixture, not microscope data.\n"
        if kind == "provisional":
            evidence = json.dumps(
                {
                    "raw_status": "unverified",
                    "allowed_for_raw_release": False,
                    "missing_evidence": "Test fixture emulating unverified acquisition provenance",
                }
            ).encode()
        (corpus / "evidence.txt").write_bytes(evidence)
        source = {
            "raw_status": "unverified" if kind == "provisional" else "synthetic-test",
            "evidence": "evidence.txt",
            "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
        }
        plan = {
            "kind": kind,
            "release": "poc-fixture",
            "selection_reason": "Use every planned test plane",
            "sources": {"fixture": source},
            "candidates": [],
        }
        for modality in ("brightfield", "fluorescence"):
            for index in range(8):
                plan["candidates"].append(
                    self.candidate
                    | {
                        "id": f"{modality}-{index}",
                        "source_id": "fixture",
                        "source_group": "fixture",
                        "field_id": f"field-{index}",
                        "modality": modality,
                        "coordinates": {"t": 0, "c": 0, "z": index % 2},
                    }
                )
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        output = self.root / "survey"
        survey(
            SimpleNamespace(
                plan=plan_path,
                output=output,
                candidates_per_modality=8,
                allow_test_data=True,
                allow_provisional=kind == "provisional",
            )
        )
        return SimpleNamespace(
            survey=output,
            corpus=corpus,
            core_per_modality=2,
            heldout_per_modality=1,
            allow_test_data=True,
            allow_provisional=kind == "provisional",
        )

    def test_full_extraction_preserves_pixels_and_field_splits(self):
        args = self.make_survey()
        build(args)
        result = verify_corpus(args.corpus, allow_test_data=True)
        self.assertEqual(result.manifest["kind"], "synthetic-test")
        self.assertEqual(result.manifest["release"], "test")
        self.assertEqual(len(result.manifest["packs"]), 4)
        fields = {"core": set(), "heldout": set()}
        for pack in result.manifest["packs"]:
            pixels = np.frombuffer(
                (args.corpus / pack["path"]).read_bytes(), dtype="<u2"
            ).reshape(-1, 3, 5)
            for plane, actual in zip(pack["planes"], pixels, strict=True):
                np.testing.assert_array_equal(
                    actual, self.image[0, 0, plane["coordinates"]["z"]]
                )
                fields[pack["split"]].add(plane["field_id"])
        self.assertFalse(fields["core"] & fields["heldout"])
        selection = result.manifest["selection"]
        saved_survey = args.corpus / selection["survey_path"]
        self.assertEqual(
            saved_survey.read_bytes(), (args.survey / "survey.json").read_bytes()
        )
        saved_survey.write_text("changed selection evidence")
        with self.assertRaisesRegex(ValueError, "Selection survey checksum"):
            verify_corpus(args.corpus, allow_test_data=True)

    def test_provisional_keeps_all_planned_planes_and_requires_opt_in(self):
        args = self.make_survey("provisional")
        args.allow_provisional = False
        with self.assertRaisesRegex(ValueError, "allow-provisional"):
            build(args)
        self.assertFalse((args.corpus / "data").exists())
        args.allow_provisional = True
        build(args)
        result = verify_corpus(args.corpus, allow_provisional=True)
        self.assertEqual(result.manifest["release"], "poc-fixture")
        self.assertEqual(len(result.manifest["packs"]), 2)
        for pack in result.manifest["packs"]:
            self.assertEqual(pack["split"], "core")
            self.assertEqual(len(pack["planes"]), 8)
            pixels = np.frombuffer(
                (args.corpus / pack["path"]).read_bytes(), dtype="<u2"
            ).reshape(-1, 3, 5)
            for plane, actual in zip(pack["planes"], pixels, strict=True):
                np.testing.assert_array_equal(
                    actual, self.image[0, 0, plane["coordinates"]["z"]]
                )

    def test_partial_survey_and_unverified_source_cannot_make_a_corpus(self):
        args = self.make_survey()
        survey_path = args.survey / "survey.json"
        document = json.loads(survey_path.read_text())
        document["status"] = "running"
        survey_path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "completed survey"):
            build(args)
        self.assertFalse((args.corpus / "data").exists())
        document.update(status="complete", kind="raw")
        document["sources"]["fixture"]["raw_status"] = "unverified"
        survey_path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "not verified raw"):
            build(args)
        self.assertFalse((args.corpus / "data").exists())


if __name__ == "__main__":
    unittest.main()

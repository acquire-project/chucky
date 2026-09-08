import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from manifest import digest_file
from opencell import catalog, download, make_plan

try:
    import numpy as np
    import tifffile
    from extract import read_plane
except ImportError as error:
    raise unittest.SkipTest(
        "OpenCell extraction tests require requirements-extract.txt"
    ) from error


class Response(io.BytesIO):
    def __init__(self, data, etag='"original"'):
        super().__init__(data)
        self.headers = {"ETag": etag}


class OpenCellTests(unittest.TestCase):
    def test_catalog_uses_distinct_gene_ids_and_only_original_stacks(self):
        objects = []
        for index in range(5):
            for field in range(2):
                for kind in ["stack", "proj"]:
                    key = f"microscopy/raw/G{index}_ENSG{index:011d}/OC-FOV_G{index}_ENSG{index:011d}_CID000001_FID{index * 2 + field:08d}_{kind}.tif"
                    objects.append(
                        f"<Contents><Key>{key}</Key><Size>100</Size><ETag>original</ETag><LastModified>2021-01-01</LastModified></Contents>"
                    )
        body = (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
            + "".join(objects)
            + "</ListBucketResult>"
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalog.json"
            with patch("opencell.urlopen", return_value=Response(body)):
                catalog(
                    SimpleNamespace(output=output, fields=4, release="opencell-test")
                )
            result = json.loads(output.read_text())
        self.assertEqual(result["catalog_stack_count"], 10)
        self.assertEqual(result["catalog_target_count"], 5)
        self.assertEqual(len({f["ensg"] for f in result["files"]}), 4)
        self.assertTrue(all(f["key"].endswith("_stack.tif") for f in result["files"]))

    def test_download_reuses_only_checksum_verified_source_files(self):
        key = "microscopy/raw/TEST_ENSG00000000001/OC-FOV_TEST_ENSG00000000001_CID000001_FID00000001_stack.tif"
        content = b"Original TIFF source bytes"
        record = {"key": key, "bytes": len(content), "etag": '"original"'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            with patch("opencell.urlopen", return_value=Response(content)) as request:
                downloaded = download(record, root, {})
            self.assertEqual(
                request.call_args.args[0].get_header("If-match"), '"original"'
            )
            self.assertEqual(
                downloaded["sha256"], digest_file(root / downloaded["path"])
            )
            with patch(
                "opencell.urlopen", side_effect=AssertionError("Unexpected redownload")
            ):
                repeated = download(downloaded, root, {})
            self.assertEqual(repeated, downloaded)
            (root / downloaded["path"]).write_bytes(b"changed" + content[7:])
            with self.assertRaisesRegex(ValueError, "checksum"):
                download(downloaded, root, {})

    def test_download_rejects_changed_remote_objects(self):
        record = {
            "key": "microscopy/raw/TEST_ENSG00000000001/OC-FOV_TEST_ENSG00000000001_CID000001_FID00000001_stack.tif",
            "bytes": 3,
            "etag": '"original"',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            with patch("opencell.urlopen", return_value=Response(b"new", '"changed"')):
                with self.assertRaisesRegex(ValueError, "object changed"):
                    download(record, root, {})
            self.assertEqual(list((root / "images").iterdir()), [])

    def test_download_keeps_an_existing_partial_file(self):
        key = "microscopy/raw/TEST_ENSG00000000001/OC-FOV_TEST_ENSG00000000001_CID000001_FID00000001_stack.tif"
        record = {"key": key, "bytes": 3, "etag": '"original"'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            partial = (root / "images" / Path(key).name).with_suffix(".part")
            partial.write_bytes(b"Previous interrupted download")
            with patch("opencell.urlopen", return_value=Response(b"raw")):
                result = download(record, root, {})
            self.assertEqual(partial.read_bytes(), b"Previous interrupted download")
            self.assertEqual((root / result["path"]).read_bytes(), b"raw")

    def test_plan_reads_correct_pixels_for_both_channel_orders(self):
        for axes in ["CZYX", "ZCYX"]:
            with self.subTest(axes=axes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                corpus = root / "corpus"
                (corpus / "provenance").mkdir(parents=True)
                evidence = {"raw_status": "verified", "allowed_for_raw_release": True}
                (corpus / "provenance/opencell-test.json").write_text(
                    json.dumps(evidence)
                )
                (corpus / "provenance/opencell-test-NOTICE.md").write_text(
                    "Test attribution"
                )
                files, originals = [], {}
                for field in range(4):
                    images = (
                        (np.arange(2 * 16 * 3 * 5) + field * 1000)
                        .astype("<u2")
                        .reshape(2, 16, 3, 5)
                    )
                    stored = images if axes == "CZYX" else images.transpose(1, 0, 2, 3)
                    path = root / f"field-{field}.tif"
                    tifffile.imwrite(
                        path, stored, metadata={"axes": axes}, photometric="minisblack"
                    )
                    identity = f"FID{field:08d}"
                    originals[identity.lower()] = images
                    files.append(
                        {
                            "path": path.name,
                            "sha256": digest_file(path),
                            "gene": f"G{field}",
                            "ensg": f"ENSG{field:011d}",
                            "cell_line": "CID000001",
                            "field": identity,
                            "url": f"https://example.invalid/{path.name}",
                        }
                    )
                downloads = root / "downloads.json"
                downloads.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "release": "opencell-test",
                            "files": files,
                            "catalog_stack_count": 4,
                            "catalog_target_count": 4,
                            "selection": "Fixture",
                        }
                    )
                )
                make_plan(SimpleNamespace(corpus=corpus, downloads=downloads))
                plan = json.loads((corpus / "source-plan.json").read_text())
                self.assertEqual(len(plan["candidates"]), 8)
                for index, candidate in enumerate(plan["candidates"]):
                    coordinates = candidate["coordinates"]
                    channel, z = index % 2, (index // 2) * 4 + 2
                    self.assertEqual(coordinates["c"], channel)
                    self.assertEqual(coordinates["z"], z)
                    actual, _ = read_plane(
                        candidate | {"path": str(root / candidate["path"])}
                    )
                    field = candidate["field_id"].split("/")[-1].lower()
                    np.testing.assert_array_equal(actual, originals[field][channel, z])
                with self.assertRaisesRegex(ValueError, "replace an existing"):
                    make_plan(SimpleNamespace(corpus=corpus, downloads=downloads))


if __name__ == "__main__":
    unittest.main()

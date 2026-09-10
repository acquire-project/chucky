import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run as runner
from test_manifest import fixture, sha


def command(*args):
    return subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run.py")), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


class RunnerDefaultsTests(unittest.TestCase):
    def test_opencell_throughput_defaults(self):
        self.assertEqual(runner.DEFAULT_REGISTRY.name, "data.json")
        self.assertTrue(runner.DEFAULT_REGISTRY.is_file())
        self.assertEqual(runner.DEFAULT_MIN_GIB, 32)
        self.assertEqual(runner.DEFAULT_REPEATS, 5)
        self.assertEqual(runner.DEFAULT_TIER, "codec")
        self.assertEqual(runner.DEFAULT_CHUNK_BYTES, ("32K",))
        self.assertEqual(
            runner.IMAGE_TIERS["compress"]["chunk_bytes"],
            tuple(runner.CHUNK_BYTES),
        )
        self.assertEqual(
            runner.IMAGE_TIERS["compress"]["backends"], ("gpu", "cpu")
        )
        self.assertEqual(
            runner.IMAGE_TIERS["backend"]["backends"], ("gpu", "cpu")
        )

    def test_explicit_axes_replace_tier_defaults(self):
        args = SimpleNamespace(
            tier="compress",
            backends=["cpu"],
            profiles=["none"],
            chunk_bytes=["16K", "1M"],
        )
        runner.resolve_axes(args)
        self.assertEqual(args.backends, ["cpu"])
        self.assertEqual(args.profiles, ["none"])
        self.assertEqual(args.chunk_bytes, ["16K", "1M"])

        defaults = SimpleNamespace(
            tier="backend", backends=None, profiles=None, chunk_bytes=None
        )
        runner.resolve_axes(defaults)
        self.assertEqual(defaults.backends, ["gpu", "cpu"])
        self.assertEqual(defaults.profiles, list(runner.PROFILES))
        self.assertEqual(defaults.chunk_bytes, list(runner.CHUNK_BYTES))


@unittest.skipUnless(
    os.environ.get("CHUCKY_IMAGE_BENCH"),
    "Set CHUCKY_IMAGE_BENCH to the built executable",
)
class RunnerTests(unittest.TestCase):
    def test_reports_compare_and_reject_changed_identity(self):
        with tempfile.TemporaryDirectory(prefix="chucky-runner-") as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            document = fixture(corpus)
            for pack in document["packs"]:
                data = bytes(range(256)) * 512 + bytes(reversed(range(256))) * 512
                (corpus / pack["path"]).write_bytes(data)
                pack.update(width=256, height=256, bytes=len(data), sha256=sha(data))
                plane = pack["planes"][0]
                plane["sha256"] = sha(data[:131072])
                plane2 = copy.deepcopy(plane)
                plane2.update(id=plane["id"] + "-second", sha256=sha(data[131072:]))
                plane2["coordinates"]["z"] = 1
                pack["planes"].append(plane2)
            for pack in list(document["packs"]):
                heldout = copy.deepcopy(pack)
                heldout.update(
                    id=pack["id"] + "-heldout",
                    split="heldout",
                    path=pack["id"] + "-heldout.raw",
                )
                for plane in heldout["planes"]:
                    plane["id"] += "-heldout"
                    plane["field_id"] += "-heldout"
                (corpus / heldout["path"]).write_bytes(
                    (corpus / pack["path"]).read_bytes()
                )
                document["packs"].append(heldout)
            (corpus / "manifest.json").write_text(json.dumps(document))
            results = []
            for index in range(2):
                output = root / f"run-{index}"
                result = command(
                    "run",
                    "--direct-corpus",
                    corpus,
                    "--executable",
                    os.environ["CHUCKY_IMAGE_BENCH"],
                    "--allow-test-data",
                    "--smoke",
                    "--min-gib",
                    "0.004",
                    "--repeats",
                    "1",
                    "--backends",
                    *os.environ.get("CHUCKY_TEST_BACKENDS", "cpu").split(),
                    "--profiles",
                    "none",
                    "--chunk-bytes",
                    "16K",
                    "32K",
                    "--machine",
                    f"fixture-{index}",
                    "--output",
                    output,
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                report = json.loads((output / "results.json").read_text())
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["schema_version"], 2)
                self.assertEqual(report["corpus"]["kind"], "synthetic-test")
                self.assertTrue(
                    all(row["scenario"] == "images" for row in report["runs"])
                )
                self.assertTrue((output / "summary.csv").is_file())
                self.assertTrue(all(row["status"] == "pass" for row in report["runs"]))
                self.assertEqual(
                    {row["chunk_bytes_label"] for row in report["runs"]},
                    {"16K", "32K"},
                )
                self.assertTrue(
                    all(
                        row["status"] == "inconclusive"
                        for row in report["representativeness"]
                    )
                )
                results.append(output / "results.json")
            result = command("compare", *results, "--output", root / "comparison.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            changed = json.loads(results[1].read_text())
            changed["corpus"]["manifest_sha256"] = "f" * 64
            results[1].write_text(json.dumps(changed))
            result = command("compare", *results)
            self.assertEqual(result.returncode, 0, result.stderr)
            changed["summary"][0]["pack_sha256"] = "f" * 64
            results[1].write_text(json.dumps(changed))
            result = command("compare", *results)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("input", result.stderr.lower())
            changed["summary"][0]["pack_sha256"] = json.loads(
                results[0].read_text()
            )["summary"][0]["pack_sha256"]
            changed["corpus"]["dataset"] = {"id": "different-selection"}
            results[1].write_text(json.dumps(changed))
            result = command("compare", *results)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dataset contracts", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()

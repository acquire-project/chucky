import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(runner.DEFAULT_LOCK.name, "opencell.lock.json")
        self.assertTrue(runner.DEFAULT_LOCK.is_file())
        self.assertEqual(runner.DEFAULT_SPLIT, "core")
        self.assertEqual(runner.DEFAULT_MIN_GIB, 32)
        self.assertEqual(runner.DEFAULT_REPEATS, 5)


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
                    "--corpus",
                    corpus,
                    "--executable",
                    os.environ["CHUCKY_IMAGE_BENCH"],
                    "--allow-unpinned",
                    "--allow-test-data",
                    "--smoke",
                    "--min-gib",
                    "0.004",
                    "--repeats",
                    "1",
                    "--backends",
                    *os.environ.get("CHUCKY_TEST_BACKENDS", "cpu").split(),
                    "--split",
                    "all",
                    "--machine",
                    f"fixture-{index}",
                    "--output",
                    output,
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                report = json.loads((output / "results.json").read_text())
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["corpus"]["kind"], "synthetic-test")
                self.assertTrue(
                    all(row["scenario"] == "images" for row in report["runs"])
                )
                self.assertTrue((output / "summary.csv").is_file())
                self.assertTrue(all(row["status"] == "pass" for row in report["runs"]))
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
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("corpus", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()

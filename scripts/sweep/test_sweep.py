# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "rich", "pydantic"]
# ///
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from pydantic import ValidationError

from columnar import decode_runs, pack
from models import CURRENT_VERSION, run_id
from summary import trim_run
from sweep import RunSpec, TIERS, deduplicate, main, run_one


def spec(**overrides):
    return RunSpec(**{
        "scenario": "orca2_single", "codec": "blosc-zstd", "fill": "xor",
        "backend": "cpu", "dtype": "u16", "chunk_label": "256K",
        "blosc_block_bytes": 16384, **overrides,
    })


class BloscSweepTests(unittest.TestCase):
    def test_block_size_is_explicit_and_valid(self):
        values = spec().model_dump()
        del values["blosc_block_bytes"]
        with self.assertRaises(ValidationError):
            RunSpec(**values)
        for value in (None, 0, -1, 127, 715827543, 16384.5, "16384", True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                spec(blosc_block_bytes=value)
        for value in (128, 4097, 16384, 715827542):
            with self.subTest(value=value):
                self.assertEqual(spec(blosc_block_bytes=value).blosc_block_bytes, value)

    def test_raw_identities_are_unchanged(self):
        for sink, throughput, suffix in (
            ("discard", 0, ""), ("fs", 0, "__fs"), ("s3", 100, "__s3__100gbps"),
        ):
            with self.subTest(sink=sink):
                run = spec(codec="zstd", blosc_block_bytes=None, sink=sink,
                           s3_throughput_gbps=throughput)
                self.assertEqual(run.id, "orca2_single__zstd__xor__cpu__u16__256K" + suffix)
                self.assertNotIn("blosc_block_bytes", run.base_result())
        with self.assertRaises(ValidationError):
            spec(codec="zstd")

    def test_block_sizes_have_distinct_identities(self):
        runs = [spec(blosc_block_bytes=value) for value in (16384, 32768)]
        self.assertNotEqual(runs[0].id, runs[1].id)
        self.assertEqual(deduplicate([*runs, runs[0]]), runs)
        for run in runs:
            self.assertEqual(run.base_result()["blosc_block_bytes"], run.blosc_block_bytes)
            self.assertEqual(run_id(run.base_result()), run.id)

    def test_archived_identity_suffixes_are_preserved(self):
        for codec, block in (("zstd", None), ("blosc-zstd", 16384)):
            with self.subTest(codec=codec):
                archived = spec(codec=codec, blosc_block_bytes=block).base_result()
                old_id = archived["id"].split("__blosc-block-")[0] + "__io"
                archived["id"] = old_id
                archived.pop("blosc_block_bytes", None)
                expected = old_id if block is None else old_id + "__blosc-block-unknown"
                self.assertEqual(run_id(archived), expected)
                self.assertEqual(trim_run(archived)["id"], expected)
                self.assertEqual(run_id({**archived, "id": expected}), expected)

    def test_all_tiers_specify_blosc_blocks(self):
        runs = deduplicate([run for matrix in TIERS.values() for run in matrix()])
        self.assertEqual(len(runs), 598)
        for run in runs:
            with self.subTest(id=run.id):
                if run.codec.startswith("blosc-"):
                    self.assertEqual(run.blosc_block_bytes, 16384)
                    self.assertTrue(run.id.endswith("__blosc-block-16384"))
                else:
                    self.assertIsNone(run.blosc_block_bytes)

    def test_command_uses_requested_block_size(self):
        for run in (spec(blosc_block_bytes=4097), spec(codec="zstd", blosc_block_bytes=None)):
            with self.subTest(codec=run.codec), patch("sweep.Path.exists", return_value=True), \
                 patch("sweep.subprocess.run", return_value=subprocess.CompletedProcess(
                     [], 0, '{"status":"pass"}', "")) as execute:
                result = run_one(run, Path("build"))
                cmd = execute.call_args.args[0]
                if run.blosc_block_bytes is None:
                    self.assertNotIn("--blosc-block-bytes", cmd)
                else:
                    self.assertEqual(cmd[cmd.index("--blosc-block-bytes") + 1], "4097")
                    self.assertEqual(result["blosc_block_bytes"], 4097)

    def test_summary_keeps_block_sizes_and_historical_unknown(self):
        new = {**spec().base_result(), "status": "pass"}
        old_id = new["id"].split("__blosc-block-")[0]
        old_known = {**new, "id": old_id}
        old_unknown = {k: v for k, v in old_known.items() if k != "blosc_block_bytes"}
        other = {**spec(blosc_block_bytes=32768).base_result(), "status": "pass"}
        rows = [trim_run(run) for run in (old_unknown, old_known, new, other)]
        strings, blocks = pack([rows])
        restored = decode_runs(blocks[0], strings)
        self.assertEqual(restored, rows)
        self.assertNotIn("blosc_block_bytes", restored[0])
        self.assertTrue(restored[0]["id"].endswith("__blosc-block-unknown"))
        self.assertEqual(restored[1]["id"], restored[2]["id"])
        self.assertEqual(restored[1]["blosc_block_bytes"], 16384)
        self.assertEqual(len({row["id"] for row in restored}), 3)
        self.assertEqual(old_unknown["id"], old_id)
        self.assertNotIn("blosc_block_bytes", old_unknown)

    def test_resume_distinguishes_unknown_and_explicit_sizes(self):
        for previous_block, calls in ((None, 1), (16384, 0), (32768, 1)):
            with self.subTest(previous_block=previous_block), tempfile.TemporaryDirectory() as directory:
                run = spec()
                previous = {**run.base_result(), "status": "pass"}
                previous["id"] = previous["id"].split("__blosc-block-")[0]
                if previous_block is None:
                    del previous["blosc_block_bytes"]
                else:
                    previous["blosc_block_bytes"] = previous_block
                output = Path(directory) / "results.json"
                output.write_text(json.dumps({"version": CURRENT_VERSION, "machine": {},
                                             "runs": [previous]}))
                with patch.dict(TIERS, {"backend": lambda: [run]}), \
                     patch("sweep.git_commit", return_value="abcdef0"), \
                     patch("sweep.run_one", return_value={**run.base_result(), "status": "pass"}) as execute:
                    result = CliRunner().invoke(main, ["--tier", "backend", "-o", str(output)])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(execute.call_count, calls)
                saved = json.loads(output.read_text())["runs"]
                self.assertEqual(saved[0], previous)
                self.assertEqual(len(saved), 1 + calls)


if __name__ == "__main__":
    unittest.main()

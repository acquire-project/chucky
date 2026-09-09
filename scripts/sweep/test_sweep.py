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
from models import CURRENT_VERSION, migrate_results, run_id
from summary import trim_run
from sweep import RunSpec, TIERS, deduplicate, main, run_one, measurement_policy


def spec(**overrides):
    return RunSpec(**{
        "scenario": "orca2_single", "codec": "blosc-zstd", "fill": "xor",
        "backend": "cpu", "dtype": "u16", "chunk_label": "256K",
        "blosc_block_bytes": 16384, **overrides,
    })


class MemcpyTimingTests(unittest.TestCase):
    def test_old_full_timing_is_not_relabelled_as_a_sample(self):
        row = {"count": 128, "in_bytes": 65536, "total_ms": 1}
        data = {"version": 10, "runs": [{"stages": {"memcpy": row.copy()}}]}
        migrate_results(data)
        self.assertEqual(data["version"], CURRENT_VERSION)
        self.assertEqual(data["runs"][0]["stages"], {"memcpy": row})
        self.assertNotIn("memcpy_work", data["runs"][0])

    def test_sample_and_exact_work_survive_columnar_round_trip(self):
        runs = [{"stages": {"memcpy_sample": {"count": 2, "in_bytes": 1024}},
                 "memcpy_work": {"calls": 128, "bytes": 65536,
                                 "timing_scope": "sampled"}}]
        strings, blocks = pack([runs])
        self.assertEqual(decode_runs(blocks[0], strings), runs)


class MeasurementPolicyTests(unittest.TestCase):
    def test_only_qualified_success_is_accepted(self):
        qualified = {"policy": measurement_policy()["policy"],
                     "coverage_status": "sufficient"}
        for window, returncode, expected in (
            (qualified, 0, "pass"),
            (qualified, 1, "error"),
            ({**qualified, "coverage_status": "insufficient"}, 0, "error"),
            ({**qualified, "policy": "drained-warmup-through-final-close-v1"}, 0, "error"),
            (None, 0, "error"),
            ([], 0, "error"),
        ):
            with self.subTest(window=window, returncode=returncode), \
                 patch("sweep.Path.exists", return_value=True), patch(
                     "sweep.subprocess.run", return_value=subprocess.CompletedProcess(
                         [], returncode, json.dumps({"status": "pass", "measurement": window}), "")):
                result = run_one(spec(), Path("build"))
            self.assertEqual(result["status"], expected)
            self.assertEqual(result["measurement"], window)

    def test_coverage_failure_retains_diagnostics(self):
        failed = {"status": "error", "error": "insufficient_coverage",
                  "measurement": {"coverage_status": "insufficient", "attempt": 5}}
        with patch("sweep.Path.exists", return_value=True), patch(
            "sweep.subprocess.run", return_value=subprocess.CompletedProcess(
                [], 1, json.dumps(failed), "")):
            result = run_one(spec(), Path("build"))
        self.assertEqual(result["measurement"], failed["measurement"])
        self.assertEqual(result["error"], failed["error"])
        self.assertEqual(result["returncode"], 1)

    def test_duration_changes_preserve_geometry_reference(self):
        run = spec(codec="none", blosc_block_bytes=None)
        for duration in (1, 5):
            with patch("sweep.Path.exists", return_value=True), patch(
                "sweep.subprocess.run", return_value=subprocess.CompletedProcess(
                    [], 0, '{"status":"pass"}', "")) as execute:
                result = run_one(run, Path("build"), warmup=0.5, duration=duration)
            cmd = execute.call_args.args[0]
            self.assertNotIn("--frames", cmd)
            self.assertEqual(cmd[cmd.index("--geometry-frames") + 1], "200")
            self.assertEqual(cmd[cmd.index("--warmup") + 1], "0.5")
            self.assertEqual(cmd[cmd.index("--duration") + 1], str(duration))
            self.assertEqual(result["geometry_frames"], 200)

    def test_resume_refuses_unknown_or_different_policy(self):
        for previous in (None, measurement_policy(duration=5),
                         {**measurement_policy(), "policy": "drained-warmup-through-final-close-v1"}):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "results.json"
                data = {"version": CURRENT_VERSION, "runs": []}
                if previous is not None:
                    data["measurement_policy"] = previous
                output.write_text(json.dumps(data))
                before = output.read_bytes()
                result = CliRunner().invoke(main, ["--tier", "backend", "-o", str(output)])
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("different or unknown timing policy", result.output)
                self.assertEqual(output.read_bytes(), before)

    def test_summary_retains_policy_and_coverage(self):
        window = {"policy": measurement_policy()["policy"],
                  "coverage_status": "insufficient", "geometry": {
                      "dimensions": [{"reference_size": 200, "chunk_size": 1}]}}
        row = {**spec().base_result(), "status": "pass", "measurement": window}
        trimmed = trim_run(row)
        self.assertEqual(trimmed["measurement"], window)
        strings, blocks = pack([[trimmed]])
        self.assertEqual(decode_runs(blocks[0], strings), [trimmed])


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
                                             "measurement_policy": measurement_policy(),
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

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
from models import CURRENT_VERSION, codec_label, run_id, validate_results
from report import load_files
from summary import trim_run
from sweep import (
    RunSpec, TIERS, backend_runs, blosc_runs, compress_runs, deduplicate,
    main, run_one,
)


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
        self.assertEqual(len(runs), 816)
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
                    self.assertNotIn("--blosc-shuffle", cmd)
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


def filter_spec(**overrides):
    return RunSpec(**{
        "scenario": "orca2_single", "codec": "blosc-lz4", "fill": "xor",
        "backend": "gpu", "dtype": "u16", "chunk_label": "16K",
        "blosc_block_bytes": 16384 if overrides.get("codec", "blosc-lz4").startswith("blosc-") else None,
        **overrides,
    })


class RunSpecTest(unittest.TestCase):
    def test_defaults_keep_archived_identity(self):
        run = filter_spec()
        self.assertEqual(run.id, "orca2_single__blosc-lz4__xor__gpu__u16__16K__blosc-block-16384")
        self.assertEqual((run.blosc_shuffle, run.level), ("none", 3))
        self.assertEqual(filter_spec(codec="lz4").level, 1)
        self.assertEqual(filter_spec(codec="zstd").level, 0)

    def test_filter_and_store_only_identities_do_not_collide(self):
        runs = [filter_spec(blosc_shuffle=s, level=level, blosc_block_bytes=block)
                for s in ("none", "byte", "bit") for level in (0, 3)
                for block in (4096, 16384)]
        self.assertEqual(len(deduplicate(runs * 2)), 12)
        self.assertIn("__shuffle-bit__level-0", filter_spec(blosc_shuffle="bit", level=0).id)
        self.assertIn("__fs__shuffle-byte", filter_spec(sink="fs", blosc_shuffle="byte").id)
        for run in runs:
            base = run.base_result()
            self.assertEqual((base["blosc_shuffle"], base["blosc_level"]), (run.blosc_shuffle, run.level))
            self.assertNotIn("shuffle", base)
            self.assertNotIn("level", base)
            self.assertEqual(run_id(base), run.id)
            self.assertEqual(trim_run(base)["id"], run.id)

    def test_recorded_settings_normalize_old_or_stale_identity_suffixes(self):
        run = filter_spec(blosc_shuffle="bit", level=0, blosc_block_bytes=4096)
        root = "orca2_single__blosc-lz4__xor__gpu__u16__16K"
        for suffix in ("", "__blosc-block-16384__shuffle-byte__level-3",
                       "__level-0__blosc-block-4096__shuffle-bit"):
            with self.subTest(suffix=suffix):
                archived = {**run.base_result(), "id": root + suffix}
                self.assertEqual(run_id(archived), run.id)
                self.assertEqual(run_id({**archived, "id": run.id}), run.id)

    def test_invalid_settings(self):
        for options in ({"blosc_shuffle": "bad"}, {"level": -1}, {"level": 256},
                        {"level": 10}, {"codec": "zstd", "blosc_shuffle": "byte"},
                        {"codec": "lz4", "level": 0}):
            with self.subTest(options=options), self.assertRaises(ValidationError):
                filter_spec(**options)


class MatrixTest(unittest.TestCase):
    def test_compress_is_subset_of_backend(self):
        self.assertLessEqual({r.id for r in compress_runs()}, {r.id for r in backend_runs()})

    def test_every_tier_can_measure_gpu_blosc(self):
        for name, generate in TIERS.items():
            runs = generate()
            with self.subTest(tier=name):
                self.assertEqual({r.codec for r in runs
                                  if r.backend == "gpu" and r.codec.startswith("blosc-")},
                                 {"blosc-lz4", "blosc-zstd"})
                if name != "blosc":
                    self.assertTrue(all(r.blosc_shuffle == "none" for r in runs))

    def test_focused_blosc_comparison(self):
        runs = blosc_runs()
        self.assertEqual(len(runs), 48)
        self.assertEqual(len(deduplicate(runs)), len(runs))
        self.assertEqual({r.chunk_label for r in runs}, {"16K", "256K", "1M"})
        for backend in ("cpu", "gpu"):
            for codec in ("blosc-lz4", "blosc-zstd"):
                self.assertEqual({r.blosc_shuffle for r in runs
                                  if r.codec == codec and r.backend == backend},
                                 {"none", "byte", "bit"})
            self.assertEqual({r.codec for r in runs if r.backend == backend},
                             {"lz4", "zstd", "blosc-lz4", "blosc-zstd"})

    @patch("sweep.git_commit", return_value="abcdef0")
    def test_cli_overrides_are_blosc_only_and_deduplicated(self, _commit):
        captured = []
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "results.json"
            with patch("sweep.gpu_and_driver", return_value=("test", "test")), \
                 patch("sweep.run_one", side_effect=lambda run, *args, **kwargs:
                       captured.append(run) or {**run.base_result(), "status": "pass"}):
                result = CliRunner().invoke(main, [
                    "--tier", "blosc", "--backend", "gpu", "--blosc-shuffle", "bit",
                    "--level", "0", "--output", str(result_file),
                ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(len(captured), 12)
            for run in captured:
                if run.codec.startswith("blosc-"):
                    self.assertEqual((run.blosc_shuffle, run.level), ("bit", 0))
                else:
                    self.assertEqual(run.blosc_shuffle, "none")
                    self.assertEqual(run.level, 1 if run.codec == "lz4" else 0)


class RunnerAndReportTest(unittest.TestCase):
    @patch("sweep.Path.exists", return_value=True)
    @patch("sweep.subprocess.run", return_value=subprocess.CompletedProcess(
        [], 0, '{"status":"pass","blosc_shuffle":"bit","blosc_level":0}', ""))
    def test_runner_forwards_flags_and_records_settings(self, execute, _exists):
        result = run_one(filter_spec(blosc_shuffle="bit", level=0), Path("build"))
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index("--blosc-shuffle") + 1], "bit")
        self.assertEqual(command[command.index("--level") + 1], "0")
        self.assertEqual(command[command.index("--blosc-block-bytes") + 1], "16384")
        self.assertEqual((result["blosc_shuffle"], result["blosc_level"]), ("bit", 0))
        self.assertEqual(result["blosc_block_bytes"], 16384)

    def test_resume_matches_all_three_settings(self):
        current = filter_spec(blosc_shuffle="bit", level=0, blosc_block_bytes=4096)
        for block, shuffle, level, calls in ((4096, "bit", 0, 0),
                                           (16384, "bit", 0, 1),
                                           (4096, "byte", 0, 1),
                                           (4096, "bit", 3, 1)):
            with self.subTest(block=block, blosc_shuffle=shuffle, level=level), \
                 tempfile.TemporaryDirectory() as directory:
                previous = {**filter_spec(blosc_block_bytes=block, blosc_shuffle=shuffle,
                                          level=level).base_result(), "status": "pass"}
                output = Path(directory) / "results.json"
                output.write_text(json.dumps({"version": CURRENT_VERSION, "machine": {},
                                             "runs": [previous]}))
                with patch.dict(TIERS, {"blosc": lambda: [current]}), \
                     patch("sweep.git_commit", return_value="abcdef0"), \
                     patch("sweep.run_one", return_value={**current.base_result(),
                                                         "status": "pass"}) as execute:
                    result = CliRunner().invoke(main, ["--tier", "blosc", "-o", str(output)])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(execute.call_count, calls)
                saved = json.loads(output.read_text())["runs"]
                self.assertEqual(saved[0], previous)
                self.assertEqual(len(saved), 1 + calls)

    def test_archives_and_variants_remain_distinct_in_reports(self):
        archived = {**filter_spec().base_result(), "status": "pass"}
        del archived["blosc_shuffle"], archived["blosc_level"]
        variant = {**filter_spec(blosc_shuffle="bit", level=0).base_result(), "status": "pass"}
        data = {"version": CURRENT_VERSION, "machine": {}, "runs": [archived, variant]}
        validated = validate_results(data)
        self.assertIsNone(validated.runs[0].blosc_shuffle)
        self.assertIsNone(validated.runs[0].blosc_level)
        self.assertEqual(codec_label(archived), "blosc-lz4")
        self.assertEqual(codec_label(filter_spec().base_result()), "blosc-lz4")
        self.assertEqual(codec_label(variant), "blosc-lz4 (bit, level 0)")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-abcdef0-20260904.json"
            path.write_text(json.dumps(data))
            loaded = load_files([path])[0][1]["runs"]
        self.assertEqual({r["codec_label"] for r in loaded},
                         {"blosc-lz4", "blosc-lz4 (bit, level 0)"})
        trimmed = trim_run(variant)
        self.assertEqual((trimmed["blosc_shuffle"], trimmed["blosc_level"]), ("bit", 0))
        self.assertEqual(trimmed["codec_label"], codec_label(variant))


if __name__ == "__main__":
    unittest.main()

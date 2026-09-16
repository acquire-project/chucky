# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "rich", "pydantic"]
# ///
import errno
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner
from pydantic import ValidationError

from columnar import decode_runs, pack
from models import CURRENT_VERSION, codec_label, run_id, validate_results
from report import load_files
from summary import trim_run
from measurements import aggregate_repetitions, measurement_policy
from test_measurements import execution
from workloads import DEFAULT_WORKLOADS, load_workloads
from sweep import (
    DEFAULT_DATA_REGISTRY, MICROSCOPY_SCENARIO, RunSpec, TIERS, _image_modules, backend_runs,
    blosc_runs, compress_runs, deduplicate, execute_with_sink, image_execution_count, image_runs,
    load_image_members, main, run_one,
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

    def test_cli_rejects_invalid_block_sizes(self):
        for value in ("0", "127", "715827543", "1G", "1.5K", "-1", "16KiB", ""):
            with self.subTest(value=value):
                result = CliRunner().invoke(main, ["--blosc-block-bytes", value, "--dry-run"])
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("--blosc-block-bytes", result.output)

    def test_image_depth_and_sinks_have_distinct_identities(self):
        runs = [image_spec(chunk_depth=depth, sink=sink)
                for depth in (1, 4) for sink in ("discard", "fs", "s3")]
        self.assertEqual(len({run.id for run in runs}), 6)
        for run in runs:
            self.assertEqual(run_id(run.base_result()), run.id)
            self.assertEqual(trim_run(run.base_result())["chunk_depth"], run.chunk_depth)
        with self.assertRaises(ValidationError):
            spec(chunk_depth=1)

    def test_filesystem_executions_use_fresh_directories_and_clean_up(self):
        directories = []

        def execute(command):
            directory = Path(command[command.index("-o") + 1])
            self.assertTrue(directory.is_dir())
            (directory / "data").write_bytes(b"test")
            directories.append(directory)
            return execution()

        with tempfile.TemporaryDirectory() as root:
            with patch("sweep.execute", side_effect=execute):
                for _ in range(2):
                    result = execute_with_sink(["bench"], image_spec(sink="fs"), Path(root))
                    self.assertEqual(result["fs_root"], str(Path(root).resolve()))
            self.assertNotEqual(directories[0], directories[1])
            self.assertTrue(all(not directory.exists() for directory in directories))
            with patch("sweep.execute", side_effect=subprocess.TimeoutExpired("bench", 1)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    execute_with_sink(["bench"], image_spec(sink="fs"), Path(root))
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_filesystem_cleanup_retries_without_repeating_measurement(self):
        cleanup = tempfile.TemporaryDirectory.cleanup
        errors = iter((errno.EBUSY, errno.ENOTEMPTY, None))

        def remove(directory):
            error = next(errors)
            if error is not None:
                raise OSError(error, "Temporary filesystem error")
            cleanup(directory)

        with tempfile.TemporaryDirectory() as root:
            with patch("sweep.tempfile.TemporaryDirectory.cleanup", autospec=True, side_effect=remove), \
                 patch("sweep.time.sleep") as sleep, \
                 patch("sweep.execute", return_value=execution()) as execute:
                result = execute_with_sink(["bench"], image_spec(sink="fs"), Path(root))
                self.assertEqual(result["fs_root"], str(Path(root).resolve()))
                execute.assert_called_once()
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.1, 0.2])
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_filesystem_cleanup_stops_on_persistent_or_other_errors(self):
        for code, attempts in ((errno.EBUSY, 8), (errno.EPERM, 1)):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as root:
                with patch("sweep.tempfile.TemporaryDirectory.cleanup", side_effect=OSError(code, "Cleanup failed")) as cleanup, \
                     patch("sweep.time.sleep") as sleep, \
                     patch("sweep.execute", return_value=execution()) as execute:
                    with self.assertRaises(OSError) as raised:
                        execute_with_sink(["bench"], image_spec(sink="fs"), Path(root))
                    self.assertEqual(raised.exception.errno, code)
                    execute.assert_called_once()
                    self.assertEqual(cleanup.call_count, attempts)
                    self.assertEqual(sleep.call_count, attempts - 1)

    def test_image_s3_execution_records_the_destination(self):
        case = image_spec(sink="s3", chunk_depth=1, s3_throughput_gbps=5)
        with patch("sweep.execute", return_value=execution()) as execute:
            result = execute_with_sink(
                ["bench"], case, s3_bucket="test", s3_region="test-region",
                s3_endpoint="http://localhost:9000",
            )
            command = execute.call_args.args[0]
            self.assertEqual(command[command.index("--s3-prefix") + 1], f"bench/{case.id}")
            self.assertEqual(command[command.index("--s3-throughput-gbps") + 1], "5.0")
            self.assertEqual(result["s3_bucket"], "test")
            self.assertEqual(result["s3_endpoint"], "http://localhost:9000")
            self.assertEqual(execute_with_sink(["bench"], case)["status"], "error")
            self.assertEqual(execute.call_count, 1)

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
                                             "measurement_policy": {**measurement_policy(), "geometry_frames": None},
                                             "repetition_policy": {"generated": 1, "images": 5},
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


def image_spec(**overrides):
    return RunSpec(**{
        "scenario": MICROSCOPY_SCENARIO, "codec": "blosc-zstd", "fill": "images",
        "input_id": "opencell-dna", "image_asset_id": "opencell-dna",
        "image_split": "core",
        "backend": "cpu", "dtype": "u16", "chunk_label": "256K",
        "blosc_block_bytes": 16384, "blosc_shuffle": "bit", "level": 3,
        **overrides,
    })


class ThreadLimitTests(unittest.TestCase):
    def test_thread_limits_are_part_of_identity_and_survive_validation(self):
        from models import run_id
        base = image_spec()
        ids = {base.id}
        for threads in (4, 8, 16, 32):
            case = RunSpec(**{**base.model_dump(), "max_threads": threads})
            result = case.base_result()
            self.assertEqual(result["max_threads"], threads)
            self.assertEqual(run_id(result), case.id)
            ids.add(case.id)
        self.assertEqual(len(ids), 5)
        for threads in (0, -1, True, 4.5, 2147483648):
            with self.assertRaises(ValueError):
                RunSpec(**{**base.model_dump(), "max_threads": threads})


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

    def test_image_identity_includes_semantic_and_physical_input(self):
        run = image_spec()
        base = run.base_result()
        self.assertEqual(base["input_id"], "opencell-dna")
        self.assertEqual(base["image_asset_id"], "opencell-dna")
        self.assertNotIn("frames", base)
        self.assertIn("__opencell-dna__opencell-dna__", run.id)
        with self.assertRaises(ValidationError):
            image_spec(input_id=None)
        with self.assertRaises(ValidationError):
            image_spec(fill="xor")
        with self.assertRaises(ValidationError):
            filter_spec(input_id="opencell-dna")


class MicroscopyTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="chucky-sweep-corpus-")
        self.addCleanup(directory.cleanup)
        self.corpus = Path(directory.name)
        datasets = []
        for name, modality, dtype, assets in (
            ("opencell", "fluorescence", "uint16", ("opencell-dna", "opencell-protein")),
            ("bbbc010", "brightfield", "uint16", ("bbbc010-brightfield",)),
            ("jump-scope", "fluorescence", "uint16", ("jump-scope-fluorescence",)),
            ("dynacell", "quantitative-phase", "float32", ("dynacell-a549-phase",)),
            ("cosem", "electron-microscopy", "uint8", ("cosem-cos7-em",)),
            ("bbbc022", "fluorescence", "uint16", ("bbbc022-mito",)),
        ):
            datasets.append({
                "id": name, "name": name, "version": 2 if name == "cosem" else 1, "modality": modality,
                "source": {},
                "assets": [{
                    "id": asset, "name": asset, "path": f"{asset}.raw",
                    "dtype": dtype, "shape": {"cosem": [32, 1024, 1024],
                                             "bbbc022": [16, 520, 696]}.get(name, [1, 2, 3]),
                    "sha256": "0" * 64,
                } for asset in assets],
            })
        document = {
            "id": "microscopy-core", "version": 3, "name": "Test microscopy metadata",
            "format": {
                "name": "chucky-image-corpus", "version": 2, "encoding": "raw",
                "byte_order": "little", "axes": ["plane", "y", "x"], "order": "C",
            },
            "datasets": datasets,
        }
        (self.corpus / "manifest.json").write_text(json.dumps(document))


class MatrixTest(MicroscopyTestCase):
    def test_workload_registry_covers_the_generated_matrix(self):
        registry = load_workloads(DEFAULT_WORKLOADS)
        registered = {
            (scenario["id"], input_id)
            for scenario in registry["scenarios"]
            for input_id in scenario["inputs"]
        }
        members = load_image_members(DEFAULT_DATA_REGISTRY, None, self.corpus)
        generated = {
            (run.scenario, run.input_id or run.fill)
            for matrix in TIERS.values()
            for run in matrix()
        }
        generated.update(
            (run.scenario, run.input_id or run.fill)
            for run in image_runs("backend", members)
        )
        self.assertEqual(registered, generated)

    def test_current_stacks_and_legacy_selection_are_distinct(self):
        data_sources, _ = _image_modules()
        dataset, root, document = data_sources.read_dataset(
            DEFAULT_DATA_REGISTRY, source_override=self.corpus)
        packs = {pack["id"]: pack for pack in data_sources.parse_compact_manifest(
            root, document, set(dataset.members))["packs"]}
        for asset, count, height, width, size in (
            ("cosem-cos7-em", 32, 1024, 1024, 32 << 20),
            ("bbbc022-mito", 16, 520, 696, 16 * 520 * 696 * 2),
        ):
            pack = packs[asset]
            self.assertEqual((len(pack["planes"]), pack["height"], pack["width"]),
                             (count, height, width))
            self.assertEqual(pack["bytes"], size)
        self.assertEqual(packs["cosem-cos7-em"]["dataset_version"], 2)
        with self.assertRaisesRegex(ValueError, "version 1.*--corpus"):
            load_image_members(DEFAULT_DATA_REGISTRY, "microscopy-core-v1", self.corpus)
        document["version"] = 1
        document["datasets"] = [item for item in document["datasets"] if item["id"] != "bbbc022"]
        cosem = next(item for item in document["datasets"] if item["id"] == "cosem")
        cosem["version"] = 1
        cosem["assets"][0]["shape"] = [1, 512, 512]
        (self.corpus / "manifest.json").write_text(json.dumps(document))
        legacy = load_image_members(DEFAULT_DATA_REGISTRY, "microscopy-core-v1", self.corpus)
        self.assertEqual(len(legacy), 6)
        self.assertNotIn("bbbc022-mito", {asset for asset, _, _ in legacy})
        with self.assertRaisesRegex(ValueError, "version 3.*--corpus"):
            load_image_members(DEFAULT_DATA_REGISTRY, None, self.corpus)

    def test_compress_is_subset_of_backend(self):
        self.assertLessEqual({r.id for r in compress_runs()}, {r.id for r in backend_runs()})
        members = [("asset", "input", "u16")]
        self.assertEqual(
            {r.id for r in image_runs("compress", members)},
            {r.id for r in image_runs("backend", members)},
        )

    def test_image_matrix_covers_chunks_inputs_codecs_and_backends(self):
        members = load_image_members(DEFAULT_DATA_REGISTRY, None, self.corpus)
        runs = image_runs("backend", members)
        self.assertEqual(len(runs), 560)
        self.assertEqual({run.chunk_label for run in runs}, {
            "16K", "32K", "64K", "128K", "256K", "512K", "1M", "2M",
        })
        self.assertEqual({run.backend for run in runs}, {"cpu", "gpu"})
        self.assertEqual({run.codec for run in runs}, {
            "none", "lz4", "zstd", "blosc-lz4", "blosc-zstd",
        })
        self.assertEqual({run.input_id for run in runs}, {
            "opencell-dna", "opencell-protein", "bbbc010-brightfield",
            "jump-scope-fluorescence", "dynacell-a549-phase", "cosem-cos7-em", "bbbc022-mito",
        })
        self.assertEqual({(run.input_id, run.dtype) for run in runs}, {
            ("opencell-dna", "u16"), ("opencell-protein", "u16"),
            ("bbbc010-brightfield", "u16"), ("jump-scope-fluorescence", "u16"),
            ("dynacell-a549-phase", "f32"), ("cosem-cos7-em", "u8"), ("bbbc022-mito", "u16"),
        })
        self.assertEqual(image_execution_count(runs, 5, False), 2800)
        for run in runs:
            if run.codec.startswith("blosc-"):
                self.assertEqual(run.blosc_block_bytes, run.chunk_bytes)
                self.assertEqual(run.base_result()["blosc_block_bytes"], run.chunk_bytes)
                self.assertIn(f"__blosc-block-{run.chunk_bytes}", run.id)
            else:
                self.assertIsNone(run.blosc_block_bytes)

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
                    "--blosc-block-bytes", "16k", "--blosc-block-bytes", "16384",
                    "--blosc-block-bytes", "64K", "--blosc-block-bytes", "chunk",
                ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(len(captured), 22)
            self.assertEqual(len({run.id for run in captured}), len(captured))
            for chunk in (16 << 10, 256 << 10, 1 << 20):
                self.assertEqual(
                    {run.blosc_block_bytes for run in captured
                     if run.codec.startswith("blosc-") and run.chunk_bytes == chunk},
                    {16 << 10, 64 << 10, chunk},
                )
            for run in captured:
                if run.codec.startswith("blosc-"):
                    self.assertEqual((run.blosc_shuffle, run.level), ("bit", 0))
                else:
                    self.assertEqual(run.blosc_shuffle, "none")
                    self.assertEqual(run.level, 1 if run.codec == "lz4" else 0)


class RunnerAndReportTest(MicroscopyTestCase):
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
                                             "measurement_policy": {**measurement_policy(), "geometry_frames": None},
                                             "repetition_policy": {"generated": 1, "images": 5},
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

    def test_image_case_uses_repetitions_and_returns_one_sweep_row(self):
        cases = [(image_spec(dtype=dtype, blosc_block_bytes=block, chunk_depth=1),
                  bpe, report_dtype, 8, False, False, 8)
                 for dtype, bpe, report_dtype, block in
                 (("u8", 1, "u8", 4096), ("u16", 2, "u16le", 65536),
                  ("f32", 4, "f32le", 262144))]
        cases.extend((image_spec(backend="cpu", max_threads=threads), 2, "u16le", 8, False, False, 8)
                     for threads in (8, 16, 32))
        cases.extend(
            (image_spec(backend=backend, codec="none", blosc_block_bytes=None,
                        blosc_shuffle="none", level=0),
             2, "u16le", requested, smoke, calibration, minimum)
            for backend, requested, smoke, calibration, minimum in
            (("cpu", 8, False, False, 32), ("cpu", 64, False, False, 64),
             ("cpu", 4, True, False, 4), ("cpu", 4, False, True, 4),
             ("gpu", 8, False, False, 8))
        )
        for case, bpe, report_dtype, requested, smoke, calibration, minimum in cases:
            dtype = case.dtype
            pack = {
                "id": "opencell-dna",
                "input_id": "opencell-dna",
                "source_group": "opencell-dna",
                "sha256": "a" * 64,
                "width": 600,
                "height": 600,
                "bytes": 6 * 600 * 600 * bpe,
                "dtype": report_dtype,
                "modality": "fluorescence",
                "dataset_id": "opencell",
                "dataset_version": 2,
                "split": "core",
                "planes": [{"id": f"plane-{index}"} for index in range(6)],
            }
            corpus = SimpleNamespace(
                manifest={"kind": "raw", "release": "opencell", "packs": [pack]},
                pack_files={"opencell-dna": Path("/data/opencell-dna.raw")},
                sha256="b" * 64,
            )
            rates = iter((5, 9, 7))

            def execute(*_args, **_kwargs):
                return execution(next(rates), image_replay={"load_s": 0.1, "shape": [100, 600, 600]})

            layout = {"chunk_shape": [1, 256, 512]}
            with patch("sweep.Path.exists", return_value=True), \
                 patch("sweep.execute", side_effect=execute) as process, \
                 patch("sweep.check_image_result", return_value=layout) as check:
                result = run_one(
                    case, Path("build"), image_corpus=corpus, image_min_gib=requested,
                    image_repeats=3, image_smoke=smoke, image_calibration=calibration,
                    image_layouts={},
                )

            self.assertEqual(process.call_count, 3)
            self.assertEqual(check.call_count, 3)
            command = process.call_args.args[0]
            self.assertEqual(Path(command[0]).stem, "bench_stream_microscopy")
            self.assertEqual(command[command.index("--dtype") + 1], dtype)
            expected_frames = ((minimum << 30) + 600 * 600 * bpe - 1) // (600 * 600 * bpe)
            self.assertEqual(int(command[command.index("--frames") + 1]), expected_frames)
            self.assertEqual(check.call_args.args[2], expected_frames)
            self.assertNotIn("--geometry-frames", command)
            self.assertNotIn("--max-attempts", command)
            self.assertEqual(command[command.index("--concurrent-shards") + 1], "16")
            self.assertEqual(int(command[command.index("--max-threads") + 1]), case.max_threads or 4)
            self.assertEqual(result["dtype"], dtype)
            self.assertEqual(Path(command[command.index("--input") + 1]).as_posix(),
                             "/data/opencell-dna.raw")
            self.assertEqual(command[command.index("--chunk-bytes") + 1], "256K")
            if case.codec.startswith("blosc-"):
                self.assertEqual(command[command.index("--shuffle") + 1], "bit")
                self.assertEqual(command[command.index("--blosc-block-bytes") + 1],
                                 str(case.blosc_block_bytes))
                self.assertEqual(result["blosc_block_bytes"], case.blosc_block_bytes)
            else:
                self.assertNotIn("--shuffle", command)
            if case.chunk_depth is not None:
                self.assertEqual(command[command.index("--chunk-depth") + 1], str(case.chunk_depth))
            else:
                self.assertNotIn("--chunk-depth", command)
            self.assertEqual(result["scenario"], "microscopy")
            self.assertEqual(result["input_id"], "opencell-dna")
            self.assertIn("(v2)", result["input_label"])
            self.assertEqual(result["image_input"]["dataset_id"], "opencell")
            self.assertEqual(result["image_input"]["dataset_version"], 2)
            self.assertEqual(result["throughput_in_gibs"], 7)
            self.assertEqual(result["compression_fold"], 2)
            self.assertEqual(result["repetitions"]["count"], 3)
            self.assertEqual(result["repetitions"]["warmups"], 0)
            self.assertEqual(result["repetitions"]["detail_repeat"], 3)
            self.assertEqual(result["repetitions"]["throughput_gibs"], [5, 9, 7])
            self.assertEqual(result["stages"]["compress"]["avg_ms"], 7)
            self.assertEqual(result["id"], run_id(result))

    @patch("sweep.git_commit", return_value="abcdef0")
    def test_cli_can_mix_images_with_an_ordinary_scenario(self, _commit):
        runner = CliRunner()
        image_only = runner.invoke(main, [
            "--tier", "compress", "--scenario", "microscopy", "--corpus", str(self.corpus), "--dry-run",
        ])
        self.assertEqual(image_only.exit_code, 0, image_only.output)
        self.assertIn("560 configurations", image_only.output)
        self.assertIn("2800 process executions", image_only.output)

        mixed = runner.invoke(main, [
            "--tier", "compress", "--scenario", "orca2_single",
            "--scenario", "microscopy", "--corpus", str(self.corpus), "--dry-run",
        ])
        self.assertEqual(mixed.exit_code, 0, mixed.output)
        self.assertIn("600 configurations", mixed.output)
        self.assertIn("2840 process executions", mixed.output)

        cpu_compress = runner.invoke(main, [
            "--tier", "compress", "--backend", "cpu",
            "--scenario", "microscopy", "--corpus", str(self.corpus), "--dry-run",
        ])
        self.assertEqual(cpu_compress.exit_code, 0, cpu_compress.output)
        self.assertIn("280 configurations", cpu_compress.output)
        self.assertIn("1400 process executions", cpu_compress.output)

        blocks = runner.invoke(main, [
            "--tier", "backend", "--scenario", "microscopy", "--corpus", str(self.corpus),
            "--backend", "cpu", "--input", "opencell-dna",
            "--chunk-bytes", "64K", "--chunk-bytes", "256K",
            "--blosc-block-bytes", "16K", "--blosc-block-bytes", "65536",
            "--blosc-block-bytes", "chunk", "--blosc-block-bytes", "16384", "--dry-run",
        ])
        self.assertEqual(blocks.exit_code, 0, blocks.output)
        self.assertIn("16 configurations", blocks.output)
        self.assertIn("80 process executions", blocks.output)

        calibration = runner.invoke(main, [
            "--tier", "backend", "--scenario", "microscopy", "--corpus", str(self.corpus),
            "--backend", "cpu", "--input", "opencell-dna", "--codec", "blosc-lz4",
            "--chunk-bytes", "256K", "--chunk-depth", "1",
            "--sink", "discard", "--sink", "fs", "--calibration",
            "--min-gib", "0.125", "--repeats", "1", "--dry-run",
        ])
        self.assertEqual(calibration.exit_code, 0, calibration.output)
        self.assertIn("2 configurations", calibration.output)
        self.assertIn("2 process executions", calibration.output)

    @patch("sweep.git_commit", return_value="abcdef0")
    def test_image_work_budget_retains_qualified_run_limits(self, _commit):
        arguments = [
            "--tier", "backend", "--scenario", "microscopy",
            "--corpus", str(self.corpus), "--dry-run",
        ]
        runner = CliRunner()
        for options, minimum in (([], 8), (["--min-gib", "8"], 8),
                                 (["--min-gib", "32"], 32)):
            with self.subTest(options=options):
                result = runner.invoke(main, [*arguments, *options])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn(f"Microscopy minimum input per execution: {minimum} GiB",
                              result.output)
                if minimum < 32:
                    self.assertIn("Uncompressed CPU minimum: 32 GiB", result.output)
                else:
                    self.assertNotIn("Uncompressed CPU minimum:", result.output)
        gpu = runner.invoke(main, [*arguments, "--backend", "gpu"])
        self.assertEqual(gpu.exit_code, 0, gpu.output)
        self.assertNotIn("Uncompressed CPU minimum:", gpu.output)
        too_small = runner.invoke(main, [*arguments, "--min-gib", "4"])
        self.assertNotEqual(too_small.exit_code, 0)
        self.assertIn("needs at least 8 GiB", too_small.output)
        too_few = runner.invoke(main, [*arguments, "--repeats", "2"])
        self.assertNotEqual(too_few.exit_code, 0)
        self.assertIn("at least three measured runs", too_few.output)
        for mode in ("--smoke", "--calibration"):
            trial = runner.invoke(main, [
                *arguments, mode, "--min-gib", "4", "--repeats", "1",
            ])
            self.assertEqual(trial.exit_code, 0, trial.output)
            self.assertNotIn("Uncompressed CPU minimum:", trial.output)

    @patch("sweep.git_commit", return_value="abcdef0")
    def test_image_scenario_records_replay_protocol_for_resume(self, _commit):
        corpus = SimpleNamespace(
            manifest={
                "kind": "raw",
                "release": "opencell",
                "packs": [{
                    "id": "opencell-dna", "input_id": "opencell-dna",
                    "source_group": "opencell-dna", "sha256": "a" * 64,
                    "width": 600, "height": 600, "split": "core", "dtype": "u16le",
                }],
            },
            record=lambda: {"kind": "raw", "release": "opencell"},
        )
        case = image_spec(codec="none", blosc_block_bytes=None,
                          blosc_shuffle="none", level=0)
        for options, minimum, cpu_minimum, repeats in (
            ([], 8 << 30, 32 << 30, 5),
            (["--smoke", "--min-gib", "4", "--repeats", "1"], 4 << 30, 4 << 30, 1),
            (["--calibration", "--min-gib", "0.001", "--repeats", "1"], 1073742, 1073742, 1),
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "sweep.json"
                arguments = ["--tier", "compress", "--scenario", "microscopy",
                             "--output", str(output), *options]
                with patch.dict(TIERS, {"compress": lambda: []}), \
                     patch("sweep.load_image_members", return_value=[
                         ("opencell-dna", "opencell-dna", "u16")
                     ]), \
                     patch("sweep.image_runs", return_value=[case]), \
                     patch("sweep.load_image_corpus", return_value=corpus), \
                     patch("sweep.run_one", return_value={
                         **case.base_result(), "status": "pass"
                     }) as execute, \
                     patch("sweep.gpu_and_driver", return_value=("test", "test")):
                    result = CliRunner().invoke(main, arguments)
                    self.assertEqual(result.exit_code, 0, result.output)
                    execute.assert_called_once()
                    self.assertIs(execute.call_args.kwargs["image_corpus"], corpus)
                    self.assertEqual(execute.call_args.kwargs["image_repeats"], repeats)
                    self.assertEqual(execute.call_args.kwargs["image_calibration"],
                                     "--calibration" in options)
                    saved = json.loads(output.read_text())
                    self.assertEqual(saved["runs"][0]["scenario"], "microscopy")
                    self.assertEqual(saved["image_protocol"]["repeats"], repeats)
                    self.assertEqual(saved["image_protocol"]["target_concurrent_shards"], 16)
                    self.assertEqual(saved["calibration"], "--calibration" in options)
                    self.assertEqual(saved["image_protocol"]["calibration"],
                                     "--calibration" in options)
                    self.assertEqual(saved["image_protocol"]["minimum_bytes"], minimum)
                    self.assertEqual(saved["image_protocol"]["cpu_uncompressed_minimum_bytes"],
                                     cpu_minimum)
                    self.assertEqual(saved["corpus"]["selected_assets"][0]["input"],
                                     "opencell-dna")
                    resumed = CliRunner().invoke(main, arguments)
                    self.assertEqual(resumed.exit_code, 0, resumed.output)
                    execute.assert_called_once()
                    for key, value in (("cpu_uncompressed_minimum_bytes", None),
                                       ("target_concurrent_shards", None),
                                       ("target_concurrent_shards", 4)):
                        previous = {**saved, "image_protocol": dict(saved["image_protocol"])}
                        if value is None:
                            del previous["image_protocol"][key]
                        else:
                            previous["image_protocol"][key] = value
                        output.write_text(json.dumps(previous))
                        before = output.read_bytes()
                        rejected = CliRunner().invoke(main, arguments)
                        self.assertNotEqual(rejected.exit_code, 0)
                        self.assertIn("different repetition or replay protocol", rejected.output)
                        execute.assert_called_once()
                        self.assertEqual(output.read_bytes(), before)

    def test_archived_image_scenario_keeps_run_identity_suffixes(self):
        current = image_spec().base_result()
        current["status"] = "pass"
        current["id"] += "__io"
        archived = {
            **current,
            "scenario": "images",
            "id": current["id"].replace("microscopy__", "images__", 1),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.json"
            path.write_text(json.dumps({
                "version": CURRENT_VERSION, "machine": {}, "runs": [archived],
            }))
            loaded = load_files([path])[0][1]["runs"][0]
        self.assertEqual(loaded["scenario"], "microscopy")
        self.assertEqual(loaded["fill"], "images")
        self.assertEqual(run_id(loaded), run_id(current))
        self.assertEqual(loaded["image_asset_id"], current["image_asset_id"])
        self.assertEqual(loaded["image_split"], current["image_split"])

    def test_archives_and_variants_share_short_report_labels(self):
        archived = {**filter_spec().base_result(), "status": "pass"}
        del archived["blosc_shuffle"], archived["blosc_level"]
        variant = {**filter_spec(blosc_shuffle="bit", level=0).base_result(), "status": "pass"}
        data = {"version": CURRENT_VERSION, "machine": {}, "runs": [archived, variant]}
        validated = validate_results(data)
        self.assertIsNone(validated.runs[0].blosc_shuffle)
        self.assertIsNone(validated.runs[0].blosc_level)
        self.assertEqual(codec_label(archived), "blosc-lz4")
        self.assertEqual(codec_label(filter_spec().base_result()), "blosc-lz4")
        self.assertEqual(codec_label(variant), "blosc-lz4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-abcdef0-20260904.json"
            path.write_text(json.dumps(data))
            loaded = load_files([path])[0][1]["runs"]
        self.assertEqual({r["codec_label"] for r in loaded}, {"blosc-lz4"})
        trimmed = trim_run(variant)
        self.assertEqual((trimmed["blosc_shuffle"], trimmed["blosc_level"]), ("bit", 0))
        self.assertEqual(trimmed["codec_label"], codec_label(variant))


class CommonMeasurementTests(unittest.TestCase):
    def test_generated_repeats_use_same_aggregator_and_preserve_windows(self):
        rows = [execution(2), execution(4)]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "bench" / "bench_stream_orca2_single"
            binary.parent.mkdir()
            binary.touch()
            with patch("sweep.execute", side_effect=rows) as execute:
                result = run_one(spec(), Path(directory), repeats=2,
                                 warmup=2, duration=5, geometry_frames=65536)
        self.assertEqual(result["throughput_in_gibs"], 3)
        self.assertEqual(result["repetitions"]["executions"], rows)
        self.assertEqual(result["measurement"], rows[0]["measurement"])
        self.assertEqual(result["geometry_frames"], 65536)
        self.assertNotIn("frames", result)
        self.assertEqual(execute.call_count, 2)
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index("--warmup") + 1], "2")
        self.assertEqual(command[command.index("--duration") + 1], "5")

    def test_resume_refuses_old_unknown_or_different_policy_without_writes(self):
        valid = {"version": CURRENT_VERSION, "machine": {}, "runs": [],
                 "measurement_policy": {**measurement_policy(), "geometry_frames": None},
                 "repetition_policy": {"generated": 1, "images": 5}}
        for override in ({"version": 10}, {"measurement_policy": None},
                         {"measurement_policy": {**measurement_policy(duration=5),
                                                 "geometry_frames": None}},
                         {"repetition_policy": {"generated": 2, "images": 5}},
                         {"smoke": True}, {"calibration": True}):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "results.json"
                before = json.dumps({**valid, **override})
                output.write_text(before)
                with (patch.dict(TIERS, {"backend": lambda: [spec()]}),
                     patch("sweep.git_commit", return_value="abcdef0"),
                     patch("sweep.run_one") as execute):
                    result = CliRunner().invoke(main, ["--tier", "backend", "-o", str(output)])
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("choose a new --output", result.output)
                execute.assert_not_called()
                self.assertEqual(output.read_text(), before)

    def test_summary_retains_contract_but_omits_raw_executions(self):
        raw = {**spec().base_result(), **aggregate_repetitions([execution(2), execution(4)])}
        trimmed = trim_run(raw)
        self.assertEqual(trimmed["measurement"], raw["measurement"])
        self.assertEqual(trimmed["repetitions"]["detail_repeat"], 1)
        self.assertNotIn("executions", trimmed["repetitions"])
        strings, blocks = pack([[trimmed]])
        restored = decode_runs(blocks[0], strings)[0]
        self.assertEqual(restored["measurement"], trimmed["measurement"])
        self.assertEqual(restored["repetitions"]["detail_repeat"], 1)
        self.assertAlmostEqual(restored["repetitions"]["throughput_spread_percent"],
                               trimmed["repetitions"]["throughput_spread_percent"], places=2)
        self.assertNotIn("executions", restored["repetitions"])
        self.assertEqual(len(raw["repetitions"]["executions"]), 2)


if __name__ == "__main__":
    unittest.main()

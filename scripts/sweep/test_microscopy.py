# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "rich", "pydantic"]
# ///
import copy
import hashlib
import json
import math
import struct
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from measurements import MEASUREMENT_POLICY
from microscopy_data import check_machine, estimate_seconds, load_entropy, summarize, validate_report, validate_study, write_datasets, write_previews
from microscopy_confirmation import candidates, representatives, select_confirmation
from microscopy_entropy import profile_corpus
from microscopy_plan import CHUNKS, DEFAULT_DEFINITION, fingerprint, make_plan, read_json, validate_plan
from microscopy_study import execute_plan, export_document, main as study_main, resume_document
from report import load_files
from sweep import RunSpec, execute_with_sink, image_executable
from microscopy_uncertainty import summarize_rounds

MEMBERS = [(name, name, dtype) for name, dtype in (
    ("opencell-dna", "u16"), ("bbbc010-brightfield", "u16"),
    ("dynacell-a549-phase", "f32"), ("cosem-cos7-em", "u8"))]


def definition(*, small=False):
    value = read_json(DEFAULT_DEFINITION)
    if small:
        value["inputs"] = ["opencell-dna"]
        value["reference"]["codecs"] = {"opencell-dna": "blosc-lz4"}
    return value


def observation(plan, task, rate=2.0):
    config = plan["cases"][task["case_id"]]
    bpe = {"u8": 1, "u16": 2, "f32": 4}[config["dtype"]]
    chunk = [1, 256, CHUNKS[config["chunk_label"]] // (256 * bpe)]
    frames, height, width = 32768, 192, 256
    minimum_elapsed = task["profile"]["duration_s"] / 0.95 + 1
    frames = max(frames, math.ceil(minimum_elapsed * rate * 2**30 / (height * width * bpe)))
    logical = frames * height * width * bpe
    frame_bytes = math.ceil(height / chunk[1]) * chunk[1] * math.ceil(width / chunk[2]) * chunk[2] * bpe
    physical, elapsed, output = frames * frame_bytes, logical / 2**30 / rate, logical // 2
    replay = {"shape": [frames, height, width], "reference_shape": [65536, height, width],
              "chunk_shape": chunk, "chunks_per_shard": [2, 1, 1], "epochs_per_batch": 1,
              "target_batch_bytes": 64 << 20, "actual_batch_bytes": 64 << 20,
              "append_elements": (64 << 20) // bpe, "dtype": config["dtype"] + ("le" if bpe > 1 else ""),
              "codec": config["codec"], "backend": config["backend"], "codec_level": config["level"],
              "shuffle": config["blosc_shuffle"], "order": "cyclic", "source_bytes": height * width * bpe,
              "source_padded_bytes": frame_bytes}
    window = {"policy": MEASUREMENT_POLICY, "coverage_status": "sufficient", "warmup_complete_batches": 2,
              "complete_batches": 4, "generation_transitions": 2, "warmup_s": task["profile"]["warmup_s"],
              "append_s": elapsed * 0.95, "drain_s": elapsed * 0.05, "elapsed_s": elapsed,
              "input_bytes": physical, "output_bytes": output, "reference_frames": 65536,
              "requested_warmup_s": task["profile"]["warmup_s"],
              "requested_duration_s": task["profile"]["duration_s"], "target_duration_s": 16,
              "geometry": {"chunk_shape": chunk}, "source_bytes": frame_bytes,
              "append_bytes": 64 << 20, "boundary_timing": True}
    result = {**RunSpec(**config).base_result(), "status": "pass", "measurement": window,
              "wall_s": elapsed, "process_wall_s": elapsed + 2,
              "throughput_in_gibs": physical / 2**30 / elapsed, "throughput_out_gibs": output / 2**30 / elapsed,
              "throughput_logical_gibs": rate, "logical_input_bytes": logical, "submitted_bytes": physical,
              "padded_input_bytes": physical, "output_bytes": output,
              "worker_threads": min(config.get("max_threads", 4), 4) if config["backend"] == "gpu" else config.get("max_threads", 4),
              "image_replay": replay,
              "stages": {"compress": {"avg_ms": 0.3, "in_gibs": 6.0, "out_gibs": 2.0}},
              "command": ["bench_stream_microscopy", "--codec", config["codec"]],
              "image_input": {"pack_sha256": "a" * 64, "pack_id": config["image_asset_id"],
                              "plane_order": ["plane"], "dtype": replay["dtype"], "input_id": config["input_id"],
                              "split": "core", "manifest_sha256": "b" * 64}}
    return {**copy.deepcopy(task), "started": "2026-09-13T00:00:00+00:00", "result": result}


def fixture(phase="discovery"):
    plan = make_plan(definition(small=True), MEMBERS, phase)
    return {"version": 1, "benchmark": "microscopy-study", "id": "fixture-" + phase,
            "created": "2026-09-13T00:00:00+00:00", "status": "complete", "plan": plan,
            "plan_sha256": fingerprint(plan),
            "machine": {"name": "Test fixture", "cpu_count": 8, "gpu": "NVIDIA L40", "hostname": "fixture"},
            "build": {"revision": "a" * 40, "executable_sha256": "b" * 64},
            "corpus": {"selected_assets": [{"asset": "opencell-dna", "input": "opencell-dna",
                                            "sha256": "a" * 64, "width": 256, "height": 192, "dtype": "u16le"}]},
            "records": [observation(plan, task) for task in plan["schedule"]]}


class PlanTests(unittest.TestCase):
    def test_discovery_bounds_raw_controls_and_blosc_pairs(self):
        plan = make_plan(definition(), MEMBERS)
        self.assertEqual(plan["counts"], {"configurations": 184, "samples": 368, "references": 32, "executions": 400})
        for config in plan["cases"].values():
            RunSpec(**config)
        configs = [config for config in plan["cases"].values() if config["input_id"] == "opencell-dna" and config["backend"] == "gpu"]
        self.assertEqual(len(configs), 23)
        self.assertEqual(sum(config["codec"] in {"lz4", "zstd", "none"} for config in configs), 9)
        pairs = {(config["chunk_label"], config["blosc_block_bytes"]) for config in configs if config["codec"] == "blosc-lz4"}
        self.assertEqual(pairs, {("16K", 4096), ("16K", 16384), ("64K", 16384), ("64K", 65536),
                                 ("256K", 16384), ("256K", 65536), ("256K", 262144)})

    def test_pilot_is_separate_and_references_surround_each_group(self):
        plan = make_plan(definition(), MEMBERS, "pilot")
        self.assertEqual(plan["counts"], {"configurations": 32, "samples": 32, "references": 16, "executions": 48})
        for batch in plan["batches"]:
            tasks = [task for task in plan["schedule"] if task["batch_id"] == batch["id"]]
            self.assertEqual(tasks[0]["role"], "reference-before")
            self.assertEqual(tasks[-1]["role"], "reference-after")
            self.assertEqual(tasks[0]["case_id"], tasks[-1]["case_id"])
            self.assertEqual({task["repeat"] for task in tasks}, {1})

    def test_changed_seed_changes_order_not_candidates(self):
        first = make_plan(definition(), MEMBERS)
        second_definition = definition()
        second_definition["seed"] += 1
        second = make_plan(second_definition, MEMBERS)
        self.assertEqual(first["cases"], second["cases"])
        self.assertNotEqual(first["schedule"], second["schedule"])
        self.assertEqual(first, make_plan(definition(), list(reversed(MEMBERS))))
        self.assertNotEqual(fingerprint(first), fingerprint(second))

    def test_unknown_options_and_schedule_tampering_fail(self):
        bad = definition()
        bad["duration"] = 99
        with self.assertRaises(ValueError):
            make_plan(bad, MEMBERS)
        bad = make_plan(definition(), MEMBERS)
        bad["schedule"][0]["profile"]["duration_s"] = 0.25
        with self.assertRaises(ValueError):
            validate_plan(bad)
        with self.assertRaises(ValueError):
            make_plan(definition(), MEMBERS[:-1])
        bad = definition()
        bad["profiles"]["gpu"]["min_gib"] = float("nan")
        with self.assertRaises(ValueError):
            make_plan(bad, MEMBERS)


class StudyDataTests(unittest.TestCase):
    def test_logical_median_fold_and_observed_detail(self):
        document = fixture()
        case_id = next(task["case_id"] for task in document["plan"]["schedule"]
                       if task["role"] == "sample" and document["plan"]["cases"][task["case_id"]]["backend"] == "cpu")
        tasks = [record for record in document["records"] if record["role"] == "sample" and record["case_id"] == case_id]
        for record, rate in zip(tasks, (1.0, 2.0, 3.0)):
            task = document["plan"]["schedule"][document["records"].index(record)]
            record["result"] = observation(document["plan"], task, rate)["result"]
        row = next(row for row in summarize(document)["measurements"] if row["case_id"] == case_id)
        self.assertEqual(row["throughput"], {"median": 2, "min": 1, "max": 3, "spread_percent": 100})
        self.assertEqual(row["compression_fold"], 2)
        self.assertGreater(row["padding_percent"], 0)
        self.assertGreater(row["detail"]["throughput_in_gibs"], row["throughput"]["median"])
        self.assertEqual(row["detail_execution"], tasks[1]["id"])
        self.assertTrue(row["needs_confirmation"])
        self.assertEqual(len(row["samples"]), 3)

    def test_reference_drift_marks_only_its_input_backend_condition(self):
        document = fixture()
        task = next(task for task in document["plan"]["schedule"] if task["role"] == "reference-after")
        index = next(i for i, record in enumerate(document["records"]) if record["id"] == task["id"])
        document["records"][index] = observation(document["plan"], task, 1.0)
        reference_spec = document["plan"]["cases"][task["case_id"]]
        affected = [key for key, spec in document["plan"]["cases"].items()
                    if spec["input_id"] == reference_spec["input_id"] and spec["backend"] == reference_spec["backend"]]
        data = summarize(document)
        self.assertEqual({row["case_id"] for row in data["measurements"] if row["reference"]["drift"]}, set(affected))
        self.assertTrue(all(row["needs_confirmation"] for row in data["measurements"] if row["reference"]["drift"]))

    def test_stable_groups_cannot_hide_between_group_reference_drift(self):
        document = fixture()
        batch = document["plan"]["batches"][0]
        for index, task in enumerate(document["plan"]["schedule"]):
            if task["batch_id"] == batch["id"] and task["role"] != "sample":
                document["records"][index] = observation(document["plan"], task, 1.9)
        rows = summarize(document)["measurements"]
        self.assertTrue(all(row["reference"]["spread_percent"] == 0 for row in rows))
        self.assertTrue(any(row["reference"]["drift"] for row in rows))
        self.assertTrue(any(not row["reference"]["drift"] for row in rows))

    def test_reject_incomplete_duplicate_changed_or_unqualified_observations(self):
        mutations = [lambda d: d["records"].pop(),
                     lambda d: d["records"].__setitem__(1, d["records"][0]),
                     lambda d: d["records"][0]["result"].__setitem__("backend", "unknown"),
                     lambda d: d["records"][0]["result"].__setitem__("throughput_logical_gibs", 999),
                     lambda d: d["records"][0]["result"]["measurement"].__setitem__("coverage_status", "insufficient"),
                     lambda d: d["records"][0]["result"]["measurement"].__setitem__("requested_duration_s", 0.25),
                     lambda d: d["records"][0]["result"]["image_input"].__setitem__("pack_sha256", "c" * 64),
                     lambda d: d["records"][0]["result"]["image_replay"].__setitem__("append_elements", 1)]
        for change in mutations:
            document = fixture()
            change(document)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_study(document)

    def test_pilot_cost_estimate_is_not_a_runtime_bound(self):
        pilot = fixture("pilot")
        estimate = estimate_seconds(pilot, make_plan(definition(small=True), MEMBERS))
        self.assertGreater(estimate["process_seconds"], 0)
        self.assertFalse(estimate["is_upper_bound"])
        bad = definition(small=True)
        bad["profiles"]["gpu"]["duration_s"] = 3
        with self.assertRaises(ValueError):
            estimate_seconds(pilot, make_plan(bad, MEMBERS))

    def test_archive_keeps_original_bytes_and_excludes_regular_report(self):
        document = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "study.json"
            raw = json.dumps(document, indent=3).encode()
            source.write_bytes(raw)
            index = root / "index.json"
            index.write_text(json.dumps({"version": 1, "studies": [{"path": "study.json", "sha256": hashlib.sha256(raw).hexdigest()}]}))
            output = root / "site"
            entries = write_datasets(output, index)
            self.assertEqual(len(entries), 1)
            self.assertEqual((output / "archives/microscopy/fixture-discovery/study.json").read_bytes(), raw)
            self.assertTrue((output / "data/microscopy/discovery.json").is_file())
            selection = [{"input": document["plan"]["definition"]["inputs"][0], "label": "Current image",
                          "sources": [{"study": document["id"], "backends": ["cpu", "gpu"]}]}]
            manifest = read_json(index)
            manifest["report"] = selection
            index.write_text(json.dumps(manifest))
            write_datasets(output, index)
            self.assertEqual(read_json(output / "data/microscopy/index.json")["report"], selection)
            self.assertEqual((output / "archives/microscopy/fixture-discovery/study.json").read_bytes(), raw)
            other = copy.deepcopy(document)
            other["id"] = "fixture-other-host"
            other["machine"]["name"] = "Other host"
            extra = root / "other.json"
            extra.write_text(json.dumps(other))
            write_datasets(output, index, extra=[extra])
            added = read_json(output / "data/microscopy/index.json")["report"][0]["sources"]
            self.assertEqual({source["study"] for source in added}, {document["id"], other["id"]})
            with self.assertRaisesRegex(ValueError, "separate from regular"):
                load_files([source])
            source.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                write_datasets(output, index)



class EntropyTests(unittest.TestCase):
    def make_corpus(self, root, raw, dtype, shape):
        corpus = root / "corpus"
        corpus.mkdir(exist_ok=True)
        (corpus / "image.raw").write_bytes(raw)
        asset = {"id": "image", "path": "image.raw", "dtype": dtype, "shape": shape,
                 "sha256": hashlib.sha256(raw).hexdigest()}
        (corpus / "manifest.json").write_text(json.dumps({"datasets": [{"assets": [asset]}]}))
        return corpus

    def test_entropy_counts_complete_pixels_and_samples_every_plane(self):
        raw = b"".join(bytes([value, plane]) for plane in range(2) for row in range(16) for value in range(256))
        with tempfile.TemporaryDirectory() as directory:
            corpus = self.make_corpus(Path(directory), raw, "uint16", [2, 16, 256])
            sample = profile_corpus(corpus)["inputs"][0]
        self.assertEqual(sample["pixel_entropy_bits"], 9.0)
        self.assertEqual(sample["unique_values"], 512)
        self.assertEqual(sample["sample_rows"], list(range(1, 16, 2)))
        self.assertEqual(sample["sample_pixels"], 4096)
        expected = b"".join(raw[(plane * 16 + row) * 512:(plane * 16 + row + 1) * 512]
                            for plane in range(2) for row in range(1, 16, 2))
        self.assertEqual(sample["sample_sha256"], hashlib.sha256(expected).hexdigest())

    def test_pixel_symbols_preserve_byte_dependence_and_float_bits(self):
        cases = [("uint8", bytes(range(256)), 256, 8.0),
                 ("uint16", struct.pack("<2H", 0x0000, 0x0101), 2, 1.0),
                 ("float32", struct.pack("<2f", 0.0, 1.0), 2, 1.0),
                 ("float32", struct.pack("<4I", 0x00000000, 0x80000000, 0x7FC00001, 0x7FC00002), 4, 2.0)]
        for dtype, raw, width, expected in cases:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                corpus = self.make_corpus(Path(directory), raw, dtype, [1, 1, width])
                sample = profile_corpus(corpus)["inputs"][0]
                self.assertEqual(sample["pixel_entropy_bits"], expected)
                self.assertEqual(sample["unique_values"], width)
                self.assertEqual(sample["sample_rows"], [0])

    def test_pixel_entropy_uses_observed_frequencies(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = self.make_corpus(Path(directory), bytes([0, 0, 0, 1]), "uint8", [1, 1, 4])
            sample = profile_corpus(corpus)["inputs"][0]
        self.assertAlmostEqual(sample["pixel_entropy_bits"], 0.8112781244591328)
        self.assertEqual(sample["unique_values"], 2)

    def test_pack_verification_follows_annex_links_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = self.make_corpus(root, b"abcd", "uint8", [1, 2, 2])
            raw = corpus / "image.raw"
            target = root / "annex-object"
            raw.replace(target)
            raw.symlink_to(target)
            document = profile_corpus(corpus)
            self.assertEqual(document["inputs"][0]["pack_sha256"], hashlib.sha256(b"abcd").hexdigest())
            target.write_bytes(b"abce")
            with self.assertRaisesRegex(ValueError, "checksum"):
                profile_corpus(corpus)

    def test_report_requires_matching_asset_hash_and_type(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = self.make_corpus(root, b"abcd", "uint8", [1, 2, 2])
            document = profile_corpus(corpus)
            sample = document["inputs"][0]
            path = root / "entropy.json"
            path.write_text(json.dumps(document))
            row = {"config": {"image_asset_id": "image", "dtype": "u8"},
                   "detail": {"image_input": {"pack_sha256": sample["pack_sha256"]}}}
            data = [{"measurements": [row]}]
            self.assertEqual(load_entropy(data, path)["inputs"], [sample])
            for changed in [{"image_asset_id": "other", "dtype": "u8"},
                            {"image_asset_id": "image", "dtype": "u16"}]:
                other = copy.deepcopy(row)
                other["config"] = changed
                self.assertEqual(load_entropy([{"measurements": [other]}], path)["inputs"], [])
            row["detail"]["image_input"]["pack_sha256"] = "a" * 64
            self.assertEqual(load_entropy(data, path)["inputs"], [])
            for changes in [{"pixel_entropy_bits": value} for value in [-1.0, 2.1, False, "2.0"]] + [
                    {"unique_values": value} for value in [0, 5, 2.5, True]]:
                invalid = copy.deepcopy(document)
                invalid["inputs"][0].update(changes)
                path.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "Invalid microscopy entropy"):
                    load_entropy(data, path)
            document["version"] = 1
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "Unsupported microscopy entropy"):
                load_entropy(data, path)


class ThumbnailExportTests(unittest.TestCase):
    def test_preview_matches_both_asset_and_measured_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            (corpus / "thumbnails").mkdir(parents=True)
            raw = b"retained thumbnail bytes"
            (corpus / "thumbnails/image.png").write_bytes(raw)
            asset = {"id": "image", "name": "Image", "sha256": "b" * 64, "dtype": "uint8",
                     "shape": [1, 2, 3], "thumbnail": "thumbnails/image.png"}
            source = {"collection": "Images", "url": "https://example.org/data", "attribution": "Image authors",
                      "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/"}
            (corpus / "manifest.json").write_text(json.dumps({"datasets": [
                {"modality": "electron-microscopy", "source": source, "assets": [asset]}]}))
            rows = [{"config": {"image_asset_id": "image"}, "detail": {"image_input": {"pack_sha256": digest}}}
                    for digest in ["a" * 64, "b" * 64]]
            output = root / "site"
            previews = write_previews(output, [{"measurements": rows}], corpus)
            self.assertEqual(len(previews), 1)
            preview = previews[0]
            self.assertEqual(preview["pack_sha256"], "b" * 64)
            self.assertEqual(preview["source"], source)
            self.assertEqual(preview["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual((output / preview["file"]).read_bytes(), raw)
            self.assertFalse(Path(preview["file"]).is_absolute())
            self.assertEqual(write_previews(output, [{"measurements": rows[:1]}], corpus), [])
            rows[1]["config"]["image_asset_id"] = "different-input"
            self.assertEqual(write_previews(output, [{"measurements": rows}], corpus), [])

    def test_missing_corpus_does_not_block_other_report_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(write_previews(root / "site", [], root / "missing"), [])


class ReportSelectionTests(unittest.TestCase):
    def setUp(self):
        self.datasets = [{"study": {"id": study_id, "machine": {"name": "L40"}},
                          "measurements": [{"config": {"input_id": "image", "backend": backend, "sink": sink,
                                                        "image_asset_id": "image", "image_split": "all", "dtype": "u16"},
                                            "detail": {"image_input": {"pack_sha256": "a" * 64}}}
                                           for backend in backends for sink in ["discard", "fs"]]}
                         for study_id, backends in [("old", ["cpu", "gpu"]), ("current", ["gpu"])]]
        self.report = [{"input": "image", "label": "Image", "sources": [
            {"study": "old", "backends": ["cpu"]}, {"study": "current", "backends": ["gpu"]}]}]

    def test_current_gpu_and_existing_cpu_evidence_share_one_input(self):
        self.assertEqual(validate_report(self.report, self.datasets), self.report)

    def test_overlapping_sources_cannot_reintroduce_superseded_comparisons(self):
        self.report[0]["sources"][0]["backends"].append("gpu")
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_report(self.report, self.datasets)

    def test_unknown_sources_inputs_and_backends_are_rejected(self):
        for change in [lambda r: r[0].update(input="missing"),
                       lambda r: r[0]["sources"][0].update(study="missing"),
                       lambda r: r[0]["sources"][1].update(backends=["cpu"]),
                       lambda r: r[0]["sources"][0].update(backends=["cpu", "cpu"]),
                       lambda r: r[0].update(label=""),
                       lambda r: r.append(r[0])]:
            report = copy.deepcopy(self.report)
            change(report)
            with self.assertRaises(ValueError):
                validate_report(report, self.datasets)

    def test_different_input_versions_cannot_be_combined(self):
        for row in self.datasets[1]["measurements"]:
            row["detail"]["image_input"]["pack_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different input content"):
            validate_report(self.report, self.datasets)

    def test_distinct_worker_budgets_can_share_a_machine(self):
        other = copy.deepcopy(self.datasets[0])
        other["study"]["id"] = "more-workers"
        other["measurements"] = [row for row in other["measurements"] if row["config"]["backend"] == "cpu"]
        for row in other["measurements"]:
            row["config"]["max_threads"] = 32
        self.datasets.append(other)
        self.report[0]["sources"].append({"study": "more-workers", "backends": ["cpu"]})
        self.assertEqual(validate_report(self.report, self.datasets), self.report)
        for row in other["measurements"]:
            row["config"]["max_threads"] = 4
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_report(self.report, self.datasets)

    def test_independent_machines_can_share_a_dataset(self):
        self.datasets[1]["study"]["machine"]["name"] = "Other machine"
        self.report[0]["sources"][0]["backends"].append("gpu")
        self.assertEqual(validate_report(self.report, self.datasets), self.report)


class ExecutionTests(unittest.TestCase):
    def test_study_cpu_control_preserves_work_and_attempts(self):
        document = fixture("pilot")
        document["records"] = []
        plan = document["plan"]
        task = next(task for task in plan["schedule"]
                    if plan["cases"][task["case_id"]]["backend"] == "cpu"
                    and plan["cases"][task["case_id"]]["codec"] == "none")
        config = plan["cases"][task["case_id"]]
        pack = {"id": "opencell-dna", "input_id": "opencell-dna", "source_group": "opencell-dna",
                "split": "core", "dtype": "u16le", "width": 256, "height": 192}
        corpus = SimpleNamespace(manifest={"packs": [pack]},
                                 pack_files={"opencell-dna": Path("pack.raw")})

        def execute(document, measure, checkpoint, max_seconds):
            return measure(config, task, max_seconds)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = image_executable(root)
            binary.parent.mkdir(parents=True)
            binary.touch()
            source = root / "plan.json"
            source.write_text(json.dumps(plan))
            arguments = ["microscopy_study", "run", "--plan", str(source), "--build-dir", str(root),
                         "--build-record", str(root / "build.json"), "--machine", "fixture",
                         "--id", "fixture-pilot", "--output", str(root / "output"), "--max-seconds", "60"]
            with patch("sys.argv", arguments), \
                 patch("microscopy_study.prepare_document", return_value=(document, corpus)), \
                 patch("microscopy_study.execute_plan", side_effect=execute), \
                 patch("sweep.execute_with_sink", side_effect=RuntimeError("command captured")) as process:
                with self.assertRaisesRegex(RuntimeError, "command captured"):
                    study_main()
            command = process.call_args.args[0]
            frames = int(command[command.index("--frames") + 1])
            expected = math.ceil(task["profile"]["min_gib"] * 2**30 / (pack["width"] * pack["height"] * 2))
            self.assertEqual(frames, expected)
            self.assertEqual(command[command.index("--max-attempts") + 1], "1")

    def test_process_budget_saves_prefix_without_calling_next_case(self):
        document = fixture("pilot")
        document["records"] = []
        checkpoints = []
        clock = iter((0, 0, 2)).__next__
        calls = []

        def measure(config, task, timeout):
            calls.append(timeout)
            return observation(document["plan"], task)["result"]

        self.assertFalse(execute_plan(document, measure, lambda data: checkpoints.append(copy.deepcopy(data)), 2, clock=clock))
        self.assertEqual(calls, [2])
        self.assertEqual(document["status"], "budget-exhausted")
        self.assertEqual(len(checkpoints[-1]["records"]), 1)
        validate_study(document, complete=False)

    def test_failure_is_retained_and_never_silently_replaced(self):
        document = fixture("pilot")
        document["records"] = []
        checkpoints = []
        with self.assertRaisesRegex(ValueError, "observations are retained"):
            execute_plan(document, lambda *_: {"status": "error"},
                         lambda data: checkpoints.append(copy.deepcopy(data)), 60)
        self.assertEqual(document["status"], "failed")
        self.assertEqual(len(checkpoints[-1]["records"]), 1)
        with self.assertRaises(ValueError):
            resume_document(document, fixture("pilot"))

    def test_resume_rejects_changed_build_and_allocation(self):
        base = fixture("pilot")
        for key in ("build", "machine", "plan_sha256"):
            changed = copy.deepcopy(base)
            changed[key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Resume changed"):
                resume_document(base, changed)

    def test_resume_checks_corpus_identity_not_verification_duration(self):
        existing = fixture("pilot")
        existing["corpus"]["verify_s"] = 1.5
        prepared = copy.deepcopy(existing)
        prepared["corpus"]["verify_s"] = 2.5
        self.assertIs(resume_document(existing, prepared), existing)
        self.assertEqual(existing["corpus"]["verify_s"], 1.5)
        for field, value in (("sha256", "b" * 64), ("asset", "another-image"), ("width", 512)):
            changed = copy.deepcopy(prepared)
            changed["corpus"]["selected_assets"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Resume changed corpus"):
                resume_document(existing, changed)
        prepared["corpus"]["decoded_bytes"] = 1024
        with self.assertRaisesRegex(ValueError, "Resume changed corpus"):
            resume_document(existing, prepared)

    def test_sink_invocation_retains_actual_command_and_timeout(self):
        config = next(iter(make_plan(definition(), MEMBERS)["cases"].values()))
        config["sink"] = "fs"
        with tempfile.TemporaryDirectory() as root:
            with patch("sweep.execute", return_value={"status": "pass"}) as execute:
                result = execute_with_sink(["bench"], RunSpec(**config), Path(root), timeout=7, record_command=True)
            self.assertEqual(execute.call_args.kwargs, {"timeout": 7})
            self.assertEqual(result["command"], execute.call_args.args[0])
            output = Path(result["command"][-1])
            self.assertFalse(output.exists())
            self.assertEqual(output.parent, Path(root))


class ExportTests(unittest.TestCase):
    def test_public_copy_preserves_measurements_and_duplicate_header_definitions(self):
        document = fixture("pilot")
        definitions = ["#define ZSTD_VERSION_NUMBER 10507 /* API version */"]
        document["build"].update(executable="/opt/build/bench_stream_microscopy", build_settings={
            "CMAKE_C_COMPILER": "C:\\Program Files\\compiler\\cl.exe",
            "CMAKE_C_FLAGS_RELEASE": "/O2 /Ob2 /DNDEBUG",
        }, toolchain={
            "c_compiler": "clang version 18.1.8\nInstalledDir:/opt/toolchain/bin",
            "compiler_configuration": {"CMAKE_C_COMPILER_VERSION": "18.1.8"},
            "codec_headers": {"/opt/toolkit/include/zstd.h": definitions,
                              "C:\\dependencies\\include\\zstd.h": definitions},
        })
        document["corpus"]["resolved_pack_paths"] = {"image": "\\\\server\\share\\image.raw"}
        document["sink_options"] = {"tmpdir": "/tmp"}
        document["storage"] = {"source": "server:/exports/share", "url": "https://example.com/data"}
        document["records"][0]["result"]["command"] = [
            "/opt/build/bench_stream_microscopy", "--input", "/images/input.raw", "-o", "/output"]
        document["records"][0]["result"]["fs_root"] = "C:\\output"
        original = copy.deepcopy(document)
        exported = export_document(document, "Network SMB share")
        self.assertEqual(document, original)
        self.assertEqual(exported["plan"], document["plan"])
        self.assertEqual(exported["plan_sha256"], document["plan_sha256"])
        self.assertEqual(exported["build"]["executable_sha256"], document["build"]["executable_sha256"])
        self.assertEqual(exported["corpus"]["selected_assets"], document["corpus"]["selected_assets"])
        self.assertEqual(exported["storage"]["description"], "Network SMB share")
        self.assertEqual(exported["storage"]["source"], "PATH_REMOVED")
        self.assertEqual(exported["storage"]["url"], "https://example.com/data")
        headers = exported["build"]["toolchain"]["codec_headers"]
        self.assertEqual(headers, {"zstd.h (1)": definitions, "zstd.h (2)": definitions})
        self.assertEqual(exported["build"]["toolchain"]["c_compiler"], "clang version 18.1.8\nPATH_REMOVED")
        self.assertEqual(exported["build"]["build_settings"]["CMAKE_C_FLAGS_RELEASE"], "/O2 /Ob2 /DNDEBUG")
        for raw, public in zip(document["records"], exported["records"]):
            self.assertEqual({key: value for key, value in raw["result"].items() if key not in ("command", "fs_root")},
                             {key: value for key, value in public["result"].items() if key not in ("command", "fs_root")})
        self.assertEqual(exported["records"][0]["result"]["command"], [
            "PATH_REMOVED", "--input", "PATH_REMOVED", "-o", "PATH_REMOVED"])
        self.assertEqual(exported["corpus"]["resolved_pack_paths"]["image"], "PATH_REMOVED")
        self.assertEqual(exported["sink_options"]["tmpdir"], "PATH_REMOVED")
        self.assertEqual(exported["records"][0]["result"]["fs_root"], "PATH_REMOVED")

    def test_export_removes_paths_from_header_metadata(self):
        document = fixture("pilot")
        document["build"]["toolchain"] = {"codec_headers": {"version.h": [
            "#define BUILD_VERSION 1", '#define BUILD_VERSION_SOURCE "/workspace/build"']}}
        exported = export_document(document, "Local SSD")
        self.assertEqual(exported["build"]["toolchain"]["codec_headers"]["version.h"], [
            "#define BUILD_VERSION 1", "PATH_REMOVED"])

    def test_export_rejects_path_labels_and_ambiguous_metadata_names(self):
        for label in ("", "/mnt/output", "C:\\output"):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "without filesystem paths"):
                export_document(fixture("pilot"), label)
        document = fixture("pilot")
        document["build"]["toolchain"] = {"codec_headers": {"/include/zstd.h": [1], "zstd.h (1)": [2]}}
        with self.assertRaisesRegex(ValueError, "collide"):
            export_document(document, "Local SSD")

    def test_export_cli_preserves_private_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "private.json", root / "public.json"
            raw = json.dumps(fixture("pilot")).encode()
            source.write_bytes(raw)
            arguments = ["microscopy_study", "export", "--study", str(source),
                         "--storage", "Local SSD", "--output", str(target)]
            with patch("sys.argv", arguments):
                study_main()
            self.assertEqual(source.read_bytes(), raw)
            validate_study(read_json(target))
            with patch("sys.argv", arguments), self.assertRaises(SystemExit):
                study_main()
            self.assertEqual(source.read_bytes(), raw)


class ComparisonTests(unittest.TestCase):
    def definition(self):
        return read_json(DEFAULT_DEFINITION.with_name("sink-comparison.json"))

    def test_fixed_rounds_pair_sinks_and_preserve_each_selected_setting(self):
        plan = make_plan(self.definition(), MEMBERS)
        self.assertEqual(plan["phase"], "comparison")
        self.assertEqual(plan["counts"], {"configurations": 48, "samples": 384, "references": 32, "executions": 416})
        self.assertEqual(plan["requested_seconds"], 2240)
        self.assertEqual(validate_plan(plan), plan)
        for number in range(1, 9):
            tasks = [task for task in plan["schedule"] if task["role"] == "sample" and task["round"] == number]
            self.assertEqual(len(tasks), 48)
            self.assertEqual(len({task["case_id"] for task in tasks}), 48)
            for left, right in zip(tasks[::2], tasks[1::2]):
                first = dict(plan["cases"][left["case_id"]])
                second = dict(plan["cases"][right["case_id"]])
                self.assertEqual({first.pop("sink"), second.pop("sink")}, {"discard", "fs"})
                self.assertEqual(first, second)
        for batch in plan["batches"]:
            tasks = [task for task in plan["schedule"] if task["batch_id"] == batch["id"]]
            self.assertEqual(tasks[0]["role"], "reference-before")
            self.assertEqual(tasks[-1]["role"], "reference-after")
            self.assertEqual({task["round"] for task in tasks}, set(batch["rounds"]))

    def test_backend_selections_do_not_expand_or_add_unused_references(self):
        definition = self.definition()
        definition["configurations"] = {
            "bbbc010-brightfield": [{"codec": "lz4", "chunk_label": "16K", "backend": "gpu"},
                                   {"codec": "zstd", "chunk_label": "64K", "backend": "cpu"}],
            "cosem-cos7-em": [{"codec": "none", "chunk_label": "256K", "backend": "gpu"}],
        }
        plan = make_plan(definition, MEMBERS)
        self.assertEqual(plan["counts"], {"configurations": 6, "samples": 48, "references": 24, "executions": 72})
        self.assertEqual(validate_plan(plan), plan)
        self.assertFalse(any(case["input_id"] == "cosem-cos7-em" and case["backend"] == "cpu"
                             for case in plan["cases"].values()))
        for number in range(1, 9):
            tasks = [task for task in plan["schedule"] if task["role"] == "sample" and task["round"] == number]
            for left, right in zip(tasks[::2], tasks[1::2]):
                first, second = (dict(plan["cases"][task["case_id"]]) for task in (left, right))
                self.assertEqual({first.pop("sink"), second.pop("sink")}, {"discard", "fs"})
                self.assertEqual(first, second)

    def test_overlapping_backend_selections_are_rejected(self):
        definition = self.definition()
        value = definition["configurations"]["cosem-cos7-em"][0]
        definition["configurations"]["cosem-cos7-em"].append({**value, "backend": "cpu"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_plan(definition, MEMBERS)
        definition["configurations"]["cosem-cos7-em"][-1]["backend"] = "other"
        with self.assertRaisesRegex(ValueError, "backend"):
            make_plan(definition, MEMBERS)

    def test_retained_plans_keep_their_original_schedule(self):
        root = DEFAULT_DEFINITION.parent
        for entry in read_json(root / "index.json")["studies"]:
            document = read_json(root / entry["path"])
            self.assertEqual(validate_plan(document["plan"]), document["plan"])

    def test_cpu_cli_omits_inputs_with_only_gpu_settings(self):
        definition = self.definition()
        definition["configurations"]["cosem-cos7-em"] = [
            {"codec": "none", "chunk_label": "256K", "backend": "gpu"}]
        with tempfile.TemporaryDirectory() as directory:
            source, output = (Path(directory) / name for name in ("definition.json", "plan.json"))
            source.write_text(json.dumps(definition))
            arguments = ["microscopy_study", "plan", "--definition", str(source),
                         "--backend", "cpu", "--sink", "discard", "--output", str(output)]
            with patch("sys.argv", arguments), patch("sweep.load_image_members", return_value=MEMBERS):
                study_main()
            plan = read_json(output)
            self.assertEqual(plan["definition"]["inputs"], ["bbbc010-brightfield"])
            self.assertEqual(validate_plan(plan), plan)

    def test_portable_cpu_plan_and_explicit_environment(self):
        definition = self.definition()
        definition["backends"] = ["cpu"]
        plan = make_plan(definition, MEMBERS)
        self.assertEqual(plan["counts"]["executions"], 208)
        check_machine({"cpu_count": 12, "gpu": None}, definition)
        definition["environment"]["cpu_count"] = 8
        with self.assertRaisesRegex(ValueError, "CPU allocation"):
            check_machine({"cpu_count": 12, "gpu": None}, definition)
        definition = self.definition()
        check_machine({"cpu_count": 20, "gpu": "Another GPU"}, definition)
        with self.assertRaisesRegex(ValueError, "available GPU"):
            check_machine({"cpu_count": 20, "gpu": "unknown"}, definition)

    def test_definition_rejects_unknown_or_duplicate_selected_settings(self):
        for mutation in [lambda d: d["configurations"]["cosem-cos7-em"].append(d["configurations"]["cosem-cos7-em"][0]),
                         lambda d: d.__setitem__("rounds", 7),
                         lambda d: d["configurations"]["bbbc010-brightfield"][1].__setitem__("blosc_block_bytes", 32768),
                         lambda d: d["codecs"]["blosc-lz4"].__setitem__("shuffle", "none")]:
            definition = self.definition()
            mutation(definition)
            with self.assertRaises(ValueError):
                make_plan(definition, MEMBERS)
        plan = make_plan(self.definition(), MEMBERS)
        next(task for task in plan["schedule"] if task["role"] == "sample")["round"] = 99
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_comparison_retains_paired_samples_and_storage_target(self):
        definition = self.definition()
        definition["backends"] = ["cpu"]
        definition["rounds"] = definition["rounds_per_group"] = 2
        plan = make_plan(definition, MEMBERS)
        document = fixture()
        document.update(plan=plan, plan_sha256=fingerprint(plan), id="fixture-comparison",
                        sink_options={"tmpdir": "/named/storage"})
        document["corpus"]["selected_assets"] = [
            {"asset": name, "input": name, "sha256": "a" * 64, "width": 256, "height": 192}
            for name in definition["inputs"]]
        document["records"] = [observation(plan, task) for task in plan["schedule"]]
        data = summarize(document)
        self.assertEqual(data["study"]["sink_options"]["tmpdir"], "/named/storage")
        for row in data["measurements"]:
            self.assertEqual({sample["round"] for sample in row["samples"]}, {1, 2})
            for sample in row["samples"]:
                self.assertEqual(sample["compression_fold"], sample["logical_input_bytes"] / sample["output_bytes"])
            self.assertNotIn("uncertainty", row)

    def test_plan_cli_can_select_cpu_and_discard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            arguments = ["microscopy_study", "plan", "--definition", str(DEFAULT_DEFINITION.with_name("sink-comparison.json")),
                         "--backend", "cpu", "--sink", "discard", "--cpu-count", "12", "--output", str(path)]
            with patch("sys.argv", arguments), patch("sweep.load_image_members", return_value=MEMBERS) as members:
                study_main()
            self.assertEqual(members.call_args.args[1], "microscopy-core-v1")
            plan = read_json(path)
            self.assertEqual(plan["counts"]["executions"], 104)
            self.assertEqual(plan["definition"]["environment"]["cpu_count"], 12)
            self.assertEqual({spec["sink"] for spec in plan["cases"].values()}, {"discard"})


class WorkerCountTests(unittest.TestCase):
    def definition(self):
        value = read_json(DEFAULT_DEFINITION.with_name("sink-comparison.json"))
        value.update(version=3, cpu_workers=[4, 8], inputs=["cosem-cos7-em"],
                     rounds=3, rounds_per_group=3)
        value["configurations"] = {"cosem-cos7-em": [
            {"codec": "zstd", "chunk_label": "256K"},
            {"codec": "blosc-lz4", "chunk_label": "16K", "blosc_block_bytes": 4096}]}
        value["reference"]["configurations"] = {"cosem-cos7-em": {"codec": "none", "chunk_label": "256K"}}
        return value

    def document(self):
        plan = make_plan(self.definition(), MEMBERS)
        document = fixture()
        document.update(plan=plan, plan_sha256=fingerprint(plan), id="fixture-workers")
        document["corpus"]["selected_assets"] = [{"asset": "cosem-cos7-em", "input": "cosem-cos7-em",
                                                  "sha256": "a" * 64, "width": 256, "height": 192}]
        document["records"] = [observation(plan, task) for task in plan["schedule"]]
        return document

    def test_worker_counts_are_distinct_and_gpu_is_not_repeated(self):
        plan = make_plan(self.definition(), MEMBERS)
        self.assertEqual(plan["counts"], {"configurations": 12, "samples": 36, "references": 8, "executions": 44})
        self.assertEqual(validate_plan(plan), plan)
        samples = [plan["cases"][task["case_id"]] for task in plan["schedule"] if task["role"] == "sample"]
        self.assertEqual({spec["max_threads"] for spec in samples if spec["backend"] == "cpu"}, {4, 8})
        self.assertEqual({spec["max_threads"] for spec in samples if spec["backend"] == "gpu"}, {4})
        self.assertEqual(len({RunSpec(**spec).id for spec in samples}), 12)
        for number in (1, 2, 3):
            tasks = [task for task in plan["schedule"] if task["role"] == "sample" and task["round"] == number]
            self.assertEqual(len({task["case_id"] for task in tasks}), 12)

    def test_focused_plan_limits_alternatives_without_repeating_gpu_settings(self):
        value = self.definition()
        value["configurations"]["cosem-cos7-em"][1]["cpu_workers"] = [8]
        plan = make_plan(value, MEMBERS)
        self.assertEqual(plan["counts"], {"configurations": 10, "samples": 30, "references": 8, "executions": 38})
        value["configurations"]["cosem-cos7-em"][1]["cpu_workers"] = [16]
        with self.assertRaisesRegex(ValueError, "outside the study range"):
            make_plan(value, MEMBERS)

    def test_worker_budgets_have_separate_frontiers_and_observations(self):
        document = self.document()
        data = summarize(document)
        cpu_rows = [row for row in data["measurements"] if row["config"]["backend"] == "cpu"]
        self.assertEqual(len({row["condition"] for row in cpu_rows}), 4)
        self.assertTrue(all(row["count"] == 3 for row in cpu_rows))
        record = next(record for record in document["records"] if record["result"]["worker_threads"] == 8)
        record["result"]["worker_threads"] = 4
        with self.assertRaisesRegex(ValueError, "worker count"):
            validate_study(document)
        record["result"]["worker_threads"] = 8
        record["result"]["max_threads"] = 4
        with self.assertRaisesRegex(ValueError, "requested thread limit"):
            validate_study(document)

    def test_invalid_counts_and_insufficient_allocations_are_rejected(self):
        for counts in ([], [0], [-1], [True], [4.0], ["4"], [4, 4], [2147483648]):
            with self.subTest(counts=counts):
                value = self.definition()
                value["cpu_workers"] = counts
                with self.assertRaises(ValueError):
                    make_plan(value, MEMBERS)
        value = self.definition()
        with self.assertRaisesRegex(ValueError, "exceed the allowed CPU count"):
            check_machine({"cpu_count": 4, "gpu": "NVIDIA L40"}, value)

    def test_scaling_analysis_pairs_rounds_without_pooling_workers(self):
        from microscopy_scaling import compare
        document = self.document()
        for record in document["records"]:
            if record["result"]["worker_threads"] == 8:
                rate = 2 * (record["round"] + 1)
                record.update(observation(document["plan"], record, rate=rate))
        report = compare([document])
        self.assertEqual(len(report["measurements"]), 12)
        self.assertEqual(len(report["choices"]), 6)
        for row in report["measurements"]:
            if row["workers"] == 8:
                ratios = row["speedup_over_four_workers"]
                self.assertEqual([ratios[key] for key in ("min", "median", "max")], [2, 3, 4])
        with self.assertRaisesRegex(ValueError, "more than once"):
            compare([document, document])

    def test_cli_changes_counts_and_records_the_new_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            arguments = ["microscopy_study", "plan", "--definition", str(DEFAULT_DEFINITION.with_name("sink-comparison.json")),
                         "--backend", "cpu", "--cpu-workers", "4", "--cpu-workers", "16", "--output", str(path)]
            with patch("sys.argv", arguments), patch("sweep.load_image_members", return_value=MEMBERS):
                study_main()
            plan = read_json(path)
            self.assertEqual(plan["version"], 3)
            self.assertEqual(plan["definition"]["cpu_workers"], [4, 16])
            self.assertEqual({spec["max_threads"] for spec in plan["cases"].values()}, {4, 16})


class ConfirmationTests(unittest.TestCase):
    def row(self, name, rate, fold, low=None, high=None, codec="blosc-lz4"):
        return {"case_id": name, "throughput": {"median": rate, "min": low or rate, "max": high or rate},
                "compression_fold": fold, "compression_range": {"min": fold, "max": fold},
                "config": {"codec": codec, "chunk_label": "16K", "backend": "gpu"}}

    def test_selection_keeps_uncertain_contenders_and_drops_clear_losses(self):
        rows = [self.row("frontier", 10, 2, 9, 11), self.row("overlap", 9, 1.9, 8, 10),
                self.row("dominated", 6, 1.8, 5, 7), self.row("compact", 5, 3)]
        selected = candidates(rows)
        self.assertEqual(set(selected), {"frontier", "overlap", "compact"})
        self.assertEqual(selected["overlap"], "overlapping-observed-ranges")

    def test_controls_keep_fast_lz4_and_compact_zstd(self):
        rows = [self.row("fast-lz4", 10, 1.5), self.row("compact-lz4", 3, 2),
                self.row("fast-zstd", 10, 1.5, codec="blosc-zstd"),
                self.row("compact-zstd", 5, 2, codec="blosc-zstd")]
        self.assertEqual({row["case_id"] for row in representatives(rows)}, {"fast-lz4", "compact-zstd"})

    def screens(self):
        documents = []
        for sink in ("fs", "discard"):
            definition = ComparisonTests().definition()
            definition["inputs"] = ["bbbc010-brightfield"]
            definition["configurations"] = {"bbbc010-brightfield": [
                {"codec": "blosc-lz4", "chunk_label": "16K", "blosc_block_bytes": 4096},
                {"codec": "blosc-lz4", "chunk_label": "16K", "blosc_block_bytes": 16384},
                {"codec": "blosc-lz4", "chunk_label": "64K", "blosc_block_bytes": 65536}]}
            definition["reference"]["configurations"] = {"bbbc010-brightfield": {"codec": "none", "chunk_label": "16K"}}
            definition.update(sinks=[sink], rounds=2, rounds_per_group=2)
            plan = make_plan(definition, MEMBERS)
            document = fixture()
            document.update(id="fixture-" + sink, plan=plan, plan_sha256=fingerprint(plan))
            document["build"].update(source_tree_sha256="c" * 64, cmake_cache_sha256="d" * 64)
            document["corpus"]["selected_assets"][0].update(asset="bbbc010-brightfield", input="bbbc010-brightfield")
            for task in plan["schedule"]:
                config = plan["cases"][task["case_id"]]
                rates = {4096: 10 if sink == "fs" else 5, 16384: 5 if sink == "fs" else 10, 65536: 1}
                task["_rate"] = rates.get(config["blosc_block_bytes"], 3)
            rates = [task.pop("_rate") for task in plan["schedule"]]
            document["records"] = [observation(plan, task, rate) for task, rate in zip(plan["schedule"], rates)]
            documents.append(document)
        return documents

    def test_confirmation_unions_sink_winners_and_respects_the_budget(self):
        documents = self.screens()
        definition, selection = select_confirmation(*documents)
        self.assertEqual(definition["rounds"], 3)
        self.assertEqual(len(selection["settings"]), 4)
        self.assertEqual({value["blosc_block_bytes"] for value in definition["configurations"]["bbbc010-brightfield"]},
                         {4096, 16384})
        self.assertEqual(selection["counts"]["samples"], 24)
        with self.assertRaisesRegex(ValueError, "exceeding the budget"):
            select_confirmation(*documents, max_settings=3)

    def test_confirmation_refuses_changed_builds_and_incomplete_screens(self):
        documents = self.screens()
        documents[1]["build"]["executable_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "same native build"):
            select_confirmation(*documents)
        documents = self.screens()
        documents[1]["records"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_confirmation(*documents)


class UncertaintyTests(unittest.TestCase):
    def test_common_round_variation_does_not_invent_ranking_reversals(self):
        rows = []
        for rate, logical, output in [(10, 200, 100), (9, 200, 100), (5, 300, 100)]:
            rows.append({"condition": "one", "samples": [
                {"round": number, "throughput_gibs": rate * factor,
                 "logical_input_bytes": logical * number, "output_bytes": output * number}
                for number, factor in enumerate([0.5, 1.3, 0.9, 1.1, 0.7, 1.2, 1.0, 0.8], 1)]})
        summarize_rounds(rows, draws=200)
        self.assertEqual([row["uncertainty"]["frontier_frequency"] for row in rows], [1, 0, 1])
        self.assertTrue(all(row["uncertainty"]["rounds"] == 8 for row in rows))
        self.assertEqual(rows[0]["uncertainty"]["compression_fold"], {"lower": 2, "upper": 2})
        self.assertLess(rows[0]["uncertainty"]["throughput"]["lower"], rows[0]["uncertainty"]["throughput"]["upper"])

    def test_sparse_and_unmatched_rounds_have_no_estimated_frequency(self):
        rows = [{"condition": "one", "samples": [{"round": 1, "throughput_gibs": 2,
                  "logical_input_bytes": 20, "output_bytes": 10}]}]
        summarize_rounds(rows)
        self.assertNotIn("uncertainty", rows[0])
        rows[0]["samples"] *= 8
        summarize_rounds(rows)
        self.assertNotIn("uncertainty", rows[0])


if __name__ == "__main__":
    unittest.main()

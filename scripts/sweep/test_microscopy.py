# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "rich", "pydantic"]
# ///
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from measurements import MEASUREMENT_POLICY
from microscopy_data import estimate_seconds, summarize, validate_study, write_datasets
from microscopy_plan import CHUNKS, DEFAULT_DEFINITION, fingerprint, make_plan, read_json, validate_plan
from microscopy_study import execute_plan, main as study_main, resume_document
from report import load_files
from sweep import RunSpec, execute_with_sink, image_executable

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
              "padded_input_bytes": physical, "output_bytes": output, "worker_threads": 4, "image_replay": replay,
              "stages": {"compress": {"avg_ms": 0.3, "in_gibs": 6.0, "out_gibs": 2.0}},
              "command": ["bench_stream_images", "--codec", config["codec"]],
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
            with self.assertRaisesRegex(ValueError, "separate from regular"):
                load_files([source])
            source.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                write_datasets(output, index)


class ExecutionTests(unittest.TestCase):
    def test_study_cpu_control_preserves_planned_work_minimum(self):
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


if __name__ == "__main__":
    unittest.main()

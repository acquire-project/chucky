# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic"]
# ///
import copy
import json
import tempfile
import unittest
from pathlib import Path

from columnar import decode_runs, pack
from image_results import image_sweep
from models import codec_label, run_id, validate_results
from report import find_results, load_files, write_data
from summary import build_summary, trim_run


def image_workloads():
    return {
        "version": 1,
        "inputs": [
            {
                "id": "cellstate",
                "scenarios": ["microscopy"],
                "default_scenario": "microscopy",
            }
        ],
        "scenarios": [
            {
                "id": "microscopy",
                "inputs": ["cellstate"],
                "default_input": "cellstate",
            }
        ],
    }


def image_document():
    layout = {
        "shape": [1024, 2048, 2048],
        "chunk_shape": [1, 256, 256],
        "chunks_per_shard": [256, 2, 2],
        "epochs_per_batch": 8,
        "target_batch_bytes": 64 << 20,
        "actual_batch_bytes": 64 << 20,
        "append_elements": 2048 * 2048,
        "dtype": "u16le",
    }
    document = {
        "benchmark": "microscopy-images",
        "schema_version": 1,
        "status": "complete",
        "created_utc": "2026-09-07T12:00:00+00:00",
        "corpus": {
            "manifest_sha256": "a" * 64,
            "revision": "b" * 40,
            "release": "cellstate-poc-v1",
            "kind": "provisional",
        },
        "protocol": {
            "minimum_bytes": 8 << 30,
            "repeats": 3,
            "warmups": 1,
            "smoke": False,
            "sink": "discard",
            "workers": 4,
            "chunk_shape": [1, 256, 256],
            "batch_bytes": 64 << 20,
            "order": "cyclic",
            "scales": 1,
            "codec_profiles": {
                "blosc-zstd": [
                    "--codec",
                    "blosc-zstd",
                    "--codec-level",
                    "3",
                    "--shuffle",
                    "bit",
                    "--blosc-block-bytes",
                    "16384",
                ]
            },
        },
        "machine": {
            "name": "reef-l40",
            "hostname": "node",
            "cpu_count": 128,
            "cpu_affinity": [1, 2, 3, 4],
            "nvidia_smi": "0, NVIDIA L40, GPU-id, 580.126.20, 46068 MiB",
        },
        "chucky": {"revision": "c" * 40, "source_tree_sha256": "d" * 64},
        "runs": [],
    }
    for iteration, rate in enumerate((1000, 5, 9, 7)):
        document["runs"].append(
            {
                "pack_id": "fluorescence-core-01-2048x2048",
                "pack_sha256": "e" * 64,
                "plane_order": ["a", "b", "c", "d"],
                "modality": "fluorescence",
                "split": "core",
                "source_group": "cellstate",
                "width": 2048,
                "height": 2048,
                "frames": 1024,
                "backend": "gpu",
                "profile": "blosc-zstd",
                "iteration": iteration,
                "warmup": iteration == 0,
                "status": "pass",
                "layout": copy.deepcopy(layout),
                "measurement": {
                    "status": "pass",
                    "throughput_in_gibs": rate,
                    "throughput_out_gibs": rate / 2,
                    "input_bytes": 8 << 30,
                    "output_bytes": 4 << 30,
                    "wall_s": 8 / rate,
                    "blosc_block_bytes": 16384,
                    "worker_threads": 4,
                    "image_replay": {
                        **copy.deepcopy(layout),
                        "codec_level": 3,
                        "shuffle": "bit",
                    },
                    "stages": {"compress": {"avg_ms": iteration + 10}},
                },
            }
        )
    return document


class ImageResultTests(unittest.TestCase):
    def test_median_preserves_one_execution_for_stage_details(self):
        original = image_document()
        before = copy.deepcopy(original)
        result = image_sweep(original)
        validate_results(result)
        self.assertEqual(original, before)
        self.assertEqual(len(result["runs"]), 1)
        run = result["runs"][0]
        self.assertEqual(run["throughput_in_gibs"], 7)
        self.assertEqual(run["compression_fold"], 2)
        self.assertEqual(run["stages"]["compress"]["avg_ms"], 13)
        self.assertEqual(run["repetitions"]["detail_repeat"], 3)
        self.assertEqual(run["repetitions"]["count"], 3)
        self.assertEqual(run["repetitions"]["throughput_max_gibs"], 9)
        self.assertEqual(run["chunk_bytes"], 128 << 10)
        self.assertEqual(run["blosc_shuffle"], "bit")
        self.assertEqual(run["blosc_level"], 3)
        self.assertEqual(codec_label(run), "blosc-zstd")
        self.assertEqual(run["id"], run_id(run))
        self.assertEqual(run["input_id"], "cellstate")
        self.assertEqual(run["input_label"], "Cellstate (provisional)")
        self.assertEqual(run["scenario"], "microscopy")
        self.assertNotIn("settings-", run["id"])
        self.assertTrue(run["id"].endswith("__blosc-block-16384"))
        self.assertEqual(result["machine"]["cpu_count"], 4)
        self.assertEqual(result["machine"]["gpu"], "NVIDIA L40")
        self.assertEqual(result["machine"]["commit"], "c" * 7)

    def test_schema_two_keeps_chunk_targets_as_separate_rows(self):
        document = image_document()
        document["schema_version"] = 2
        for record in document["runs"]:
            record.update(
                scenario="images",
                chunk_bytes=128 << 10,
                chunk_bytes_label="128K",
            )
        smaller = copy.deepcopy(document["runs"])
        for record in smaller:
            record.update(chunk_bytes=32 << 10, chunk_bytes_label="32K")
            record["layout"]["chunk_shape"] = [1, 128, 128]
            record["measurement"]["image_replay"]["chunk_shape"] = [1, 128, 128]
        document["runs"].extend(smaller)

        runs = image_sweep(document)["runs"]
        self.assertEqual(len(runs), 2)
        self.assertEqual(
            {(run["chunk_bytes_label"], run["chunk_bytes"]) for run in runs},
            {("32K", 32 << 10), ("128K", 128 << 10)},
        )
        self.assertEqual(len({run["id"] for run in runs}), 2)

    def test_schema_two_rejects_a_mislabeled_chunk(self):
        document = image_document()
        document["schema_version"] = 2
        for record in document["runs"]:
            record.update(
                scenario="images", chunk_bytes=128 << 10, chunk_bytes_label="32K"
            )
        with self.assertRaisesRegex(ValueError, "chunk label"):
            image_sweep(document)

    def test_recorded_scenario_controls_grouping_and_identity(self):
        archived = image_sweep(image_document())["runs"][0]

        document = image_document()
        for record in document["runs"]:
            record["scenario"] = "images"
        current = image_sweep(document)["runs"][0]
        self.assertEqual(current["id"], archived["id"])
        self.assertEqual(current["scenario"], "microscopy")
        for record in document["runs"]:
            record["scenario"] = "microscopy"
        renamed = image_sweep(document)["runs"][0]
        self.assertEqual(renamed["id"], archived["id"])

        for record in document["runs"]:
            record["scenario"] = "images_v2"
        changed = image_sweep(document)["runs"][0]
        self.assertEqual(changed["scenario"], "images_v2")
        self.assertNotEqual(changed["id"], current["id"])

        document["runs"][-1]["scenario"] = "images"
        with self.assertRaisesRegex(ValueError, "Incomplete repetitions"):
            image_sweep(document)

    def test_archived_codec_settings_remain_distinct(self):
        document = image_document()
        for record in document["runs"]:
            record["profile"] = "zstd"
            record["measurement"].pop("blosc_block_bytes")
            record["measurement"]["image_replay"]["shuffle"] = "none"
        first = image_sweep(document)["runs"][0]
        self.assertEqual(first["level"], 3)
        self.assertEqual(codec_label(first), "zstd")
        for record in document["runs"]:
            record["measurement"]["image_replay"]["codec_level"] = 0
        other = image_sweep(document)["runs"][0]
        self.assertEqual(codec_label(other), "zstd")
        self.assertEqual(first["input_id"], other["input_id"])
        self.assertNotEqual(first["id"], other["id"])

    def test_input_identity_is_portable_and_allows_code_comparisons(self):
        document = image_document()
        first = image_sweep(document)["runs"][0]
        document["machine"]["name"] = "auk"
        document["chucky"]["revision"] = "f" * 40
        document["corpus"]["resolved_pack_paths"] = {"pack": "C:/data/object"}
        for record in document["runs"]:
            record["input_path"] = "C:/data/object"
        other = image_sweep(document)["runs"][0]
        self.assertEqual(first["id"], other["id"])
        self.assertEqual(first["input_id"], other["input_id"])
        for record in document["runs"]:
            record["backend"] = "cpu"
        other = image_sweep(document)["runs"][0]
        self.assertEqual(first["input_id"], other["input_id"])
        self.assertNotEqual(first["id"], other["id"])

    def test_exact_corpus_pack_and_protocol_details_do_not_split_inputs(self):
        before = image_sweep(image_document())["runs"][0]
        for change in ("corpus", "pack", "order", "protocol", "layout", "smoke"):
            with self.subTest(change=change):
                document = image_document()
                if change == "corpus":
                    document["corpus"]["manifest_sha256"] = "f" * 64
                elif change == "protocol":
                    document["protocol"]["workers"] = 8
                elif change == "smoke":
                    document["protocol"]["smoke"] = True
                else:
                    for record in document["runs"]:
                        if change == "pack":
                            record["pack_sha256"] = "f" * 64
                        elif change == "order":
                            record["plane_order"].reverse()
                        else:
                            record["layout"]["append_elements"] //= 2
                after = image_sweep(document)["runs"][0]
                self.assertEqual(before["input_id"], after["input_id"])

    def test_source_group_names_provenance_and_content(self):
        document = image_document()
        document["corpus"]["kind"] = "raw"
        for record in document["runs"]:
            record["source_group"] = "opencell-dna"
        core = image_sweep(document)["runs"][0]
        self.assertEqual(core["input_id"], "opencell-dna")
        self.assertEqual(core["input_label"], "OpenCell DNA")

        for record in document["runs"]:
            record["split"] = "heldout"
        heldout = image_sweep(document)["runs"][0]
        self.assertEqual(heldout["input_id"], core["input_id"])
        self.assertEqual(heldout["input_label"], core["input_label"])
        self.assertNotEqual(heldout["id"], core["id"])

        for record in document["runs"]:
            record["source_group"] = "opencell-protein"
        protein = image_sweep(document)["runs"][0]
        self.assertEqual(protein["input_id"], "opencell-protein")
        self.assertEqual(protein["input_label"], "OpenCell Protein")
        self.assertNotEqual(protein["input_id"], heldout["input_id"])

    def test_chucky_input_id_overrides_external_source_group(self):
        document = image_document()
        document["corpus"]["kind"] = "raw"
        for record in document["runs"]:
            record["source_group"] = "external-description"
            record["input_id"] = "opencell-dna"
        run = image_sweep(document)["runs"][0]
        self.assertEqual(run["input_id"], "opencell-dna")
        self.assertEqual(run["input_label"], "OpenCell DNA")
        self.assertEqual(run["image_input"]["source_group"], "external-description")

    def test_incomplete_or_inconsistent_measurements_are_rejected(self):
        for change in ("status", "repeats", "duplicate", "layout", "rate"):
            with self.subTest(change=change):
                document = image_document()
                if change == "status":
                    document["status"] = "running"
                elif change == "repeats":
                    document["runs"].pop()
                elif change == "duplicate":
                    document["runs"][-1]["iteration"] = 1
                elif change == "layout":
                    document["runs"][-1]["layout"]["append_elements"] = 1
                else:
                    document["runs"][-1]["measurement"]["throughput_in_gibs"] = float(
                        "nan"
                    )
                with self.assertRaises(ValueError):
                    image_sweep(document)

    def test_overview_keeps_input_identity_and_repeat_spread(self):
        sweep = image_sweep(image_document())
        row = trim_run(sweep["runs"][0])
        strings, blocks = pack([[row]])
        restored = decode_runs(blocks[0], strings)[0]
        self.assertEqual(restored["input_id"], row["input_id"])
        self.assertEqual(restored["input_label"], row["input_label"])
        self.assertEqual(restored["repetitions"]["count"], 3)
        self.assertAlmostEqual(
            restored["repetitions"]["throughput_spread_percent"],
            row["repetitions"]["throughput_spread_percent"],
            places=2,
        )

        summary = build_summary([(Path("reef.json"), sweep)])
        self.assertFalse(summary["sweeps"][0]["smoke"])
        smoke = image_document()
        smoke["protocol"]["smoke"] = True
        summary = build_summary([(Path("reef-smoke.json"), image_sweep(smoke))])
        self.assertTrue(summary["sweeps"][0]["smoke"])

    def test_report_finds_canonical_results_and_keeps_distinct_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("auk", "oreb"):
                folder = root / "images" / name
                folder.mkdir(parents=True)
                document = image_document()
                document["machine"]["name"] = name
                (folder / "results.json").write_text(json.dumps(document))
                (folder / "results-gpu.json").write_text(json.dumps(document))
                (folder / "validation.json").write_text("{}")
            paths = find_results(root)
            self.assertEqual(len(paths), 2)
            loaded = load_files([*paths, paths[0]])
            self.assertEqual(len(loaded), 2)
            self.assertEqual(len({path.name for path, _ in loaded}), 2)
            workloads = image_workloads()
            write_data(loaded, [], root / "site", workloads=workloads)
            overview = json.loads((root / "site/overview.json").read_text())
            explorer = json.loads((root / "site/sweeps.json").read_text())
            self.assertEqual(overview["version"], 5)
            self.assertEqual(explorer["version"], 8)
            self.assertEqual(overview["workloads"], workloads)
            self.assertEqual(explorer["workloads"], workloads)
            self.assertEqual(
                {s["machine"] for s in overview["sweeps"]}, {"auk", "oreb"}
            )
            for sweep in overview["sweeps"]:
                rows = decode_runs(sweep["runs"], overview["strings"])
                self.assertEqual(rows[0]["throughput_in_gibs"], 7)
                self.assertIn("provisional", rows[0]["input_label"])
                self.assertTrue((root / "site/sweeps" / sweep["filename"]).is_file())


if __name__ == "__main__":
    unittest.main()

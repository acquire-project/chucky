"""Archive validation and extensibility tests; standard library, no GPU needed."""
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pareto_data import (DEFAULT_MANIFEST, ADAPTERS, GIB, check_hash, identity, load_datasets,
                         normalize_experiment, normalize_samples, raw_records, read_csv,
                         summary_row, validate_repetitions, write_datasets)


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.datasets = load_datasets()
        cls.base = DEFAULT_MANIFEST.parent
        cls.specs = [{**s, "matrix": cls.manifest["matrices"][s["matrix"]]} for s in cls.manifest["experiments"]]
        cls.raw = {s["id"]: raw_records(cls.base / s["directory"] / s["raw"]) for s in cls.specs if "raw" in s}

    def test_all_archives_and_summary_only_limits(self):
        self.assertEqual([len(d["measurements"]) for d in self.datasets], [200, 200, 200])
        self.assertEqual([d["experiment"]["validated_executions"] for d in self.datasets], [None, 800, 1200])
        self.assertEqual(self.datasets[1]["experiment"]["start_utc"][:10], "2026-09-06")
        self.assertEqual(len({r["workload_id"] for d in self.datasets for r in d["measurements"]}), 4)
        for data in self.datasets:
            self.assertEqual({w["scenario"] for w in data["workloads"]}, {"orca2_single"})
            self.assertEqual({w["source_url"] for w in data["workloads"]},
                             {"https://github.com/acquire-project/chucky/blob/main/bench/bench_stream_orca2_single.c"})
        for row in self.datasets[0]["measurements"]:
            self.assertIsNone(row["samples"])
            self.assertIsNone(row["measured_device_gib"]["min"])
            self.assertIsNone(row["estimated_pinned_gib"])
            self.assertNotIn("raw", row["provenance"])

    def test_raw_metrics_and_no_rounding(self):
        for data in self.datasets[1:]:
            for row in data["measurements"]:
                values = [r["throughput_gibs"] for r in row["samples"]]
                self.assertEqual(row["throughput_gibs"]["min"], min(values))
                self.assertEqual(row["throughput_gibs"]["max"], max(values))
                self.assertEqual(row["throughput_gibs"]["median"], sorted(values)[len(values) // 2])
                self.assertEqual(len(values), data["experiment"]["repetitions"])
                self.assertNotIn(0, [s["repeat"] for s in row["samples"]])
                self.assertTrue(row["source_metrics"])

    def validate(self, spec, records):
        sources = read_csv(self.base / spec["directory"] / spec["summary"])
        return validate_repetitions(records, sources, spec, self.manifest["workloads"][spec["workload"]],
                                    node=spec["format"] == "node-jsonl-v1")

    def test_warmups_are_excluded_even_if_extreme(self):
        for spec in self.specs[1:]:
            with self.subTest(format=spec["format"]):
                records = copy.deepcopy(self.raw[spec["id"]])
                for record in records:
                    if record["warmup"]:
                        record["result"]["throughput_in_gibs"] = 1e9
                groups = self.validate(spec, records)
                self.assertTrue(all(len(rs) == spec["repeats"] for rs in groups.values()))
                self.assertTrue(all(r["result"]["throughput_in_gibs"] < 1e9 for rs in groups.values() for r in rs))

    def test_missing_duplicate_and_misflagged_repetitions(self):
        for spec in self.specs[1:]:
            for mutation in ("missing", "duplicate", "warmup", "unknown", "geometry", "codec", "failed"):
                with self.subTest(format=spec["format"], mutation=mutation):
                    records = copy.deepcopy(self.raw[spec["id"]])
                    if mutation == "missing": records.pop()
                    if mutation == "duplicate": records[-1] = copy.deepcopy(records[0])
                    if mutation == "warmup": records[0]["warmup"] = False
                    if mutation == "unknown": records[0]["config"]["block_kib"] = 999
                    if mutation == "geometry": records[0]["result"]["chunks_per_epoch"] += 1
                    if mutation == "codec": records[0]["result"]["blosc_block_bytes"] = 1
                    if mutation == "failed": records[0]["result"]["status"] = "fail"
                    with self.assertRaises(ValueError): self.validate(spec, records)

    def test_summary_metrics_are_checked_against_repetitions(self):
        spec = self.specs[1]
        source = read_csv(self.base / spec["directory"] / "summary.csv")[0]
        samples = [r for r in self.raw[spec["id"]] if not r["warmup"] and identity(r["config"]) == identity(source)]
        for field in ["speed", "lo", "hi", "fold", "device_gib", "estimate_gib", "compress_ms", "padded_bytes"]:
            with self.subTest(field=field):
                bad = {**source, field: str(float(source[field]) * 1.001)}
                with self.assertRaises(ValueError):
                    normalize_samples(summary_row(bad, spec["format"]), bad, samples, node=True)

    def test_missing_optional_metrics_are_null(self):
        source = self.datasets[0]["measurements"][0]["source_metrics"].copy()
        del source["device_gib"]
        del source["estimated_total_gib"]
        row = summary_row(source, "summary-v1")
        self.assertEqual(row["measured_device_gib"], {"median": None, "min": None, "max": None})
        self.assertIsNone(row["estimated_device_gib"])
        spec = self.specs[1]
        source = read_csv(self.base / spec["directory"] / "summary.csv")[0]
        samples = copy.deepcopy([r for r in self.raw[spec["id"]] if not r["warmup"] and identity(r["config"]) == identity(source)])
        for record in samples:
            for field in ("memory_device_used_bytes", "memory_device_overhead_bytes", "memory_estimate_total_bytes", "memory_estimate_pinned_bytes"):
                record["result"][field] = None
        for field in ("device_gib", "device_min_gib", "device_max_gib", "estimate_gib", "pinned_gib", "overhead_mib"):
            source[field] = ""
        row = summary_row(source, spec["format"])
        normalize_samples(row, source, samples, node=True)
        self.assertIsNone(row["estimated_device_gib"])
        self.assertIsNone(row["measured_device_gib"]["median"])

    def test_duplicate_summary_identity(self):
        spec = self.specs[0]
        sources = read_csv(self.base / spec["directory"] / "summary.csv")
        sources[-1] = sources[0]
        with patch("pareto_data.read_csv", return_value=sources), self.assertRaisesRegex(ValueError, "Duplicate identities"):
            normalize_experiment(self.base, spec, self.manifest["workloads"][spec["workload"]])

    def test_corrupt_and_nonfinite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "bad.jsonl.gz"
            for content in (b"invalid gzip", gzip.compress(b"{bad json}")):
                corrupt.write_bytes(content)
                with self.assertRaises((ValueError, OSError)): raw_records(corrupt)
            file = root / "sample.csv"; file.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"): check_hash(file, "0" * 64)
        source = self.datasets[0]["measurements"][0]["source_metrics"].copy()
        for value in ("nan", "inf", "", "-1"):
            source["throughput_median_gibs"] = value
            with self.assertRaises(ValueError): summary_row(source, "summary-v1")

    def test_another_manifest_entry_needs_no_presentation_changes(self):
        spec = {**self.specs[0], "id": "another-supported-experiment", "label": "Another system"}
        data = normalize_experiment(self.base, spec, self.manifest["workloads"][spec["workload"]])
        self.assertEqual(data["experiment"]["label"], "Another system")
        self.assertEqual(len(data["measurements"]), 200)
        self.assertTrue(all(r["id"].startswith(spec["id"] + ":") for r in data["measurements"]))
        geometry = copy.deepcopy(self.manifest["workloads"][spec["workload"]])
        geometry["shape_note"] = "different geometry identity"
        different = normalize_experiment(self.base, spec, geometry)
        self.assertNotEqual(data["measurements"][0]["workload_id"], different["measurements"][0]["workload_id"])

    def test_build_copies_original_bytes_and_complete_index(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_datasets(output)
            index = json.loads((output / "data/pareto/index.json").read_text())
            self.assertEqual(len(index["experiments"]), 3)
            for spec in self.specs:
                for file in spec["retained_files"]:
                    self.assertEqual((self.base / spec["directory"] / file["path"]).read_bytes(),
                                     (output / "archives" / spec["id"] / file["path"]).read_bytes())
            future = copy.deepcopy(self.manifest)
            future["experiments"].append({**future["experiments"][0], "id": "fourth", "label": "Another system"})
            manifest_path = output / "archives/datasets.json"
            manifest_path.write_text(json.dumps(future), encoding="utf-8")
            write_datasets(output / "future-site", manifest_path)
            future_index = json.loads((output / "future-site/data/pareto/index.json").read_text())
            self.assertEqual(len(future_index["experiments"]), 4)
            fourth = json.loads((output / "future-site/data/pareto/fourth.json").read_text())
            self.assertEqual(len(fourth["measurements"]), 200)
            self.assertTrue(all(r["experiment_id"] == "fourth" for r in fourth["measurements"]))


if __name__ == "__main__":
    unittest.main()

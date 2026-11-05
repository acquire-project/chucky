# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic"]
# ///
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from migrate import main as migrate_main

from compression_metrics import migrate_compression
from models import CURRENT_VERSION, migrate_results


class CompressionMigrationTests(unittest.TestCase):
    def test_old_generated_gib_values_preserve_precision_and_legacy_value(self):
        data = {"version": 1, "runs": [{"fill": "xor", "input_gib": 3.51562,
                 "compressed_gib": 3.65804, "compression_fold": 0.999512,
                 "total_chunks": 100}]}
        migrate_results(data)
        row = data["runs"][0]
        self.assertEqual(data["version"], CURRENT_VERSION)
        self.assertEqual(data["migrated_from"], 1)
        self.assertEqual(row["logical_compression_fold"], 3.51562 / 3.65804)
        self.assertEqual(row["logical_compression_source"], "recorded_gib")
        self.assertEqual(row["legacy_compression_fold"], 0.999512)
        self.assertEqual(row["legacy_total_chunks"], 100)
        self.assertNotIn("compression_fold", row)
        self.assertNotIn("chunk_compression_fold", row)
        self.assertNotIn("total_chunks", row)
        before = copy.deepcopy(data)
        migrate_results(data)
        self.assertEqual(data, before)

    def test_unequal_repeats_pool_bytes_not_ratios_or_detail(self):
        row = {"logical_input_bytes": 10, "output_bytes": 2,
               "logical_compression_fold": 99, "compression_fold": 100,
               "repetitions": {"count": 2, "executions": [
                   {"logical_input_bytes": 10, "output_bytes": 2, "padded_input_bytes": 16},
                   {"logical_input_bytes": 90, "output_bytes": 30, "padded_input_bytes": 128}]}}
        migrate_compression(row)
        self.assertEqual(row["logical_compression_fold"], 100 / 32)
        self.assertEqual(row["logical_compression_source"], "pooled_bytes")
        self.assertNotIn("chunk_compression_fold", row)
        self.assertNotIn("decoded_chunk_bytes", row)
        self.assertEqual(row["repetitions"]["executions"][0]["legacy_padded_input_bytes"], 16)

    def test_missing_repeats_do_not_replace_pooled_summary_with_detail(self):
        row = {"logical_input_bytes": 10, "output_bytes": 2,
               "logical_compression_fold": 3, "repetitions": {"count": 2}}
        self.assertEqual(migrate_compression(row)["logical_compression_fold"], 3)
        del row["logical_compression_fold"]
        self.assertNotIn("logical_compression_fold", migrate_compression(row))

    def test_image_padding_is_never_assumed_to_be_logical_input(self):
        row = {"fill": "images", "input_bytes": 128, "output_bytes": 50,
               "input_gib": 128, "compressed_gib": 50, "compression_fold": 2.56}
        self.assertNotIn("logical_compression_fold", migrate_compression(row))
        row["logical_input_bytes"] = 100
        self.assertEqual(migrate_compression(row)["logical_compression_fold"], 2)

    def test_new_counters_and_ratios_are_preserved(self):
        row = {"logical_input_bytes": 100, "output_bytes": 50,
               "logical_compression_fold": 2, "chunk_compression_fold": 2.56,
               "decoded_chunk_bytes": 128, "padded_input_bytes": 128, "total_chunks": 8}
        before = copy.deepcopy(row)
        self.assertEqual(migrate_compression(row), before)

    def test_copy_command_and_report_use_the_same_migration(self):
        from report import load_files
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old.json"
            output = root / "migrated"
            row = {"id": "old", "scenario": "orca2_single", "codec": "none",
                   "fill": "xor", "backend": "cpu", "dtype": "u16",
                   "chunk_bytes": 1024, "chunk_bytes_label": "1K", "status": "pass",
                   "compression_fold": 1, "input_gib": 3, "compressed_gib": 4}
            original = json.dumps({"version": 1, "machine": {}, "runs": [row]})
            source.write_text(original)
            with patch("sys.argv", ["migrate.py", str(source), "--output-dir", str(output)]):
                migrate_main()
                with self.assertRaises(SystemExit):
                    migrate_main()  # Never overwrite an existing source or copy.
            self.assertEqual(source.read_text(), original)
            migrated = json.loads((output / source.name).read_text())
            report = load_files([source])[0][1]
            self.assertEqual(migrated["runs"][0]["logical_compression_fold"], 0.75)
            self.assertEqual(migrated["runs"][0]["logical_compression_fold"],
                             report["runs"][0]["logical_compression_fold"])
            self.assertEqual(migrated["migrated_from"], 1)

    def test_zero_or_missing_denominators_remain_unavailable(self):
        for output in (None, 0):
            row = {"fill": "zeros", "logical_input_bytes": 100, "output_bytes": output}
            self.assertNotIn("logical_compression_fold", migrate_compression(row))


if __name__ == "__main__":
    unittest.main()

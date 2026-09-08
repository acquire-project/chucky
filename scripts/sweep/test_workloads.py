import tempfile
import unittest
from pathlib import Path

from workloads import DEFAULT_WORKLOADS, load_workloads, validate_run_pairs


VALID = """\
version = 1

[[input]]
id = "xor"
scenarios = ["orca2_single"]
default_scenario = "orca2_single"

[[scenario]]
id = "orca2_single"
inputs = ["xor"]
default_input = "xor"
"""


class WorkloadRegistryTests(unittest.TestCase):
    def load(self, text: str) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workloads.toml"
            path.write_text(text)
            return load_workloads(path)

    def test_repository_registry_has_explicit_defaults_and_pairs(self):
        registry = load_workloads(DEFAULT_WORKLOADS)
        inputs = {entry["id"]: entry for entry in registry["inputs"]}
        scenarios = {entry["id"]: entry for entry in registry["scenarios"]}
        self.assertEqual(inputs["opencell-dna"]["default_scenario"], "images")
        self.assertEqual(inputs["opencell-protein"]["scenarios"], ["images"])
        self.assertEqual(inputs["xor"]["default_scenario"], "orca2_single")
        self.assertEqual(inputs["zeros"]["default_scenario"], "orca2_single")
        self.assertEqual(inputs["rand"]["default_scenario"], "orca2_single")
        self.assertEqual(scenarios["images"]["default_input"], "opencell-dna")
        self.assertEqual(
            scenarios["images"]["inputs"],
            ["opencell-dna", "opencell-protein"],
        )
        for identifier, scenario in scenarios.items():
            if identifier != "images":
                self.assertEqual(scenario["default_input"], "xor")

    def test_missing_and_malformed_registries_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No workload registry"):
                load_workloads(Path(directory) / "missing.toml")
        with self.assertRaisesRegex(ValueError, "Invalid value"):
            self.load("version = !!!")

    def test_duplicate_entries_and_references_fail(self):
        duplicate_entry = VALID + """
[[input]]
id = "xor"
scenarios = ["orca2_single"]
default_scenario = "orca2_single"
"""
        with self.assertRaisesRegex(ValueError, "duplicate input id xor"):
            self.load(duplicate_entry)
        with self.assertRaisesRegex(ValueError, "repeats orca2_single"):
            self.load(VALID.replace(
                'scenarios = ["orca2_single"]',
                'scenarios = ["orca2_single", "orca2_single"]',
            ))

    def test_unknown_and_one_sided_references_fail(self):
        with self.assertRaisesRegex(ValueError, "unknown scenario missing"):
            self.load(VALID.replace(
                'scenarios = ["orca2_single"]',
                'scenarios = ["orca2_single", "missing"]',
            ))
        with self.assertRaisesRegex(ValueError, "does not list the input"):
            self.load(VALID.replace(
                'inputs = ["xor"]',
                'inputs = ["other"]',
            ).replace(
                'default_input = "xor"',
                'default_input = "other"',
            ).replace(
                '[[scenario]]',
                '[[input]]\nid = "other"\nscenarios = ["orca2_single"]\n'
                'default_scenario = "orca2_single"\n\n[[scenario]]',
            ))

    def test_invalid_defaults_fail(self):
        with self.assertRaisesRegex(ValueError, "default missing is not in scenarios"):
            self.load(VALID.replace(
                'default_scenario = "orca2_single"',
                'default_scenario = "missing"',
            ))
        with self.assertRaisesRegex(ValueError, "default other is not in inputs"):
            self.load(VALID.replace('default_input = "xor"', 'default_input = "other"'))

    def test_unknown_result_pairs_fail_and_input_id_wins(self):
        registry = self.load(VALID)
        path = Path("result.json")
        validate_run_pairs(
            [(path, {"runs": [{"scenario": "orca2_single", "fill": "xor"}]})],
            registry,
        )
        with self.assertRaisesRegex(
            ValueError, r"\(orca2_single, mystery\) in result.json"
        ):
            validate_run_pairs(
                [(path, {"runs": [{
                    "scenario": "orca2_single",
                    "input_id": "mystery",
                    "fill": "xor",
                }]})],
                registry,
            )


if __name__ == "__main__":
    unittest.main()

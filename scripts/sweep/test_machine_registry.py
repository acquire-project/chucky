import unittest
from pathlib import Path

from machine_registry import load_registry, machine_catalog, machine_id, match_registry


class MachineRegistryTests(unittest.TestCase):
    def test_confirmed_aliases_resolve_across_sweeps_and_archives(self):
        registry = load_registry(Path(__file__).resolve().parents[2] / "bench/machines.toml")
        cases = [("rtx5080", "", "", "oreb"), ("", "oreb", "", "oreb"),
                 ("", "", "blosc-rtx5080-20260905", "oreb"),
                 ("", "", "blosc-rtx5080-20260916", "oreb"),
                 ("", "", "blosc-rtx5070-20260905", "auk"),
                 ("reef-turin", "", "", "turin-raid10"),
                 ("", "cw-us-e4a2-l40-234-085", "", "reef-l40")]
        for name, host, run, expected in cases:
            with self.subTest(name=name, host=host, run=run):
                self.assertEqual(machine_id(registry, name, host, run), expected)
        self.assertEqual(machine_id(registry, "another RTX 5080 host"), "another RTX 5080 host")
        self.assertEqual(machine_catalog(registry)["machines"][0]["specs"], registry[0]["specs"])

    def test_canonical_names_work_without_repeating_them_as_aliases(self):
        entry = {"name": "host", "description": "", "names": [], "hosts": [], "specs": {}}
        self.assertEqual(match_registry([entry], "HOST", ""), entry)


if __name__ == "__main__":
    unittest.main()

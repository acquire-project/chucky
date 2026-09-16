"""The dataset entry point delegates execution to the shared sweep runner."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run as runner


class RunnerTests(unittest.TestCase):
    def test_delegates_sweep_options_and_failure_status(self):
        arguments = ["--tier", "backend", "--backend", "cpu", "--chunk-bytes", "16K",
                     "--codec", "none", "--smoke", "--min-gib", "0.004",
                     "--repeats", "1", "--dry-run"]
        with patch("run.subprocess.run", return_value=subprocess.CompletedProcess([], 7)) as execute:
            self.assertEqual(runner.run(arguments), 7)
        command = execute.call_args.args[0]
        self.assertEqual(command[:2], ["uv", "run"])
        self.assertEqual(Path(command[2]).name, "sweep.py")
        self.assertEqual(command[3:], ["--scenario", "microscopy", *arguments])

    def test_main_dispatches_run_without_a_second_corpus_or_matrix(self):
        with patch("run.sys.argv", ["run.py", "run", "--help"]), \
             patch("run.run", return_value=3) as execute:
            self.assertEqual(runner.main(), 3)
        execute.assert_called_once_with(["--help"])


class TopologyTests(unittest.TestCase):
    def test_counts_physical_cores_and_records_allowed_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cpu, core, socket in ((0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 0), (4, 0, 1)):
                path = root / f"cpu{cpu}"
                topology = path / "topology"
                topology.mkdir(parents=True)
                (topology / "core_id").write_text(str(core))
                (topology / "physical_package_id").write_text(str(socket))
                (path / f"node{socket}").mkdir()
            (root / "cpu4" / "online").write_text("0")
            result = runner.cpu_topology([0, 2], root)
            self.assertEqual(result["logical_cpus"], 4)
            self.assertEqual(result["physical_cores"], 2)
            self.assertEqual(result["allowed_physical_cores"], 1)
            self.assertEqual([cpu["cpu"] for cpu in result["cpus"] if cpu["allowed"]], [0, 2])
            self.assertEqual(runner.cpu_topology([0, 1], root)["allowed_physical_cores"], 2)
            self.assertEqual(runner.cpu_topology([99], root), {})
            self.assertEqual(runner.cpu_topology(None, root), {})


if __name__ == "__main__":
    unittest.main()

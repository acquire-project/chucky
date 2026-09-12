"""The dataset entry point delegates execution to the shared sweep runner."""

from pathlib import Path
import subprocess
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


if __name__ == "__main__":
    unittest.main()

"""CPU smoke coverage for the shared benchmark CLI and its JSON settings."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


BENCH = Path(sys.argv.pop(1)).resolve()


class CodecOptionsTest(unittest.TestCase):
    def run_bench(self, *options):
        return subprocess.run(
            [str(BENCH), "--backend", "cpu", "--frames", "128",
             "--chunk-bytes", "16K", "--batch-bytes", "1M",
             "--memory-budget", "64M", "--max-threads", "2", "--json",
             "--blosc-block-bytes", "16K",
             *options],
            capture_output=True, text=True, timeout=30,
        )

    def passed(self, *options):
        result = self.run_bench(*options)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "pass")
        return data

    def test_blosc_filters_and_defaults(self):
        for codec in ("blosc-lz4", "blosc-zstd"):
            for shuffle in ("none", "byte", "bit"):
                with self.subTest(codec=codec, shuffle=shuffle):
                    data = self.passed("--shuffle", shuffle, "--codec", codec)
                    self.assertEqual((data["shuffle"], data["level"]), (shuffle, 3))

    def test_explicit_zero_is_order_independent(self):
        for codec in ("blosc-lz4", "blosc-zstd"):
            for options in (("--level", "0", "--codec", codec),
                            ("--codec", codec, "--level", "0")):
                with self.subTest(options=options):
                    data = self.passed(*options, "--shuffle", "bit")
                    self.assertEqual(data["level"], 0)
                    # Store-only Blosc adds framing; it cannot compress input.
                    self.assertLessEqual(data["compression_fold"], 1)

    def test_final_codec_supplies_default(self):
        for options, expected in (
            (("--codec", "blosc-lz4", "--codec", "zstd"), 0),
            (("--codec", "lz4"), 1),
            (("--level", "1", "--codec", "blosc-zstd"), 1),
        ):
            with self.subTest(options=options):
                self.assertEqual(self.passed(*options)["level"], expected)

    def test_invalid_options(self):
        cases = [
            ("--shuffle", "invalid"), ("--shuffle", "byte"),
            ("--codec", "blosc-lz4", "--level", "10"),
            ("--codec", "blosc-zstd", "--level", "-1"),
            ("--level", "256"), ("--level", "1x"),
            ("--level", "9999999999999999999999999"),
            ("--level", ""), ("--level",), ("--shuffle",),
        ]
        for options in cases:
            with self.subTest(options=options):
                result = self.run_bench(*options)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)


if __name__ == "__main__":
    unittest.main()

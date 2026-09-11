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
                    data = self.passed("--blosc-block-bytes", "4K",
                                       "--blosc-shuffle", shuffle, "--codec", codec)
                    self.assertEqual((data["blosc_shuffle"], data["blosc_level"]), (shuffle, 3))
                    self.assertEqual(data["blosc_block_bytes"], 4096)
                    self.assertNotIn("shuffle", data)
                    self.assertNotIn("level", data)

    def test_explicit_zero_is_order_independent(self):
        for codec in ("blosc-lz4", "blosc-zstd"):
            for options in (("--level", "0", "--codec", codec),
                            ("--codec", codec, "--level", "0")):
                with self.subTest(options=options):
                    data = self.passed(*options, "--blosc-shuffle", "bit",
                                       "--blosc-block-bytes", "16K")
                    self.assertEqual(data["blosc_level"], 0)
                    self.assertEqual(data["blosc_block_bytes"], 16384)
                    self.assertLessEqual(data["compression_fold"], 1)

    def test_final_codec_supplies_default(self):
        for options, key, expected in (
            (("--codec", "blosc-lz4", "--codec", "zstd"), "level", 0),
            (("--codec", "lz4"), "level", 1),
            (("--level", "1", "--codec", "blosc-zstd",
              "--blosc-block-bytes", "16K"), "blosc_level", 1),
        ):
            with self.subTest(options=options):
                data = self.passed(*options)
                self.assertEqual(data[key], expected)
                if key == "level":
                    self.assertNotIn("blosc_shuffle", data)
                    self.assertNotIn("blosc_level", data)

    def test_invalid_options(self):
        cases = [
            ("--blosc-shuffle", "invalid"), ("--blosc-shuffle", "byte"),
            ("--codec", "lz4", "--level", "0"),
            ("--codec", "blosc-lz4", "--blosc-block-bytes", "16K", "--level", "10"),
            ("--codec", "blosc-zstd", "--blosc-block-bytes", "16K", "--level", "-1"),
            ("--level", "256"), ("--level", "1x"),
            ("--level", "9999999999999999999999999"),
            ("--level", ""), ("--level",), ("--blosc-shuffle",),
        ]
        for options in cases:
            with self.subTest(options=options):
                result = self.run_bench(*options)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)

    def test_blosc_requires_explicit_valid_blocks_with_every_filter(self):
        for codec in ("blosc-lz4", "blosc-zstd"):
            for shuffle in ("none", "byte", "bit"):
                for block in ((), ("--blosc-block-bytes", "0"),
                              ("--blosc-block-bytes", "127")):
                    with self.subTest(codec=codec, shuffle=shuffle, block=block):
                        result = self.run_bench("--codec", codec, "--blosc-shuffle", shuffle,
                                                "--level", "0", *block)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertTrue(result.stderr)


if __name__ == "__main__":
    unittest.main()

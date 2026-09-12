import copy
import json
import subprocess
import unittest
from unittest.mock import patch

from measurements import (
    MEASUREMENT_POLICY, aggregate_repetitions, execute, validate_measurement,
)


def execution(rate=2, **overrides):
    elapsed = 32 / rate
    result = {
        "status": "pass", "input_bytes": 32 << 30, "submitted_bytes": 32 << 30,
        "logical_input_bytes": 30 << 30, "padded_input_bytes": 32 << 30,
        "output_bytes": 16 << 30, "wall_s": elapsed, "process_wall_s": 60,
        "throughput_in_gibs": rate, "throughput_out_gibs": rate / 2,
        "throughput_logical_gibs": 30 / elapsed,
        "stages": {"compress": {"avg_ms": rate}},
        "measurement": {
            "policy": MEASUREMENT_POLICY, "coverage_status": "sufficient",
            "warmup_complete_batches": 2, "complete_batches": 4,
            "generation_transitions": 2, "warmup_s": 0.25,
            "append_s": elapsed * 0.95, "drain_s": elapsed * 0.05,
            "elapsed_s": elapsed, "input_bytes": 32 << 30,
            "output_bytes": 16 << 30, "reference_frames": 65536,
            "geometry": {"epoch_bytes": 1 << 20},
            "source_bytes": 64 << 20, "append_bytes": 1 << 20,
            "boundary_timing": True,
        },
    }
    result.update(overrides)
    return result


class MeasurementTests(unittest.TestCase):
    def test_rejects_unqualified_success_and_bad_denominators(self):
        for mutate in (
            lambda r: r.pop("measurement"),
            lambda r: r["measurement"].update(policy="old"),
            lambda r: r["measurement"].update(coverage_status="insufficient"),
            lambda r: r["measurement"].update(complete_batches=3),
            lambda r: r["measurement"].update(generation_transitions=1),
            lambda r: r["measurement"].update(warmup_s=0),
            lambda r: r["measurement"].update(drain_s=4),
            lambda r: r.update(throughput_in_gibs=99),
            lambda r: r.update(throughput_in_gibs=float("nan")),
        ):
            result = execution()
            mutate(result)
            with self.subTest(result=result), self.assertRaises(ValueError):
                validate_measurement(result)

    def test_nonzero_exit_overrides_pass_and_keeps_diagnostics(self):
        raw = execution()
        with patch("measurements.subprocess.run", return_value=subprocess.CompletedProcess(
                [], 1, json.dumps(raw), "out of space")):
            result = execute(["benchmark"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["measurement"], raw["measurement"])

    def test_even_repeats_keep_observed_detail_and_all_raw_executions(self):
        rows = [execution(2), execution(4)]
        before = copy.deepcopy(rows)
        result = aggregate_repetitions(rows)
        self.assertEqual(result["throughput_in_gibs"], 3)
        self.assertEqual(result["measurement"], rows[0]["measurement"])
        self.assertEqual(result["repetitions"]["detail_repeat"], 1)
        self.assertEqual(result["repetitions"]["executions"], rows)
        self.assertEqual(result["compression_fold"], 2)
        self.assertEqual(result["logical_compression_fold"], 30 / 16)
        self.assertEqual(rows, before)
        result["repetitions"]["executions"][0]["status"] = "changed"
        self.assertEqual(rows, before)

    def test_repeats_may_extend_but_must_share_geometry_and_source(self):
        rows = [execution(2), execution(4)]
        rows[1]["measurement"]["target_duration_s"] = 8
        aggregate_repetitions(rows)
        for key in ("geometry", "source_bytes", "append_bytes", "boundary_timing"):
            changed = copy.deepcopy(rows)
            changed[1]["measurement"][key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate_repetitions(changed)


if __name__ == "__main__":
    unittest.main()

import copy
import unittest

from agent_lab.experiments import compare, digest, summarize, validate_result


def result(tokens=10):
    return {
        "schema_version": 1,
        "run_id": "fixture",
        "task": {"id": "sum", "version": 1, "fixture_hash": "a" * 64},
        "protocol": {"id": "exact-v1", "split": "development", "repeat": 0},
        "configuration": {"model": "fixture", "seed": 7},
        "hardware": {"profile": "fixture"},
        "software": {"runtime": "v1"},
        "correctness": {"status": "pass", "score": 1, "evaluator": "exact-v1"},
        "work": {
            "input_tokens": 20,
            "output_tokens": tokens,
            "generated_tokens": tokens,
            "model_calls": 1,
            "retries": 0,
            "tool_calls": 0,
        },
        "measurements": {
            key: None
            for key in (
                "wall_seconds",
                "inference_seconds",
                "tool_seconds",
                "setup_seconds",
                "cpu_seconds",
                "gpu_seconds",
                "peak_ram_bytes",
                "peak_vram_bytes",
                "cpu_utilization",
                "gpu_utilization",
                "energy_joules",
            )
        },
        "measurement_sources": {},
        "artifacts": {},
    }


class ExperimentTests(unittest.TestCase):
    def test_unknown_telemetry_is_incomparable(self):
        a, b = result(10), result(5)
        self.assertEqual(compare(a, b)["generated_tokens"], "candidate_dominates")
        self.assertEqual(compare(a, b)["wall_seconds"], "unavailable")

    def test_environment_or_fixture_drift_is_rejected(self):
        for key in ("hardware", "software", "task", "protocol"):
            a, b = result(), result()
            b[key] = {"different": True}
            with self.assertRaises(ValueError):
                compare(a, b)

    def test_invalid_contract_is_rejected(self):
        for field, value in (("generated_tokens", -1), ("retries", 0.5)):
            r = result()
            r["work"][field] = value
            with self.assertRaises(ValueError):
                validate_result(r)
        r = result()
        r["measurements"]["wall_seconds"] = 1
        with self.assertRaises(ValueError):
            validate_result(r)

    def test_configuration_changes_are_visible(self):
        a, b = result(), result()
        b["configuration"]["model"] = "other"
        self.assertEqual(compare(a, b)["configuration_changes"], ["model"])

    def test_repeat_summary_reports_variance_and_missing_samples(self):
        a, b = result(10), copy.deepcopy(result(20))
        b["protocol"]["repeat"] = 1
        report = summarize([a, b])
        self.assertEqual(report["generated_tokens"]["mean"], 15)
        self.assertEqual(report["generated_tokens"]["samples"], 2)
        self.assertIsNone(report["wall_seconds"]["mean"])
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))

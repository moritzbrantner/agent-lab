import io
import json
import unittest
from unittest.mock import patch

from test_experiments import result

from agent_lab.baselines import (
    baseline_report,
    unload_experiment_models,
    validate_manifest,
)


class BaselineTests(unittest.TestCase):
    def test_deferred_unload_waits_for_confirmed_isolation(self):
        class Opener:
            def __init__(self):
                self.responses = iter(
                    [
                        {"models": [{"name": "ours"}]},
                        {"models": [{"name": "ours"}]},
                        {"models": []},
                    ]
                )

            def open(self, *args, **kwargs):
                return io.StringIO(json.dumps(next(self.responses)))

        with (
            patch(
                "agent_lab.baselines.urllib.request.build_opener", return_value=Opener()
            ),
            patch("agent_lab.baselines.unload") as unload,
            patch("agent_lab.baselines.time.sleep") as sleep,
        ):
            unload_experiment_models({"ours"})
            unload.assert_called_once_with("ours")
            sleep.assert_called_once_with(0.2)

    def test_report_keeps_failures_and_repeated_variance(self):
        runs = []
        for repeat in range(5):
            run = result(10 + repeat)
            run["protocol"]["repeat"] = repeat
            run["configuration"]["seed"] = repeat
            run["configuration"]["backend"] = "ollama"
            run["configuration"]["model_digest"] = "b" * 64
            run["configuration"]["quantization"] = "Q4_K_M"
            runs.append(run)
        runs[0]["correctness"] = {"status": "fail", "score": 0, "evaluator": "exact-v1"}
        report = baseline_report(runs)
        self.assertEqual(report[0]["samples"], 5)
        self.assertEqual(report[0]["passed"], 4)
        self.assertEqual(report[0]["summary"]["generated_tokens"]["mean"], 12)
        with self.assertRaises(ValueError):
            baseline_report(runs[:1])

    def test_manifest_rejects_wrong_model_identity(self):
        config = {
            "model": "qwen",
            "model_digest": "a" * 64,
            "backend_version": "1",
            "quantization": "Q4",
        }
        tags = {
            "models": [
                {
                    "name": "qwen",
                    "digest": "a" * 64,
                    "details": {"quantization_level": "Q4"},
                }
            ]
        }
        validate_manifest(config, tags, "1")
        with self.assertRaises(ValueError):
            validate_manifest(config, tags, "2")
        tags["models"][0]["digest"] = "b" * 64
        with self.assertRaises(ValueError):
            validate_manifest(config, tags, "1")

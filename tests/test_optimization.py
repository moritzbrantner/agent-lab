import unittest

from agent_lab.backends import OllamaAdapter
from agent_lab.optimization import negotiate
from agent_lab.runtime import AgentConfig, SequenceAdapter


class OptimizationTests(unittest.TestCase):
    def test_supported_context_and_offload_are_adapter_configuration(self):
        config, report = negotiate(
            OllamaAdapter(),
            AgentConfig(),
            {"context_size": 2048, "gpu_layers": 0, "speculation": True},
        )
        self.assertEqual(config.context_size, 2048)
        self.assertEqual(config.options["num_gpu"], 0)
        self.assertEqual(report["speculation"]["status"], "unavailable")
        with self.assertRaises(ValueError):
            negotiate(OllamaAdapter(), config, {"context_size": -1})

    def test_adapter_without_features_keeps_correct_fallback(self):
        original = AgentConfig()
        config, report = negotiate(
            SequenceAdapter([]), original, {"context_size": 2048}
        )
        self.assertEqual(config, original)
        self.assertEqual(report["context_size"]["status"], "unavailable")

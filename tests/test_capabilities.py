import json
import tempfile
import unittest
from pathlib import Path

from agent_lab.benchmarks import benchmark, load_task
from agent_lab.capabilities import CapabilityRunner, discover_capabilities
from agent_lab.runtime import AgentConfig


class CapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_helper_bypasses_inference_with_independent_correctness(self):
        capability = discover_capabilities()[0]
        task = load_task("development", "batch-sum")
        config = AgentConfig(
            model="none",
            model_digest=capability.identity,
            backend="capability",
            options={"capability": capability.id},
        )
        result = await benchmark(
            task, None, config, executor=CapabilityRunner(capability, task["input"])
        )
        self.assertEqual(result["correctness"]["status"], "pass")
        self.assertEqual(result["work"]["model_calls"], 0)
        self.assertEqual(result["work"]["generated_tokens"], 0)
        self.assertEqual(result["work"]["tool_calls"], 1)

    def test_cache_provenance_and_corruption_do_not_change_output(self):
        capability = discover_capabilities()[0]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first = capability.invoke(
                {"batches": [[1, 2], [3]]}, cache=cache, source_runs=["verified-run"]
            )
            second = capability.invoke({"batches": [[1, 2], [3]]}, cache=cache)
            self.assertEqual(first["output"], "6")
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(second["source_runs"], ["verified-run"])
            file = next(cache.glob("*.json"))
            data = json.loads(file.read_text())
            data["output"] = "999"
            file.write_text(json.dumps(data))
            self.assertEqual(
                capability.invoke({"batches": [[1, 2], [3]]}, cache=cache)["output"],
                "6",
            )
            file.write_text("[]")
            self.assertEqual(
                capability.invoke({"batches": [[1, 2], [3]]}, cache=cache)["output"],
                "6",
            )
        with self.assertRaises(ValueError):
            capability.invoke({"batches": [["untrusted"]]})

import unittest

from agent_lab.benchmarks import load_task
from agent_lab.generation import generate_verify
from agent_lab.runtime import AgentConfig, Reply, SequenceAdapter


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_candidate_and_verifier_cost_are_retained(self):
        task = load_task("development", "sum")
        configs = [AgentConfig(model="wrong"), AgentConfig(model="right")]

        def factory(config):
            return SequenceAdapter(
                [
                    Reply(
                        "0" if config.model == "wrong" else "49",
                        input_tokens=2,
                        output_tokens=3,
                    )
                ]
            )

        report = await generate_verify(task, configs, factory)
        self.assertEqual(report["result"]["correctness"]["status"], "pass")
        self.assertEqual(report["result"]["work"]["generated_tokens"], 6)
        self.assertEqual(report["verification"]["calls"], 2)
        self.assertEqual(len(report["candidates"]), 2)
        self.assertEqual(report["selected_index"], 1)
        self.assertIn("failed", report["candidates"][1]["configuration"]["feedback"])

    async def test_parallel_selection_is_schedule_order_and_budget_is_bounded(self):
        task = load_task("development", "sum")
        configs = [AgentConfig(model=str(i)) for i in range(3)]
        report = await generate_verify(
            task,
            configs,
            lambda _: SequenceAdapter([Reply("49", input_tokens=1, output_tokens=1)]),
            max_parallel=2,
        )
        self.assertEqual(len(report["candidates"]), 2)
        self.assertEqual(report["selected_index"], 0)
        self.assertEqual(report["result"]["work"]["model_calls"], 2)
        with self.assertRaises(ValueError):
            await generate_verify(task, configs, lambda _: None, max_parallel=0)
